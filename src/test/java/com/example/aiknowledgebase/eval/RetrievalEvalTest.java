package com.example.aiknowledgebase.eval;

import com.example.aiknowledgebase.config.RagProperties;
import com.example.aiknowledgebase.model.SearchResult;
import com.example.aiknowledgebase.service.EmbeddingService;
import com.example.aiknowledgebase.service.RagService;
import com.example.aiknowledgebase.store.VectorStore;
import com.example.aiknowledgebase.util.TextSplitter;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.env.Environment;
import org.springframework.test.context.ActiveProfiles;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.example.aiknowledgebase.eval.EvalMetrics.Level.CHUNK;
import static com.example.aiknowledgebase.eval.EvalMetrics.Level.DOC;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * P0 检索评估基线 —— 在固定数据集、固定 embedding 模型、固定检索配置下产出可重复比较的数字。
 * <p>
 * 隔离：{@code @ActiveProfiles("eval")} → Redis DB15 + {@code rag-eval} 前缀，
 * 与生产（DB0 / {@code rag}）和 Step3VerificationTest（DB15 / {@code rag-test}）互不干扰。
 * <p>
 * 本类只走生产代码路径（{@link RagService#indexDocument} / {@link EmbeddingService#embed} /
 * {@link VectorStore#search}），不重新实现切块与检索，否则测的不是生产行为。
 * <p>
 * 依赖：本地 Redis 与 Ollama 均已启动，且 {@code nomic-embed-text:latest} 已 pull。
 * 本测试不打进 CI：{@code @Tag("eval")}，因为环境不可重复。
 */
@SpringBootTest
@ActiveProfiles("eval")
@Tag("eval")
class RetrievalEvalTest {

    private static final Logger log = LoggerFactory.getLogger(RetrievalEvalTest.class);

    /**
     * 评测检索深度固定为 5：一次检索即可同时得出 @1/@3/@5。
     * 依据 {@link VectorStore#search} 的实现是"先按阈值过滤 → 全量排序 → 取前 topK"，
     * 排序结果与 topK 无关，因此 top-5 的前 3 条与直接用 topK=3 检索完全一致，
     * 也不会因为评测用了 5 而改变生产 {@code rag.top-k=3} 的口径。
     */
    private static final int EVAL_DEPTH = 5;

    /** 预热次数：首次调用包含连接建立与服务端模型加载，必须丢弃后再计时 */
    private static final int WARMUP_ROUNDS = 3;

    /** 样本规模是评测集的一部分：改样本集必须同步改这里，并重新归档基线报告 */
    private static final int EXPECTED_ANSWERABLE = 30;
    private static final int EXPECTED_UNANSWERABLE = 10;

    /** 生成层拒答观察中，正样本抽样条数（按 id 顺序取前 N 条，不用随机抽样，否则不可重复） */
    private static final int REFUSAL_POSITIVE_SAMPLE = 10;

    private static final String REPORT_DIR = "target/eval";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Autowired
    private EmbeddingService embeddingService;

    @Autowired
    private VectorStore vectorStore;

    @Autowired
    private RagService ragService;

    @Autowired
    private RagProperties ragProperties;

    @Autowired
    private Environment environment;

    private EvalDataset dataset;
    private int embeddingDimension;
    private int chunkCount;
    private double minScore;

    // ==================== 索引准备 ====================

    @BeforeEach
    void buildIndex() {
        dataset = EvalDataset.load();

        // 隔离保险丝：评测只允许落在 rag-eval 前缀上。若 profile 没生效，
        // 这里会直接失败，而不是把生产知识库清空。
        String keyPrefix = ragProperties.getRedis().getKeyPrefix();
        assertTrue(keyPrefix != null && keyPrefix.startsWith("rag-eval"),
                "评测必须使用 rag-eval 前缀，当前为 " + keyPrefix
                        + "：请确认 @ActiveProfiles(\"eval\") 生效且加载了 application-eval.properties");

        // 清库后全量重建：保证库里只有本次语料，不含上次运行的残留（残留会污染召回与延迟）
        vectorStore.clear();
        int chunkSize = ragProperties.getChunkSize();
        int overlap = ragProperties.getChunkOverlap();
        int expectedChunks = 0;
        int indexed = 0;
        for (Map.Entry<String, String> doc : dataset.corpus().entrySet()) {
            expectedChunks += TextSplitter.split(doc.getValue(), chunkSize, overlap).size();
            indexed += ragService.indexDocument(doc.getKey(), doc.getValue());
        }
        chunkCount = expectedChunks;

        assertEquals(expectedChunks, indexed,
                "有文本块 embedding 失败，索引不完整（检索指标会因此失真）");
        assertEquals(expectedChunks, vectorStore.size(),
                "库内文本块数与语料不一致，可能存在残留数据");
        assertEquals(EXPECTED_ANSWERABLE, dataset.answerable().size(), "answerable 样本数不符");
        assertEquals(EXPECTED_UNANSWERABLE, dataset.unanswerable().size(), "unanswerable 样本数不符");

        minScore = ragProperties.getMinScore();
        log.info("评测索引就绪: 语料 {} 篇 / {} 块, 样本 {} 条(answerable={}, unanswerable={}), "
                        + "keyPrefix={}, minScore={}, chunkSize={}, overlap={}",
                dataset.corpus().size(), chunkCount, dataset.cases().size(),
                dataset.answerable().size(), dataset.unanswerable().size(),
                keyPrefix, minScore, chunkSize, overlap);
    }

    // ==================== 主基线 ====================

    @Test
    @DisplayName("检索基线：Doc/Chunk 的 HitRate 与 MRR、延迟分位、top1 分数分布")
    void retrievalBaseline() throws IOException {
        // ── 预热：只用于取维度并让连接就绪，结果一律丢弃 ──────────────
        EvalDataset.Case probe = dataset.answerable().get(0);
        for (int i = 0; i < WARMUP_ROUNDS; i++) {
            float[] warmupEmbedding = embeddingService.embed(probe.question());
            embeddingDimension = warmupEmbedding.length;
            vectorStore.search(warmupEmbedding, EVAL_DEPTH, minScore);
        }

        // ── 逐条检索并计时 ────────────────────────────────────────
        List<EvalMetrics.RetrievalRecord> answerableRecords = new ArrayList<>();
        List<Double> answerableTop1 = new ArrayList<>();
        List<Double> unanswerableTop1 = new ArrayList<>();
        List<Double> embedMillis = new ArrayList<>();
        List<Double> searchMillis = new ArrayList<>();
        List<Double> totalMillis = new ArrayList<>();
        List<Map<String, Object>> details = new ArrayList<>();
        int negativeNoRecall = 0;

        for (EvalDataset.Case evalCase : dataset.cases()) {
            long t0 = System.nanoTime();
            float[] queryEmbedding = embeddingService.embed(evalCase.question());
            long t1 = System.nanoTime();
            List<SearchResult> hits = vectorStore.search(queryEmbedding, EVAL_DEPTH, minScore);
            long t2 = System.nanoTime();

            double embedCost = (t1 - t0) / 1_000_000.0;
            double searchCost = (t2 - t1) / 1_000_000.0;
            embedMillis.add(embedCost);
            searchMillis.add(searchCost);
            totalMillis.add(embedCost + searchCost);

            List<String> topDocIds = new ArrayList<>();
            List<String> topChunkIds = new ArrayList<>();
            List<Double> topScores = new ArrayList<>();
            for (SearchResult hit : hits) {
                topDocIds.add(hit.getChunk().getDocumentId());
                topChunkIds.add(hit.getChunk().getId());
                topScores.add(hit.getScore());
            }

            EvalMetrics.RetrievalRecord record = new EvalMetrics.RetrievalRecord(
                    evalCase.id(), evalCase.goldDocIds(), evalCase.goldChunkIds(),
                    topDocIds, topChunkIds, topScores);

            Double top1 = record.top1Score();
            if (evalCase.answerable()) {
                answerableRecords.add(record);
                if (top1 != null) {
                    answerableTop1.add(top1);
                }
            } else {
                if (hits.isEmpty()) {
                    negativeNoRecall++;
                }
                if (top1 != null) {
                    unanswerableTop1.add(top1);
                }
            }

            Map<String, Object> detail = new LinkedHashMap<>();
            detail.put("caseId", evalCase.id());
            detail.put("type", evalCase.type());
            detail.put("question", evalCase.question());
            detail.put("top1DocId", topDocIds.isEmpty() ? null : topDocIds.get(0));
            detail.put("top1ChunkId", topChunkIds.isEmpty() ? null : topChunkIds.get(0));
            detail.put("top1Score", top1);
            detail.put("docRank", record.firstRelevantRank(DOC, EVAL_DEPTH));
            detail.put("chunkRank", record.firstRelevantRank(CHUNK, EVAL_DEPTH));
            details.add(detail);
        }

        // ── 指标 ──────────────────────────────────────────────────
        Map<String, Double> metrics = new LinkedHashMap<>();
        for (EvalMetrics.Level level : EvalMetrics.Level.values()) {
            for (int k : EvalMetrics.K_VALUES) {
                metrics.put(levelPrefix(level) + "HitRate@" + k,
                        EvalMetrics.hitRateAtK(answerableRecords, level, k));
                metrics.put(levelPrefix(level) + "MRR@" + k,
                        EvalMetrics.mrrAtK(answerableRecords, level, k));
            }
        }
        // 自洽性断言：K 增大时命中率与 MRR 只可能不变或变好。违反说明统计实现有 bug，
        // 这类错误在报告上看起来和"检索变差"一模一样，必须在这里拦住。
        for (EvalMetrics.Level level : EvalMetrics.Level.values()) {
            for (String metric : List.of("HitRate", "MRR")) {
                double at1 = metrics.get(levelPrefix(level) + metric + "@1");
                double at3 = metrics.get(levelPrefix(level) + metric + "@3");
                double at5 = metrics.get(levelPrefix(level) + metric + "@5");
                assertTrue(at1 <= at3 + 1e-9 && at3 <= at5 + 1e-9,
                        levelPrefix(level) + metric + " 随 K 的单调性被破坏: @1=" + at1 + ", @3=" + at3 + ", @5=" + at5);
            }
        }

        // ── 分数分布与延迟 ────────────────────────────────────────
        EvalMetrics.ScoreSummary answerableSummary = EvalMetrics.ScoreSummary.of(answerableTop1);
        EvalMetrics.ScoreSummary unanswerableSummary = EvalMetrics.ScoreSummary.of(unanswerableTop1);
        double[] thresholdInterval = EvalMetrics.exploratoryThresholdInterval(unanswerableTop1, answerableTop1);

        Map<String, Double> latency = new LinkedHashMap<>();
        reportLatency(latency, "embed", embedMillis);
        reportLatency(latency, "search", searchMillis);
        reportLatency(latency, "total", totalMillis);

        // ── 报告 ──────────────────────────────────────────────────
        Map<String, Object> json = new LinkedHashMap<>();
        json.put("meta", baselineMeta("检索基线（主表）"));
        json.put("metrics", metrics);
        json.put("latency", latency);
        json.put("scoreDistribution", Map.of(
                "answerable", summaryToMap(answerableSummary),
                "unanswerable", summaryToMap(unanswerableSummary),
                "exploratoryThresholdInterval", thresholdInterval == null
                        ? "n/a" : List.of(thresholdInterval[0], thresholdInterval[1])));
        String refusalNote = minScore <= RagProperties.MIN_SCORE_DISABLED
                ? "minScore=" + minScore + " 时 VectorStore.search 的阈值过滤从不生效，只要库非空必有结果，"
                        + "故 noRecall 必然为 0：当前配置下检索层没有拒答能力，拒答完全依赖生成模型。"
                        + "这是本阶段要暴露的事实，不是缺陷。"
                : "minScore=" + minScore + " 时阈值过滤已启用，noRecall 反映负样本 top-K 被阈值清空的比例。";
        json.put("retrievalLayerRefusal", Map.of(
                "negativeTotal", dataset.unanswerable().size(),
                "noRecall", negativeNoRecall,
                "noRecallRate", (double) negativeNoRecall / dataset.unanswerable().size(),
                "note", refusalNote));
        json.put("details", details);

        StringBuilder md = new StringBuilder();
        appendHeader(md, "RAG 检索评测基线（Stage 2 / P0）",
                "本报告只回答一个问题：在固定数据集、固定 embedding 模型、固定检索配置下，检索质量与延迟是多少。",
                "不含生成质量评测；不含 Hybrid Search、Rerank、Query Rewrite、ANN 的任何改动。");
        appendSection(md, "1. 元信息", baselineMeta("检索基线（主表）"));
        appendMetricsTable(md, metrics);
        appendLatencyTable(md, latency, dataset.cases().size(), WARMUP_ROUNDS, chunkCount);
        appendDistribution(md, answerableSummary, unanswerableSummary, thresholdInterval, minScore);
        appendRetrievalRefusal(md, dataset.unanswerable().size(), negativeNoRecall, minScore);
        appendDetails(md, details);

        Path reportPath = writeReport("retrieval-baseline", md, json);

        log.info("{}", metrics);
        log.info("延迟(ms) embed.p50={} embed.p95={} search.p50={} search.p95={} total.p50={} total.p95={}",
                EvalMetrics.format(latency.get("embed.p50"), 2), EvalMetrics.format(latency.get("embed.p95"), 2),
                EvalMetrics.format(latency.get("search.p50"), 2), EvalMetrics.format(latency.get("search.p95"), 2),
                EvalMetrics.format(latency.get("total.p50"), 2), EvalMetrics.format(latency.get("total.p95"), 2));
        log.info("检索基线报告已写入: {}", reportPath.toAbsolutePath());
    }

    // ==================== 生成层拒答（观察项，不参与验收） ====================

    @Test
    @DisplayName("[观察项] 生成层拒答：RefusalAccuracy / FalseRefusalRate（非确定性，不参与检索基线）")
    void refusalObservation() throws IOException {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (EvalDataset.Case evalCase : dataset.unanswerable()) {
            rows.add(askOnce(evalCase));
        }
        for (EvalDataset.Case evalCase : dataset.answerable().subList(0, REFUSAL_POSITIVE_SAMPLE)) {
            rows.add(askOnce(evalCase));
        }

        long negativeTotal = 0;
        long negativeRefused = 0;
        long positiveTotal = 0;
        long positiveRefused = 0;
        long failed = 0;
        for (Map<String, Object> row : rows) {
            if (row.get("refused") == null) {
                failed++;
                continue;
            }
            boolean unanswerable = EvalDataset.TYPE_UNANSWERABLE.equals(row.get("type"));
            if (unanswerable) {
                negativeTotal++;
                if (Boolean.TRUE.equals(row.get("refused"))) {
                    negativeRefused++;
                }
            } else {
                positiveTotal++;
                if (Boolean.TRUE.equals(row.get("refused"))) {
                    positiveRefused++;
                }
            }
        }

        double refusalAccuracy = negativeTotal == 0 ? Double.NaN : (double) negativeRefused / negativeTotal;
        double falseRefusalRate = positiveTotal == 0 ? Double.NaN : (double) positiveRefused / positiveTotal;

        if (failed == rows.size()) {
            log.error("生成层拒答观察：全部 {} 条调用失败，请确认 Ollama 与生成模型是否可用", rows.size());
        }

        Map<String, Object> json = new LinkedHashMap<>();
        json.put("meta", baselineMeta("生成层拒答观察（observational，不参与验收）"));
        json.put("refusalAccuracy", num(refusalAccuracy));
        json.put("falseRefusalRate", num(falseRefusalRate));
        json.put("negativeTotal", negativeTotal);
        json.put("negativeRefused", negativeRefused);
        json.put("positiveTotal", positiveTotal);
        json.put("positiveRefused", positiveRefused);
        json.put("failed", failed);
        json.put("rows", rows);

        StringBuilder md = new StringBuilder();
        appendHeader(md, "生成层拒答观察（observational）",
                "本文件是**观察项**：OllamaClient.generate 未传 options，采样参数由服务端模型默认值决定，"
                        + "同一条问题两次运行可能得到不同回答，因此这些数字不具备严格的可重复性。",
                "它**不是 Stage 2 的验收对象**，也不得混入检索基线主表；检索基线是否通过只取决于 RetrievalEvalTest#retrievalBaseline。");
        appendSection(md, "1. 元信息", baselineMeta("生成层拒答观察（observational，不参与验收）"));
        md.append("## 2. 拒答指标\n\n");
        md.append("| 指标 | 值 | 说明 |\n|---|---|---|\n");
        md.append("| RefusalAccuracy | ").append(EvalMetrics.format(refusalAccuracy, 4))
                .append(" | ").append(negativeRefused).append("/").append(negativeTotal)
                .append("，10 条负样本中被拒答的比例 |\n");
        md.append("| FalseRefusalRate | ").append(EvalMetrics.format(falseRefusalRate, 4))
                .append(" | ").append(positiveRefused).append("/").append(positiveTotal)
                .append("，正样本被误拒的比例（必须与上一行成对阅读）|\n");
        md.append("| 调用失败 | ").append(failed).append(" | 异常样本已记录在明细中，不计入上述分母 |\n\n");
        md.append("## 3. 逐条明细\n\n");
        md.append("| caseId | type | 是否拒答 | 耗时(ms) | 回答摘要 / 错误 |\n|---|---|---|---|---|\n");
        for (Map<String, Object> row : rows) {
            md.append("| ").append(row.get("caseId"))
                    .append(" | ").append(row.get("type"))
                    .append(" | ").append(row.get("refused"))
                    .append(" | ").append(row.get("latencyMs") == null ? "-"
                            : EvalMetrics.format((Double) row.get("latencyMs"), 1))
                    .append(" | ").append(cell(row.get("answerExcerpt") != null
                            ? row.get("answerExcerpt") : row.get("error")))
                    .append(" |\n");
        }

        Path reportPath = writeReport("refusal-observation", md, json);
        log.info("生成层拒答观察: RefusalAccuracy={}, FalseRefusalRate={}（非确定性，仅供参考）",
                EvalMetrics.format(refusalAccuracy, 4), EvalMetrics.format(falseRefusalRate, 4));
        log.info("拒答观察报告已写入: {}", reportPath.toAbsolutePath());
    }

    private Map<String, Object> askOnce(EvalDataset.Case evalCase) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("caseId", evalCase.id());
        row.put("type", evalCase.type());
        row.put("question", evalCase.question());
        try {
            long t0 = System.nanoTime();
            String answer = ragService.ask(evalCase.question());
            row.put("latencyMs", (System.nanoTime() - t0) / 1_000_000.0);
            row.put("refused", EvalMetrics.isRefusal(answer));
            row.put("answerExcerpt", excerpt(answer));
        } catch (Exception e) {
            // 生成层失败不抛出：它不该影响检索基线的结论
            row.put("refused", null);
            row.put("error", e.getClass().getSimpleName() + ": " + e.getMessage());
            log.warn("生成层观察样本 {} 调用失败（不影响检索基线）: {}", evalCase.id(), e.getMessage());
        }
        return row;
    }

    // ==================== 报告构建 ====================

    private Map<String, Object> baselineMeta(String scope) {
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("scope", scope);
        meta.put("runAt", LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        meta.put("gitCommit", "（人工归档时补充）");
        meta.put("datasetHash", dataset.hash());
        meta.put("corpusDocs", String.join(", ", dataset.corpus().keySet()));
        meta.put("corpusDocCount", dataset.corpus().size());
        meta.put("corpusChunkCount", chunkCount);
        meta.put("samplesAnswerable", dataset.answerable().size());
        meta.put("samplesUnanswerable", dataset.unanswerable().size());
        meta.put("embeddingModel", environment.getProperty("ollama.embedding-model"));
        meta.put("embeddingDimension", embeddingDimension);
        meta.put("chunkSize", ragProperties.getChunkSize());
        meta.put("chunkOverlap", ragProperties.getChunkOverlap());
        meta.put("evalDepth", EVAL_DEPTH);
        meta.put("productionTopK", ragProperties.getTopK());
        meta.put("minScore", minScore);
        meta.put("minScoreFilterEnabled", ragProperties.isMinScoreEnabled());
        meta.put("redisDatabase", environment.getProperty("spring.data.redis.database"));
        meta.put("redisKeyPrefix", ragProperties.getRedis().getKeyPrefix());
        meta.put("ollamaBaseUrl", environment.getProperty("ollama.base-url"));
        meta.put("generationModel", environment.getProperty("ollama.model"));
        meta.put("javaVersion", System.getProperty("java.version"));
        meta.put("os", System.getProperty("os.name") + " " + System.getProperty("os.version"));
        meta.putAll(probeEmbeddingModelInfo());
        return meta;
    }

    /**
     * 取 embedding 模型的 digest 等信息。
     * <p>
     * 只用测试端的一次 HTTP GET（{@code /api/tags}），不改动 OllamaClient：
     * {@code latest} 这类 tag 是可变的，只记 tag 无法说明"这份报告到底跑的哪个模型"。
     * 探测失败不抛异常，如实标记 unavailable，由人工归档时补齐。
     */
    private Map<String, Object> probeEmbeddingModelInfo() {
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("embeddingModelDigest", "unavailable");
        String baseUrl = environment.getProperty("ollama.base-url");
        String configured = environment.getProperty("ollama.embedding-model");
        if (baseUrl == null || configured == null) {
            info.put("embeddingModelProbeNote", "ollama.base-url 或 ollama.embedding-model 未配置");
            return info;
        }
        try {
            HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
            HttpRequest request = HttpRequest.newBuilder(URI.create(trimTrailingSlash(baseUrl) + "/api/tags"))
                    .timeout(Duration.ofSeconds(3))
                    .GET()
                    .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                info.put("embeddingModelProbeNote", "/api/tags 返回 HTTP " + response.statusCode());
                return info;
            }
            JsonNode models = MAPPER.readTree(response.body()).path("models");
            for (JsonNode model : models) {
                String name = model.path("name").asText("");
                String alias = model.path("model").asText("");
                if (configured.equals(name) || configured.equals(alias)) {
                    info.put("embeddingModelDigest", model.path("digest").asText("unavailable"));
                    info.put("embeddingModelQuantization",
                            model.path("details").path("quantization_level").asText("unavailable"));
                    info.put("embeddingModelParameterSize",
                            model.path("details").path("parameter_size").asText("unavailable"));
                    return info;
                }
            }
            info.put("embeddingModelProbeNote", "/api/tags 未找到模型 " + configured);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            info.put("embeddingModelProbeNote", "探测被中断");
        } catch (Exception e) {
            info.put("embeddingModelProbeNote", "探测失败: " + e.getClass().getSimpleName() + " " + e.getMessage());
        }
        return info;
    }

    private void reportLatency(Map<String, Double> target, String name, List<Double> values) {
        target.put(name + ".p50", EvalMetrics.percentileNearestRank(values, 50));
        target.put(name + ".p95", EvalMetrics.percentileNearestRank(values, 95));
    }

    private static void appendHeader(StringBuilder md, String title, String line1, String line2) {
        md.append("# ").append(title).append("\n\n");
        md.append("> ").append(line1).append("\n>\n> ").append(line2).append("\n\n");
    }

    private static void appendSection(StringBuilder md, String title, Map<String, Object> values) {
        md.append("## ").append(title).append("\n\n");
        md.append("| 项 | 值 |\n|---|---|\n");
        values.forEach((key, value) -> md.append("| ").append(key).append(" | ").append(value).append(" |\n"));
        md.append('\n');
    }

    private static void appendMetricsTable(StringBuilder md, Map<String, Double> metrics) {
        md.append("## 2. 检索指标（主表，分母 = ").append(EXPECTED_ANSWERABLE).append(" 条 answerable）\n\n");
        md.append("| 指标 | @1 | @3 | @5 |\n|---|---|---|---|\n");
        for (EvalMetrics.Level level : EvalMetrics.Level.values()) {
            for (String metric : List.of("HitRate", "MRR")) {
                md.append("| ").append(levelPrefix(level)).append(metric);
                for (int k : EvalMetrics.K_VALUES) {
                    md.append(" | ").append(EvalMetrics.format(metrics.get(levelPrefix(level) + metric + "@" + k), 4));
                }
                md.append(" |\n");
            }
        }
        md.append("\n- `Doc*`：命中的文本块属于 gold 文档（是否定位到正确文档）\n");
        md.append("- `Chunk*`：命中的文本块本身就是 gold 文本块（真正含答案的上下文是否被召回）；")
                .append("gold 文本块允许多个，用于覆盖答案跨切分边界的情况\n");
        md.append("- MRR 中名次为 0 表示前 K 条内没有任何相关结果\n\n");
    }

    private static void appendLatencyTable(StringBuilder md, Map<String, Double> latency,
                                           int sampleCount, int warmupRounds, int chunkCount) {
        md.append("## 3. 延迟（单线程，N=").append(sampleCount)
                .append("，含 HTTP 往返，已丢弃 ").append(warmupRounds).append(" 次预热）\n\n");
        md.append("| 量 | p50 (ms) | p95 (ms) |\n|---|---|---|\n");
        for (String name : List.of("embed", "search", "total")) {
            md.append("| ").append(name)
                    .append(" | ").append(EvalMetrics.format(latency.get(name + ".p50"), 2))
                    .append(" | ").append(EvalMetrics.format(latency.get(name + ".p95"), 2))
                    .append(" |\n");
        }
        md.append("\n- 分位数用 nearest-rank 法（升序第 ceil(p/100×N) 个），算法固定，两次运行可比\n");
        md.append("- 延迟随语料块数增长（本次 ").append(chunkCount)
                .append(" 块），语料规模变化后延迟不可比\n\n");
    }

    private static void appendDistribution(StringBuilder md,
                                           EvalMetrics.ScoreSummary answerable,
                                           EvalMetrics.ScoreSummary unanswerable,
                                           double[] thresholdInterval, double minScore) {
        md.append("## 4. top-1 分数分布（minScore=").append(minScore).append("，")
                .append(minScore <= RagProperties.MIN_SCORE_DISABLED ? "未启用阈值过滤" : "已启用阈值过滤")
                .append("）\n\n");
        md.append("| 样本类型 | n | min | p25 | p50 | p75 | max |\n|---|---|---|---|---|---|---|\n");
        md.append("| answerable | ").append(answerable.n())
                .append(" | ").append(EvalMetrics.format(answerable.min(), 4))
                .append(" | ").append(EvalMetrics.format(answerable.p25(), 4))
                .append(" | ").append(EvalMetrics.format(answerable.p50(), 4))
                .append(" | ").append(EvalMetrics.format(answerable.p75(), 4))
                .append(" | ").append(EvalMetrics.format(answerable.max(), 4)).append(" |\n");
        md.append("| unanswerable | ").append(unanswerable.n())
                .append(" | ").append(EvalMetrics.format(unanswerable.min(), 4))
                .append(" | ").append(EvalMetrics.format(unanswerable.p25(), 4))
                .append(" | ").append(EvalMetrics.format(unanswerable.p50(), 4))
                .append(" | ").append(EvalMetrics.format(unanswerable.p75(), 4))
                .append(" | ").append(EvalMetrics.format(unanswerable.max(), 4)).append(" |\n\n");
        md.append("探索性阈值区间（exploratory threshold interval）：");
        if (thresholdInterval == null) {
            md.append("n/a\n\n");
        } else {
            md.append("[p90(unanswerable)=").append(EvalMetrics.format(thresholdInterval[0], 4))
                    .append(", p10(answerable)=").append(EvalMetrics.format(thresholdInterval[1], 4))
                    .append("]\n\n");
        }
        md.append("> ⚠ 该区间只是「值得拿去做对比实验」的方向，**不是推荐阈值，也不是最佳阈值**。\n");
        md.append("> 本阶段固定 `rag.min-score=-1`，不据此修改任何配置；正式阈值选择留到后续对比实验。\n\n");
    }

    private static void appendRetrievalRefusal(StringBuilder md, int negativeTotal,
                                               int noRecall, double minScore) {
        md.append("## 5. 检索层拒答观察（确定性，不属于主表）\n\n");
        md.append("- NoRecallRate（负样本 top-K 为空的比例）= ").append(noRecall).append("/").append(negativeTotal)
                .append(" = ").append(EvalMetrics.format((double) noRecall / negativeTotal, 4)).append("\n");
        md.append("- 说明：minScore=").append(minScore);
        if (minScore <= RagProperties.MIN_SCORE_DISABLED) {
            md.append(" 时 `VectorStore.search` 的阈值过滤从不生效，只要库非空就一定有结果，")
                    .append("该值必然为 0，也就是说**当前配置下检索层没有拒答能力**，")
                    .append("拒答完全依赖生成模型遵守 prompt 规则。这是本阶段要暴露的事实，不是缺陷。\n\n");
        } else {
            md.append(" 时阈值过滤已启用，该值反映的是「负样本 top-K 被阈值清空」的比例，")
                    .append("与上面的分数分布对照阅读即可。\n\n");
        }
    }

    private static void appendDetails(StringBuilder md, List<Map<String, Object>> details) {
        md.append("## 6. 逐条明细\n\n");
        md.append("> docRank / chunkRank：前 ").append(EVAL_DEPTH)
                .append(" 条内首个相关结果的名次，0 表示未命中\n\n");
        md.append("| caseId | type | docRank | chunkRank | top1 文档 | top1 分数 | 问题 |\n");
        md.append("|---|---|---|---|---|---|---|\n");
        for (Map<String, Object> detail : details) {
            Object score = detail.get("top1Score");
            md.append("| ").append(detail.get("caseId"))
                    .append(" | ").append(detail.get("type"))
                    .append(" | ").append(detail.get("docRank"))
                    .append(" | ").append(detail.get("chunkRank"))
                    .append(" | ").append(detail.get("top1DocId"))
                    .append(" | ").append(score == null ? "-" : EvalMetrics.format((Double) score, 4))
                    .append(" | ").append(cell(detail.get("question")))
                    .append(" |\n");
        }
    }

    private Path writeReport(String baseName, StringBuilder markdown, Map<String, Object> json) throws IOException {
        Path dir = Paths.get(REPORT_DIR);
        Files.createDirectories(dir);
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        Path markdownPath = dir.resolve(timestamp + "-" + baseName + ".md");
        Path jsonPath = dir.resolve(timestamp + "-" + baseName + ".json");
        Files.writeString(markdownPath, markdown.toString(), StandardCharsets.UTF_8);
        Files.writeString(jsonPath, MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(json),
                StandardCharsets.UTF_8);
        return markdownPath;
    }

    // ==================== 小工具 ====================

    private static String levelPrefix(EvalMetrics.Level level) {
        return level == DOC ? "Doc" : "Chunk";
    }

    private static Map<String, Object> summaryToMap(EvalMetrics.ScoreSummary summary) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("n", summary.n());
        map.put("min", num(summary.min()));
        map.put("p25", num(summary.p25()));
        map.put("p50", num(summary.p50()));
        map.put("p75", num(summary.p75()));
        map.put("max", num(summary.max()));
        return map;
    }

    /** JSON 里不允许出现 NaN，空集统一写成 "n/a" */
    private static Object num(double value) {
        return Double.isNaN(value) ? "n/a" : value;
    }

    /** 表格单元格：转义竖线并压掉换行，避免内容把 Markdown 表格切碎 */
    private static String cell(Object value) {
        if (value == null) {
            return "-";
        }
        return String.valueOf(value).replace("|", "\\|").replaceAll("\\s+", " ");
    }

    private static String excerpt(String text) {
        if (text == null) {
            return null;
        }
        String oneLine = text.replaceAll("\\s+", " ").trim();
        return oneLine.length() <= 80 ? oneLine : oneLine.substring(0, 80) + "…";
    }

    private static String trimTrailingSlash(String url) {
        String trimmed = url.trim();
        while (trimmed.endsWith("/")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        return trimmed;
    }
}
