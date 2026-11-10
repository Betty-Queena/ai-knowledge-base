package com.example.aiknowledgebase.eval;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static com.example.aiknowledgebase.eval.EvalMetrics.Level.CHUNK;
import static com.example.aiknowledgebase.eval.EvalMetrics.Level.DOC;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 指标算法的纯单测 —— 不依赖 Spring、Redis、Ollama。
 * <p>
 * 指标算错和"检索变差"在报告上表现一样，所以这里用**手算好答案的样例**把算法钉死。
 * 全部期望值都是手工推导出来的，不能靠跑一遍取数来"对答案"。
 */
class EvalMetricsTest {

    /**
     * 构造一条记录，4 条样本刻意覆盖四种情形：
     * a 文档级与块级都命中 rank1；b 都命中 rank3；c 都没命中；
     * d 文档级命中 rank1 但块级未命中（定位到正确文档、却取错文本块）。
     */
    private static List<EvalMetrics.RetrievalRecord> fixture() {
        return List.of(
                record("a", List.of("doc-1"), List.of("doc-1-chunk-0"),
                        List.of("doc-1", "doc-2", "doc-3"), List.of("doc-1-chunk-0", "doc-2-chunk-0", "doc-3-chunk-0")),
                record("b", List.of("doc-2"), List.of("doc-2-chunk-0"),
                        List.of("doc-1", "doc-3", "doc-2"), List.of("doc-1-chunk-0", "doc-3-chunk-0", "doc-2-chunk-0")),
                record("c", List.of("doc-9"), List.of("doc-9-chunk-0"),
                        List.of("doc-1", "doc-2", "doc-3"), List.of("doc-1-chunk-0", "doc-2-chunk-0", "doc-3-chunk-0")),
                record("d", List.of("doc-1"), List.of("doc-1-chunk-2"),
                        List.of("doc-1", "doc-2", "doc-3"), List.of("doc-1-chunk-0", "doc-2-chunk-0", "doc-3-chunk-0")));
    }

    private static EvalMetrics.RetrievalRecord record(String id,
                                                      List<String> goldDocs, List<String> goldChunks,
                                                      List<String> topDocs, List<String> topChunks) {
        List<Double> scores = new ArrayList<>();
        for (int i = 0; i < topDocs.size(); i++) {
            scores.add(0.9 - i * 0.1);
        }
        return new EvalMetrics.RetrievalRecord(id, goldDocs, goldChunks, topDocs, topChunks, scores);
    }

    @Test
    @DisplayName("HitRate@K：按前 K 条截断，文档级与块级分别统计")
    void hitRate() {
        List<EvalMetrics.RetrievalRecord> records = fixture();

        // 文档级：@1 命中 a、d → 2/4；@3 再命中 b → 3/4
        assertEquals(0.5, EvalMetrics.hitRateAtK(records, DOC, 1), 1e-9);
        assertEquals(0.75, EvalMetrics.hitRateAtK(records, DOC, 3), 1e-9);
        assertEquals(0.75, EvalMetrics.hitRateAtK(records, DOC, 5), 1e-9);

        // 块级：d 虽然定位到正确文档，但取到的是别的块，不计命中
        assertEquals(0.25, EvalMetrics.hitRateAtK(records, CHUNK, 1), 1e-9);
        assertEquals(0.5, EvalMetrics.hitRateAtK(records, CHUNK, 3), 1e-9);
    }

    @Test
    @DisplayName("MRR@K：取首个相关结果名次的倒数，前 K 条内无相关结果记 0")
    void mrr() {
        List<EvalMetrics.RetrievalRecord> records = fixture();

        // 文档级 @3：(1 + 1/3 + 0 + 1) / 4
        assertEquals((1.0 + 1.0 / 3 + 0 + 1) / 4, EvalMetrics.mrrAtK(records, DOC, 3), 1e-9);
        // 文档级 @1：(1 + 0 + 0 + 1) / 4，b 的相关结果在 rank3，被截断
        assertEquals(0.5, EvalMetrics.mrrAtK(records, DOC, 1), 1e-9);
        // 块级 @3：(1 + 1/3 + 0 + 0) / 4
        assertEquals((1.0 + 1.0 / 3) / 4, EvalMetrics.mrrAtK(records, CHUNK, 3), 1e-9);
    }

    @Test
    @DisplayName("firstRelevantRank：1 起计数，未命中返回 0，空结果列表不抛异常")
    void firstRelevantRank() {
        List<EvalMetrics.RetrievalRecord> records = fixture();
        assertEquals(1, records.get(0).firstRelevantRank(DOC, 5));
        assertEquals(3, records.get(1).firstRelevantRank(DOC, 5));
        assertEquals(0, records.get(2).firstRelevantRank(DOC, 5));
        assertEquals(1, records.get(3).firstRelevantRank(DOC, 5));
        assertEquals(0, records.get(3).firstRelevantRank(CHUNK, 5));

        EvalMetrics.RetrievalRecord empty = record("e", List.of("doc-1"), List.of("doc-1-chunk-0"),
                List.of(), List.of());
        assertEquals(0, empty.firstRelevantRank(DOC, 5));
        assertEquals(0, empty.firstRelevantRank(CHUNK, 5));
        assertEquals(null, empty.top1Score());
    }

    @Test
    @DisplayName("分位数：nearest-rank 法与手算结果一致")
    void percentile() {
        List<Double> even = List.of(1.0, 2.0, 3.0, 4.0);
        assertEquals(1.0, EvalMetrics.percentileNearestRank(even, 0), 1e-9);
        assertEquals(1.0, EvalMetrics.percentileNearestRank(even, 25), 1e-9);
        assertEquals(2.0, EvalMetrics.percentileNearestRank(even, 50), 1e-9);
        assertEquals(3.0, EvalMetrics.percentileNearestRank(even, 75), 1e-9);
        assertEquals(4.0, EvalMetrics.percentileNearestRank(even, 100), 1e-9);

        List<Double> odd = List.of(50.0, 10.0, 40.0, 30.0, 20.0);
        assertEquals(30.0, EvalMetrics.percentileNearestRank(odd, 50), 1e-9);
        assertEquals(50.0, EvalMetrics.percentileNearestRank(odd, 95), 1e-9);

        assertTrue(Double.isNaN(EvalMetrics.percentileNearestRank(List.of(), 50)));
    }

    @Test
    @DisplayName("分数分布：五数概括与探索性阈值区间")
    void scoreSummaryAndInterval() {
        EvalMetrics.ScoreSummary summary = EvalMetrics.ScoreSummary.of(List.of(0.1, 0.2, 0.3, 0.4));
        assertEquals(4, summary.n());
        assertEquals(0.1, summary.min(), 1e-9);
        assertEquals(0.4, summary.max(), 1e-9);

        assertEquals(0, EvalMetrics.ScoreSummary.of(List.of()).n());
        assertTrue(Double.isNaN(EvalMetrics.ScoreSummary.of(List.of()).p50()));

        // 负样本 top1 全在 0.2 以下，正样本 top1 都在 0.5 以上 → 区间下界 < 上界
        double[] interval = EvalMetrics.exploratoryThresholdInterval(
                List.of(0.10, 0.15, 0.20), List.of(0.50, 0.60, 0.70));
        assertEquals(0.20, interval[0], 1e-9);
        assertEquals(0.50, interval[1], 1e-9);

        // 任一侧为空时不做任何推断
        assertEquals(null, EvalMetrics.exploratoryThresholdInterval(List.of(), List.of(0.5)));
    }

    @Test
    @DisplayName("拒答判定：命中固定标志串才算拒答")
    void refusalDetection() {
        assertTrue(EvalMetrics.isRefusal("根据提供的资料，无法回答该问题"));
        assertTrue(EvalMetrics.isRefusal("抱歉，知识库中未找到与问题相关的内容。"));
        assertFalse(EvalMetrics.isRefusal("SpringBootApplication 由三个注解组合而成。"));
        assertFalse(EvalMetrics.isRefusal(null));
        assertFalse(EvalMetrics.isRefusal("   "));
    }
}
