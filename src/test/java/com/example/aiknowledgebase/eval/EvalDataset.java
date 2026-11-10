package com.example.aiknowledgebase.eval;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * 检索评测集 —— 语料 + 标注样本的加载与校验。
 * <p>
 * 数据位置（测试资源，不进产物）：
 * <ul>
 *   <li>{@code eval/corpus/<docId>.md} —— 语料，文件名即 documentId</li>
 *   <li>{@code eval/questions.jsonl} —— 样本，一行一条</li>
 * </ul>
 * <p>
 * 本类只做"读 + 校验"，不计算指标（见 {@link EvalMetrics}），也不依赖 Spring。
 * 校验的目的不是形式好看，而是防止标注漂移导致指标悄悄失真：
 * 语料改名、gold 文档被删、证据句与语料不一致，都会在加载阶段直接失败，
 * 而不是让 HitRate 变成 0 之后才被发现。
 * <p>
 * 所有文件显式按 UTF-8 读取：Windows 上 JVM 的默认字符集不是 UTF-8，
 * 依赖默认值会让中文语料变成乱码，且错误非常隐蔽。
 */
public final class EvalDataset {

    /** 语料目录（相对 classpath） */
    public static final String CORPUS_DIR = "eval/corpus";

    /** 样本文件（相对 classpath） */
    public static final String QUESTIONS_FILE = "eval/questions.jsonl";

    public static final String TYPE_ANSWERABLE = "answerable";
    public static final String TYPE_UNANSWERABLE = "unanswerable";

    /** chunk id 的构成规则，必须与 {@code RagService.indexDocument} 保持一致 */
    private static final String CHUNK_ID_MARKER = "-chunk-";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * 一条评测样本。
     *
     * @param id          样本 id
     * @param type        answerable / unanswerable
     * @param question    问句
     * @param goldDocIds  可接受的相关文档（文档级相关性的判定依据）
     * @param goldChunkIds 可接受的文本块，允许多个（答案跨切分边界时）
     * @param evidence    语料中的证据句，用于人工复核；加载时校验它确实存在于 gold 文档中
     * @param notes       标注说明（问法类型、负样本类别等）
     */
    public record Case(String id,
                       String type,
                       String question,
                       List<String> goldDocIds,
                       List<String> goldChunkIds,
                       String evidence,
                       String notes) {

        public boolean answerable() {
            return TYPE_ANSWERABLE.equals(type);
        }
    }

    private final Map<String, String> corpus;
    private final List<Case> cases;
    private final String datasetHash;

    private EvalDataset(Map<String, String> corpus, List<Case> cases, String datasetHash) {
        this.corpus = Collections.unmodifiableMap(corpus);
        this.cases = Collections.unmodifiableList(cases);
        this.datasetHash = datasetHash;
    }

    // ==================== 加载入口 ====================

    /**
     * 从测试资源加载并校验评测集。
     *
     * @throws IllegalStateException 资源缺失、JSON 非法或标注校验不通过
     */
    public static EvalDataset load() {
        Path corpusDir = resourcePath(CORPUS_DIR);
        Path questionsFile = resourcePath(QUESTIONS_FILE);

        Map<String, String> corpus = loadCorpus(corpusDir);
        List<Case> cases = loadCases(questionsFile);

        List<String> problems = validate(corpus, cases);
        if (!problems.isEmpty()) {
            throw new IllegalStateException("评测集校验失败，共 " + problems.size() + " 处问题：\n  - "
                    + String.join("\n  - ", problems));
        }

        return new EvalDataset(corpus, cases, sha256(corpus, questionsFile));
    }

    // ==================== 访问器 ====================

    /** docId → 文档全文，按文件名排序 */
    public Map<String, String> corpus() {
        return corpus;
    }

    public List<Case> cases() {
        return cases;
    }

    public List<Case> answerable() {
        return cases.stream().filter(Case::answerable).collect(Collectors.toList());
    }

    public List<Case> unanswerable() {
        return cases.stream().filter(c -> !c.answerable()).collect(Collectors.toList());
    }

    /**
     * 数据集指纹（questions.jsonl 与全部语料内容的 SHA-256）。
     * <p>
     * 报告里记录它，"两份报告能不能比较"就不再依赖人记版本号。
     */
    public String hash() {
        return datasetHash;
    }

    // ==================== 内部实现 ====================

    private static Path resourcePath(String name) {
        URL url = EvalDataset.class.getClassLoader().getResource(name);
        if (url == null) {
            throw new IllegalStateException("测试资源不存在: " + name
                    + "（请确认 src/test/resources 下存在该路径，且已被构建复制到测试类路径）");
        }
        try {
            return Paths.get(url.toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("测试资源路径无法解析: " + name, e);
        }
    }

    private static Map<String, String> loadCorpus(Path corpusDir) {
        List<Path> files;
        try (Stream<Path> stream = Files.list(corpusDir)) {
            files = stream
                    .filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().endsWith(".md"))
                    .sorted()
                    .collect(Collectors.toList());
        } catch (IOException e) {
            throw new UncheckedIOException("读取语料目录失败: " + corpusDir, e);
        }
        if (files.isEmpty()) {
            throw new IllegalStateException("语料目录为空: " + corpusDir);
        }

        Map<String, String> corpus = new LinkedHashMap<>();
        for (Path file : files) {
            String fileName = file.getFileName().toString();
            String docId = fileName.substring(0, fileName.length() - ".md".length());
            try {
                corpus.put(docId, Files.readString(file, StandardCharsets.UTF_8));
            } catch (IOException e) {
                throw new UncheckedIOException("读取语料失败: " + file, e);
            }
        }
        return corpus;
    }

    private static List<Case> loadCases(Path questionsFile) {
        List<String> lines;
        try {
            lines = Files.readAllLines(questionsFile, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("读取样本文件失败: " + questionsFile, e);
        }

        List<Case> cases = new ArrayList<>();
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }
            JsonNode node;
            try {
                node = MAPPER.readTree(line);
            } catch (Exception e) {
                throw new IllegalStateException("样本文件第 " + (i + 1) + " 行不是合法 JSON: " + e.getMessage(), e);
            }
            cases.add(new Case(
                    text(node, "id"),
                    text(node, "type"),
                    text(node, "question"),
                    stringList(node.get("goldDocIds")),
                    stringList(node.get("goldChunkIds")),
                    text(node, "evidence"),
                    text(node, "notes")));
        }
        if (cases.isEmpty()) {
            throw new IllegalStateException("样本文件没有任何样本: " + questionsFile);
        }
        return cases;
    }

    /**
     * 标注校验。用"收集全部问题再一次性抛出"的方式，避免改一条跑一次的来回。
     */
    private static List<String> validate(Map<String, String> corpus, List<Case> cases) {
        List<String> problems = new ArrayList<>();
        Set<String> seenIds = new HashSet<>();

        for (Case c : cases) {
            String where = "样本 " + c.id();

            if (c.id() == null || c.id().isBlank()) {
                problems.add("存在 id 为空的样本");
                continue;
            }
            if (!seenIds.add(c.id())) {
                problems.add(where + ": id 重复");
            }
            if (c.question() == null || c.question().isBlank()) {
                problems.add(where + ": question 为空");
            }
            if (!TYPE_ANSWERABLE.equals(c.type()) && !TYPE_UNANSWERABLE.equals(c.type())) {
                problems.add(where + ": type 必须是 " + TYPE_ANSWERABLE + " 或 " + TYPE_UNANSWERABLE
                        + "，当前为 " + c.type());
                continue;
            }

            if (c.answerable()) {
                if (c.goldDocIds().isEmpty()) {
                    problems.add(where + ": answerable 样本必须标注 goldDocIds");
                }
                for (String docId : c.goldDocIds()) {
                    if (!corpus.containsKey(docId)) {
                        problems.add(where + ": goldDocIds 中的 " + docId + " 在语料目录中不存在");
                    }
                }
                if (c.goldChunkIds().isEmpty()) {
                    problems.add(where + ": answerable 样本必须标注 goldChunkIds");
                }
                for (String chunkId : c.goldChunkIds()) {
                    int marker = chunkId.lastIndexOf(CHUNK_ID_MARKER);
                    if (marker <= 0) {
                        problems.add(where + ": goldChunkId 不符合 <docId>-chunk-<n> 规则: " + chunkId);
                        continue;
                    }
                    String ownerDoc = chunkId.substring(0, marker);
                    if (!c.goldDocIds().contains(ownerDoc)) {
                        problems.add(where + ": goldChunkId " + chunkId + " 不属于 goldDocIds " + c.goldDocIds());
                    }
                    if (!corpus.containsKey(ownerDoc)) {
                        problems.add(where + ": goldChunkId " + chunkId + " 的文档 " + ownerDoc + " 不存在");
                    }
                }
                // 证据句必须真实存在于 gold 文档中：这是防止"改了语料忘了改标注"的那道闸
                if (c.evidence() != null && !c.evidence().isBlank()) {
                    boolean found = c.goldDocIds().stream()
                            .map(corpus::get)
                            .anyMatch(text -> text != null && text.contains(c.evidence()));
                    if (!found) {
                        problems.add(where + ": evidence 在 gold 文档中找不到，标注与实际语料已不一致");
                    }
                }
            } else {
                if (!c.goldDocIds().isEmpty() || !c.goldChunkIds().isEmpty()) {
                    problems.add(where + ": unanswerable 样本不应标注 goldDocIds / goldChunkIds");
                }
            }
        }
        return problems;
    }

    private static String sha256(Map<String, String> corpus, Path questionsFile) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(Files.readAllBytes(questionsFile));
            for (Map.Entry<String, String> entry : corpus.entrySet()) {
                digest.update(entry.getKey().getBytes(StandardCharsets.UTF_8));
                digest.update(entry.getValue().getBytes(StandardCharsets.UTF_8));
            }
            return "sha256:" + HexFormat.of().formatHex(digest.digest());
        } catch (IOException e) {
            throw new UncheckedIOException("计算数据集指纹失败", e);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("当前 JVM 不支持 SHA-256", e);
        }
    }

    private static String text(JsonNode node, String field) {
        JsonNode value = node.get(field);
        return value == null || value.isNull() ? null : value.asText();
    }

    private static List<String> stringList(JsonNode node) {
        if (node == null || !node.isArray()) {
            return List.of();
        }
        List<String> values = new ArrayList<>();
        for (JsonNode item : node) {
            values.add(item.asText());
        }
        return List.copyOf(values);
    }
}
