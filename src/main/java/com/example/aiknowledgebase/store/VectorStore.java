package com.example.aiknowledgebase.store;

import com.example.aiknowledgebase.config.RagProperties;
import com.example.aiknowledgebase.model.SearchResult;
import com.example.aiknowledgebase.model.TextChunk;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;

/**
 * 向量存储 — 基于 Redis Hash 的持久化存储。
 * <p>
 * 所有 {@link TextChunk} 以 JSON 形式存储在 Redis Hash 中（key 为
 * {@code <rag.redis.key-prefix>:chunks}），field 为 chunk ID，value 为序列化后的 JSON 字符串。
 * 检索时从 Redis 读取全部 chunk，逐条计算余弦相似度并返回 Top-K。
 * <p>
 * 注意：Redis 在这里只承担<b>存储</b>职责，相似度计算全部发生在 JVM 应用层（暴力全量扫描）。
 * 项目重启后数据不丢失（依赖 Redis 持久化）。
 */
@Component
public class VectorStore {

    private static final Logger log = LoggerFactory.getLogger(VectorStore.class);

    /** Redis Hash key 的后缀，完整 key 为 {@code <keyPrefix>:chunks} */
    private static final String CHUNKS_KEY_SUFFIX = ":chunks";

    private final StringRedisTemplate stringRedisTemplate;
    private final ObjectMapper objectMapper;
    private final String chunksKey;

    public VectorStore(StringRedisTemplate stringRedisTemplate,
                       ObjectMapper objectMapper,
                       RagProperties ragProperties) {
        this.stringRedisTemplate = stringRedisTemplate;
        this.objectMapper = objectMapper;
        this.chunksKey = ragProperties.getRedis().getKeyPrefix() + CHUNKS_KEY_SUFFIX;
    }

    /**
     * 存入一个文本块到 Redis。
     *
     * @param chunk 文本块（需包含非 null 的 embedding）
     */
    public void add(TextChunk chunk) {
        if (chunk == null) {
            throw new IllegalArgumentException("chunk 不能为 null");
        }
        if (chunk.getEmbedding() == null) {
            throw new IllegalArgumentException("chunk 的 embedding 不能为 null");
        }
        try {
            String json = objectMapper.writeValueAsString(chunk);
            stringRedisTemplate.opsForHash().put(chunksKey, chunk.getId(), json);
        } catch (Exception e) {
            throw new RuntimeException("TextChunk 序列化失败: " + e.getMessage(), e);
        }
    }

    /**
     * 批量存入文本块。
     *
     * @param chunkList 文本块列表
     */
    public void addAll(List<TextChunk> chunkList) {
        if (chunkList == null) {
            throw new IllegalArgumentException("chunkList 不能为 null");
        }
        for (TextChunk chunk : chunkList) {
            add(chunk);
        }
    }

    /**
     * 删除某个文档已索引的全部文本块。
     * <p>
     * 用于重新索引前清理旧版本数据：若不做这一步，重传一份更短的文档时，
     * 旧版本多出来的文本块会残留在库中并被检索到。
     * <p>
     * 实现为全量扫描后按 documentId 匹配删除。删除是低频操作，因此不额外维护
     * documentId → chunkId 的索引。
     *
     * @param documentId 文档标识
     * @return 实际删除的文本块数量
     */
    public int deleteByDocumentId(String documentId) {
        if (documentId == null || documentId.isBlank()) {
            throw new IllegalArgumentException("documentId 不能为空");
        }

        Map<Object, Object> entries = stringRedisTemplate.opsForHash().entries(chunksKey);
        if (entries.isEmpty()) {
            return 0;
        }

        List<Object> fieldsToDelete = new ArrayList<>();
        for (Map.Entry<Object, Object> entry : entries.entrySet()) {
            String json = (String) entry.getValue();
            if (json == null) {
                continue;
            }
            try {
                TextChunk chunk = objectMapper.readValue(json, TextChunk.class);
                if (documentId.equals(chunk.getDocumentId())) {
                    fieldsToDelete.add(entry.getKey());
                }
            } catch (Exception e) {
                log.warn("按文档删除时反序列化失败, field={}: {}", entry.getKey(), e.getMessage());
            }
        }

        if (fieldsToDelete.isEmpty()) {
            return 0;
        }
        Long removed = stringRedisTemplate.opsForHash().delete(chunksKey, fieldsToDelete.toArray());
        return removed != null ? removed.intValue() : 0;
    }

    /**
     * 查询与 queryEmbedding 最相似的 Top-K 个文本块（不做阈值过滤）。
     *
     * @param queryEmbedding 查询向量
     * @param topK           返回前 K 条结果
     * @return 按相似度降序排列的结果列表
     */
    public List<SearchResult> search(float[] queryEmbedding, int topK) {
        return search(queryEmbedding, topK, RagProperties.MIN_SCORE_DISABLED);
    }

    /**
     * 查询与 queryEmbedding 最相似的 Top-K 个文本块。
     * <p>
     * 先按阈值过滤，再排序取 Top-K——顺序不能反，否则可能在仍有合格候选的情况下
     * 返回少于 topK 条结果。
     *
     * @param queryEmbedding 查询向量
     * @param topK           返回前 K 条结果
     * @param minScore       相似度阈值；{@link RagProperties#MIN_SCORE_DISABLED} 表示不过滤
     * @return 按相似度降序排列的结果列表
     */
    public List<SearchResult> search(float[] queryEmbedding, int topK, double minScore) {
        if (queryEmbedding == null) {
            throw new IllegalArgumentException("queryEmbedding 不能为 null");
        }
        if (topK <= 0) {
            throw new IllegalArgumentException("topK 必须大于 0，当前值: " + topK);
        }

        // 从 Redis 读取全部 chunk
        Map<Object, Object> entries = stringRedisTemplate.opsForHash().entries(chunksKey);
        if (entries.isEmpty()) {
            return List.of();
        }

        boolean thresholdEnabled = minScore > RagProperties.MIN_SCORE_DISABLED;
        List<SearchResult> results = new ArrayList<>();
        int deserializeFailed = 0;
        int dimensionMismatch = 0;
        int filteredByScore = 0;

        for (Map.Entry<Object, Object> entry : entries.entrySet()) {
            String json = (String) entry.getValue();
            if (json == null) {
                continue;
            }

            // 反序列化失败与向量维度不符是两类不同的问题，必须分开统计：
            // 前者说明存储数据损坏，后者通常说明 embedding 模型被更换但索引没有重建。
            TextChunk chunk;
            try {
                chunk = objectMapper.readValue(json, TextChunk.class);
            } catch (Exception e) {
                deserializeFailed++;
                log.debug("TextChunk 反序列化失败, field={}: {}", entry.getKey(), e.getMessage());
                continue;
            }

            float[] embedding = chunk.getEmbedding();
            if (embedding == null || embedding.length != queryEmbedding.length) {
                dimensionMismatch++;
                continue;
            }

            double similarity = cosineSimilarity(queryEmbedding, embedding);
            if (thresholdEnabled && similarity < minScore) {
                filteredByScore++;
                continue;
            }
            results.add(new SearchResult(chunk, similarity));
        }

        if (dimensionMismatch > 0) {
            log.warn("检索跳过 {} 条向量维度不一致的文本块（查询维度={}）。"
                            + "这通常意味着 embedding 模型已更换，请清理并重建索引，否则检索结果会不完整。",
                    dimensionMismatch, queryEmbedding.length);
        }
        if (deserializeFailed > 0) {
            log.warn("检索跳过 {} 条反序列化失败的文本块，存储数据可能已损坏（field={}）。",
                    deserializeFailed, chunksKey);
        }

        // 按相似度降序排序，取 Top-K
        results.sort(Comparator.comparingDouble(SearchResult::getScore).reversed());
        List<SearchResult> topResults = topK >= results.size()
                ? results
                : new ArrayList<>(results.subList(0, topK));

        log.info("向量检索: 候选={}, 维度不符={}, 反序列化失败={}, 阈值过滤={}, 返回={}, top1Score={}",
                entries.size(), dimensionMismatch, deserializeFailed, filteredByScore,
                topResults.size(),
                topResults.isEmpty() ? "-" : String.format("%.4f", topResults.get(0).getScore()));

        return topResults;
    }

    /**
     * 获取当前存储的文本块总数。
     */
    public int size() {
        Long size = stringRedisTemplate.opsForHash().size(chunksKey);
        return size != null ? size.intValue() : 0;
    }

    /**
     * 采样一个已存文本块的向量维度，用于启动时校验 embedding 模型是否被更换。
     *
     * @return 维度；库为空或无可用向量时返回 {@link OptionalInt#empty()}
     */
    public OptionalInt sampleEmbeddingDimension() {
        Map<Object, Object> entries = stringRedisTemplate.opsForHash().entries(chunksKey);
        for (Object value : entries.values()) {
            if (value == null) {
                continue;
            }
            try {
                TextChunk chunk = objectMapper.readValue((String) value, TextChunk.class);
                float[] embedding = chunk.getEmbedding();
                if (embedding != null && embedding.length > 0) {
                    return OptionalInt.of(embedding.length);
                }
            } catch (Exception e) {
                log.debug("采样向量维度时反序列化失败: {}", e.getMessage());
            }
        }
        return OptionalInt.empty();
    }

    /**
     * 清空所有文本块。
     */
    public void clear() {
        stringRedisTemplate.delete(chunksKey);
    }

    // ==================== 工具方法 ====================

    /**
     * 计算两个 float 向量的余弦相似度。
     * <p>
     * cosine_similarity(A, B) = (A·B) / (||A|| × ||B||)
     * <p>
     * 当任一向量为零向量时返回 0。
     */
    static double cosineSimilarity(float[] a, float[] b) {
        if (a == null || b == null || a.length == 0 || b.length == 0) {
            return 0;
        }
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "向量维度不一致: a=" + a.length + ", b=" + b.length);
        }

        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        for (int i = 0; i < a.length; i++) {
            dotProduct += (double) a[i] * b[i];
            normA += (double) a[i] * a[i];
            normB += (double) b[i] * b[i];
        }

        double denominator = Math.sqrt(normA) * Math.sqrt(normB);
        return denominator == 0 ? 0 : dotProduct / denominator;
    }
}
