package com.example.aiknowledgebase.store;

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

/**
 * 向量存储 — 基于 Redis Hash 的持久化存储。
 * <p>
 * 所有 {@link TextChunk} 以 JSON 形式存储在 Redis Hash {@code rag:chunks} 中，
 * field 为 chunk ID，value 为序列化后的 JSON 字符串。
 * 检索时从 Redis 读取全部 chunk，逐条计算余弦相似度并返回 Top-K。
 * <p>
 * 项目重启后数据不丢失（依赖 Redis 持久化）。
 */
@Component
public class VectorStore {

    private static final Logger log = LoggerFactory.getLogger(VectorStore.class);

    /** Redis Hash key，存储所有 chunk：field=chunkId, value=JSON */
    private static final String REDIS_KEY_CHUNKS = "rag:chunks";

    private final StringRedisTemplate stringRedisTemplate;
    private final ObjectMapper objectMapper;

    public VectorStore(StringRedisTemplate stringRedisTemplate,
                       ObjectMapper objectMapper) {
        this.stringRedisTemplate = stringRedisTemplate;
        this.objectMapper = objectMapper;
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
            stringRedisTemplate.opsForHash().put(REDIS_KEY_CHUNKS, chunk.getId(), json);
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
     * 查询与 queryEmbedding 最相似的 Top-K 个文本块。
     * <p>
     * 从 Redis 读取全部 chunk，使用余弦相似度计算后排序返回。
     *
     * @param queryEmbedding 查询向量
     * @param topK           返回前 K 条结果
     * @return 按相似度降序排列的结果列表
     */
    public List<SearchResult> search(float[] queryEmbedding, int topK) {
        if (queryEmbedding == null) {
            throw new IllegalArgumentException("queryEmbedding 不能为 null");
        }
        if (topK <= 0) {
            throw new IllegalArgumentException("topK 必须大于 0，当前值: " + topK);
        }

        // 从 Redis 读取全部 chunk
        Map<Object, Object> entries = stringRedisTemplate.opsForHash().entries(REDIS_KEY_CHUNKS);
        if (entries.isEmpty()) {
            return List.of();
        }

        List<SearchResult> results = new ArrayList<>();
        for (Map.Entry<Object, Object> entry : entries.entrySet()) {
            String json = (String) entry.getValue();
            if (json == null) {
                continue;
            }
            try {
                TextChunk chunk = objectMapper.readValue(json, TextChunk.class);
                double similarity = cosineSimilarity(queryEmbedding, chunk.getEmbedding());
                results.add(new SearchResult(chunk, similarity));
            } catch (Exception e) {
                log.warn("TextChunk 反序列化失败, field={}: {}", entry.getKey(), e.getMessage());
            }
        }

        // 按相似度降序排序，取 Top-K
        results.sort(Comparator.comparingDouble(SearchResult::getScore).reversed());
        return topK >= results.size() ? results : results.subList(0, topK);
    }

    /**
     * 获取当前存储的文本块总数。
     */
    public int size() {
        Long size = stringRedisTemplate.opsForHash().size(REDIS_KEY_CHUNKS);
        return size != null ? size.intValue() : 0;
    }

    /**
     * 清空所有文本块。
     */
    public void clear() {
        stringRedisTemplate.delete(REDIS_KEY_CHUNKS);
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
