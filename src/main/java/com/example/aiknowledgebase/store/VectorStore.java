package com.example.aiknowledgebase.store;

import com.example.aiknowledgebase.model.SearchResult;
import com.example.aiknowledgebase.model.TextChunk;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * 向量存储 — 基于内存的余弦相似度检索。
 * <p>
 * 当前使用 {@link CopyOnWriteArrayList} 存储所有 {@link TextChunk}，
 * 检索时遍历全部块计算余弦相似度并返回 Top-K 结果。
 * <p>
 * 未来可替换为基于 FAISS / pgvector / Elasticsearch 的实现。
 */
@Component
public class VectorStore {

    /** 所有文本块的集合（线程安全） */
    private final List<TextChunk> chunks = new CopyOnWriteArrayList<>();

    /**
     * 存入一个文本块。
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
        chunks.add(chunk);
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

        // 计算每个 chunk 的余弦相似度
        List<SearchResult> results = new ArrayList<>();
        for (TextChunk chunk : chunks) {
            double similarity = cosineSimilarity(queryEmbedding, chunk.getEmbedding());
            results.add(new SearchResult(chunk, similarity));
        }

        // 按相似度降序排序，取 Top-K
        results.sort(Comparator.comparingDouble(SearchResult::getScore).reversed());

        return topK >= results.size() ? results : results.subList(0, topK);
    }

    /**
     * 获取当前存储的文本块总数。
     */
    public int size() {
        return chunks.size();
    }

    /**
     * 清空所有文本块。
     */
    public void clear() {
        chunks.clear();
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
