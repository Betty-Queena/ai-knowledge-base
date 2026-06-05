package com.example.aiknowledgebase.store;

import com.example.aiknowledgebase.model.SearchResult;
import com.example.aiknowledgebase.model.TextChunk;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link VectorStore} 单元测试。
 * <p>
 * 使用已知向量验证余弦相似度计算和 Top-K 检索逻辑的正确性。
 */
class VectorStoreTest {

    private VectorStore store;

    @BeforeEach
    void setUp() {
        store = new VectorStore();
    }

    // ==================== cosineSimilarity ====================

    @Test
    @DisplayName("相同向量余弦相似度为 1")
    void identicalVectors() {
        float[] a = {1, 2, 3};
        double sim = VectorStore.cosineSimilarity(a, a);
        assertEquals(1.0, sim, 1e-10);
    }

    @Test
    @DisplayName("正交向量余弦相似度为 0")
    void orthogonalVectors() {
        float[] a = {1, 0, 0};
        float[] b = {0, 1, 0};
        double sim = VectorStore.cosineSimilarity(a, b);
        assertEquals(0.0, sim, 1e-10);
    }

    @Test
    @DisplayName("相反向量余弦相似度为 -1")
    void oppositeVectors() {
        float[] a = {1, 2, 3};
        float[] b = {-1, -2, -3};
        double sim = VectorStore.cosineSimilarity(a, b);
        assertEquals(-1.0, sim, 1e-10);
    }

    @Test
    @DisplayName("零向量返回 0")
    void zeroVector() {
        float[] a = {1, 2, 3};
        float[] b = {0, 0, 0};
        double sim = VectorStore.cosineSimilarity(a, b);
        assertEquals(0.0, sim, 1e-10);
    }

    @Test
    @DisplayName("已知角度：45 度近似值")
    void knownAngle() {
        float[] a = {1, 0};
        float[] b = {1, 1};
        // cos(45°) = 1/√2 ≈ 0.7071
        double sim = VectorStore.cosineSimilarity(a, b);
        assertEquals(1 / Math.sqrt(2), sim, 1e-10);
    }

    @Test
    @DisplayName("维度不一致应抛出异常")
    void dimensionMismatch() {
        float[] a = {1, 2, 3};
        float[] b = {1, 2};
        assertThrows(IllegalArgumentException.class,
                () -> VectorStore.cosineSimilarity(a, b));
    }

    @Test
    @DisplayName("null 向量返回 0")
    void nullVectors() {
        assertEquals(0, VectorStore.cosineSimilarity(null, new float[]{1, 2}));
        assertEquals(0, VectorStore.cosineSimilarity(new float[]{1, 2}, null));
        assertEquals(0, VectorStore.cosineSimilarity(null, null));
    }

    // ==================== add / search ====================

    @Test
    @DisplayName("存入和检索的基本流程")
    void addAndSearch() {
        // 三个向量：[1,0] 与 [1,1] 相似度高，[1,0] 与 [-1,0] 相似度低
        store.add(new TextChunk("1", "doc1", "A 正方向", new float[]{1, 0}));
        store.add(new TextChunk("2", "doc1", "B 对角线", new float[]{1, 1}));
        store.add(new TextChunk("3", "doc2", "C 反方向", new float[]{-1, 0}));

        assertEquals(3, store.size());

        // 查询 [1, 0]
        List<SearchResult> results = store.search(new float[]{1, 0}, 3);

        assertEquals(3, results.size());
        // 最相似的应该是 chunk "A 正方向"
        assertEquals("1", results.get(0).getChunk().getId());
        assertEquals(1.0, results.get(0).getScore(), 1e-10);
        // 第三应该是 "C 反方向"
        assertEquals("3", results.get(2).getChunk().getId());
        assertTrue(results.get(2).getScore() < 0);
    }

    @Test
    @DisplayName("Top-K 只返回前 K 条")
    void topK() {
        store.add(new TextChunk("1", "doc1", "A", new float[]{1, 0}));
        store.add(new TextChunk("2", "doc1", "B", new float[]{0, 1}));
        store.add(new TextChunk("3", "doc2", "C", new float[]{-1, 0}));

        List<SearchResult> top1 = store.search(new float[]{1, 0}, 1);
        assertEquals(1, top1.size());
        assertEquals("1", top1.get(0).getChunk().getId());

        List<SearchResult> top2 = store.search(new float[]{1, 0}, 2);
        assertEquals(2, top2.size());
    }

    @Test
    @DisplayName("大于总条数的 topK 返回全部")
    void topKLargerThanSize() {
        store.add(new TextChunk("1", "doc1", "A", new float[]{1, 0}));
        List<SearchResult> results = store.search(new float[]{1, 0}, 100);
        assertEquals(1, results.size());
    }

    @Test
    @DisplayName("addAll 批量存入")
    void addAll() {
        List<TextChunk> list = List.of(
                new TextChunk("1", "doc1", "A", new float[]{1, 0}),
                new TextChunk("2", "doc1", "B", new float[]{0, 1})
        );
        store.addAll(list);
        assertEquals(2, store.size());
    }

    @Test
    @DisplayName("embedding 为 null 的 chunk 应拒绝")
    void rejectNullEmbedding() {
        assertThrows(IllegalArgumentException.class,
                () -> store.add(new TextChunk("1", "doc1", "no vector", null)));
    }

    @Test
    @DisplayName("queryEmbedding 为 null 应拒绝")
    void rejectNullQuery() {
        assertThrows(IllegalArgumentException.class,
                () -> store.search(null, 1));
    }

    @Test
    @DisplayName("topK <= 0 应拒绝")
    void rejectInvalidTopK() {
        assertThrows(IllegalArgumentException.class,
                () -> store.search(new float[]{1, 0}, 0));
        assertThrows(IllegalArgumentException.class,
                () -> store.search(new float[]{1, 0}, -1));
    }

    @Test
    @DisplayName("clear 后 size 为 0")
    void clear() {
        store.add(new TextChunk("1", "doc1", "A", new float[]{1, 0}));
        store.clear();
        assertEquals(0, store.size());
    }
}
