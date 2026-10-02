package com.example.aiknowledgebase.store;

import com.example.aiknowledgebase.model.SearchResult;
import com.example.aiknowledgebase.model.TextChunk;
import com.example.aiknowledgebase.service.EmbeddingService;
import org.junit.jupiter.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Step 3 集成验证 — 最小闭环测试。
 * <p>
 * 前提：本地 Ollama 服务已启动（默认 http://localhost:11434）。
 * 如果 Ollama 未运行，测试会因 {@link com.example.aiknowledgebase.common.BusinessException} 而失败，
 * 这是预期行为 —— 启动 Ollama 即可。
 * <p>
 * 验证链路：
 * 1. 存入文本块 + 生成 embedding → VectorStore.add()
 * 2. 查询文本 → EmbeddingService.embed() → VectorStore.search()
 * 3. 输出 → SearchResult
 */
@SpringBootTest
@ActiveProfiles("test")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class Step3VerificationTest {

    private static final Logger log = LoggerFactory.getLogger(Step3VerificationTest.class);

    @Autowired
    private EmbeddingService embeddingService;

    @Autowired
    private VectorStore vectorStore;

    @BeforeEach
    void setUp() {
        // 每次测试前清理，避免互相干扰。
        // 依赖 test profile 提供的独立 key 前缀（rag-test）与独立 Redis 库，
        // 否则这里的 clear() 会直接删掉生产知识库的数据。
        vectorStore.clear();
    }

    @Test
    @Order(1)
    @DisplayName("Step 3 最小闭环：存储 → Embedding → 检索 → 输出得分")
    void fullPipeline() {
        // ── 1. 准备文本块内容 ──────────────────────────────────────
        String chunkContent = "Spring Boot 是一个用于快速开发 Java Web 应用的框架";

        // ── 2. 生成文本块的 embedding 并存入 VectorStore ──────────
        float[] chunkEmbedding = embeddingService.embed(chunkContent);
        assertNotNull(chunkEmbedding);
        assertTrue(chunkEmbedding.length > 0, "embedding 维度应大于 0");

        TextChunk chunk = new TextChunk("chunk-1", "doc-spring", chunkContent, chunkEmbedding);
        vectorStore.add(chunk);
        assertEquals(1, vectorStore.size());
        log.info("✅ 已存入文本块，embedding 维度: {}", chunkEmbedding.length);

        // ── 3. 生成查询的 embedding ────────────────────────────────
        String query = "什么是 Spring Boot？";
        float[] queryEmbedding = embeddingService.embed(query);
        assertNotNull(queryEmbedding);
        assertEquals(chunkEmbedding.length, queryEmbedding.length);

        // ── 4. 检索 Top-1 ──────────────────────────────────────────
        List<SearchResult> results = vectorStore.search(queryEmbedding, 1);

        // ── 5. 断言 + 输出 ────────────────────────────────────────
        assertFalse(results.isEmpty(), "应返回至少一条结果");

        SearchResult top1 = results.get(0);
        assertNotNull(top1.getChunk());
        assertTrue(top1.getScore() > 0, "相似度应大于 0");

        log.info("=" .repeat(60));
        log.info("Step 3 闭环验证通过 ✅");
        log.info("  查询: {}", query);
        log.info("  Top-1 内容: {}", top1.getChunk().getContent());
        log.info("  相似度得分: {}", fmt(top1.getScore()));
        log.info("=" .repeat(60));
    }

    @Test
    @Order(2)
    @DisplayName("多个文本块检索：应返回最相关的结果")
    void multipleChunks() {
        // ── 存入多个文本块 ──────────────────────────────────────────
        String[][] docs = {
                {"chunk-1", "Spring Boot 是一个用于快速开发 Java Web 应用的框架"},
                {"chunk-2", "Python 是一种解释型、面向对象的高级编程语言"},
                {"chunk-3", "Docker 是一个开源的应用容器引擎，用于打包、分发和运行应用"},
        };

        for (String[] doc : docs) {
            float[] emb = embeddingService.embed(doc[1]);
            vectorStore.add(new TextChunk(doc[0], "doc-mix", doc[1], emb));
        }

        // ── 查询 Spring Boot 相关内容 ──────────────────────────────
        float[] queryEmb = embeddingService.embed("Java Web 开发框架");
        List<SearchResult> results = vectorStore.search(queryEmb, 3);

        assertEquals(3, results.size());

        // Top-1 应该是 Spring Boot 相关
        SearchResult top1 = results.get(0);
        assertTrue(top1.getChunk().getContent().contains("Spring Boot"),
                "Top-1 应是与查询最相关的 Spring Boot 文本块");
        assertTrue(top1.getScore() > results.get(1).getScore(),
                "Top-1 分数应高于 Top-2");

        log.info("多文本块检索验证 ✅");
        log.info("  Top-1: {} (score={})", top1.getChunk().getContent(), fmt(top1.getScore()));
        log.info("  Top-2: {} (score={})", results.get(1).getChunk().getContent(), fmt(results.get(1).getScore()));
        log.info("  Top-3: {} (score={})", results.get(2).getChunk().getContent(), fmt(results.get(2).getScore()));
    }

    /**
     * SLF4J 不支持 {:.4f} 这类格式说明符，会把占位符原样打印，因此先格式化再传入。
     */
    private static String fmt(double score) {
        return String.format("%.4f", score);
    }

    @Test
    @Order(3)
    @DisplayName("EmbeddingService 空文本应抛 BusinessException")
    void emptyTextThrows() {
        assertThrows(com.example.aiknowledgebase.common.BusinessException.class,
                () -> embeddingService.embed(""));
        assertThrows(com.example.aiknowledgebase.common.BusinessException.class,
                () -> embeddingService.embed(null));
        assertThrows(com.example.aiknowledgebase.common.BusinessException.class,
                () -> embeddingService.embed("   "));
    }
}
