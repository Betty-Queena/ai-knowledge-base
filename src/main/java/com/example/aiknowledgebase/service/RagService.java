package com.example.aiknowledgebase.service;

import com.example.aiknowledgebase.client.OllamaClient;
import com.example.aiknowledgebase.common.BusinessException;
import com.example.aiknowledgebase.model.SearchResult;
import com.example.aiknowledgebase.model.TextChunk;
import com.example.aiknowledgebase.store.VectorStore;
import com.example.aiknowledgebase.util.TextSplitter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

/**
 * RAG 编排服务 — 知识库文档索引 + 检索增强问答。
 * <p>
 * 职责：
 * <ul>
 *   <li>{@link #indexDocument(String, String)} — 将文档切块、向量化、存入内存向量库</li>
 *   <li>{@link #ask(String)} — 接收用户问题，检索相关上下文，让大模型基于上下文回答</li>
 * </ul>
 */
@Service
public class RagService {

    private static final Logger log = LoggerFactory.getLogger(RagService.class);

    /** 文本切分参数：每块最大字符数 */
    private static final int CHUNK_SIZE = 500;

    /** 文本切分参数：相邻块重叠字符数 */
    private static final int CHUNK_OVERLAP = 100;

    /** 检索时返回的 Top-K 文本块数量 */
    private static final int TOP_K = 3;

    private final EmbeddingService embeddingService;
    private final VectorStore vectorStore;
    private final OllamaClient ollamaClient;

    public RagService(EmbeddingService embeddingService,
                      VectorStore vectorStore,
                      OllamaClient ollamaClient) {
        this.embeddingService = embeddingService;
        this.vectorStore = vectorStore;
        this.ollamaClient = ollamaClient;
    }

    // ==================== 公开方法 ====================

    /**
     * 将一篇文档切分、向量化后存入向量库。
     * <p>
     * 流程：
     * <ol>
     *   <li>用 {@link TextSplitter#split(String, int, int)} 按 500/100 切分文档</li>
     *   <li>每个文本块调用 {@link EmbeddingService#embed(String)} 生成向量</li>
     *   <li>封装为 {@link TextChunk} 存入 {@link VectorStore}</li>
     * </ol>
     *
     * @param documentId 文档唯一标识（由调用方传入，如文件名或 DB 主键）
     * @param content    文档全文内容
     * @return 切分后存入的文本块数量
     * @throws BusinessException 当 documentId 为空或 content 为空时抛出
     */
    public int indexDocument(String documentId, String content) {
        // ── 参数校验 ──────────────────────────────────────────────
        if (documentId == null || documentId.isBlank()) {
            throw new BusinessException(400, "documentId 不能为空");
        }
        if (content == null || content.isBlank()) {
            throw new BusinessException(400, "文档内容不能为空");
        }

        // ── 1. 切分 ────────────────────────────────────────────────
        List<String> rawChunks = TextSplitter.split(content, CHUNK_SIZE, CHUNK_OVERLAP);
        if (rawChunks.isEmpty()) {
            log.warn("文档 {} 切分后为空，跳过索引", documentId);
            return 0;
        }
        log.info("文档 {} 切分为 {} 个文本块", documentId, rawChunks.size());

        // ── 2. 逐块 embedding + 存入 VectorStore ──────────────────
        int successCount = 0;
        for (int i = 0; i < rawChunks.size(); i++) {
            String text = rawChunks.get(i);
            try {
                float[] embedding = embeddingService.embed(text);
                String chunkId = documentId + "-chunk-" + i;

                TextChunk chunk = new TextChunk(chunkId, documentId, text, embedding);
                vectorStore.add(chunk);
                successCount++;
            } catch (Exception e) {
                log.warn("文本块 [{}/{}] embedding 失败: {}", i + 1, rawChunks.size(), e.getMessage());
                // 单个块失败不影响后续块
            }
        }

        log.info("文档 {} 索引完成：成功 {}/{} 块", documentId, successCount, rawChunks.size());
        return successCount;
    }

    /**
     * 基于知识库内容回答用户问题。
     * <p>
     * 流程：
     * <ol>
     *   <li>对问题做 embedding</li>
     *   <li>从向量库检索最相关的 Top-3 文本块</li>
     *   <li>拼接检索结果作为上下文</li>
     *   <li>构造约束 prompt，要求大模型仅基于上下文回答</li>
     *   <li>调用 Ollama 生成回答</li>
     * </ol>
     *
     * @param question 用户提问，不能为空
     * @return 大模型基于知识库生成的回答
     * @throws BusinessException 当 question 为空或知识库为空时抛出
     */
    public String ask(String question) {
        // ── 参数校验 ──────────────────────────────────────────────
        if (question == null || question.isBlank()) {
            throw new BusinessException(400, "问题不能为空");
        }
        if (vectorStore.size() == 0) {
            throw new BusinessException(400, "知识库为空，请先索引文档");
        }

        // ── 1. 问题 embedding ─────────────────────────────────────
        float[] queryEmbedding = embeddingService.embed(question);

        // ── 2. 检索 Top-3 ─────────────────────────────────────────
        List<SearchResult> searchResults = vectorStore.search(queryEmbedding, TOP_K);
        if (searchResults.isEmpty()) {
            return "抱歉，知识库中未找到与问题相关的内容。";
        }

        // ── 3. 拼接上下文 ──────────────────────────────────────────
        String context = searchResults.stream()
                .map(r -> r.getChunk().getContent())
                .collect(Collectors.joining("\n\n---\n\n"));

        // ── 4. 构造约束 prompt ─────────────────────────────────────
        String prompt = buildPrompt(context, question);

        // ── 5. 调用大模型 ──────────────────────────────────────────
        log.info("RAG ask: question={}, contextLength={}chars, top1Score={}",
                truncate(question, 50), context.length(),
                String.format("%.4f", searchResults.get(0).getScore()));

        String answer = ollamaClient.generate(prompt);

        log.info("RAG answer: {}chars", answer != null ? answer.length() : 0);
        return answer;
    }

    // ==================== 私有方法 ====================

    /**
     * 构造 RAG 问答 prompt。
     * <p>
     * 要求模型：
     * <ul>
     *   <li>仅基于给定的上下文回答</li>
     *   <li>上下文不足时明确说明，不编造信息</li>
     *   <li>使用中文回答</li>
     * </ul>
     */
    private String buildPrompt(String context, String question) {
        StringBuilder sb = new StringBuilder();
        sb.append("你是一个知识库问答助手。请根据以下提供的上下文内容回答用户的问题。\n\n");
        sb.append("规则：\n");
        sb.append("1. 只基于上下文中的信息回答，不要使用你自己的知识\n");
        sb.append("2. 如果上下文中没有足够的信息来回答问题，请明确说")
                .append("“根据提供的资料，无法回答该问题”\n");
        sb.append("3. 回答要简洁、准确，使用中文\n");
        sb.append("4. 可以引用上下文中的原文来支撑你的回答\n\n");
        sb.append("上下文：\n").append(context).append("\n\n");
        sb.append("问题：").append(question).append("\n\n");
        sb.append("回答：");
        return sb.toString();
    }

    /**
     * 截断字符串到指定长度（用于日志输出）。
     */
    private static String truncate(String s, int maxLen) {
        if (s == null) return "null";
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }
}
