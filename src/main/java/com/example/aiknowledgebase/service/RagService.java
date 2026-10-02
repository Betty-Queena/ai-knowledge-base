package com.example.aiknowledgebase.service;

import com.example.aiknowledgebase.client.OllamaClient;
import com.example.aiknowledgebase.common.BusinessException;
import com.example.aiknowledgebase.common.ErrorCode;
import com.example.aiknowledgebase.config.RagProperties;
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
 *   <li>{@link #indexDocument(String, String)} — 将文档切块、向量化、存入向量库</li>
 *   <li>{@link #ask(String, String)} — 接收用户问题，检索相关上下文，让大模型基于上下文回答</li>
 * </ul>
 * 切分参数、Top-K、相似度阈值均来自 {@link RagProperties}，便于通过配置做对比实验。
 */
@Service
public class RagService {

    private static final Logger log = LoggerFactory.getLogger(RagService.class);

    private final EmbeddingService embeddingService;
    private final VectorStore vectorStore;
    private final OllamaClient ollamaClient;
    private final RagProperties ragProperties;

    public RagService(EmbeddingService embeddingService,
                      VectorStore vectorStore,
                      OllamaClient ollamaClient,
                      RagProperties ragProperties) {
        this.embeddingService = embeddingService;
        this.vectorStore = vectorStore;
        this.ollamaClient = ollamaClient;
        this.ragProperties = ragProperties;
    }

    // ==================== 公开方法 ====================

    /**
     * 将一篇文档切分、向量化后存入向量库。
     * <p>
     * 流程：
     * <ol>
     *   <li>先清理该 documentId 的旧文本块（保证重传后不残留上一版内容）</li>
     *   <li>用 {@link TextSplitter#split(String, int, int)} 按配置切分文档</li>
     *   <li>每个文本块调用 {@link EmbeddingService#embed(String)} 生成向量</li>
     *   <li>封装为 {@link TextChunk} 存入 {@link VectorStore}</li>
     * </ol>
     *
     * @param documentId 文档唯一标识（由调用方传入，如文件名）
     * @param content    文档全文内容
     * @return 切分后存入的文本块数量
     * @throws BusinessException 当 documentId 为空或 content 为空时抛出
     */
    public int indexDocument(String documentId, String content) {
        // ── 参数校验 ──────────────────────────────────────────────
        if (documentId == null || documentId.isBlank()) {
            throw new BusinessException(ErrorCode.PARAM_INVALID, "documentId 不能为空");
        }
        if (content == null || content.isBlank()) {
            throw new BusinessException(ErrorCode.PARAM_INVALID, "文档内容不能为空");
        }

        // ── 1. 清理旧版本 ──────────────────────────────────────────
        // 必须先删后加：否则重传一份更短的文档时，旧版本多出来的文本块会残留在库中
        // 并被检索到（检索会返回上一版的内容）。
        int removed = vectorStore.deleteByDocumentId(documentId);
        if (removed > 0) {
            log.info("文档 {} 重新索引，已清理旧文本块 {} 个", documentId, removed);
        }

        // ── 2. 切分 ────────────────────────────────────────────────
        List<String> rawChunks = TextSplitter.split(
                content, ragProperties.getChunkSize(), ragProperties.getChunkOverlap());
        if (rawChunks.isEmpty()) {
            log.warn("文档 {} 切分后为空，跳过索引", documentId);
            return 0;
        }
        log.info("文档 {} 切分为 {} 个文本块（chunkSize={}, overlap={}）",
                documentId, rawChunks.size(),
                ragProperties.getChunkSize(), ragProperties.getChunkOverlap());

        // ── 3. 逐块 embedding + 存入 VectorStore ──────────────────
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
     *
     * @param question 用户提问，不能为空
     * @return 大模型基于知识库生成的回答
     * @throws BusinessException 当 question 为空或知识库为空时抛出
     */
    public String ask(String question) {
        return ask(question, null);
    }

    /**
     * 基于知识库内容 + 历史会话上下文回答用户问题。
     * <p>
     * 与 {@link #ask(String)} 的区别在于：
     * <ul>
     *   <li>检索时会将历史对话与当前问题拼接以提高 recall</li>
     *   <li>构造 prompt 时额外注入历史对话，让模型理解上下文指代</li>
     *   <li>回答仍然优先基于知识库上下文</li>
     * </ul>
     *
     * @param question            用户当前问题
     * @param conversationContext 最近几轮对话的文本（可为 null/空）
     * @return 大模型生成的回答
     * @throws BusinessException 当 question 为空或知识库为空时抛出
     */
    public String ask(String question, String conversationContext) {
        // ── 参数校验 ──────────────────────────────────────────────
        if (question == null || question.isBlank()) {
            throw new BusinessException(ErrorCode.PARAM_INVALID, "问题不能为空");
        }
        if (vectorStore.size() == 0) {
            throw new BusinessException(ErrorCode.KNOWLEDGE_BASE_EMPTY, "知识库为空，请先索引文档");
        }

        // ── 1. 构造检索查询 ─────────────────────────────────────────
        String retrievalQuery;
        if (conversationContext != null && !conversationContext.isBlank()) {
            retrievalQuery = conversationContext + "\n当前问题：" + question;
        } else {
            retrievalQuery = question;
        }

        // ── 2. embedding 并检索 ─────────────────────────────────────
        float[] queryEmbedding = embeddingService.embed(retrievalQuery);

        // ── 3. 检索 Top-K（按配置的相似度阈值过滤）───────────────────
        List<SearchResult> searchResults = vectorStore.search(
                queryEmbedding, ragProperties.getTopK(), ragProperties.getMinScore());
        if (searchResults.isEmpty()) {
            log.info("检索无合格结果: topK={}, minScore={}, 阈值过滤后为空",
                    ragProperties.getTopK(),
                    ragProperties.isMinScoreEnabled() ? ragProperties.getMinScore() : "disabled");
            return "抱歉，知识库中未找到与问题相关的内容。";
        }

        // ── 4. 拼接上下文 ──────────────────────────────────────────
        String context = searchResults.stream()
                .map(r -> r.getChunk().getContent())
                .collect(Collectors.joining("\n\n---\n\n"));

        // ── 5. 构造约束 prompt（含历史会话）─────────────────────────
        String prompt = buildPrompt(context, conversationContext, question);

        // ── 6. 调用大模型 ──────────────────────────────────────────
        log.info("RAG ask: question={}, retrievalQueryLength={}chars, contextLength={}chars, hits={}, hasHistory={}",
                truncate(question, 50), retrievalQuery.length(), context.length(),
                searchResults.size(),
                conversationContext != null && !conversationContext.isBlank());

        String answer = ollamaClient.generate(prompt);

        log.info("RAG answer: {}chars", answer != null ? answer.length() : 0);
        return answer;
    }

    // ==================== 私有方法 ====================

    /**
     * 构造 RAG 问答 prompt（无历史会话）。
     */
    private String buildPrompt(String context, String question) {
        return buildPrompt(context, null, question);
    }

    /**
     * 构造 RAG 问答 prompt（含历史会话上下文）。
     * <p>
     * 在原有约束基础上增加历史会话，帮助模型理解"它/这个/上面说的"等指代。
     * 检索仍基于当前问题，历史仅作为额外参考。
     */
    private String buildPrompt(String context, String conversationContext, String question) {
        StringBuilder sb = new StringBuilder();
        sb.append("你是一个知识库问答助手。请根据以下提供的上下文内容回答用户的问题。\n\n");
        sb.append("规则：\n");
        sb.append("1. 只基于上下文中的信息回答，不要使用你自己的知识\n");
        sb.append("2. 如果上下文中没有足够的信息来回答问题，请明确说")
                .append("“根据提供的资料，无法回答该问题”\n");
        sb.append("3. 回答要简洁、准确，使用中文\n");
        sb.append("4. 可以引用上下文中的原文来支撑你的回答\n\n");

        // 注入历史会话（如果有），帮助模型理解上下文指代
        if (conversationContext != null && !conversationContext.isBlank()) {
            sb.append("以下是最近的对话历史，供参考（但仍优先使用上下文中的知识库资料）：\n");
            sb.append(conversationContext).append("\n\n");
        }

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
