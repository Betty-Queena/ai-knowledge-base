package com.example.aiknowledgebase.service.impl;

import com.example.aiknowledgebase.client.OllamaClient;
import com.example.aiknowledgebase.common.BusinessException;
import com.example.aiknowledgebase.common.ErrorCode;
import com.example.aiknowledgebase.dto.ChatMessage;
import com.example.aiknowledgebase.service.ChatService;
import com.example.aiknowledgebase.service.RagService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class ChatServiceImpl implements ChatService {

    private static final Logger log = LoggerFactory.getLogger(ChatServiceImpl.class);

    /** 每个会话在 Redis 中保留的最大消息条数，避免无限增长 */
    private static final int MAX_STORED_MESSAGES = 20;

    /** 构造会话上下文时取最近的消息条数 */
    private static final int MAX_HISTORY_SIZE = 6;

    /** 会话上下文文本的最大字符数，防止 prompt 过长 */
    private static final int MAX_CONTEXT_CHARS = 2000;

    private final OllamaClient ollamaClient;
    private final RedisTemplate<String, Object> redisTemplate;
    private final RagService ragService;

    public ChatServiceImpl(OllamaClient ollamaClient,
                           RedisTemplate<String, Object> redisTemplate,
                           RagService ragService) {
        this.ollamaClient = ollamaClient;
        this.redisTemplate = redisTemplate;
        this.ragService = ragService;
    }

    @Override
    public String ask(String question, String sessionId) {
        if (!StringUtils.hasText(question)) {
            throw new BusinessException(ErrorCode.PARAM_INVALID, "问题不能为空");
        }

        // 从 Redis 中构建最近会话上下文
        String conversationContext = buildConversationContext(sessionId);

        String answer = generateAnswer(question, conversationContext);

        ChatMessage message = new ChatMessage(
                question,
                answer,
                System.currentTimeMillis()
        );

        String historyKey = buildHistoryKey(sessionId);
        redisTemplate.opsForList().leftPush(historyKey, message);

        // 只保留最近 N 条，避免 Redis 无限增长
        redisTemplate.opsForList().trim(historyKey, 0, MAX_STORED_MESSAGES - 1);

        return answer;
    }

    /**
     * 从 Redis 读取最近 N 条聊天记录，格式化为历史会话文本（时间正序）。
     * <p>
     * 返回形如：
     * <pre>
     * 用户：xxx
     * 助手：xxx
     *
     * 用户：yyy
     * 助手：yyy
     * </pre>
     */
    private String buildConversationContext(String sessionId) {
        // Redis 中最新记录在链表头部（index 0），取 top-N
        List<Object> objects = redisTemplate.opsForList().range(
                buildHistoryKey(sessionId), 0, MAX_HISTORY_SIZE - 1);

        if (objects == null || objects.isEmpty()) {
            return "";
        }

        // 反转：变为时间正序（旧→新）
        List<ChatMessage> messages = objects.stream()
                .filter(ChatMessage.class::isInstance)
                .map(ChatMessage.class::cast)
                .collect(Collectors.toList());
        Collections.reverse(messages);

        // 拼接历史文本，超过长度限制时截断
        StringBuilder sb = new StringBuilder();
        for (ChatMessage msg : messages) {
            String entry = "用户：" + msg.getQuestion() + "\n助手：" + msg.getAnswer() + "\n";
            if (sb.length() + entry.length() > MAX_CONTEXT_CHARS) {
                break;
            }
            sb.append(entry);
        }
        return sb.toString();
    }

    /**
     * 优先使用 RAG 检索增强回答，失败则降级为普通对话。
     * <p>
     * 和纯 LLM 对话的区别：
     * <ul>
     *   <li>RAG 路径：调用 {@link RagService#ask(String, String)} 带上会话上下文</li>
     *   <li>降级路径：将会话上下文拼入普通对话 prompt</li>
     * </ul>
     * 降级判定基于 {@link ErrorCode}，不再依赖异常消息文本比较。
     */
    private String generateAnswer(String question, String conversationContext) {
        try {
            String answer = ragService.ask(question, conversationContext);
            log.info("RAG 回答成功: question=\"{}\"", truncate(question, 50));
            return answer;
        } catch (BusinessException e) {
            // 知识库为空是预期情况，非异常
            if (e.is(ErrorCode.KNOWLEDGE_BASE_EMPTY)) {
                log.info("知识库为空，降级为普通对话: question=\"{}\"", truncate(question, 50));
            } else {
                log.warn("RAG 回答失败 (code={}, {}), 降级为普通对话: {}",
                        e.getCode(), e.getMessage(), truncate(question, 50));
            }
        } catch (Exception e) {
            log.warn("RAG 回答异常 ({}), 降级为普通对话: {}",
                    e.getMessage(), truncate(question, 50));
        }

        // 降级：走普通 Ollama 对话（带上历史上下文）
        String promptWithHistory = buildFallbackPrompt(question, conversationContext);
        String fallbackAnswer = ollamaClient.generate(promptWithHistory);
        log.info("普通对话回答成功: question=\"{}\"", truncate(question, 50));
        return fallbackAnswer;
    }

    /**
     * 构造降级（非 RAG）场景的对话 prompt。
     * 包含历史上下文，让模型能理解上下文指代。
     */
    private static String buildFallbackPrompt(String question, String conversationContext) {
        StringBuilder sb = new StringBuilder();
        sb.append("你是一个智能问答助手。请回答用户的问题。\n\n");
        if (conversationContext != null && !conversationContext.isBlank()) {
            sb.append("以下是最近的对话历史，供参考：\n");
            sb.append(conversationContext).append("\n\n");
        }
        sb.append("用户：").append(question).append("\n");
        sb.append("助手：");
        return sb.toString();
    }

    @Override
    public List<ChatMessage> history(String sessionId) {
        // 读取与写入必须使用同一个 key：写入走 buildHistoryKey(sessionId)。
        List<Object> list = redisTemplate.opsForList().range(
                buildHistoryKey(sessionId), 0, MAX_STORED_MESSAGES - 1);

        if (list == null || list.isEmpty()) {
            return List.of();
        }

        // Redis 中最新记录在头部，反转为时间正序（旧 → 新）后返回
        List<ChatMessage> messages = list.stream()
                .filter(ChatMessage.class::isInstance)
                .map(ChatMessage.class::cast)
                .collect(Collectors.toList());
        Collections.reverse(messages);
        return messages;
    }

    private String buildHistoryKey(String sessionId) {
        if (sessionId == null || sessionId.isBlank()) {
            sessionId = "default";
        }
        return "ai:chat:history:" + sessionId;
    }

    private static String truncate(String s, int maxLen) {
        if (s == null) return "null";
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }
}
