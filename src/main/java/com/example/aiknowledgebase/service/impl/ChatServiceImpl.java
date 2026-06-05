package com.example.aiknowledgebase.service.impl;

import com.example.aiknowledgebase.client.OllamaClient;
import com.example.aiknowledgebase.common.BusinessException;
import com.example.aiknowledgebase.dto.ChatMessage;
import com.example.aiknowledgebase.dto.ChatRequest;
import com.example.aiknowledgebase.dto.ChatResponse;
import com.example.aiknowledgebase.service.ChatService;
import com.example.aiknowledgebase.service.RagService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.List;

@Service
public class ChatServiceImpl implements ChatService {

    private static final Logger log = LoggerFactory.getLogger(ChatServiceImpl.class);

    private static final String CHAT_HISTORY_KEY = "chat:history";

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
    public ChatResponse ask(ChatRequest request) {
        if (request == null || !StringUtils.hasText(request.getQuestion())) {
            throw new BusinessException(400, "问题不能为空");
        }

        String question = request.getQuestion();
        String answer = generateAnswer(question);

        ChatMessage message = new ChatMessage(
                question,
                answer,
                System.currentTimeMillis()
        );

        redisTemplate.opsForList().leftPush(CHAT_HISTORY_KEY, message);

        // 只保留最近 20 条，避免 Redis 无限增长
        redisTemplate.opsForList().trim(CHAT_HISTORY_KEY, 0, 19);

        return new ChatResponse(answer);
    }

    /**
     * 优先使用 RAG 检索增强回答，失败则降级为普通对话。
     * <p>
     * 降级条件：
     * <ul>
     *   <li>知识库为空（未上传文档）</li>
     *   <li>Embedding 或 Ollama 调用异常</li>
     * </ul>
     */
    private String generateAnswer(String question) {
        try {
            String answer = ragService.ask(question);
            log.info("RAG 回答成功: question=\"{}\"", truncate(question, 50));
            return answer;
        } catch (BusinessException e) {
            // 知识库为空是预期情况，非异常
            if ("知识库为空，请先索引文档".equals(e.getMessage())) {
                log.info("知识库为空，降级为普通对话: question=\"{}\"", truncate(question, 50));
            } else {
                log.warn("RAG 回答失败 ({}), 降级为普通对话: {}",
                        e.getMessage(), truncate(question, 50));
            }
        } catch (Exception e) {
            log.warn("RAG 回答异常 ({}), 降级为普通对话: {}",
                    e.getMessage(), truncate(question, 50));
        }

        // 降级：走普通 Ollama 对话
        String fallbackAnswer = ollamaClient.generate(question);
        log.info("普通对话回答成功: question=\"{}\"", truncate(question, 50));
        return fallbackAnswer;
    }

    private static String truncate(String s, int maxLen) {
        if (s == null) return "null";
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }

    @Override
    public List<ChatMessage> history() {
        List<Object> list = redisTemplate.opsForList().range(CHAT_HISTORY_KEY, 0, 19);

        if (list == null) {
            return List.of();
        }

        return list.stream()
                .filter(ChatMessage.class::isInstance)
                .map(ChatMessage.class::cast)
                .toList();
    }
}