package com.example.aiknowledgebase.service;

import com.example.aiknowledgebase.client.OllamaClient;
import com.example.aiknowledgebase.common.BusinessException;
import org.springframework.stereotype.Service;

/**
 * Embedding 服务 — 将文本向量化。
 * <p>
 * 封装 {@link OllamaClient#embed(String)} 调用，提供参数校验。
 */
@Service
public class EmbeddingService {

    private final OllamaClient ollamaClient;

    public EmbeddingService(OllamaClient ollamaClient) {
        this.ollamaClient = ollamaClient;
    }

    /**
     * 将指定文本转为向量。
     *
     * @param text 待向量化的文本，不能为 null 或空
     * @return float 数组，维度取决于底层模型
     * @throws BusinessException 当 text 为空时抛出
     */
    public float[] embed(String text) {
        if (text == null || text.isBlank()) {
            throw new BusinessException(400, "待向量化的文本不能为空");
        }
        return ollamaClient.embed(text);
    }
}
