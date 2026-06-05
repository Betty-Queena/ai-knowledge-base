package com.example.aiknowledgebase.service.impl;

import com.example.aiknowledgebase.common.BusinessException;
import com.example.aiknowledgebase.dto.DocumentInfo;
import com.example.aiknowledgebase.service.DocumentService;
import com.example.aiknowledgebase.service.RagService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.nio.charset.StandardCharsets;

@Service
public class DocumentServiceImpl implements DocumentService {

    private static final Logger log = LoggerFactory.getLogger(DocumentServiceImpl.class);

    private static final String DOCUMENT_KEY_PREFIX = "kb:document:";

    private final RedisTemplate<String, Object> redisTemplate;
    private final RagService ragService;

    public DocumentServiceImpl(RedisTemplate<String, Object> redisTemplate,
                               RagService ragService) {
        this.redisTemplate = redisTemplate;
        this.ragService = ragService;
    }

    @Override
    public DocumentInfo upload(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException(400, "上传文件不能为空");
        }

        String fileName = file.getOriginalFilename();

        if (fileName == null ||
                !(fileName.endsWith(".txt") || fileName.endsWith(".md"))) {
            throw new BusinessException(400, "当前仅支持上传 txt 或 md 文件");
        }

        try {
            String content = new String(file.getBytes(), StandardCharsets.UTF_8);

            // ── 1. 写入 Redis（原有逻辑） ───────────────────────────
            String redisKey = DOCUMENT_KEY_PREFIX + fileName;
            redisTemplate.opsForValue().set(redisKey, content);

            // ── 2. 自动接入 RAG 知识库 ──────────────────────────────
            try {
                int chunkCount = ragService.indexDocument(fileName, content);
                log.info("文档 {} 已自动索引至 RAG 知识库（{} 个文本块）",
                        fileName, chunkCount);
            } catch (Exception e) {
                // RAG 索引失败不影响上传结果，记录日志由运维排查
                log.error("文档 {} RAG 索引失败: {}", fileName, e.getMessage());
            }

            return new DocumentInfo(
                    fileName,
                    file.getSize(),
                    System.currentTimeMillis()
            );

        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new BusinessException("文件上传失败：" + e.getMessage());
        }
    }
}