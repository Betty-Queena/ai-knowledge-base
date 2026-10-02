package com.example.aiknowledgebase.config;

import com.example.aiknowledgebase.service.EmbeddingService;
import com.example.aiknowledgebase.store.VectorStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

import java.util.OptionalInt;

/**
 * 启动自检 — 校验当前 embedding 模型维度与库内已存向量是否一致。
 * <p>
 * 背景：更换 {@code ollama.embedding-model} 后，旧索引里的向量维度与新模型不一致，
 * 检索会跳过全部旧向量并返回空结果（表现为"知识库突然什么都查不到"）。
 * 该问题的排查成本很高，因此在启动时主动检查并给出明确提示。
 * <p>
 * 检查失败不阻断启动：Ollama 未启动属于预期情况（例如只做单元测试时），
 * 此时仅记录警告。
 */
@Component
public class RagStartupCheck implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(RagStartupCheck.class);

    /** 自检用的探针文本，内容本身不重要，只用于取维度 */
    private static final String PROBE_TEXT = "向量维度自检";

    private final VectorStore vectorStore;
    private final EmbeddingService embeddingService;
    private final RagProperties ragProperties;

    public RagStartupCheck(VectorStore vectorStore,
                           EmbeddingService embeddingService,
                           RagProperties ragProperties) {
        this.vectorStore = vectorStore;
        this.embeddingService = embeddingService;
        this.ragProperties = ragProperties;
    }

    @Override
    public void run(ApplicationArguments args) {
        if (!ragProperties.isStartupCheckEnabled()) {
            log.info("RAG 启动自检已关闭（rag.startup-check-enabled=false）");
            return;
        }

        try {
            OptionalInt storedDimension = vectorStore.sampleEmbeddingDimension();
            if (storedDimension.isEmpty()) {
                log.info("知识库为空，跳过向量维度自检");
                return;
            }

            float[] probeEmbedding = embeddingService.embed(PROBE_TEXT);
            int currentDimension = probeEmbedding.length;

            if (currentDimension == storedDimension.getAsInt()) {
                log.info("向量维度自检通过: dim={}, 已索引文本块 {} 个",
                        currentDimension, vectorStore.size());
            } else {
                log.error("向量维度不一致：已存索引 dim={}，当前 embedding 模型 dim={}。"
                                + "检索会跳过全部旧向量并返回空结果，请清理并重建索引"
                                + "（或把 ollama.embedding-model 改回原模型）。",
                        storedDimension.getAsInt(), currentDimension);
            }
        } catch (Exception e) {
            log.warn("向量维度自检未执行（无法访问 embedding 模型）: {}", e.getMessage());
        }
    }
}
