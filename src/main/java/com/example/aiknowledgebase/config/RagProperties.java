package com.example.aiknowledgebase.config;

import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * RAG 检索相关配置。
 * <p>
 * 只暴露会被实际调整的参数。后续每新增一个检索策略（Hybrid / Rerank / Query Rewrite）
 * 时，再按需追加它自己的开关，不提前占位没有数据可调的配置项。
 * <p>
 * 注意：embedding 与生成模型名不在这里，它们统一由 {@code ollama.model} 与
 * {@code ollama.embedding-model} 定义，避免同一配置出现两个来源。
 */
@Component
@ConfigurationProperties(prefix = "rag")
public class RagProperties {

    private static final Logger log = LoggerFactory.getLogger(RagProperties.class);

    /**
     * 余弦相似度下界，用于标记"不做阈值过滤"。
     * <p>
     * 余弦相似度的取值范围是 [-1, 1]，因此 -1 表示"任何分数都保留"。
     * 不能用 0 表示关闭——0 会静默丢弃所有负相关的文本块，那是一次行为变更而非关闭过滤。
     */
    public static final double MIN_SCORE_DISABLED = -1.0;

    /** 文本切分：每块最大字符数 */
    private int chunkSize = 500;

    /** 文本切分：相邻块重叠字符数，必须小于 chunkSize */
    private int chunkOverlap = 100;

    /** 检索返回的 Top-K 文本块数量 */
    private int topK = 3;

    /**
     * 相似度阈值：低于该值的召回结果会被丢弃。
     * <p>
     * 默认 {@link #MIN_SCORE_DISABLED}（不过滤）。该取值必须由检索评估集校准后确定，
     * 不要凭经验拍一个数——在中文短查询上不同 embedding 模型的分数分布差异很大。
     */
    private double minScore = MIN_SCORE_DISABLED;

    private final Redis redis = new Redis();

    /**
     * 启动时校验当前 embedding 模型维度与库内已存向量是否一致（不一致时仅告警，不阻断启动）。
     */
    private boolean startupCheckEnabled = true;

    /**
     * Redis 存储相关配置。
     */
    public static class Redis {

        /**
         * 所有 RAG 相关 key 的统一前缀。
         * <p>
         * 测试必须覆盖该值（见 application-test.properties），否则测试清理动作
         * 会直接删除生产知识库数据。
         */
        private String keyPrefix = "rag";

        public String getKeyPrefix() {
            return keyPrefix;
        }

        public void setKeyPrefix(String keyPrefix) {
            this.keyPrefix = keyPrefix;
        }
    }

    /**
     * 配置校验：参数不合法时在启动阶段就失败，而不是等到第一次上传文档才暴露。
     */
    @PostConstruct
    void validate() {
        if (chunkSize <= 0) {
            throw new IllegalStateException("rag.chunk-size 必须大于 0，当前值: " + chunkSize);
        }
        if (chunkOverlap < 0) {
            throw new IllegalStateException("rag.chunk-overlap 不能为负数，当前值: " + chunkOverlap);
        }
        if (chunkOverlap >= chunkSize) {
            throw new IllegalStateException(
                    "rag.chunk-overlap (" + chunkOverlap + ") 必须小于 rag.chunk-size (" + chunkSize + ")");
        }
        if (topK <= 0) {
            throw new IllegalStateException("rag.top-k 必须大于 0，当前值: " + topK);
        }
        if (minScore < -1.0 || minScore > 1.0) {
            throw new IllegalStateException(
                    "rag.min-score 必须在 [-1, 1] 区间内，当前值: " + minScore);
        }
        if (redis.getKeyPrefix() == null || redis.getKeyPrefix().isBlank()) {
            throw new IllegalStateException("rag.redis.key-prefix 不能为空");
        }
        log.info("RAG 配置: chunkSize={}, chunkOverlap={}, topK={}, minScore={}, keyPrefix={}, startupCheckEnabled={}",
                chunkSize, chunkOverlap, topK,
                minScore <= MIN_SCORE_DISABLED ? "disabled" : minScore,
                redis.getKeyPrefix(), startupCheckEnabled);
    }

    // ==================== getter / setter ====================

    public int getChunkSize() {
        return chunkSize;
    }

    public void setChunkSize(int chunkSize) {
        this.chunkSize = chunkSize;
    }

    public int getChunkOverlap() {
        return chunkOverlap;
    }

    public void setChunkOverlap(int chunkOverlap) {
        this.chunkOverlap = chunkOverlap;
    }

    public int getTopK() {
        return topK;
    }

    public void setTopK(int topK) {
        this.topK = topK;
    }

    public double getMinScore() {
        return minScore;
    }

    public void setMinScore(double minScore) {
        this.minScore = minScore;
    }

    public Redis getRedis() {
        return redis;
    }

    public boolean isStartupCheckEnabled() {
        return startupCheckEnabled;
    }

    public void setStartupCheckEnabled(boolean startupCheckEnabled) {
        this.startupCheckEnabled = startupCheckEnabled;
    }

    /**
     * 是否启用相似度阈值过滤。
     */
    public boolean isMinScoreEnabled() {
        return minScore > MIN_SCORE_DISABLED;
    }
}
