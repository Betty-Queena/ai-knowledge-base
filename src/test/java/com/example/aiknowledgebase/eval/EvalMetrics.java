package com.example.aiknowledgebase.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 评测指标计算 —— 纯函数集合：不读文件、不依赖 Spring、不含随机性。
 * <p>
 * 之所以把它独立成"无 IO 的纯函数"，是因为指标算错时表现和"检索变差"一模一样，
 * 把它与 IO 分开后可以用 {@link EvalMetricsTest} 用手算样例直接验证。
 * <p>
 * 相关性判定分两层，缺一不可：
 * <ul>
 *   <li>{@link Level#DOC}：命中的 chunk 属于 gold 文档 —— 回答"有没有定位到正确文档"</li>
 *   <li>{@link Level#CHUNK}：命中的 chunk 本身就是 gold chunk —— 回答"真正含答案的那段上下文有没有被召回"</li>
 * </ul>
 */
public final class EvalMetrics {

    private EvalMetrics() {
    }

    /** 评测的 K 取值 */
    public static final int[] K_VALUES = {1, 3, 5};

    /** 相关性判定层级 */
    public enum Level {
        /** 文档级 */
        DOC,
        /** 文本块级 */
        CHUNK
    }

    /**
     * 生成层拒答的标志串。
     * <p>
     * 两条路径：prompt 规则要求模型回答"根据提供的资料，无法回答该问题"；
     * 检索结果为空时 {@code RagService.ask} 直接返回"知识库中未找到与问题相关的内容"。
     * 判定写死为字符串匹配，不用 LLM 判 LLM —— 后者不可重复、也无法作为基线。
     */
    public static final List<String> REFUSAL_MARKERS = List.of("无法回答", "未找到");

    /**
     * 一条样本的检索原始记录，rank = 列表下标 + 1。
     */
    public record RetrievalRecord(String caseId,
                                  List<String> goldDocIds,
                                  List<String> goldChunkIds,
                                  List<String> topDocIds,
                                  List<String> topChunkIds,
                                  List<Double> topScores) {

        public List<String> topIds(Level level) {
            return level == Level.DOC ? topDocIds : topChunkIds;
        }

        public List<String> goldIds(Level level) {
            return level == Level.DOC ? goldDocIds : goldChunkIds;
        }

        /**
         * 前 k 条中第一条相关结果的名次（1 起）；前 k 条内没有相关结果返回 0。
         */
        public int firstRelevantRank(Level level, int k) {
            Set<String> gold = new HashSet<>(goldIds(level));
            List<String> top = topIds(level);
            int limit = Math.min(k, top.size());
            for (int i = 0; i < limit; i++) {
                if (gold.contains(top.get(i))) {
                    return i + 1;
                }
            }
            return 0;
        }

        /** top-1 分数；一条结果都没有时返回 null */
        public Double top1Score() {
            return topScores.isEmpty() ? null : topScores.get(0);
        }
    }

    /** 分数分布的五数概括 */
    public record ScoreSummary(int n, double min, double p25, double p50, double p75, double max) {

        public static ScoreSummary of(List<Double> values) {
            if (values == null || values.isEmpty()) {
                return new ScoreSummary(0, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN);
            }
            return new ScoreSummary(values.size(),
                    percentileNearestRank(values, 0),
                    percentileNearestRank(values, 25),
                    percentileNearestRank(values, 50),
                    percentileNearestRank(values, 75),
                    percentileNearestRank(values, 100));
        }
    }

    // ==================== 指标 ====================

    /**
     * HitRate@K = 前 K 条中存在相关结果的样本数 / 样本总数。
     */
    public static double hitRateAtK(List<RetrievalRecord> records, Level level, int k) {
        if (records.isEmpty()) {
            return Double.NaN;
        }
        int hit = 0;
        for (RetrievalRecord r : records) {
            if (r.firstRelevantRank(level, k) > 0) {
                hit++;
            }
        }
        return (double) hit / records.size();
    }

    /**
     * MRR@K = 前 K 条中首个相关结果名次倒数的平均值；前 K 条内无相关结果记 0。
     */
    public static double mrrAtK(List<RetrievalRecord> records, Level level, int k) {
        if (records.isEmpty()) {
            return Double.NaN;
        }
        double sum = 0;
        for (RetrievalRecord r : records) {
            int rank = r.firstRelevantRank(level, k);
            if (rank > 0) {
                sum += 1.0 / rank;
            }
        }
        return sum / records.size();
    }

    /**
     * 分位数（nearest-rank 法）：升序排序后取第 {@code ceil(p/100 × n)} 个（1 起）。
     * <p>
     * 明确写死算法是刻意的：不同分位数算法在同一批数据上会给出不同的 p95，
     * 两次运行若用了不同算法，数字就不可比。
     *
     * @param p 0–100
     */
    public static double percentileNearestRank(List<Double> values, double p) {
        if (values == null || values.isEmpty()) {
            return Double.NaN;
        }
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int n = sorted.size();
        int index = (int) Math.ceil(p / 100.0 * n) - 1;
        index = Math.max(0, Math.min(n - 1, index));
        return sorted.get(index);
    }

    /**
     * 探索性阈值区间（exploratory threshold interval）：
     * {@code [p90(负样本 top1), p10(正样本 top1)]}。
     * <p>
     * <b>它只是"值得进一步做对比实验"的方向，不是推荐阈值，也不是最佳阈值</b>；
     * 本阶段固定 {@code rag.min-score=-1}，不据此修改任何配置。
     *
     * @return 长度为 2 的数组；任一侧样本为空时返回 null
     */
    public static double[] exploratoryThresholdInterval(List<Double> unanswerableTop1,
                                                        List<Double> answerableTop1) {
        if (unanswerableTop1 == null || unanswerableTop1.isEmpty()
                || answerableTop1 == null || answerableTop1.isEmpty()) {
            return null;
        }
        return new double[]{
                percentileNearestRank(unanswerableTop1, 90),
                percentileNearestRank(answerableTop1, 10)};
    }

    /**
     * 判断一次生成结果是否属于拒答（字符串匹配，见 {@link #REFUSAL_MARKERS}）。
     */
    public static boolean isRefusal(String answer) {
        if (answer == null || answer.isBlank()) {
            return false;
        }
        for (String marker : REFUSAL_MARKERS) {
            if (answer.contains(marker)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 保留指定小数位，仅用于报告展示（不参与任何计算）。
     */
    public static String format(double value, int digits) {
        if (Double.isNaN(value)) {
            return "n/a";
        }
        return String.format("%." + digits + "f", value);
    }
}
