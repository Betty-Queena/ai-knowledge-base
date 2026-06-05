
package com.example.aiknowledgebase.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 文本分块工具。
 * <p>
 * 将长文本按指定块大小和重叠字符数切分为多个文本块，
 * 用于后续向量化（Embedding）和检索增强生成（RAG）。
 */
public class TextSplitter {

    /**
     * 将文本切分为多个重叠的文本块。
     *
     * @param text      待切分的原始文本
     * @param chunkSize 每个文本块的最大字符数，必须大于 0
     * @param overlap   相邻文本块之间的重叠字符数，必须 ≥ 0 且 < chunkSize
     * @return 文本块列表。若 text 为 null 或空，返回空列表；
     *         若 text 长度 ≤ chunkSize，返回仅包含 text 本身的单元素列表。
     * @throws IllegalArgumentException 当 chunkSize ≤ 0、overlap < 0 或 overlap ≥ chunkSize 时抛出
     */
    public static List<String> split(String text, int chunkSize, int overlap) {
        // ── 参数校验 ──────────────────────────────────────────────
        if (text == null || text.isEmpty()) {
            return Collections.emptyList();
        }
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunkSize 必须大于 0，当前值: " + chunkSize);
        }
        if (overlap < 0) {
            throw new IllegalArgumentException("overlap 不能为负数，当前值: " + overlap);
        }
        if (overlap >= chunkSize) {
            throw new IllegalArgumentException(
                    "overlap (" + overlap + ") 必须小于 chunkSize (" + chunkSize + ")");
        }

        // ── 文本长度 <= 块大小，无需切分 ────────────────────────────
        if (text.length() <= chunkSize) {
            return List.of(text);
        }

        // ── 滑动窗口切分 ───────────────────────────────────────────
        List<String> chunks = new ArrayList<>();
        int stride = chunkSize - overlap; // 每次前进的步长
        int start = 0;

        while (start < text.length()) {
            int end = Math.min(start + chunkSize, text.length());
            chunks.add(text.substring(start, end));

            // 已到达文本末尾，结束循环
            if (end == text.length()) {
                break;
            }

            start += stride;
        }

        return Collections.unmodifiableList(chunks);
    }
}
