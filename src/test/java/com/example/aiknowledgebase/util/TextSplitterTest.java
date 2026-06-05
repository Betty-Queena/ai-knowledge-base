package com.example.aiknowledgebase.util;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link TextSplitter} 的单元测试。
 */
class TextSplitterTest {

    // ──────────────── 异常参数 ────────────────

    @Test
    @DisplayName("null 文本应返回空列表")
    void nullText() {
        assertTrue(TextSplitter.split(null, 10, 2).isEmpty());
    }

    @Test
    @DisplayName("空文本应返回空列表")
    void emptyText() {
        assertTrue(TextSplitter.split("", 10, 2).isEmpty());
    }

    @Test
    @DisplayName("chunkSize 为 0 应抛出异常")
    void chunkSizeZero() {
        assertThrows(IllegalArgumentException.class,
                () -> TextSplitter.split("hello", 0, 0));
    }

    @Test
    @DisplayName("chunkSize 为负数应抛出异常")
    void chunkSizeNegative() {
        assertThrows(IllegalArgumentException.class,
                () -> TextSplitter.split("hello", -1, 0));
    }

    @Test
    @DisplayName("overlap 为负数应抛出异常")
    void overlapNegative() {
        assertThrows(IllegalArgumentException.class,
                () -> TextSplitter.split("hello", 10, -1));
    }

    @Test
    @DisplayName("overlap ≥ chunkSize 应抛出异常")
    void overlapTooLarge() {
        assertThrows(IllegalArgumentException.class,
                () -> TextSplitter.split("hello", 5, 5));
        assertThrows(IllegalArgumentException.class,
                () -> TextSplitter.split("hello", 5, 6));
    }

    // ──────────────── 正常切分 ────────────────

    @Test
    @DisplayName("文本长度 ≤ chunkSize 应返回单个元素")
    void textShorterThanChunkSize() {
        List<String> result = TextSplitter.split("hello", 10, 0);
        assertEquals(List.of("hello"), result);
    }

    @Test
    @DisplayName("文本长度 == chunkSize 应返回单个元素")
    void textEqualsChunkSize() {
        List<String> result = TextSplitter.split("hello", 5, 0);
        assertEquals(List.of("hello"), result);
    }

    @Test
    @DisplayName("无重叠切分：多个 chunk 不应有重叠字符")
    void splitWithoutOverlap() {
        String text = "HelloWorldJava";
        List<String> chunks = TextSplitter.split(text, 5, 0);

        // 预期: ["Hello", "World", "Java"]
        assertEquals(3, chunks.size());
        assertEquals("Hello", chunks.get(0));
        assertEquals("World", chunks.get(1));
        assertEquals("Java", chunks.get(2));
    }

    @Test
    @DisplayName("有重叠切分：相邻 chunk 之间存在指定长度的重叠")
    void splitWithOverlap() {
        String text = "HelloWorld";
        List<String> chunks = TextSplitter.split(text, 5, 2);

        // chunkSize=5, overlap=2 → stride=3
        // start=0, end=5 → "Hello"      [0-5)
        // start=3, end=8 → "loWor"      [3-8)   ← 与上一块重叠 "lo"（位置 3-5）
        // start=6, end=10 → "orld"      [6-10)
        assertEquals(3, chunks.size());
        assertEquals("Hello", chunks.get(0));
        assertEquals("loWor", chunks.get(1));
        assertEquals("orld", chunks.get(2));

        // 验证重叠：chunk0[3-5) == chunk1[0-2) == "lo"
        assertEquals(
                chunks.get(0).substring(3),
                chunks.get(1).substring(0, 2)
        );
    }

    @Test
    @DisplayName("重叠字符数 = 1 的基本场景")
    void overlapOne() {
        String text = "abcdefghij";
        List<String> chunks = TextSplitter.split(text, 4, 1);

        // stride=3
        // start=0, end=4 → "abcd"
        // start=3, end=7 → "defg"
        // start=6, end=10 → "ghij"
        assertEquals(3, chunks.size());
        assertEquals("abcd", chunks.get(0));
        assertEquals("defg", chunks.get(1));
        assertEquals("ghij", chunks.get(2));

        // chunk0[3] == chunk1[0] == 'd'
        assertEquals(chunks.get(0).charAt(3), chunks.get(1).charAt(0));
    }

    @Test
    @DisplayName("大重叠比例：overlap 接近 chunkSize")
    void largeOverlap() {
        String text = "abcdefgh";
        List<String> chunks = TextSplitter.split(text, 6, 5);

        // stride=1
        // start=0, end=6 → "abcdef"
        // start=1, end=7 → "bcdefg"
        // start=2, end=8 → "cdefgh"
        assertEquals(3, chunks.size());
        assertEquals("abcdef", chunks.get(0));
        assertEquals("bcdefg", chunks.get(1));
        assertEquals("cdefgh", chunks.get(2));
    }

    @Test
    @DisplayName("无重叠 & 文本恰好被整除")
    void exactDivisionNoOverlap() {
        // "12345678" 长度 8，chunkSize=4 恰好 2 块
        List<String> chunks = TextSplitter.split("12345678", 4, 0);
        assertEquals(2, chunks.size());
        assertEquals("1234", chunks.get(0));
        assertEquals("5678", chunks.get(1));
    }

    @Test
    @DisplayName("单字符 chunk")
    void singleCharChunk() {
        List<String> chunks = TextSplitter.split("abc", 1, 0);
        assertEquals(3, chunks.size());
        assertEquals("a", chunks.get(0));
        assertEquals("b", chunks.get(1));
        assertEquals("c", chunks.get(2));
    }

    @Test
    @DisplayName("返回的列表不可修改")
    void resultIsUnmodifiable() {
        List<String> result = TextSplitter.split("hello world", 5, 0);
        assertThrows(UnsupportedOperationException.class, () -> result.add("x"));
    }

    // ──────────────── 参数化测试：随机校验 ────────────────

    @ParameterizedTest
    @CsvSource({
            "HelloWorldJava, 5, 0, 3",
            "HelloWorldJava, 5, 1, 4",
            "HelloWorldJava, 5, 2, 4",
            "HelloWorld,      4, 1, 3",
            "abcdefghij,      3, 1, 5",
            "a,               2, 0, 1",
    })
    @DisplayName("参数化：切分后的 chunk 数量符合预期")
    void chunkCount(String text, int chunkSize, int overlap, int expectedCount) {
        List<String> chunks = TextSplitter.split(text, chunkSize, overlap);
        assertEquals(expectedCount, chunks.size(),
                () -> "文本 \"" + text + "\" chunkSize=" + chunkSize
                        + " overlap=" + overlap + " 预期 " + expectedCount + " 块");
    }

    @Test
    @DisplayName("所有 chunk 的长度均不超过 chunkSize")
    void allChunksWithinSizeLimit() {
        String text = "这是一段用于测试分块工具的中文文本，包含了一些标点符号。" +
                "我们的目标是验证每个切分后的文本块都不超过指定的最大字符数。" +
                "同时还要确保相邻块之间有正确的重叠字符。";
        int chunkSize = 20;
        int overlap = 5;
        List<String> chunks = TextSplitter.split(text, chunkSize, overlap);

        assertFalse(chunks.isEmpty());
        for (int i = 0; i < chunks.size(); i++) {
            assertTrue(chunks.get(i).length() <= chunkSize,
                    "chunk[" + i + "] 长度 " + chunks.get(i).length() + " 超过 " + chunkSize);
        }
    }

    @Test
    @DisplayName("相邻 chunk 之间的重叠字符数符合预期")
    void overlapCorrectness() {
        String text = "这是一个用于测试重叠正确性的长文本，我们需要确保每个相邻块之间的重叠字符数符合预期。";
        int chunkSize = 15;
        int overlap = 5;
        List<String> chunks = TextSplitter.split(text, chunkSize, overlap);

        for (int i = 0; i < chunks.size() - 1; i++) {
            String current = chunks.get(i);
            String next = chunks.get(i + 1);
            int idx = i;

            // 当前块的末尾 overlap 个字符 = 下一块的开头 overlap 个字符
            String tailOfCurrent = current.substring(Math.max(0, current.length() - overlap));
            String headOfNext = next.substring(0, Math.min(overlap, next.length()));

            int actualOverlap = Math.min(tailOfCurrent.length(), headOfNext.length());
            assertEquals(
                    tailOfCurrent.substring(0, actualOverlap),
                    headOfNext.substring(0, actualOverlap),
                    () -> "chunk[" + idx + "] 与 chunk[" + (idx + 1) + "] 之间的 overlap 不匹配"
            );
        }
    }
}
