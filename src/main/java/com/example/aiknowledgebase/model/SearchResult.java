package com.example.aiknowledgebase.model;

/**
 * 向量检索结果 — 一个匹配的文本块及其相似度得分。
 */
public class SearchResult {

    /** 匹配到的文本块 */
    private TextChunk chunk;

    /** 与查询向量的余弦相似度得分，范围 [-1, 1] */
    private double score;

    public SearchResult() {
    }

    public SearchResult(TextChunk chunk, double score) {
        this.chunk = chunk;
        this.score = score;
    }

    public TextChunk getChunk() {
        return chunk;
    }

    public void setChunk(TextChunk chunk) {
        this.chunk = chunk;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    @Override
    public String toString() {
        return "SearchResult{" +
                "score=" + String.format("%.4f", score) +
                ", content='" + (chunk != null ? chunk.getContent() : "null") + '\'' +
                '}';
    }
}
