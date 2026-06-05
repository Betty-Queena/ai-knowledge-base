package com.example.aiknowledgebase.model;

/**
 * 文本块 — 知识库中最小的检索单元。
 * <p>
 * 将一篇文档切分为多个 TextChunk，每个 Chunk 对应一段文本及其向量表示。
 */
public class TextChunk {

    /** 唯一标识 */
    private String id;

    /** 所属文档的 ID */
    private String documentId;

    /** 文本内容 */
    private String content;

    /** 向量表示（embedding） */
    private float[] embedding;

    public TextChunk() {
    }

    public TextChunk(String id, String documentId, String content, float[] embedding) {
        this.id = id;
        this.documentId = documentId;
        this.content = content;
        this.embedding = embedding;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getDocumentId() {
        return documentId;
    }

    public void setDocumentId(String documentId) {
        this.documentId = documentId;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public float[] getEmbedding() {
        return embedding;
    }

    public void setEmbedding(float[] embedding) {
        this.embedding = embedding;
    }

    @Override
    public String toString() {
        return "TextChunk{" +
                "id='" + id + '\'' +
                ", documentId='" + documentId + '\'' +
                ", content='" + content + '\'' +
                ", embedding=" + (embedding != null ? embedding.length + "d" : "null") +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TextChunk textChunk = (TextChunk) o;
        return java.util.Objects.equals(id, textChunk.id);
    }

    @Override
    public int hashCode() {
        return java.util.Objects.hash(id);
    }
}
