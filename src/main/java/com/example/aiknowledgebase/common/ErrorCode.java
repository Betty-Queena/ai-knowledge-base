package com.example.aiknowledgebase.common;

/**
 * 业务错误码。
 * <p>
 * 调用方应基于 {@link #getCode()} 判断错误类型，而不是比较异常消息文本
 * （消息文本会随文案调整而变化，字符串比较会让分支静默失效）。
 */
public enum ErrorCode {

    /** 请求参数不合法 */
    PARAM_INVALID(400, "参数错误"),

    /** 知识库中没有任何已索引的文本块 */
    KNOWLEDGE_BASE_EMPTY(4001, "知识库为空"),

    /** 本地大模型（Ollama）不可用或返回异常 */
    LLM_UNAVAILABLE(5001, "大模型服务不可用"),

    /** 向量化（Embedding）失败 */
    EMBEDDING_FAILED(5002, "向量化失败"),

    /** 系统内部异常 */
    INTERNAL_ERROR(500, "系统异常");

    private final int code;
    private final String defaultMessage;

    ErrorCode(int code, String defaultMessage) {
        this.code = code;
        this.defaultMessage = defaultMessage;
    }

    public int getCode() {
        return code;
    }

    public String getDefaultMessage() {
        return defaultMessage;
    }
}
