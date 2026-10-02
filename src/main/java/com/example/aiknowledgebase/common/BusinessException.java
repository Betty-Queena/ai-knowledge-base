package com.example.aiknowledgebase.common;

public class BusinessException extends RuntimeException {

    private final Integer code;

    public BusinessException(String message) {
        super(message);
        this.code = 500;
    }

    public BusinessException(Integer code, String message) {
        super(message);
        this.code = code;
    }

    /**
     * 使用统一错误码构造。推荐用法，便于调用方按 {@link ErrorCode} 判断分支。
     */
    public BusinessException(ErrorCode errorCode, String message) {
        super(message);
        this.code = errorCode.getCode();
    }

    /**
     * 使用统一错误码构造并保留原始异常。
     * <p>
     * 保留 cause 是为了不丢失底层诊断信息（如连接超时、HTTP 状态码），
     * 便于日志中直接看到根因。
     */
    public BusinessException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.code = errorCode.getCode();
    }

    public Integer getCode() {
        return code;
    }

    /**
     * 判断本异常是否属于指定错误码。
     */
    public boolean is(ErrorCode errorCode) {
        return errorCode != null && this.code != null && this.code == errorCode.getCode();
    }
}
