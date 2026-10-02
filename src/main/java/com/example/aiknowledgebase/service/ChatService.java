package com.example.aiknowledgebase.service;

import com.example.aiknowledgebase.dto.ChatMessage;

import java.util.List;

public interface ChatService {

    String ask(String question, String sessionId);

    /**
     * 查询指定会话的历史消息。
     * <p>
     * 会话历史按 sessionId 分开存储，因此查询必须带 sessionId；sessionId 为空时
     * 会落到默认会话（与 {@link #ask(String, String)} 的规则一致）。
     *
     * @param sessionId 会话标识，可为空
     * @return 时间正序（旧 → 新）的历史消息列表
     */
    List<ChatMessage> history(String sessionId);
}
