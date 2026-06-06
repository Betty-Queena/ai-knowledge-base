package com.example.aiknowledgebase.service;

import com.example.aiknowledgebase.dto.ChatMessage;

import java.util.List;

public interface ChatService {
    String ask(String question, String sessionId);

    List<ChatMessage> history();
}