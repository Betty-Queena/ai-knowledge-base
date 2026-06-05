package com.example.aiknowledgebase.service;

import com.example.aiknowledgebase.dto.ChatMessage;
import com.example.aiknowledgebase.dto.ChatRequest;
import com.example.aiknowledgebase.dto.ChatResponse;

import java.util.List;

public interface ChatService {
    ChatResponse ask(ChatRequest request);

    List<ChatMessage> history();
}