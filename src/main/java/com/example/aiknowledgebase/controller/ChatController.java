package com.example.aiknowledgebase.controller;

import com.example.aiknowledgebase.common.Result;
import com.example.aiknowledgebase.dto.ChatMessage;
import com.example.aiknowledgebase.dto.ChatRequest;
import com.example.aiknowledgebase.dto.ChatResponse;
import com.example.aiknowledgebase.service.ChatService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final ChatService chatService;

    public ChatController(ChatService chatService) {
        this.chatService = chatService;
    }

    @PostMapping("/ask")
    public Result<ChatResponse> ask(@RequestBody ChatRequest request) {
        String answer = chatService.ask(request.getQuestion(), request.getSessionId());
        return Result.success(new ChatResponse(answer));
    }

    @GetMapping("/history")
    public Result<List<ChatMessage>> history() {
        return Result.success(chatService.history());
    }
}