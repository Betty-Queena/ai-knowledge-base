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

    /**
     * 查询指定会话的历史消息（时间正序）。
     * <p>
     * 会话历史按 sessionId 分开存储，因此需要带上 sessionId；不传则落到默认会话。
     */
    @GetMapping("/history")
    public Result<List<ChatMessage>> history(
            @RequestParam(value = "sessionId", required = false) String sessionId) {
        return Result.success(chatService.history(sessionId));
    }
}