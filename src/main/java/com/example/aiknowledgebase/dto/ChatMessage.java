package com.example.aiknowledgebase.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ChatMessage {
    private String question;
    private String answer;
    private Long createTime;
}