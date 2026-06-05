package com.example.aiknowledgebase.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class DocumentInfo {
    private String fileName;
    private Long fileSize;
    private Long uploadTime;
}