package com.example.aiknowledgebase.service;

import com.example.aiknowledgebase.dto.DocumentInfo;
import org.springframework.web.multipart.MultipartFile;

public interface DocumentService {
    DocumentInfo upload(MultipartFile file);
}