package com.example.aiknowledgebase.controller;

import com.example.aiknowledgebase.common.Result;
import com.example.aiknowledgebase.dto.DocumentInfo;
import com.example.aiknowledgebase.service.DocumentService;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/documents")
public class DocumentController {

    private final DocumentService documentService;

    public DocumentController(DocumentService documentService) {
        this.documentService = documentService;
    }

    @PostMapping("/upload")
    public Result<DocumentInfo> upload(@RequestParam("file") MultipartFile file) {
        return Result.success(documentService.upload(file));
    }
}