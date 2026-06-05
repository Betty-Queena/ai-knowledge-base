package com.example.aiknowledgebase.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Jackson 配置 — 向 Spring 容器注册 ObjectMapper Bean。
 * <p>
 * VectorStore 通过构造器注入 ObjectMapper，用于 TextChunk 的 JSON 序列化与反序列化。
 */
@Configuration
public class JacksonConfig {

    @Bean
    public ObjectMapper objectMapper() {
        return new ObjectMapper();
    }
}
