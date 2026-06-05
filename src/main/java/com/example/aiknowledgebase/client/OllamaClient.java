package com.example.aiknowledgebase.client;

import com.example.aiknowledgebase.common.BusinessException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

@Component
public class OllamaClient {

    @Value("${ollama.url}")
    private String ollamaUrl;

    @Value("${ollama.model}")
    private String modelName;

    @Value("${ollama.embedding-model:nomic-embed-text:latest}")
    private String embeddingModelName;

    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(OllamaClient.class);

    private final ObjectMapper objectMapper = new ObjectMapper();

    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    /**
     * 调用 Ollama /api/generate，生成文本回复（非流式）
     */
    public String generate(String prompt) {
        try {
            Map<String, Object> body = new HashMap<>();
            body.put("model", modelName);
            body.put("prompt", prompt);
            body.put("stream", false);

            String jsonBody = objectMapper.writeValueAsString(body);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(ollamaUrl))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(60))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString()
            );

            JsonNode jsonNode = objectMapper.readTree(response.body());

            if (!jsonNode.has("response")) {
                throw new BusinessException("Ollama 返回结果异常");
            }

            return jsonNode.get("response").asText();

        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new BusinessException("调用本地大模型失败，请检查 Ollama 是否启动");
        }
    }

    /**
     * 调用 Ollama /api/embeddings，将文本转为向量
     *
     * @param text 要编码的文本
     * @return float 数组（nomic-embed-text 为 768 维）
     */
    public float[] embed(String text) {
        try {
            Map<String, Object> body = new HashMap<>();
            body.put("model", embeddingModelName);
            body.put("prompt", text);

            String jsonBody = objectMapper.writeValueAsString(body);

            // 从已有的 generate URL 派生出 embeddings URL
            String embeddingsUrl = ollamaUrl.replace("/api/generate", "/api/embeddings");

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(embeddingsUrl))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(30))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString()
            );

            JsonNode jsonNode = objectMapper.readTree(response.body());

            if (!jsonNode.has("embedding") || !jsonNode.get("embedding").isArray()) {
                String responseBody = response.body();
                log.error("Ollama Embedding 返回结果异常，响应内容: {}", responseBody);
                throw new BusinessException("Ollama Embedding 返回结果异常，响应: "
                        + (responseBody.length() > 200 ? responseBody.substring(0, 200) : responseBody));
            }

            JsonNode embeddingArray = jsonNode.get("embedding");
            float[] result = new float[embeddingArray.size()];
            for (int i = 0; i < embeddingArray.size(); i++) {
                result[i] = embeddingArray.get(i).floatValue();
            }
            return result;

        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new BusinessException("调用 Ollama Embedding 失败，请检查 Ollama 是否启动");
        }
    }
}