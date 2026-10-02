package com.example.aiknowledgebase.client;

import com.example.aiknowledgebase.common.BusinessException;
import com.example.aiknowledgebase.common.ErrorCode;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

/**
 * Ollama 调用客户端。
 * <p>
 * 服务地址统一由 {@code ollama.base-url} 提供，各接口路径在此基础上拼接，
 * 避免在某个完整 URL 字符串上做替换来派生其他接口地址。
 */
@Component
public class OllamaClient {

    private static final Logger log = LoggerFactory.getLogger(OllamaClient.class);

    /** 生成接口路径 */
    private static final String GENERATE_PATH = "/api/generate";

    /** 向量化接口路径（/api/embeddings 为 legacy 端点，/api/embed 支持批量输入） */
    private static final String EMBED_PATH = "/api/embed";

    /** 日志中响应体的最大输出长度 */
    private static final int MAX_LOGGED_BODY = 300;

    @Value("${ollama.base-url}")
    private String baseUrl;

    @Value("${ollama.model}")
    private String modelName;

    @Value("${ollama.embedding-model}")
    private String embeddingModelName;

    private final ObjectMapper objectMapper = new ObjectMapper();

    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    /**
     * 调用 Ollama /api/generate，生成文本回复（非流式）
     */
    public String generate(String prompt) {
        String url = resolveUrl(GENERATE_PATH);

        Map<String, Object> body = new HashMap<>();
        body.put("model", modelName);
        body.put("prompt", prompt);
        body.put("stream", false);

        try {
            String jsonBody = objectMapper.writeValueAsString(body);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(60))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString()
            );

            ensureSuccess(response, "generate", modelName);

            JsonNode jsonNode = objectMapper.readTree(response.body());
            if (!jsonNode.has("response")) {
                log.error("Ollama generate 响应缺少 response 字段: url={}, model={}, body={}",
                        url, modelName, abbreviate(response.body()));
                throw new BusinessException(ErrorCode.LLM_UNAVAILABLE,
                        "Ollama 返回结果异常：响应中缺少 response 字段");
            }

            return jsonNode.get("response").asText();

        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            // 保留原始异常作为 cause，否则超时、连接拒绝等根因会在日志中彻底丢失
            log.error("调用 Ollama generate 失败: url={}, model={}", url, modelName, e);
            throw new BusinessException(ErrorCode.LLM_UNAVAILABLE,
                    "调用本地大模型失败，请检查 Ollama 是否启动", e);
        }
    }

    /**
     * 调用 Ollama 向量化接口，将文本转为向量。
     * <p>
     * 默认走 {@code /api/embed}（请求体 {@code input}，响应 {@code embeddings} 数组）。
     * 为兼容仍在使用 legacy {@code /api/embeddings} 端点（响应 {@code embedding}）的服务端，
     * 解析时对两种响应结构都做兼容。
     *
     * @param text 要编码的文本
     * @return float 数组，维度取决于底层模型
     */
    public float[] embed(String text) {
        String url = resolveUrl(EMBED_PATH);

        Map<String, Object> body = new HashMap<>();
        body.put("model", embeddingModelName);
        body.put("input", text);

        try {
            String jsonBody = objectMapper.writeValueAsString(body);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(30))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString()
            );

            ensureSuccess(response, "embed", embeddingModelName);

            JsonNode jsonNode = objectMapper.readTree(response.body());
            JsonNode embeddingArray = extractEmbeddingArray(jsonNode);

            if (embeddingArray == null) {
                log.error("Ollama Embedding 响应结构无法识别: url={}, model={}, body={}",
                        url, embeddingModelName, abbreviate(response.body()));
                throw new BusinessException(ErrorCode.EMBEDDING_FAILED,
                        "Ollama Embedding 返回结果异常，响应: " + abbreviate(response.body()));
            }

            float[] result = new float[embeddingArray.size()];
            for (int i = 0; i < embeddingArray.size(); i++) {
                result[i] = embeddingArray.get(i).floatValue();
            }
            return result;

        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            log.error("调用 Ollama Embedding 失败: url={}, model={}", url, embeddingModelName, e);
            throw new BusinessException(ErrorCode.EMBEDDING_FAILED,
                    "调用 Ollama Embedding 失败，请检查 Ollama 是否启动", e);
        }
    }

    // ==================== 私有方法 ====================

    /**
     * 拼接接口地址，并去掉 base-url 结尾多余的 '/'。
     */
    private String resolveUrl(String path) {
        String base = baseUrl == null ? "" : baseUrl.trim();
        while (base.endsWith("/")) {
            base = base.substring(0, base.length() - 1);
        }
        return base + path;
    }

    /**
     * 校验 HTTP 状态码。此前未做校验，服务端返回 4xx/5xx 时会被误报成"响应格式异常"。
     */
    private void ensureSuccess(HttpResponse<String> response, String api, String model) {
        int status = response.statusCode();
        if (status < 200 || status >= 300) {
            log.error("Ollama {} 返回非 2xx: status={}, model={}, body={}",
                    api, status, model, abbreviate(response.body()));
            throw new BusinessException(ErrorCode.LLM_UNAVAILABLE,
                    "Ollama " + api + " 调用失败，HTTP 状态码 " + status);
        }
    }

    /**
     * 从响应中提取向量数组，兼容两种响应结构：
     * <ul>
     *   <li>{@code /api/embed} → {@code {"embeddings": [[...]]}}</li>
     *   <li>{@code /api/embeddings}（legacy）→ {@code {"embedding": [...]}}</li>
     * </ul>
     *
     * @return 向量数组节点；无法识别时返回 null
     */
    private JsonNode extractEmbeddingArray(JsonNode root) {
        JsonNode batch = root.path("embeddings").path(0);
        if (batch.isArray() && batch.size() > 0) {
            return batch;
        }
        JsonNode single = root.path("embedding");
        if (single.isArray() && single.size() > 0) {
            return single;
        }
        return null;
    }

    /**
     * 截断过长的响应体，避免日志被模型输出淹没。
     */
    private static String abbreviate(String body) {
        if (body == null) {
            return "null";
        }
        return body.length() <= MAX_LOGGED_BODY
                ? body
                : body.substring(0, MAX_LOGGED_BODY) + "...(已截断)";
    }
}
