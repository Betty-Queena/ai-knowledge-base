# AI Knowledge Base — 基于 Spring Boot 与 Ollama 的 RAG 知识库

基于 **Java 17、Spring Boot 4、Ollama 和 Redis** 实现的本地检索增强生成（Retrieval-Augmented Generation，RAG）应用。支持文档上传、文本分块、向量化、语义检索、知识库问答和按会话隔离的对话历史，并提供可复现的检索评估基线。

> 项目定位：面向学习和工程实践的轻量级 RAG 系统。当前检索采用 **Redis Hash 存储向量数据 + Java 进程内余弦相似度排序**，未使用 Redis Stack / RediSearch 的原生向量索引。

## 核心功能

- **文档导入**：支持 TXT、Markdown、PDF 文档解析与上传。
- **文本切分**：固定字符滑动窗口切分，支持配置块大小与重叠长度。
- **向量化与存储**：调用 Ollama Embedding 模型生成向量，将文本块和向量数据保存到 Redis。
- **检索增强问答**：对问题进行向量化，从 Redis 读取候选文本块，在 JVM 中计算余弦相似度，按 Top-K 选取上下文交给生成模型回答。
- **会话记忆**：按 `sessionId` 隔离和查询对话历史。
- **异常降级**：知识库检索不可用时支持退化为普通模型对话。
- **检索评估**：提供独立测试语料和问答集，计算文档级/文本块级 HitRate@K、MRR@K，以及检索延迟分位数。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 后端 | Java 17、Spring Boot 4.0.6、Maven |
| 生成模型 | Ollama / `qwen2.5:1.5b` |
| 向量模型 | Ollama / `nomic-embed-text:latest`（768 维） |
| 数据存储 | Redis Hash |
| PDF 解析 | Apache PDFBox |
| 测试与评估 | JUnit、Maven Surefire、自建检索评估数据集 |

## 系统流程

**文档索引链路**

```text
TXT / MD / PDF 文件
      ↓
文档解析与文本提取
      ↓
滑动窗口文本分块
      ↓
Ollama Embedding
      ↓
Redis Hash（文本块及向量）
```

**检索问答链路**

```text
用户问题 + sessionId
      ↓
问题向量化
      ↓
Redis HGETALL 读取候选块
      ↓
JVM 余弦相似度计算、Top-K 排序
      ↓
拼接检索上下文与会话历史
      ↓
Ollama 生成回答
      ↓
返回答案、维护会话历史
```

> 说明：当前实现是轻量级全量候选检索，数据规模增大后，`HGETALL` 和 JVM 全量相似度计算可能成为性能瓶颈。

## 快速开始

### 环境要求

- JDK 17
- Redis（默认 `localhost:6379`）
- Ollama（默认 `http://localhost:11434`）
- Windows 可使用仓库中的 `mvnw.cmd`；macOS/Linux 使用 `./mvnw`

### 1. 准备模型

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text:latest
```

确保 Ollama 服务已启动，Redis 服务可连接。

### 2. 启动后端

Windows CMD：

```cmd
mvnw.cmd spring-boot:run
```

默认服务端口：`8081`。相关配置位于 `src/main/resources/application.properties`。

### 3. 调用 API

**上传文档**

```bash
curl -X POST "http://localhost:8081/api/documents/upload" -F "file=@example.md"
```

**知识库问答**

```bash
curl -X POST "http://localhost:8081/api/chat/ask" -H "Content-Type: application/json" -d "{\"question\":\"什么是 Spring Boot？\",\"sessionId\":\"demo-session\"}"
```

**查询会话历史**

```bash
curl "http://localhost:8081/api/chat/history?sessionId=demo-session"
```

接口返回采用项目的 `Result<T>` 统一响应包装；具体字段结构以代码中的 DTO 和 `Result` 定义为准。

## 检索评估

项目包含一组可复现的本地评估数据：**9 篇文档、31 个文本块、40 条 QA 样本**（其中 30 条可回答、10 条不可回答）。评估使用 `nomic-embed-text:latest`、`chunkSize=500`、`overlap=100`，独立 Redis key 前缀 `rag-eval`。

最近一次完整测试的检索基线（2026-11-10）：

| 指标 | @1 | @3 | @5 |
| --- | ---: | ---: | ---: |
| 文档级 HitRate | 66.67% | 80.00% | 83.33% |
| 文档级 MRR | 66.67% | 72.22% | 73.06% |
| 文本块级 HitRate | 46.67% | 63.33% | 70.00% |
| 文本块级 MRR | 46.67% | 54.44% | 56.11% |

该次运行的检索链路总耗时为 **p50 32.52 ms / p95 54.30 ms**（本地测试环境，统计范围为 Embedding + Search，不代表端到端生成耗时）。生成层拒答准确率观察值为 **10%**，属于非确定性观察指标，仍有改进空间。

运行完整测试：

```cmd
mvnw.cmd test
```

已验证结果：**35 tests, 0 failures, 0 errors**。检索评估报告生成在 `target/eval/` 目录（构建产物，不提交到仓库）。运行评估需要可用的 Redis 与 Ollama Embedding 模型；首次加载模型可能产生额外延迟。

## 配置说明

常用参数位于 `src/main/resources/application.properties`：

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `server.port` | `8081` | HTTP 端口 |
| `spring.data.redis.host` | `localhost` | Redis 主机 |
| `spring.data.redis.port` | `6379` | Redis 端口 |
| `ollama.base-url` | `http://localhost:11434` | Ollama 地址 |
| `ollama.model` | `qwen2.5:1.5b` | 生成模型 |
| `ollama.embedding-model` | `nomic-embed-text:latest` | 向量模型 |
| `rag.chunk-size` | `500` | 文本块长度 |
| `rag.chunk-overlap` | `100` | 文本块重叠长度 |
| `rag.top-k` | `3` | 返回的候选文本块数 |
| `rag.min-score` | `-1.0` | 相似度过滤阈值，-1 表示不启用过滤 |

测试使用独立的 Redis DB/key 前缀，避免与默认知识库数据混用。

## 后续改进方向

- 混合检索（语义检索 + 词法检索）与 RRF 融合
- 两阶段召回与重排序（Rerank）
- 增量索引、内容哈希去重及大规模索引性能优化
- 检索阈值校准、引用溯源与更可靠的拒答策略

以上属于**规划项，当前版本尚未实现**。

## 说明

本项目主要用于 RAG 系统原理、Java 后端开发及检索评估实践。评估数据与指标反映特定语料和本地测试环境下的结果，不应直接推断为生产环境性能。
