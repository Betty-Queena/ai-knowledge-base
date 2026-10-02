package com.example.aiknowledgebase;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

/**
 * 激活 test profile，使用独立 Redis 库与前缀，避免测试影响生产知识库数据。
 */
@SpringBootTest
@ActiveProfiles("test")
class AiKnowledgeBaseApplicationTests {

    @Test
    void contextLoads() {
    }

}
