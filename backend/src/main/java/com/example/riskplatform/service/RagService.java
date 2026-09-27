package com.example.riskplatform.service;

import com.example.riskplatform.service.agent.PythonAgentClient;
import com.fasterxml.jackson.databind.JsonNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * 知识库检索（已迁 Python，这里是转调桩）。
 *
 * <p>原来的实现包含查询改写、多路召回、RRF 融合、二阶段精排、MMR 去重，
 * 是 Java 侧最大的一块（1143 行）。现在整条流水线在 Python 侧的
 * {@code RagService} 里，索引也从 Lucene 换成了 SQLite FTS5 + numpy。</p>
 *
 * <p><b>为什么保留这个类</b>：{@code KnowledgeController} 的
 * {@code GET /api/knowledge/search} 依赖它，前端 {@code searchKnowledge} 会调。
 * 删掉类就要改 Controller，而这个端点本身并没有迁到 Python（它不属于 {@code /api/ai/*}）。</p>
 *
 * <p><b>取舍</b>：Python 侧的 {@code /api/ai/rag/probe} 只返回 ref / title / score，
 * 不给正文片段，所以 {@code snippet} 为 null。前端该端点主要用于看"命中了哪些文档"，
 * 标题与相关度足够；真要片段再给 probe 端点加字段即可。</p>
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RagService {

    private final PythonAgentClient py;

    public record RagHit(Long documentId, String title, String snippet, double score,
                         Double rerankScore, List<String> recallPaths, String sourceRef, Double rawScore) {
        /** 兼容旧调用：不关心精排细节时只用四参数即可。 */
        public RagHit(Long documentId, String title, String snippet, double score) {
            this(documentId, title, snippet, score, null, null, null, null);
        }
    }

    /** 检索：转调 Python 侧的零 token 检索端点。Python 不可用时返回空列表，不抛错。 */
    public List<RagHit> search(Long companyId, String query, int topK) {
        if (query == null || query.isBlank()) return List.of();
        String path = "/api/ai/rag/probe?q=" + URLEncoder.encode(query, StandardCharsets.UTF_8)
                + "&companyId=" + (companyId == null ? 1 : companyId)
                + "&topK=" + (topK <= 0 ? 5 : topK);
        JsonNode data = py.get(path);
        if (data == null) return List.of();

        List<RagHit> out = new ArrayList<>();
        for (JsonNode h : data.path("hits")) {
            String ref = h.path("ref").asText(null);
            out.add(new RagHit(documentIdOf(ref),
                    h.path("title").asText(null),
                    null,
                    h.path("score").asDouble(0),
                    null,
                    List.of("python"),
                    ref,
                    h.path("rawScore").isNumber() ? h.path("rawScore").asDouble() : null));
        }
        return out;
    }

    /**
     * 从引用标识里还原知识库文档 ID。
     *
     * <p>引用标识形如 {@code k:7#0}（知识库第 7 号文档的第 0 片）、{@code m:12}（指标）、
     * {@code agg:3:metric:16}（聚合）。只有 {@code k:} 前缀才对应 KnowledgeDocument 的主键，
     * 其余返回 null —— 宁可让 documentId 为空，也不能把指标 ID 当成文档 ID 返回给前端。</p>
     */
    private static Long documentIdOf(String ref) {
        if (ref == null || !ref.startsWith("k:")) return null;
        String body = ref.substring(2);
        int hash = body.indexOf('#');
        String num = hash >= 0 ? body.substring(0, hash) : body;
        try {
            return Long.parseLong(num.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }
}
