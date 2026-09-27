package com.example.riskplatform.service.rag;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.example.riskplatform.entity.KnowledgeDocument;
import com.example.riskplatform.entity.RagIngestJob;
import com.example.riskplatform.entity.RagOutboxEvent;
import com.example.riskplatform.mapper.KnowledgeDocumentMapper;
import com.example.riskplatform.mapper.RagIngestJobMapper;
import com.example.riskplatform.mapper.RagOutboxEventMapper;
import com.example.riskplatform.service.agent.PythonAgentClient;
import com.fasterxml.jackson.databind.JsonNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
@Service
@RequiredArgsConstructor
public class RagOutboxDispatcher {
    private final RagOutboxEventMapper outboxMapper;
    private final KnowledgeDocumentMapper documentMapper;
    private final RagIngestJobMapper jobMapper;
    private final PythonAgentClient python;
    private final AtomicBoolean working = new AtomicBoolean();

    @EventListener(ApplicationReadyEvent.class)
    public void recoverInterruptedEvents() {
        for (RagOutboxEvent event : outboxMapper.selectList(new LambdaQueryWrapper<RagOutboxEvent>()
                .eq(RagOutboxEvent::getStatus, "SENDING"))) {
            event.setStatus("RETRY");
            event.setNextRetryAt(LocalDateTime.now());
            event.setLastError("服务重启后自动恢复中断的投递事件");
            outboxMapper.updateById(event);
        }
    }

    @Scheduled(fixedDelayString = "${app.knowledge.outbox-delay-ms:1500}")
    public void tick() {
        if (!working.compareAndSet(false, true)) return;
        try {
            RagOutboxEvent event = outboxMapper.selectOne(new LambdaQueryWrapper<RagOutboxEvent>()
                    .in(RagOutboxEvent::getStatus, "PENDING", "RETRY")
                    .and(w -> w.isNull(RagOutboxEvent::getNextRetryAt).or()
                            .le(RagOutboxEvent::getNextRetryAt, LocalDateTime.now()))
                    .orderByAsc(RagOutboxEvent::getId).last("LIMIT 1"));
            if (event != null) dispatch(event);
        } catch (Exception e) { log.error("[RAG Outbox] 调度异常", e); }
        finally { working.set(false); }
    }

    private void dispatch(RagOutboxEvent event) {
        try {
            event.setStatus("SENDING"); outboxMapper.updateById(event);
            JsonNode result = python.post("/api/ai/rag/event", event.getPayload());
            int chunks = result == null ? 0 : result.path("chunks").asInt(0);
            event.setStatus("SENT"); event.setPublishedAt(LocalDateTime.now());
            outboxMapper.update(null, new LambdaUpdateWrapper<RagOutboxEvent>()
                    .eq(RagOutboxEvent::getId, event.getId())
                    .set(RagOutboxEvent::getStatus, "SENT")
                    .set(RagOutboxEvent::getPublishedAt, event.getPublishedAt())
                    .set(RagOutboxEvent::getLastError, null));
            int version = result == null ? 0 : result.path("documentVersion").asInt(0);
            if ("UPSERT".equals(event.getEventType())) markIndexed(event.getAggregateId(), chunks, version);
            else if ("DELETE".equals(event.getEventType())) markDeleted(event.getAggregateId());
        } catch (Exception e) {
            int retries = value(event.getRetryCount(), 0) + 1, max = value(event.getMaxRetries(), 8);
            event.setRetryCount(retries); event.setStatus(retries >= max ? "DEAD" : "RETRY");
            event.setNextRetryAt(LocalDateTime.now().plusSeconds(Math.min(300, 1L << Math.min(retries, 8))));
            event.setLastError(shorten(e.getMessage(), 900)); outboxMapper.updateById(event);
            markIndexFailure(event.getAggregateId(), event.getStatus(), e.getMessage());
            log.warn("[RAG Outbox] 事件 {} 投递失败，第 {} 次: {}", event.getEventKey(), retries, e.getMessage());
        }
    }

    private void markIndexed(Long id, int chunks, int version) {
        KnowledgeDocument doc = documentMapper.selectById(id); if (doc == null) return;
        if (version > 0 && !Integer.valueOf(version).equals(doc.getDocumentVersion())) return;
        LocalDateTime now = LocalDateTime.now();
        documentMapper.update(null, new LambdaUpdateWrapper<KnowledgeDocument>()
                .eq(KnowledgeDocument::getId, id)
                .set(KnowledgeDocument::getIngestStatus, "INDEXED")
                .set(KnowledgeDocument::getIngestError, null)
                .set(KnowledgeDocument::getChunkCount, chunks)
                .set(KnowledgeDocument::getIndexedAt, now));
        RagIngestJob job = currentJob(doc);
        if (job != null) {
            jobMapper.update(null, new LambdaUpdateWrapper<RagIngestJob>()
                    .eq(RagIngestJob::getId, job.getId())
                    .set(RagIngestJob::getStatus, "SUCCESS")
                    .set(RagIngestJob::getStage, "COMPLETE")
                    .set(RagIngestJob::getProgress, 100)
                    .set(RagIngestJob::getChunkCount, chunks)
                    .set(RagIngestJob::getErrorMessage, null)
                    .set(RagIngestJob::getFinishedAt, now));
        }
    }

    private void markDeleted(Long id) {
        KnowledgeDocument doc = documentMapper.selectById(id);
        if (doc != null) { doc.setIngestStatus("DELETED"); documentMapper.updateById(doc); }
    }

    private void markIndexFailure(Long id, String eventStatus, String error) {
        KnowledgeDocument doc = documentMapper.selectById(id);
        if (doc == null || "DELETING".equals(doc.getIngestStatus())) return;
        doc.setIngestStatus("DEAD".equals(eventStatus) ? "FAILED" : "INDEX_RETRY");
        doc.setIngestError(shorten(error, 900)); documentMapper.updateById(doc);
        RagIngestJob job = currentJob(doc);
        if (job != null) {
            job.setStatus("DEAD".equals(eventStatus) ? "FAILED" : "WAITING_INDEX");
            job.setErrorMessage(shorten(error, 1900));
            if ("DEAD".equals(eventStatus)) job.setFinishedAt(LocalDateTime.now());
            jobMapper.updateById(job);
        }
    }

    private RagIngestJob currentJob(KnowledgeDocument doc) {
        return jobMapper.selectOne(new LambdaQueryWrapper<RagIngestJob>()
                .eq(RagIngestJob::getDocumentId, doc.getId())
                .eq(RagIngestJob::getDocumentVersion, doc.getDocumentVersion()).last("LIMIT 1"));
    }
    private static int value(Integer n, int fallback) { return n == null ? fallback : n; }
    private static String shorten(String s, int n) { return s == null ? "未知错误" : s.substring(0, Math.min(n, s.length())); }
}
