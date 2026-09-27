package com.example.riskplatform.service.rag;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.example.riskplatform.entity.KnowledgeDocument;
import com.example.riskplatform.entity.RagIngestJob;
import com.example.riskplatform.entity.RagOutboxEvent;
import com.example.riskplatform.mapper.KnowledgeDocumentMapper;
import com.example.riskplatform.mapper.RagIngestJobMapper;
import com.example.riskplatform.mapper.RagOutboxEventMapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
@Service
@RequiredArgsConstructor
public class KnowledgeIngestWorker {
    private final RagIngestJobMapper jobMapper;
    private final KnowledgeDocumentMapper documentMapper;
    private final RagOutboxEventMapper outboxMapper;
    private final KnowledgeFileService files;
    private final ObjectMapper json;
    private final AtomicBoolean working = new AtomicBoolean();

    @EventListener(ApplicationReadyEvent.class)
    public void recoverInterruptedJobs() {
        for (RagIngestJob job : jobMapper.selectList(new LambdaQueryWrapper<RagIngestJob>()
                .eq(RagIngestJob::getStatus, "RUNNING"))) {
            job.setStatus("RETRY");
            job.setErrorMessage("服务重启后自动恢复中断的解析任务");
            jobMapper.updateById(job);
        }
    }

    @Scheduled(fixedDelayString = "${app.knowledge.worker-delay-ms:1000}")
    public void tick() {
        if (!working.compareAndSet(false, true)) return;
        try {
            RagIngestJob job = jobMapper.selectOne(new LambdaQueryWrapper<RagIngestJob>()
                    .in(RagIngestJob::getStatus, "PENDING", "RETRY")
                    .orderByAsc(RagIngestJob::getId).last("LIMIT 1"));
            if (job != null) process(job);
            else repairWaitingIndex();
        } catch (Exception e) { log.error("[RAG入库] 调度异常", e); }
        finally { working.set(false); }
    }

    private void repairWaitingIndex() {
        RagIngestJob job = jobMapper.selectOne(new LambdaQueryWrapper<RagIngestJob>()
                .eq(RagIngestJob::getStatus, "WAITING_INDEX")
                .orderByAsc(RagIngestJob::getUpdatedAt).last("LIMIT 1"));
        if (job == null) return;
        KnowledgeDocument document = documentMapper.selectById(job.getDocumentId());
        if (document == null || Boolean.TRUE.equals(document.getDeleted())
                || !job.getDocumentVersion().equals(document.getDocumentVersion())) return;
        try { createOutbox(document); }
        catch (Exception e) { log.warn("[RAG入库] 补建 Outbox 失败: {}", e.getMessage()); }
    }

    private void process(RagIngestJob job) {
        KnowledgeDocument document = documentMapper.selectById(job.getDocumentId());
        if (document == null || Boolean.TRUE.equals(document.getDeleted())
                || !job.getDocumentVersion().equals(document.getDocumentVersion())) {
            finish(job, "CANCELLED", "文档已删除或已有新版本", 100);
            return;
        }
        try {
            job.setStatus("RUNNING"); job.setStage("PARSE"); job.setProgress(30);
            job.setStartedAt(LocalDateTime.now()); job.setHeartbeatAt(LocalDateTime.now()); jobMapper.updateById(job);
            document.setIngestStatus("PARSING"); documentMapper.updateById(document);
            KnowledgeFileService.Extraction extraction = files.extract(document.getStorageUri(), document.getMimeType());
            document.setContent(extraction.text());
            document.setParseQualityScore(BigDecimal.valueOf(extraction.qualityScore()));
            document.setParseWarning(extraction.warning()); document.setIngestStatus("INDEX_PENDING");
            document.setIngestError(null); documentMapper.updateById(document);
            job.setStage("INDEX"); job.setStatus("WAITING_INDEX"); job.setProgress(75);
            job.setQualityScore(document.getParseQualityScore()); job.setHeartbeatAt(LocalDateTime.now());
            jobMapper.updateById(job); createOutbox(document);
        } catch (Exception e) {
            int retries = value(job.getRetryCount(), 0) + 1;
            job.setRetryCount(retries); job.setStatus(retries < value(job.getMaxRetries(), 5) ? "RETRY" : "FAILED");
            job.setStage("PARSE"); job.setProgress(30); job.setErrorMessage(shorten(e.getMessage(), 1900));
            job.setFinishedAt("FAILED".equals(job.getStatus()) ? LocalDateTime.now() : null); jobMapper.updateById(job);
            document.setIngestStatus(job.getStatus()); document.setIngestError(shorten(e.getMessage(), 900));
            documentMapper.updateById(document);
            log.warn("[RAG入库] 文档 {} v{} 解析失败: {}", document.getId(), document.getDocumentVersion(), e.getMessage());
        }
    }

    private void createOutbox(KnowledgeDocument document) throws Exception {
        String key = "knowledge:" + document.getId() + ":v" + document.getDocumentVersion() + ":upsert";
        if (outboxMapper.selectCount(new LambdaQueryWrapper<RagOutboxEvent>().eq(RagOutboxEvent::getEventKey, key)) > 0) return;
        Map<String, Object> payload = new HashMap<>();
        payload.put("sourceType", "knowledge"); payload.put("sourceId", document.getId());
        payload.put("companyId", document.getCompanyId()); payload.put("kind", "UPSERT");
        payload.put("documentVersion", document.getDocumentVersion());
        RagOutboxEvent event = new RagOutboxEvent();
        event.setEventKey(key); event.setAggregateType("KNOWLEDGE"); event.setAggregateId(document.getId());
        event.setEventType("UPSERT"); event.setPayload(json.writeValueAsString(payload));
        event.setStatus("PENDING"); event.setRetryCount(0); event.setMaxRetries(8); outboxMapper.insert(event);
    }

    private void finish(RagIngestJob job, String status, String error, int progress) {
        job.setStatus(status); job.setErrorMessage(error); job.setProgress(progress);
        job.setFinishedAt(LocalDateTime.now()); jobMapper.updateById(job);
    }
    private static int value(Integer n, int fallback) { return n == null ? fallback : n; }
    private static String shorten(String s, int n) { return s == null ? "未知错误" : s.substring(0, Math.min(n, s.length())); }
}
