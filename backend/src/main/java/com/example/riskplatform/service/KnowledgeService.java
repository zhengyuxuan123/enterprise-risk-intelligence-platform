package com.example.riskplatform.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.example.riskplatform.common.BusinessException;
import com.example.riskplatform.entity.KnowledgeDocument;
import com.example.riskplatform.entity.RagIngestJob;
import com.example.riskplatform.entity.RagOutboxEvent;
import com.example.riskplatform.mapper.KnowledgeDocumentMapper;
import com.example.riskplatform.mapper.RagIngestJobMapper;
import com.example.riskplatform.mapper.RagOutboxEventMapper;
import com.example.riskplatform.security.CurrentUserService;
import com.example.riskplatform.service.rag.KnowledgeFileService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@Service
@RequiredArgsConstructor
public class KnowledgeService {
    private final KnowledgeDocumentMapper mapper;
    private final RagIngestJobMapper jobMapper;
    private final RagOutboxEventMapper outboxMapper;
    private final KnowledgeFileService files;
    private final DataScopeService scope;
    private final CurrentUserService current;
    private final AuditService audit;

    public List<KnowledgeDocument> list(Long companyId) {
        LambdaQueryWrapper<KnowledgeDocument> query = new LambdaQueryWrapper<KnowledgeDocument>()
                .eq(KnowledgeDocument::getDeleted, false)
                .le(KnowledgeDocument::getSecurityLevel, current.securityLevel())
                .orderByDesc(KnowledgeDocument::getId).last("LIMIT 200");
        if (companyId != null) {
            requireCompany(companyId);
            query.eq(KnowledgeDocument::getCompanyId, companyId);
        }
        List<KnowledgeDocument> rows = mapper.selectList(query);
        List<Long> allowed = scope.allowedCompanyIds();
        return allowed == null ? rows : rows.stream()
                .filter(d -> d.getCompanyId() == null || allowed.contains(d.getCompanyId())).toList();
    }

    public List<RagIngestJob> jobs(Long documentId) {
        if (documentId != null) requireDocument(documentId);
        LambdaQueryWrapper<RagIngestJob> query = new LambdaQueryWrapper<RagIngestJob>()
                .orderByDesc(RagIngestJob::getId).last("LIMIT 200");
        if (documentId != null) query.eq(RagIngestJob::getDocumentId, documentId);
        List<RagIngestJob> jobs = jobMapper.selectList(query);
        if (documentId != null) return jobs;
        var visible = list(null).stream().map(KnowledgeDocument::getId).collect(java.util.stream.Collectors.toSet());
        return jobs.stream().filter(j -> visible.contains(j.getDocumentId())).toList();
    }

    @Transactional
    public KnowledgeDocument upload(Long companyId, Long deptId, String type, Integer securityLevel,
                                    String title, MultipartFile file) {
        requireCompany(companyId);
        int level = securityLevel == null ? 1 : securityLevel;
        if (level < 1 || level > current.securityLevel()) throw new BusinessException("文档密级超出当前用户权限");
        KnowledgeDocument document = new KnowledgeDocument();
        document.setCompanyId(companyId);
        document.setDeptId(deptId);
        document.setDocType(blank(type) ? "业务文档" : type.trim());
        document.setSecurityLevel(level);
        document.setTitle(blank(title) ? file.getOriginalFilename() : title.trim());
        document.setOriginalFilename(file.getOriginalFilename());
        document.setUploaderUserId(current.userId());
        // 旧版表结构要求 content NOT NULL；正文由后台解析任务稍后覆盖。
        document.setContent("");
        document.setDocumentVersion(1);
        document.setIngestStatus("STORING");
        document.setChunkCount(0);
        document.setDeleted(false);
        mapper.insert(document);
        KnowledgeFileService.StoredFile stored = files.store(document.getId(), 1, file);
        try { rejectDuplicate(document.getCompanyId(), stored.sha256(), document.getId()); }
        catch (RuntimeException e) { files.deleteQuietly(stored.storageUri()); throw e; }
        applyStored(document, stored);
        document.setIngestStatus("PENDING");
        mapper.updateById(document);
        createJob(document);
        audit.log("UPLOAD", "KNOWLEDGE", document.getId(), document.getTitle());
        return document;
    }

    @Transactional
    public KnowledgeDocument replace(Long id, MultipartFile file) {
        KnowledgeDocument document = requireDocument(id);
        int version = (document.getDocumentVersion() == null ? 1 : document.getDocumentVersion()) + 1;
        KnowledgeFileService.StoredFile stored = files.store(id, version, file);
        try { rejectDuplicate(document.getCompanyId(), stored.sha256(), id); }
        catch (RuntimeException e) { files.deleteQuietly(stored.storageUri()); throw e; }
        applyStored(document, stored);
        document.setDocumentVersion(version);
        document.setIngestStatus("PENDING");
        document.setIngestError(null);
        document.setParseWarning(null);
        mapper.updateById(document);
        createJob(document);
        audit.log("REPLACE", "KNOWLEDGE", id, document.getTitle() + " v" + version);
        return document;
    }

    @Transactional
    public void retry(Long documentId) {
        KnowledgeDocument document = requireDocument(documentId);
        RagIngestJob job = jobMapper.selectOne(new LambdaQueryWrapper<RagIngestJob>()
                .eq(RagIngestJob::getDocumentId, documentId)
                .eq(RagIngestJob::getDocumentVersion, document.getDocumentVersion()).last("LIMIT 1"));
        if (job == null) createJob(document);
        else {
            jobMapper.update(null, new LambdaUpdateWrapper<RagIngestJob>()
                    .eq(RagIngestJob::getId, job.getId())
                    .set(RagIngestJob::getStatus, "PENDING")
                    .set(RagIngestJob::getStage, "PARSE")
                    .set(RagIngestJob::getProgress, 15)
                    .set(RagIngestJob::getErrorMessage, null)
                    .set(RagIngestJob::getFinishedAt, null));
        }
        outboxMapper.update(null, new LambdaUpdateWrapper<RagOutboxEvent>()
                .eq(RagOutboxEvent::getAggregateType, "KNOWLEDGE")
                .eq(RagOutboxEvent::getAggregateId, documentId)
                .ne(RagOutboxEvent::getStatus, "SENT")
                .set(RagOutboxEvent::getStatus, "PENDING")
                .set(RagOutboxEvent::getRetryCount, 0)
                .set(RagOutboxEvent::getNextRetryAt, null)
                .set(RagOutboxEvent::getLastError, null));
        mapper.update(null, new LambdaUpdateWrapper<KnowledgeDocument>()
                .eq(KnowledgeDocument::getId, documentId)
                .set(KnowledgeDocument::getIngestStatus, "PENDING")
                .set(KnowledgeDocument::getIngestError, null));
        audit.log("RETRY", "KNOWLEDGE", documentId, document.getTitle());
    }

    @Transactional
    public void delete(Long id) {
        KnowledgeDocument document = requireDocument(id);
        document.setDeleted(true); document.setIngestStatus("DELETING"); mapper.updateById(document);
        RagOutboxEvent event = new RagOutboxEvent();
        event.setEventKey("knowledge:" + id + ":delete:" + System.currentTimeMillis());
        event.setAggregateType("KNOWLEDGE"); event.setAggregateId(id); event.setEventType("DELETE");
        event.setPayload("{\"sourceType\":\"knowledge\",\"sourceId\":" + id
                + ",\"companyId\":" + (document.getCompanyId() == null ? "null" : document.getCompanyId())
                + ",\"kind\":\"DELETE\",\"documentVersion\":" + document.getDocumentVersion() + "}");
        event.setStatus("PENDING"); event.setRetryCount(0); event.setMaxRetries(8); outboxMapper.insert(event);
        audit.log("DELETE", "KNOWLEDGE", id, document.getTitle());
    }

    private void createJob(KnowledgeDocument document) {
        RagIngestJob job = new RagIngestJob();
        job.setDocumentId(document.getId()); job.setDocumentVersion(document.getDocumentVersion());
        job.setStage("PARSE"); job.setStatus("PENDING"); job.setProgress(15);
        job.setRetryCount(0); job.setMaxRetries(5); job.setChunkCount(0); job.setCreatedBy(current.userId());
        jobMapper.insert(job);
    }

    private KnowledgeDocument requireDocument(Long id) {
        KnowledgeDocument document = mapper.selectById(id);
        if (document == null || Boolean.TRUE.equals(document.getDeleted())) throw new BusinessException(404, "文档不存在");
        requireCompany(document.getCompanyId());
        if (document.getSecurityLevel() != null && document.getSecurityLevel() > current.securityLevel())
            throw new BusinessException(403, "无权访问该密级文档");
        return document;
    }

    private void requireCompany(Long companyId) {
        if (companyId != null && !scope.canAccess(companyId)) throw new BusinessException(403, "无权访问该企业");
    }

    private void rejectDuplicate(Long companyId, String hash, Long selfId) {
        Long count = mapper.selectCount(new LambdaQueryWrapper<KnowledgeDocument>()
                .eq(KnowledgeDocument::getFileSha256, hash).eq(KnowledgeDocument::getDeleted, false)
                .ne(KnowledgeDocument::getId, selfId)
                .eq(companyId != null, KnowledgeDocument::getCompanyId, companyId)
                .isNull(companyId == null, KnowledgeDocument::getCompanyId));
        if (count > 0) throw new BusinessException("同一企业已存在内容完全相同的文件");
    }

    private static void applyStored(KnowledgeDocument d, KnowledgeFileService.StoredFile f) {
        d.setStorageUri(f.storageUri()); d.setFileSha256(f.sha256()); d.setFileSize(f.size());
        d.setMimeType(f.mimeType()); d.setOriginalFilename(f.originalName());
    }
    private static boolean blank(String value) { return value == null || value.isBlank(); }
}
