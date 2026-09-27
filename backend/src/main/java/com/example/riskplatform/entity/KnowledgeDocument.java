package com.example.riskplatform.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("knowledge_document")
public class KnowledgeDocument {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long companyId;
    private Long deptId;
    private String title;
    private String docType;
    private String originalFilename;
    private Integer securityLevel;
    private String content;
    private Long uploaderUserId;
    private String storageUri;
    private String fileSha256;
    private Long fileSize;
    private String mimeType;
    private Integer documentVersion;
    private String ingestStatus;
    private String ingestError;
    private BigDecimal parseQualityScore;
    private String parseWarning;
    private Integer chunkCount;
    private LocalDateTime indexedAt;
    private Boolean deleted;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
