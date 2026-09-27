package com.example.riskplatform.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("rag_ingest_job")
public class RagIngestJob {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long documentId;
    private Integer documentVersion;
    private String stage;
    private String status;
    private Integer progress;
    private Integer retryCount;
    private Integer maxRetries;
    private BigDecimal qualityScore;
    private Integer chunkCount;
    private String errorMessage;
    private Long createdBy;
    private LocalDateTime startedAt;
    private LocalDateTime heartbeatAt;
    private LocalDateTime finishedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
