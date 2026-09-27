package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;import java.math.BigDecimal;
@Data @TableName("ai_tool_log") public class AiToolLog{@TableId(type=IdType.AUTO)private Long id;private Long userId;private Long companyId;private String toolName;private String requestSummary;private String resultSummary;
private String traceId;private Long analysisId;private Long durationMs;private Integer success;private String errorMessage;
private LocalDateTime createdAt;}
