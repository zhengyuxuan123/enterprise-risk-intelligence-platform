package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;import java.math.BigDecimal;
@Data @TableName("ai_analysis") public class AiAnalysis{@TableId(type=IdType.AUTO)private Long id;private Long companyId;private Long userId;private Long conversationId;private String question;private String answer;private String evidenceJson;private String toolTraceJson;private String answerJson;private String provider;private String confidence;private Integer grounded;
/** 全链路追踪号，与 ai_analysis_trace.trace_id 对齐。 */
private String traceId;
/** NONE / PARTIAL / FULL：本次分析是否走了降级路径。 */
private String degradeLevel;
private String degradeReason;
private Long durationMs;
private Integer llmCalls;
private Integer toolCalls;
private String model;
private Integer reviewRevised;
private LocalDateTime createdAt;}
