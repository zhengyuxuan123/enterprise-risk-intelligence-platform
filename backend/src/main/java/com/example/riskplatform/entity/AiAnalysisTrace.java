package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;
/**
 * 一次 AI 分析的完整决策留痕。
 * <p>存在的意义：分析报告给出去以后，企业一定会追问"这个结论是怎么来的"。
 * 只存最终答案无法回答这个问题——必须把路由决策、工具调用、模型切换、
 * 降级原因、护栏与引用核对结果一并留下来，才能复盘和审计。</p>
 */
@Data @TableName("ai_analysis_trace") public class AiAnalysisTrace{
@TableId(type=IdType.AUTO)private Long id;
/** 全链路追踪号，贯穿日志 / 落库 / 响应头 / 前端展示。 */
private String traceId;
private Long analysisId;
private Long companyId;
private Long userId;
private String question;
/** 路由决策：派了哪些专员、开放了哪些工具。 */
private String routeJson;
private Integer iterations;
private Integer llmCalls;
private String model;
/** 模型降级切换记录（如 deepseek-v3 -> doubao-pro-32k）。 */
private String modelSwitched;
private Integer toolCalls;
private Integer toolFailures;
/** 每个工具的入参摘要 / 耗时 / 成败，JSON 数组。 */
private String toolDetail;
/** 召回的 chunk id 与分数，供结果复现。 */
private String retrievalJson;
/** NONE / PARTIAL / FULL。 */
private String degradeLevel;
private String degradeReason;
private String guardrailFlags;
private String groundingJson;
private String reviewTrigger;
private Integer reviewRevised;
private Integer promptChars;
private Long durationMs;
private LocalDateTime createdAt;}
