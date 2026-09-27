package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;
/**
 * 评估跑批留档。原来评估报告只落文件（换机器 / 换容器就丢），
 * 企业场景下"哪次上线评估没通过"必须可审计，因此落到库里。
 */
@Data @TableName("ai_eval_run") public class AiEvalRun{
@TableId(type=IdType.AUTO)private Long id;
private String runId;
private Integer total;
private Integer passed;
private Double passRate;
/** 六维加权综合分（0-100）。 */
private Double score;
/** A / B / C / D。 */
private String grade;
/** PASSED / FAILED（门禁判定）。 */
private String gate;
/** 各维度得分明细 JSON。 */
private String dimensions;
private String model;
private Long wallClockMs;
private Long operatorId;
private LocalDateTime createdAt;}
