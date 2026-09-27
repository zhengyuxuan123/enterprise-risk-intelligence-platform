package com.example.riskplatform.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 主动预警记录：不是人提问产生，而是业务事件触发 Agent 研判产生。
 *
 * <p>同时承担<b>成本闸门</b>的职责——按 {@code company_id + created_at} 统计当日研判次数，
 * 超过配额直接不再调用模型。没有这张表，"每天最多研判几次"就无从落地。</p>
 */
@Data
@TableName("ai_proactive_alert")
public class AiProactiveAlert {

    @TableId(type = IdType.AUTO)
    private Long id;
    private Long companyId;
    /** METRIC_THRESHOLD / NEW_COMPLAINT / KB_UPDATED */
    private String triggerType;
    /** 触发对象，如 metric:123 */
    private String triggerRef;
    private String question;
    private String riskLevel;
    private String headline;
    private String answerJson;
    /** ARCHIVED（低危归档） / PENDING_APPROVAL（中危待审批） / ALERTED（高危告警） */
    private String dispatched;
    private Long approvalId;
    private LocalDateTime createdAt;
}
