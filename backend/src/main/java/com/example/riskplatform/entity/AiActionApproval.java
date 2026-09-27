package com.example.riskplatform.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * Agent 提议的「写动作」审批队列。
 *
 * <p>Agent 的工具本来全是只读的，结论止于「建议关注现金流」。但企业的诉求是风险<b>被处置</b>，
 * 所以这里开放了有限的写能力——不过<b>Agent 永远只提议，不执行</b>：
 * 它生成的动作先落到本表（PENDING），由人确认后才真正生效，
 * 执行结果、审批人、审批意见全部留痕，随时可回溯「这条处置是谁让 AI 提的、谁批的」。</p>
 */
@Data
@TableName("ai_action_approval")
public class AiActionApproval {

    @TableId(type = IdType.AUTO)
    private Long id;
    private Long companyId;
    /** 提议人：用户提问场景为提问用户，事件预警场景为空（系统发起）。 */
    private Long requesterUserId;
    private Long analysisId;
    /** CREATE_TICKET / NOTIFY_OWNER / UPDATE_EVENT_STATUS */
    private String actionType;
    /** 动作入参 JSON */
    private String actionArgs;
    /** Agent 给出的依据（要求带引用） */
    private String reason;
    private String riskLevel;
    /** PENDING / APPROVED / REJECTED / EXECUTED / FAILED */
    private String status;
    /** agent=用户提问触发 / proactive=事件预警触发 */
    private String source;
    private Long approverUserId;
    private String approveComment;
    private String executeResult;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private LocalDateTime approvedAt;
    private LocalDateTime executedAt;
}
