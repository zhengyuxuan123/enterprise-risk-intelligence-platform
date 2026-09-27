package com.example.riskplatform.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("ai_message")
public class AiMessage {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long conversationId;
    private String role; // SYSTEM / USER / ASSISTANT
    private String content;
    private String toolTraceJson;
    private String answerJson;
    private String provider;
    private LocalDateTime createdAt;
}
