package com.example.riskplatform.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("ai_conversation")
public class AiConversation {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long userId;
    private Long companyId;
    private String title;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
