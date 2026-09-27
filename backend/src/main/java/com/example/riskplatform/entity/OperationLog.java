package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;import java.math.BigDecimal;
@Data @TableName("operation_log") public class OperationLog{@TableId(type=IdType.AUTO)private Long id;private Long userId;private String action;private String resourceType;private Long resourceId;private String detail;private String ipAddress;private LocalDateTime createdAt;}
