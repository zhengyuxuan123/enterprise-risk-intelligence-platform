package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;import java.math.BigDecimal;
@Data @TableName("risk_rule") public class RiskRule{@TableId(type=IdType.AUTO)private Long id;private String ruleName;private String riskType;private String metricCode;private String operatorCode;private BigDecimal thresholdValue;private String riskLevel;private Integer enabled;private String description;private LocalDateTime createdAt;private LocalDateTime updatedAt;}
