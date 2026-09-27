package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;import java.math.BigDecimal;
@Data @TableName("business_metric") public class Metric{@TableId(type=IdType.AUTO)private Long id;private Long companyId;private LocalDate metricDate;private String metricCode;private String metricName;private BigDecimal metricValue;private String unit;private String sourceType;private LocalDateTime createdAt;}
