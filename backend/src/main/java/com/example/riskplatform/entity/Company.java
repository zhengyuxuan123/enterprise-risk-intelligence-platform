package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;import java.math.BigDecimal;
@Data @TableName("company") public class Company{@TableId(type=IdType.AUTO)private Long id;private String companyCode;private String companyName;private String industry;private String region;private String customerLevel;private String companyScale;private Long deptId;private Long ownerUserId;private Integer status;private LocalDateTime createdAt;private LocalDateTime updatedAt;}
