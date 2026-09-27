package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;import java.math.BigDecimal;
@Data @TableName("sys_role") public class SysRole{@TableId(type=IdType.AUTO)private Long id;private String roleName;private String roleCode;private String dataScope;private Integer readonlyFlag;private LocalDateTime createdAt;}
