package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;import java.math.BigDecimal;
@Data @TableName("sys_user") public class SysUser{@TableId(type=IdType.AUTO)private Long id;private String username;private String passwordHash;private String realName;private Long deptId;private Integer securityLevel;private Integer status;private LocalDateTime createdAt;private LocalDateTime updatedAt;}
