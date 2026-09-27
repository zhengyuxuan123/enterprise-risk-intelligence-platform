package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;import java.math.BigDecimal;
@Data @TableName("sys_permission") public class SysPermission{@TableId(type=IdType.AUTO)private Long id;private String permissionName;private String permissionCode;private String permissionType;}
