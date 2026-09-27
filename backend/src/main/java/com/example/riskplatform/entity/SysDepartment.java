package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;import java.math.BigDecimal;
@Data @TableName("sys_department") public class SysDepartment{@TableId(type=IdType.AUTO)private Long id;private Long parentId;private String deptName;private Integer sortNo;private Integer status;}
