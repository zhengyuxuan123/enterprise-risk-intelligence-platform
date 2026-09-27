package com.example.riskplatform.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("import_task")
public class ImportTask {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String taskNo;
    private String importType;
    private String originalFilename;
    private Integer totalRows;
    private Integer successRows;
    private Integer failedRows;
    private Integer skippedRows;
    private String fileHash;
    private String errorMessage;
    private String status;
    private Long createdBy;
    private LocalDateTime createdAt;

    /** 本次导入的说明信息（重复跳过原因等），不落库。 */
    @TableField(exist = false)
    private String message;
}
