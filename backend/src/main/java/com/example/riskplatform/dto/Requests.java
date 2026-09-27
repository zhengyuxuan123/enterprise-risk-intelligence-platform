package com.example.riskplatform.dto;import jakarta.validation.constraints.*;import lombok.Data;import java.math.BigDecimal;import java.time.LocalDate;import java.util.List;
public class Requests{
 @Data public static class CompanyRequest{@NotBlank private String companyCode;@NotBlank private String companyName;private String industry;private String region;private String customerLevel;private String companyScale;private Long deptId;private Long ownerUserId;private Integer status=1;}
 @Data public static class MetricRequest{@NotNull private Long companyId;@NotNull private LocalDate metricDate;@NotBlank private String metricCode;@NotBlank private String metricName;@NotNull private BigDecimal metricValue;private String unit;private String sourceType="MANUAL";}
 @Data public static class RiskRuleRequest{@NotBlank private String ruleName;@NotBlank private String riskType;@NotBlank private String metricCode;@NotBlank private String operatorCode;@NotNull private BigDecimal thresholdValue;@NotBlank private String riskLevel;private Integer enabled=1;private String description;}
 @Data public static class AssignRequest{@NotNull private Long assigneeUserId;}@Data public static class HandleRequest{@NotBlank private String handleResult;}@Data public static class ReviewRequest{@NotBlank private String reviewComment;private Boolean approved=true;}
 @Data public static class ComplaintRequest{@NotNull private Long companyId;@NotNull private LocalDate complaintDate;private String customerName;private String productName;@NotBlank private String category;@NotBlank private String description;private String severity="P3";private Integer repeatFlag=0;private Integer firstResponseMinutes;private BigDecimal solveHours;private Integer slaExceeded=0;private String rootCause;private String churnRisk="中";private Integer satisfaction;private String status="处理中";}
 @Data public static class CompetitorRequest{@NotNull private Long companyId;@NotBlank private String competitorName;@NotBlank private String productName;private String targetCustomer;private BigDecimal price;private String priceUnit;private String sellingPoint;private String weakness;private String promotion;private String deliveryCycle;private String serviceCommitment;private String riskLevel="MEDIUM";private LocalDate updatedDate;}
 /**
  * 流式/同步分析入参。
  *
  * @param depth 分析深度：quick=快答（少工具、短文、默认首选）/ standard=标准 / deep=深度。
  *              实测耗时几乎全部来自模型生成字数（2960 字≈169s，5356 字≈743s），
  *              所以「要不要等」本质是「要不要那么长」，把这个选择权交给调用方。
  */
 @Data public static class AiQueryRequest{@NotNull private Long companyId;@NotBlank private String question;private Integer topK=5;private Long sessionId;private Boolean useMultiAgent=false;private String depth="standard";}
 @Data public static class CreateUserRequest{@NotBlank private String username;@NotBlank private String password;@NotBlank private String realName;private Long deptId;private Integer securityLevel=1;@NotEmpty private List<Long>roleIds;}@Data public static class AssignRolesRequest{@NotEmpty private List<Long>roleIds;}
}
