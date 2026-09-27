package com.example.riskplatform.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.example.riskplatform.common.BusinessException;
import com.example.riskplatform.dto.Requests.*;
import com.example.riskplatform.entity.*;
import com.example.riskplatform.mapper.*;
import com.example.riskplatform.security.CurrentUserService;
import lombok.RequiredArgsConstructor;
import org.apache.poi.ss.usermodel.*;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.security.MessageDigest;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Excel 批量导入。
 *
 * <p>两层去重，避免同一个文件被反复导入产生重复数据：</p>
 * <ol>
 *   <li><b>文件级</b>：对上传文件取 SHA-256，若历史上同类型已成功导入过同一份文件，直接标记 SKIPPED，不再解析。</li>
 *   <li><b>行级</b>：按业务主键（指标=编码+日期，投诉=日期+客户+产品+类别+描述，竞品=竞品公司+产品+更新日期）
 *       与库中已有数据比对，已存在或同一文件内重复的行记为「跳过」，不写入。</li>
 * </ol>
 */
@Service
@RequiredArgsConstructor
public class ImportService {
    private final ImportTaskMapper taskMapper;
    private final CompanyMapper companyMapper;
    private final MetricService metricService;
    private final ComplaintService complaintService;
    private final CompetitorService competitorService;
    private final MetricMapper metricMapper;
    private final ComplaintMapper complaintMapper;
    private final CompetitorMapper competitorMapper;
    private final CurrentUserService current;
    private final AuditService audit;

    /** 视为「已成功落库」的任务状态，用于文件级去重判断。 */
    private static final Set<String> FINISHED = Set.of("SUCCESS", "PARTIAL", "SKIPPED");
    private static final char SEP = '\u0001';

    public ImportTask importExcel(String type, MultipartFile file) {
        String normalized = type == null ? "" : type.trim().toUpperCase(Locale.ROOT);
        if (!Set.of("METRIC", "COMPLAINT", "COMPETITOR").contains(normalized)) {
            throw new BusinessException("importType仅支持 METRIC / COMPLAINT / COMPETITOR");
        }
        byte[] bytes;
        try {
            bytes = file.getBytes();
        } catch (IOException e) {
            throw new BusinessException("读取上传文件失败: " + e.getMessage());
        }
        String hash = sha256(bytes);

        ImportTask task = new ImportTask();
        task.setTaskNo("IMP-" + System.currentTimeMillis());
        task.setImportType(normalized);
        task.setOriginalFilename(file.getOriginalFilename());
        task.setFileHash(hash);
        task.setStatus("RUNNING");
        task.setCreatedBy(current.userId());
        task.setTotalRows(0);
        task.setSuccessRows(0);
        task.setFailedRows(0);
        task.setSkippedRows(0);
        taskMapper.insert(task);

        // ---------- 第 1 层：文件级去重 ----------
        ImportTask previous = taskMapper.selectOne(new LambdaQueryWrapper<ImportTask>()
                .eq(ImportTask::getFileHash, hash)
                .eq(ImportTask::getImportType, normalized)
                .in(ImportTask::getStatus, FINISHED)
                .ne(ImportTask::getId, task.getId())
                .orderByAsc(ImportTask::getId)
                .last("LIMIT 1"));
        if (previous != null) {
            task.setStatus("SKIPPED");
            task.setMessage("该文件内容此前已导入过（任务号 " + previous.getTaskNo()
                    + "，成功 " + nz(previous.getSuccessRows()) + " 行），本次未重复写入。");
            taskMapper.updateById(task);
            audit.log("IMPORT", "IMPORT_TASK", task.getId(), normalized + ": duplicate file skipped");
            return task;
        }

        // ---------- 第 2 层：行级去重 + 入库 ----------
        List<String> errors = new ArrayList<>();
        int total = 0, success = 0, skipped = 0;
        Map<String, Set<String>> keyCache = new HashMap<>();
        try (InputStream in = new ByteArrayInputStream(bytes); Workbook wb = WorkbookFactory.create(in)) {
            Sheet sheet = wb.getSheetAt(0);
            if (sheet.getPhysicalNumberOfRows() < 2) throw new BusinessException("Excel至少需要表头和1行数据");
            Map<String, Integer> header = headerMap(sheet.getRow(sheet.getFirstRowNum()));
            for (int rn = sheet.getFirstRowNum() + 1; rn <= sheet.getLastRowNum(); rn++) {
                Row row = sheet.getRow(rn);
                if (row == null || isBlank(row)) continue;
                total++;
                try {
                    Long companyId = companyId(row, header);
                    String cacheKey = normalized + ":" + companyId;
                    Set<String> keys = keyCache.computeIfAbsent(cacheKey, k -> loadExistingKeys(normalized, companyId));
                    String rowKey = normalized + "@" + companyId + SEP + naturalKey(normalized, row, header);
                    if (keys.contains(rowKey)) {
                        skipped++;
                        continue;
                    }
                    switch (normalized) {
                        case "METRIC" -> importMetric(row, header);
                        case "COMPLAINT" -> importComplaint(row, header);
                        default -> importCompetitor(row, header);
                    }
                    keys.add(rowKey);
                    success++;
                } catch (Exception e) {
                    if (errors.size() < 20) errors.add("第" + (rn + 1) + "行: " + e.getMessage());
                }
            }
        } catch (BusinessException e) {
            errors.add(e.getMessage());
        } catch (Exception e) {
            errors.add("文件解析失败: " + e.getMessage());
        }

        task.setTotalRows(total);
        task.setSuccessRows(success);
        task.setSkippedRows(skipped);
        task.setFailedRows(total - success - skipped);
        task.setErrorMessage(errors.isEmpty() ? null : String.join("\n", errors));
        task.setStatus(resolveStatus(errors.isEmpty(), success, skipped));
        if (skipped > 0) {
            task.setMessage("检测到 " + skipped + " 行与库中已有数据内容相同，已自动跳过、未重复写入。");
        }
        taskMapper.updateById(task);
        audit.log("IMPORT", "IMPORT_TASK", task.getId(),
                normalized + ": success=" + success + ", skipped=" + skipped + ", total=" + total);
        return task;
    }

    private String resolveStatus(boolean noError, int success, int skipped) {
        if (!noError && success == 0) return "FAILED";
        if (success == 0 && skipped > 0) return "SKIPPED";
        if (noError && skipped == 0) return "SUCCESS";
        return "PARTIAL";
    }

    public List<ImportTask> history() {
        return taskMapper.selectList(new LambdaQueryWrapper<ImportTask>()
                .eq(ImportTask::getCreatedBy, current.userId()).orderByDesc(ImportTask::getId).last("LIMIT 50"));
    }

    // ==================== 行级去重：业务主键 ====================

    /**
     * 从 Excel 行计算业务主键。
     *
     * <p>投诉刻意不把「客户」纳入主键：历史数据中该列大量为空（源文件表头为“客户ID”，
     * 早期导入未映射），纳入会导致新旧数据永远凑不上、去重失效。其余判别字段
     * （日期/产品/类别/描述/严重度/响应时长/解决时长/满意度/状态）已足以区分真实行。</p>
     */
    private String naturalKey(String type, Row r, Map<String, Integer> h) {
        return switch (type) {
            case "METRIC" -> join(value(r, h, "metric_code", "指标编码"),
                    normDate(value(r, h, "metric_date", "指标日期", "日期")));
            case "COMPLAINT" -> join(normDate(value(r, h, "complaint_date", "投诉日期", "日期")),
                    value(r, h, "product_name", "产品/服务", "产品"),
                    value(r, h, "category", "投诉类别"),
                    value(r, h, "description", "投诉描述"),
                    defaultValue(value(r, h, "severity", "严重度"), "P3"),
                    num(value(r, h, "first_response_minutes", "首次响应分钟")),
                    num(value(r, h, "solve_hours", "解决时长小时")),
                    num(value(r, h, "satisfaction", "满意度")),
                    defaultValue(value(r, h, "status", "处理状态"), "处理中"));
            default -> join(value(r, h, "competitor_name", "竞品公司"),
                    value(r, h, "product_name", "产品/方案", "产品方案"),
                    normDate(value(r, h, "updated_date", "最近更新时间", "更新日期")));
        };
    }

    /** 读取该企业已有数据的业务主键集合（每种类型只需一次查询，之后在内存中比对）。 */
    private Set<String> loadExistingKeys(String type, Long companyId) {
        Set<String> keys = new HashSet<>();
        switch (type) {
            case "METRIC" -> metricMapper.selectList(new LambdaQueryWrapper<Metric>()
                            .eq(Metric::getCompanyId, companyId))
                    .forEach(m -> keys.add(composeKey(type, companyId,
                            join(m.getMetricCode(), m.getMetricDate() == null ? "" : m.getMetricDate().toString()))));
            case "COMPLAINT" -> complaintMapper.selectList(new LambdaQueryWrapper<Complaint>()
                            .eq(Complaint::getCompanyId, companyId))
                    .forEach(x -> keys.add(composeKey(type, companyId, join(
                            x.getComplaintDate() == null ? "" : x.getComplaintDate().toString(),
                            x.getProductName(),
                            x.getCategory(),
                            x.getDescription(),
                            x.getSeverity(),
                            num(x.getFirstResponseMinutes()),
                            num(x.getSolveHours()),
                            num(x.getSatisfaction()),
                            x.getStatus()))));
            default -> competitorMapper.selectList(new LambdaQueryWrapper<Competitor>()
                            .eq(Competitor::getCompanyId, companyId))
                    .forEach(x -> keys.add(composeKey(type, companyId, join(
                            x.getCompetitorName(), x.getProductName(),
                            x.getUpdatedDate() == null ? "" : x.getUpdatedDate().toString()))));
        }
        return keys;
    }

    private String composeKey(String type, Long companyId, String naturalKey) {
        return type + "@" + companyId + SEP + naturalKey;
    }

    private String join(String... parts) {
        StringBuilder b = new StringBuilder();
        for (String p : parts) b.append(p == null ? "" : p.trim()).append(SEP);
        return b.toString();
    }

    // ==================== 各类型入库 ====================

    private void importMetric(Row r, Map<String, Integer> h) {
        MetricRequest x = new MetricRequest();
        x.setCompanyId(companyId(r, h));
        x.setMetricDate(date(value(r, h, "metric_date", "指标日期", "日期")));
        x.setMetricCode(required(value(r, h, "metric_code", "指标编码"), "指标编码"));
        x.setMetricName(required(value(r, h, "metric_name", "指标名称"), "指标名称"));
        x.setMetricValue(decimal(required(value(r, h, "metric_value", "指标值"), "指标值")));
        x.setUnit(value(r, h, "unit", "单位"));
        String source = value(r, h, "source_type", "来源");
        x.setSourceType(source == null || source.isBlank() ? "IMPORT" : source);
        metricService.create(x);
    }

    private void importComplaint(Row r, Map<String, Integer> h) {
        ComplaintRequest x = new ComplaintRequest();
        x.setCompanyId(companyId(r, h));
        x.setComplaintDate(date(value(r, h, "complaint_date", "投诉日期", "日期")));
        x.setCustomerName(value(r, h, "customer_name", "客户名称", "客户ID"));
        x.setProductName(value(r, h, "product_name", "产品/服务", "产品"));
        x.setCategory(required(value(r, h, "category", "投诉类别"), "投诉类别"));
        x.setDescription(required(value(r, h, "description", "投诉描述"), "投诉描述"));
        x.setSeverity(defaultValue(value(r, h, "severity", "严重度"), "P3"));
        x.setRepeatFlag(boolInt(value(r, h, "repeat_flag", "是否重复投诉")));
        x.setFirstResponseMinutes(integer(value(r, h, "first_response_minutes", "首次响应分钟")));
        x.setSolveHours(decimalNullable(value(r, h, "solve_hours", "解决时长小时")));
        x.setSlaExceeded(boolInt(value(r, h, "sla_exceeded", "是否超SLA")));
        x.setRootCause(value(r, h, "root_cause", "根因标签", "根因"));
        x.setChurnRisk(defaultValue(value(r, h, "churn_risk", "流失风险"), "中"));
        x.setSatisfaction(integer(value(r, h, "satisfaction", "满意度")));
        x.setStatus(defaultValue(value(r, h, "status", "处理状态"), "处理中"));
        complaintService.create(x);
    }

    private void importCompetitor(Row r, Map<String, Integer> h) {
        CompetitorRequest x = new CompetitorRequest();
        x.setCompanyId(companyId(r, h));
        x.setCompetitorName(required(value(r, h, "competitor_name", "竞品公司"), "竞品公司"));
        x.setProductName(required(value(r, h, "product_name", "产品/方案", "产品方案"), "产品方案"));
        x.setTargetCustomer(value(r, h, "target_customer", "目标客户"));
        x.setPrice(decimalNullable(value(r, h, "price", "标准价", "价格")));
        x.setPriceUnit(value(r, h, "price_unit", "计费单位", "价格单位"));
        x.setSellingPoint(value(r, h, "selling_point", "核心卖点", "卖点"));
        x.setWeakness(value(r, h, "weakness", "主要弱点"));
        x.setPromotion(value(r, h, "promotion", "当前促销", "促销"));
        x.setDeliveryCycle(value(r, h, "delivery_cycle", "交付周期"));
        x.setServiceCommitment(value(r, h, "service_commitment", "服务承诺"));
        x.setRiskLevel(defaultValue(value(r, h, "risk_level", "竞争风险等级", "风险等级"), "MEDIUM"));
        String d = value(r, h, "updated_date", "最近更新时间", "更新日期");
        if (d != null && !d.isBlank()) x.setUpdatedDate(date(d));
        competitorService.create(x);
    }

    // ==================== 通用工具 ====================

    private Long companyId(Row r, Map<String, Integer> h) {
        String raw = value(r, h, "company_id", "企业ID");
        if (raw != null && !raw.isBlank()) return Long.valueOf(raw.replace(".0", ""));
        String code = value(r, h, "company_code", "企业编码");
        if (code != null && !code.isBlank()) {
            Company c = companyMapper.selectOne(new LambdaQueryWrapper<Company>()
                    .eq(Company::getCompanyCode, code).last("LIMIT 1"));
            if (c != null) return c.getId();
        }
        throw new BusinessException("缺少有效企业ID/企业编码");
    }

    private Map<String, Integer> headerMap(Row row) {
        Map<String, Integer> m = new HashMap<>();
        DataFormatter f = new DataFormatter();
        for (Cell c : row) {
            String v = f.formatCellValue(c).trim();
            if (!v.isEmpty()) m.put(v.toLowerCase(Locale.ROOT), c.getColumnIndex());
        }
        return m;
    }

    private String value(Row r, Map<String, Integer> h, String... names) {
        DataFormatter f = new DataFormatter();
        for (String n : names) {
            Integer i = h.get(n.toLowerCase(Locale.ROOT));
            if (i != null) {
                Cell c = r.getCell(i);
                return c == null ? null : f.formatCellValue(c).trim();
            }
        }
        return null;
    }

    private boolean isBlank(Row r) {
        DataFormatter f = new DataFormatter();
        for (Cell c : r) if (!f.formatCellValue(c).isBlank()) return false;
        return true;
    }

    private String required(String v, String n) {
        if (v == null || v.isBlank()) throw new BusinessException(n + "不能为空");
        return v;
    }

    private String defaultValue(String v, String d) {
        return v == null || v.isBlank() ? d : v;
    }

    private BigDecimal decimal(String v) {
        return new BigDecimal(v.replace(",", ""));
    }

    private BigDecimal decimalNullable(String v) {
        return v == null || v.isBlank() ? null : decimal(v);
    }

    private Integer integer(String v) {
        if (v == null || v.isBlank()) return null;
        return new BigDecimal(v).intValue();
    }

    private Integer boolInt(String v) {
        if (v == null || v.isBlank()) return 0;
        return Set.of("1", "是", "true", "TRUE", "Y", "yes").contains(v) ? 1 : 0;
    }

    private LocalDate date(String v) {
        if (v == null || v.isBlank()) return LocalDate.now();
        String s = v.trim();
        for (String p : List.of("yyyy-MM-dd", "yyyy/M/d", "yyyy/MM/dd")) {
            try {
                return LocalDate.parse(s, DateTimeFormatter.ofPattern(p));
            } catch (Exception ignored) {
            }
        }
        throw new BusinessException("无法解析日期: " + v);
    }

    /** 把各种日期写法统一成 ISO 字符串，仅用于去重比对，不抛异常。 */
    private String normDate(String v) {
        if (v == null || v.isBlank()) return "";
        String s = v.trim();
        for (String p : List.of("yyyy-MM-dd", "yyyy/M/d", "yyyy/MM/dd")) {
            try {
                return LocalDate.parse(s, DateTimeFormatter.ofPattern(p)).toString();
            } catch (Exception ignored) {
            }
        }
        return s;
    }

    // 数值统一成“无多余小数位”的写法，避免 Excel 的 18.9 与 DECIMAL 的 18.90 对不上。

    private String num(BigDecimal v) {
        return v == null ? "" : v.stripTrailingZeros().toPlainString();
    }

    private String num(Integer v) {
        return v == null ? "" : String.valueOf(v);
    }

    private String num(String v) {
        if (v == null || v.isBlank()) return "";
        try {
            return new BigDecimal(v.replace(",", "")).stripTrailingZeros().toPlainString();
        } catch (Exception e) {
            return v.trim();
        }
    }

    private String nz(Integer v) {
        return v == null ? "0" : String.valueOf(v);
    }

    private String sha256(byte[] data) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(data);
            StringBuilder b = new StringBuilder(digest.length * 2);
            for (byte x : digest) b.append(String.format("%02x", x));
            return b.toString();
        } catch (Exception e) {
            throw new BusinessException("计算文件指纹失败: " + e.getMessage());
        }
    }
}
