package com.example.riskplatform.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.example.riskplatform.entity.*;
import com.example.riskplatform.mapper.*;
import com.example.riskplatform.service.cache.CacheService;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class DashboardService {
    private final CompanyMapper companyMapper;
    private final RiskEventMapper riskEventMapper;
    private final ComplaintMapper complaintMapper;
    private final DataScopeService scope;
    private final CacheService cache;
    private final ObjectMapper objectMapper;

    public Map<String, Object> summary() {
        List<Long> ids = scope.allowedCompanyIds();
        String key = "dashboard:" + String.valueOf(ids);
        String cached = cache.get(key);
        if (cached != null) {
            try { return objectMapper.readValue(cached, new TypeReference<Map<String, Object>>() {}); }
            catch (Exception ignored) {}
        }

        LambdaQueryWrapper<Company> companyQ = new LambdaQueryWrapper<>();
        LambdaQueryWrapper<RiskEvent> riskQ = new LambdaQueryWrapper<>();
        LambdaQueryWrapper<Complaint> complaintQ = new LambdaQueryWrapper<>();
        if (ids != null) {
            if (ids.isEmpty()) return emptySummary();
            companyQ.in(Company::getId, ids);
            riskQ.in(RiskEvent::getCompanyId, ids);
            complaintQ.in(Complaint::getCompanyId, ids);
        }

        List<RiskEvent> allRisks = riskEventMapper.selectList(riskQ.orderByDesc(RiskEvent::getCreatedAt).last("LIMIT 500"));
        Map<String, Long> riskByLevel = allRisks.stream().collect(Collectors.groupingBy(
                x -> Optional.ofNullable(x.getRiskLevel()).orElse("UNKNOWN"), LinkedHashMap::new, Collectors.counting()));
        Map<String, Long> riskByStatus = allRisks.stream().collect(Collectors.groupingBy(
                x -> Optional.ofNullable(x.getStatus()).orElse("UNKNOWN"), LinkedHashMap::new, Collectors.counting()));
        long open = allRisks.stream().filter(x -> !"CLOSED".equalsIgnoreCase(x.getStatus())).count();
        long high = allRisks.stream().filter(x -> "HIGH".equalsIgnoreCase(x.getRiskLevel())).count();
        long pending = allRisks.stream().filter(x -> "PENDING".equalsIgnoreCase(x.getStatus()) || "OPEN".equalsIgnoreCase(x.getStatus())).count();

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("companyCount", companyMapper.selectCount(companyQ));
        result.put("riskCount", allRisks.size());
        result.put("openRiskCount", open);
        result.put("highRiskCount", high);
        result.put("pendingRiskCount", pending);
        result.put("complaintCount", complaintMapper.selectCount(complaintQ));
        result.put("riskLevelDistribution", riskByLevel);
        result.put("riskStatusDistribution", riskByStatus);
        result.put("riskByLevel", riskByLevel);
        result.put("topRisks", allRisks.stream().filter(x -> !"CLOSED".equalsIgnoreCase(x.getStatus())).limit(8).toList());
        try { cache.set(key, objectMapper.writeValueAsString(result), Duration.ofSeconds(60)); }
        catch (Exception ignored) {}
        return result;
    }

    private Map<String, Object> emptySummary() {
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("companyCount", 0); r.put("riskCount", 0); r.put("openRiskCount", 0);
        r.put("highRiskCount", 0); r.put("pendingRiskCount", 0); r.put("complaintCount", 0);
        r.put("riskLevelDistribution", Map.of()); r.put("riskStatusDistribution", Map.of()); r.put("topRisks", List.of());
        return r;
    }
}
