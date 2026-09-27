package com.example.riskplatform.service;

import com.example.riskplatform.entity.KnowledgeDocument;
import com.example.riskplatform.entity.SysDepartment;
import com.example.riskplatform.mapper.SysDepartmentMapper;
import com.example.riskplatform.security.CurrentUserService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 知识库文档的<b>文档级</b>访问控制。
 *
 * <h3>为什么要单独做这一层</h3>
 * <p>{@link DataScopeService} 只回答「这个用户能不能看这家企业」——它是<b>企业维度</b>的。
 * 但知识库是<b>文档维度</b>的：同一家企业下，既有全公司可看的制度文件，也有只有风控部门
 * 和高密级人员能看的处置预案。表结构里早就预留了 {@code knowledge_document.security_level}
 * 与 {@code knowledge_document.dept_id}，可召回链路此前完全没有读这两个字段，
 * 于是低权限用户提问时，RAG 会把高密级文档一并召回进上下文，模型还会给它标注「来源ID=x」——
 * 这等于把无权查看的内容经由模型输出了出去。</p>
 *
 * <h3>判定规则</h3>
 * <ol>
 *   <li><b>密级</b>：{@code doc.security_level <= 用户 security_level} 才可见（null 视为最低的 1）。</li>
 *   <li><b>部门</b>：{@code doc.dept_id == null} 视为全公司公共文档，放行；否则必须落在用户的可见部门集合内。
 *       {@code ALL / ALL_READONLY} 返回 null 表示不限制；{@code DEPT_AND_CHILD} 展开部门子树。</li>
 * </ol>
 *
 * <p>部门树一次性载入并缓存 60 秒，部门表通常只有几十行，成本可忽略。</p>
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class DocumentAccessGuard {

    private final CurrentUserService current;
    private final SysDepartmentMapper deptMapper;

    /** 总开关：关掉即回到「只看企业维度」的旧行为。 */
    @Value("${app.rag.document-acl:true}")
    private boolean enabled;

    /**
     * 无登录上下文时（例如定时任务、MQ 消费者里调用检索）的策略：
     * <ul>
     *   <li>{@code strict}（默认）：只保留最低密级且无部门归属的公共文档。</li>
     *   <li>{@code skip}：不过滤，保持旧行为。</li>
     * </ul>
     */
    @Value("${app.rag.document-acl-anonymous:strict}")
    private String anonymousPolicy;

    private static final long TREE_TTL_MS = 60_000L;
    private volatile long treeCachedAt = 0L;
    private volatile Map<Long, List<Long>> childMap = Map.of();

    /** 过滤结果：保留的文档 + 各类原因的拦截计数，便于排查「为什么这篇没被召回」。 */
    public record FilterResult(List<KnowledgeDocument> kept, int blockedByLevel, int blockedByDept, String mode) {
        public int blocked() {
            return blockedByLevel + blockedByDept;
        }
    }

    public FilterResult filter(List<KnowledgeDocument> docs) {
        List<KnowledgeDocument> src = (docs == null) ? List.of() : docs;
        if (!enabled || src.isEmpty()) {
            return new FilterResult(src, 0, 0, enabled ? "on" : "off");
        }
        Integer userLevel = tryUserLevel();
        boolean anon = (userLevel == null);
        Set<Long> visible = anon ? null : tryVisibleDeptIds();

        List<KnowledgeDocument> kept = new ArrayList<>(src.size());
        int byLevel = 0;
        int byDept = 0;
        for (KnowledgeDocument d : src) {
            if (d == null) continue;
            int lv = (d.getSecurityLevel() == null) ? 1 : d.getSecurityLevel();
            if (anon) {
                // 没有登录上下文时无从判定身份，按配置走保守或放开
                if ("strict".equalsIgnoreCase(anonymousPolicy) && (lv > 1 || d.getDeptId() != null)) {
                    byLevel++;
                    continue;
                }
                kept.add(d);
                continue;
            }
            if (lv > userLevel) {
                byLevel++;
                continue;
            }
            if (visible != null && d.getDeptId() != null && !visible.contains(d.getDeptId())) {
                byDept++;
                continue;
            }
            kept.add(d);
        }
        String mode = anon ? ("anonymous:" + anonymousPolicy) : "on";
        if (byLevel + byDept > 0) {
            log.info("文档级权限过滤：{} 篇中拦下 {} 篇（密级 {} / 部门 {}），mode={}",
                    src.size(), byLevel + byDept, byLevel, byDept, mode);
        }
        return new FilterResult(kept, byLevel, byDept, mode);
    }

    /** 单篇判定，供工具与控制器复用。 */
    public boolean canAccess(KnowledgeDocument d) {
        if (!enabled || d == null) return true;
        return !filter(List.of(d)).kept().isEmpty();
    }

    /**
     * 当前用户的密级；{@code null} = 不限制（功能关闭 / 无登录上下文）。
     *
     * <p>向量索引要在查询侧复刻同样的判定，所以这里必须对外暴露——判定口径只能有一份，
     * 否则「内存过滤」和「索引过滤」迟早会不一致。</p>
     */
    public Integer currentLevel() {
        return enabled ? tryUserLevel() : null;
    }

    /** 当前用户可见部门集合；{@code null} = 不限制。 */
    public Set<Long> currentVisibleDepts() {
        return enabled ? tryVisibleDeptIds() : null;
    }

    private Integer tryUserLevel() {
        try {
            return current.securityLevel();
        } catch (Exception e) {
            // 无认证上下文（匿名 / 系统内部任务）
            return null;
        }
    }

    /** 可见部门集合；null 表示不限制。 */
    private Set<Long> tryVisibleDeptIds() {
        try {
            String ds = current.dataScope();
            if ("ALL".equalsIgnoreCase(ds) || "ALL_READONLY".equalsIgnoreCase(ds)) return null;
            Long me = current.deptId();
            if (me == null) return Set.of();
            if ("DEPT_AND_CHILD".equalsIgnoreCase(ds)) {
                Set<Long> out = new HashSet<>();
                out.add(me);
                collectSubtree(me, out);
                return out;
            }
            return Set.of(me);
        } catch (Exception e) {
            return null;
        }
    }

    private void collectSubtree(Long root, Set<Long> out) {
        Map<Long, List<Long>> m = tree();
        Deque<Long> stack = new ArrayDeque<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            Long x = stack.pop();
            for (Long ch : m.getOrDefault(x, List.of())) {
                if (out.add(ch)) stack.push(ch);
            }
        }
    }

    private Map<Long, List<Long>> tree() {
        long now = System.currentTimeMillis();
        Map<Long, List<Long>> m = childMap;
        if (m.isEmpty() || now - treeCachedAt > TREE_TTL_MS) {
            try {
                List<SysDepartment> all = deptMapper.selectList(null);
                Map<Long, List<Long>> nm = new HashMap<>();
                if (all != null) {
                    for (SysDepartment d : all) {
                        if (d == null || d.getParentId() == null || d.getId() == null) continue;
                        nm.computeIfAbsent(d.getParentId(), k -> new ArrayList<>()).add(d.getId());
                    }
                }
                childMap = nm;
                treeCachedAt = now;
                return nm;
            } catch (Exception e) {
                log.warn("部门树加载失败，文档级部门过滤退化为仅本部门：{}", e.getMessage());
                return Map.of();
            }
        }
        return m;
    }
}
