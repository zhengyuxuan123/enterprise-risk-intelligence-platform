package com.example.riskplatform.service.agent;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * 主动预警的 Java 侧入口（已迁 Python，这里是转调桩）。
 *
 * <p>原来的实现在 Java 侧完成「取高危事件 → 组提问 → 调 Agent 研判 → 落预警」。
 * 现在这套逻辑在 Python 侧的 {@code ProactiveRiskService} 里，
 * 本类只负责把 MQ 里的指标事件转过去。</p>
 *
 * <p><b>为什么保留这个类而不是删掉</b>：{@code MetricRiskListener} 用
 * {@code @Autowired(required=false)} 持有它，删掉类会直接编译失败；
 * 而"预警是加分项"这个语义不能变 —— 转调失败必须被吞掉，绝不能打断规则引擎落库。</p>
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ProactiveRiskService {

    private final PythonAgentClient py;

    /**
     * 某条指标入库并跑完规则引擎后，若产生了高危事件，唤起一次自动研判。
     *
     * <p>成本闸门仍在 Python 侧生效：单企业每日上限、时间窗新鲜度，
     * 以及 {@code app.ai.auto-consume}（未开启则不消耗模型额度）。</p>
     */
    public void evaluateByMetric(Long metricId) {
        if (metricId == null) return;
        if (!py.available()) {
            log.debug("[proactive] Python agent 未启用，跳过指标 {} 的自动研判", metricId);
            return;
        }
        py.postAsync("/api/ai/proactive/evaluate-metric/" + metricId);
    }
}
