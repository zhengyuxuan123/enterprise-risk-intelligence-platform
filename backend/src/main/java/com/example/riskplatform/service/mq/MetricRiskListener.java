package com.example.riskplatform.service.mq;import com.example.riskplatform.config.RabbitConfig;import com.example.riskplatform.service.RiskEngineService;import lombok.RequiredArgsConstructor;import org.springframework.amqp.rabbit.annotation.RabbitListener;import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;import org.springframework.stereotype.Component;
@Component @RequiredArgsConstructor @ConditionalOnProperty(name="app.mq.enabled",havingValue="true")public class MetricRiskListener{
    private final RiskEngineService engine;

    /** 主动预警组件；MQ 链路不该因为它缺失而启动失败。 */
    @org.springframework.beans.factory.annotation.Autowired(required = false)
    private com.example.riskplatform.service.agent.ProactiveRiskService proactive;

    @RabbitListener(queues=RabbitConfig.QUEUE)
    public void listen(Long id){
        engine.evaluateMetric(id);
        // 规则引擎跑完之后，若本次真的产生了高危事件，再唤起 Agent 做一次研判。
        // 放在主流程之后且完全吞掉异常：预警是加分项，绝不能拖累或打断规则引擎的落库。
        if (proactive != null) {
            try {
                proactive.evaluateByMetric(id);
            } catch (Exception ignored) {
            }
        }
    }
}
