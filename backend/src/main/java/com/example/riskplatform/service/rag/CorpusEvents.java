package com.example.riskplatform.service.rag;

import com.example.riskplatform.service.agent.PythonAgentClient;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

/**
 * 业务侧发布语料变更事件的统一入口（已迁 Python，这里是转调桩）。
 *
 * <p>各业务 Service 仍然只管在写库之后调 {@code corpus.upsert(...)}，
 * 不用关心索引怎么落 —— 但落索引这件事现在由 Python 侧的
 * {@code RagIndexingService} 负责，本类把变更转成一次增量重建通知。</p>
 *
 * <p><b>必须写在数据库操作之后</b>（这一条没有变）：Python 侧是回查业务表拿最新内容的，
 * 先通知再落库，要么读到旧数据，要么读到"记录不存在"而把切片删掉。</p>
 *
 * <p><b>为什么用"重建该企业"而不是"增量这一条"</b>：Python 侧的 reindex 端点按企业粒度工作，
 * 内部自带内容 hash 比对（内容没变就不重算向量），所以重复通知的代价可控；
 * 而"只更新一条"需要额外的事件协议，收益不抵复杂度。</p>
 *
 * <p>通知是异步且静默失败的：Python 没起也不会影响写库。
 * 万一漏了，Python 侧的定时 reconcile（默认 360 分钟）会把索引补回来。</p>
 */
@Service
@RequiredArgsConstructor
public class CorpusEvents {

    private final PythonAgentClient py;

    /** 新增或更新了一条记录。 */
    public void upsert(String sourceType, Long sourceId, Long companyId) {
        if (sourceId == null) return;
        reindex(companyId);
    }

    /** 删除了一条记录。 */
    public void remove(String sourceType, Long sourceId, Long companyId) {
        if (sourceId == null) return;
        reindex(companyId);
    }

    /** 该企业下的跨行聚合切片（指标走势 / 事件分布 / 投诉分布）失效。 */
    public void aggregate(Long companyId) {
        reindex(companyId);
    }

    private void reindex(Long companyId) {
        if (companyId == null) return;
        py.postAsync("/api/ai/rag/reindex?companyId=" + companyId);
    }
}
