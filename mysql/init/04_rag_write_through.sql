-- ============================================================================
-- 04_rag_write_through.sql
-- 「写入即索引」改造：本地资料库（Lucene 向量索引）的增量记账表
--
-- 背景：改造前索引只在「被提问那一刻」由 RagService.searchDetailed() 同步一次，
--       业务模块写入的数据不入库，索引也只反映"最后一次查询那位用户的可见集"。
-- 改造后：业务写入 → 发 CorpusChangeEvent → 事务提交后异步增量入索引。
--       本表记录每个语料块的内容指纹与入库状态，用于：
--         ① 增量判定（hash 没变就不重新向量化）
--         ② 删除定位（物理删除后已查不到原记录，只能靠这里反查 chunk_id）
--         ③ 失败补偿（status=FAILED 的由巡检任务重跑）
--         ④ 巡检对账（比对 state 与业务表，补漏删多）
-- ============================================================================

CREATE TABLE IF NOT EXISTS rag_chunk_state (
    id            BIGINT       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    chunk_id      VARCHAR(128) NOT NULL                COMMENT '语料块主键：k:12#0 / m:12 / e:7 / agg:metric:3',
    source_type   VARCHAR(32)  NOT NULL                COMMENT 'knowledge/metric/event/complaint/competitor/company/rule',
    source_id     VARCHAR(64)  NOT NULL                COMMENT '业务记录主键（聚合切片存企业ID）',
    company_id    BIGINT       NULL                    COMMENT '所属企业；NULL = 全局语料（如风险规则）',
    content_hash  VARCHAR(32)  NOT NULL                COMMENT 'title+text 的 SHA-256 前 8 字节，增量判定依据',
    status        VARCHAR(16)  NOT NULL DEFAULT 'PENDING' COMMENT 'PENDING=待入库 / INDEXED=已入库 / FAILED=失败待重试',
    retry_count   INT          NOT NULL DEFAULT 0      COMMENT '连续失败次数，超过阈值不再自动重试',
    error         VARCHAR(512) NULL                    COMMENT '最后一次失败原因（截断）',
    updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_chunk (chunk_id),
    KEY idx_source (source_type, source_id),
    KEY idx_company (company_id),
    KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RAG 语料块入库状态（写入即索引的记账表）';
