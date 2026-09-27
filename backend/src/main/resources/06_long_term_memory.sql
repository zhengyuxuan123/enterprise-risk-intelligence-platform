-- ============================================================================
-- 长期记忆（跨会话）：把「这家企业曾经发生过什么」沉淀成可复用的记忆条目
--
-- 与 ai_conversation / ai_message 的分工：
--   * ai_message       = 短期记忆，只在同一会话内回看最近若干轮，会话结束即失忆；
--   * ai_memory（本表）= 长期记忆，按企业 / 用户维度沉淀，跨会话、跨提问复用。
--
-- 四条设计约束（改动前请先读完）：
--   1) **零 token**：写入与召回全部走本地规则与本地向量，任何时候都不会因
--      记忆而多调一次模型 —— 记忆是"省钱的东西"，不能自己先烧掉额度。
--   2) **冲突消解而不是覆盖**：同一 (作用域, 主题) 出现新值时，旧条目置
--      SUPERSEDED 并指向新条目，**不删除**。否则"上次说 8.3%、这次说 12%"
--      这种口径变化会被无声抹掉，而那正是最该被看见的信号。
--   3) **company_id / user_id 用 0 而不是 NULL**：MySQL 里 NULL 不参与唯一性判定，
--      用 NULL 会让"同一主题只能有一条有效记忆"的约束静默失效。
--   4) 本表**只是记忆**，不是证据：条目一律不参与正文的 [n] / 来源ID 编号，
--      引用它的历史数字时必须标注为历史快照。
-- 幂等设计：全部 CREATE TABLE IF NOT EXISTS，可重复执行（docker init 重跑也安全）
-- ============================================================================

USE risk_platform;

CREATE TABLE IF NOT EXISTS ai_memory(
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    -- 作用域：COMPANY（企业事实）/ USER（用户偏好）
    scope VARCHAR(16) NOT NULL DEFAULT 'COMPANY',
    -- 企业级记忆归属企业；用户级记忆同样记录企业（便于按企业清理）
    company_id BIGINT NOT NULL DEFAULT 0,
    -- 记忆归属用户；企业级事实写 0
    user_id BIGINT NOT NULL DEFAULT 0,
    -- FACT=事实断言 / EPISODE=一次分析处置经过 / PREFERENCE=用户偏好
    kind VARCHAR(16) NOT NULL DEFAULT 'FACT',
    -- 归一化主题（人可读），用于同主题覆盖判定
    topic VARCHAR(160) NOT NULL DEFAULT '',
    -- 主题指纹（稳定、可比），冲突消解与去重的依据
    topic_key VARCHAR(64) NOT NULL DEFAULT '',
    content VARCHAR(1000) NOT NULL,
    -- 重要度 0.00~1.00：高风险、被反复命中的条目更高
    importance DECIMAL(4, 2) NOT NULL DEFAULT 0.50,
    -- 被召回注入的次数（用于"这条记忆到底有没有用"的复盘）
    hits INT NOT NULL DEFAULT 0,
    status VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',
    -- 被哪条新记忆取代（status=SUPERSEDED 时有值）
    superseded_by BIGINT,
    -- 出处：来自哪次分析，用于血缘与"这条记忆凭什么存在"
    source_analysis_id BIGINT,
    source_trace_id VARCHAR(40),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_used_at DATETIME,
    INDEX idx_mem_scope(scope, company_id, status),
    INDEX idx_mem_user(scope, user_id, status),
    INDEX idx_mem_topic(scope, company_id, user_id, kind, topic_key, status)
)ENGINE=InnoDB;
