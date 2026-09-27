USE risk_platform;

-- ============================================================================
-- Agent 增量：多轮会话记忆 + ai_analysis 扩展 + 写工具审批队列 + 主动预警记录
--
-- 这段原本只躺在 backend/src/main/resources/03_agent_upgrade.sql 里，
-- 不在 mysql/init/ 的初始化链路上 —— 全新部署会缺表，多轮对话直接报错。
-- 现在并入初始化，并且每一处都写成幂等，重复执行不会炸。
-- ============================================================================

-- ---------------------- 多轮会话记忆 ----------------------
CREATE TABLE IF NOT EXISTS ai_conversation(
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    company_id BIGINT,
    title VARCHAR(200),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_conv_user(user_id),
    INDEX idx_conv_company(company_id)
)ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ai_message(
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    conversation_id BIGINT NOT NULL,
    role VARCHAR(20) NOT NULL,
    content MEDIUMTEXT,
    tool_trace_json TEXT,
    answer_json MEDIUMTEXT,
    provider VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_msg_conv(conversation_id)
)ENGINE=InnoDB;

-- ---------------------- ai_analysis 扩展列（存在才加） ----------------------
SET @db = 'risk_platform';

SELECT COUNT(*) INTO @c FROM information_schema.columns WHERE table_schema=@db AND table_name='ai_analysis' AND column_name='conversation_id';
SET @sql = IF(@c=0, 'ALTER TABLE ai_analysis ADD COLUMN conversation_id BIGINT NULL', 'SELECT 1');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @c FROM information_schema.columns WHERE table_schema=@db AND table_name='ai_analysis' AND column_name='answer_json';
SET @sql = IF(@c=0, 'ALTER TABLE ai_analysis ADD COLUMN answer_json MEDIUMTEXT', 'SELECT 1');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @c FROM information_schema.columns WHERE table_schema=@db AND table_name='ai_analysis' AND column_name='confidence';
SET @sql = IF(@c=0, 'ALTER TABLE ai_analysis ADD COLUMN confidence VARCHAR(20)', 'SELECT 1');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @c FROM information_schema.columns WHERE table_schema=@db AND table_name='ai_analysis' AND column_name='grounded';
SET @sql = IF(@c=0, 'ALTER TABLE ai_analysis ADD COLUMN grounded TINYINT DEFAULT 1', 'SELECT 1');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ---------------------- 写工具人工审批队列 ----------------------
-- Agent 只允许「提议动作」，不允许直接改业务数据。所有写工具的产物先落这里，
-- 由人确认后才真正执行，执行前后都留痕。
CREATE TABLE IF NOT EXISTS ai_action_approval(
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    company_id BIGINT NOT NULL,
    requester_user_id BIGINT,
    analysis_id BIGINT,
    action_type VARCHAR(50) NOT NULL COMMENT 'CREATE_TICKET / NOTIFY_OWNER / UPDATE_EVENT_STATUS',
    action_args TEXT COMMENT '动作入参 JSON',
    reason VARCHAR(2000) COMMENT 'Agent 给出的依据（含引用）',
    risk_level VARCHAR(20),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT 'PENDING/APPROVED/REJECTED/EXECUTED/FAILED',
    source VARCHAR(20) NOT NULL DEFAULT 'agent' COMMENT 'agent=用户提问 / proactive=事件预警',
    approver_user_id BIGINT,
    approve_comment VARCHAR(500),
    execute_result TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    approved_at DATETIME,
    executed_at DATETIME,
    INDEX idx_appr_status(status),
    INDEX idx_appr_company(company_id),
    INDEX idx_appr_created(created_at)
)ENGINE=InnoDB;

-- ---------------------- 主动预警记录（含每日配额计数） ----------------------
CREATE TABLE IF NOT EXISTS ai_proactive_alert(
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    company_id BIGINT NOT NULL,
    trigger_type VARCHAR(40) NOT NULL COMMENT 'METRIC_THRESHOLD / NEW_COMPLAINT / KB_UPDATED',
    trigger_ref VARCHAR(120) COMMENT '触发对象，如 metric:123 / complaint:45',
    question VARCHAR(1000),
    risk_level VARCHAR(20),
    headline VARCHAR(500),
    answer_json MEDIUMTEXT,
    dispatched VARCHAR(20) COMMENT 'ARCHIVED / PENDING_APPROVAL / ALERTED',
    approval_id BIGINT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_pa_company_date(company_id, created_at),
    INDEX idx_pa_level(risk_level),
    INDEX idx_pa_trigger(trigger_type)
)ENGINE=InnoDB;
