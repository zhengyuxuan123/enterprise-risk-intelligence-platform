USE risk_platform;

-- 多轮会话记忆表
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

-- ai_analysis 扩展字段（存在才加，幂等）
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
