-- ============================================================================
-- 企业级治理：可追溯（trace / 决策留痕）+ 多方面评估留档
-- 幂等设计：ALTER 用存储过程判断列是否存在，可重复执行（docker init 重跑也安全）
-- ============================================================================

-- 幂等加列存储过程
DROP PROCEDURE IF EXISTS add_col_if_missing;
DELIMITER $$
CREATE PROCEDURE add_col_if_missing(IN tname VARCHAR(64), IN cname VARCHAR(64), IN cdef TEXT)
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = tname AND COLUMN_NAME = cname
    ) THEN
        SET @sql = CONCAT('ALTER TABLE `', tname, '` ADD COLUMN `', cname, '` ', cdef);
        PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;
    END IF;
END$$
DELIMITER ;

-- 幂等加索引存储过程
DROP PROCEDURE IF EXISTS add_idx_if_missing;
DELIMITER $$
CREATE PROCEDURE add_idx_if_missing(IN tname VARCHAR(64), IN iname VARCHAR(64), IN idef TEXT)
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = tname AND INDEX_NAME = iname
    ) THEN
        SET @sql = CONCAT('ALTER TABLE `', tname, '` ADD INDEX `', iname, '` ', idef);
        PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;
    END IF;
END$$
DELIMITER ;

-- ---------------------------------------------------------------------------
-- 1) ai_analysis：补追溯列
-- ---------------------------------------------------------------------------
CALL add_col_if_missing('ai_analysis', 'trace_id',      'VARCHAR(40) NULL');
CALL add_col_if_missing('ai_analysis', 'degrade_level', "VARCHAR(16) NOT NULL DEFAULT 'NONE'");
CALL add_col_if_missing('ai_analysis', 'degrade_reason','VARCHAR(500) NULL');
CALL add_col_if_missing('ai_analysis', 'duration_ms',  'BIGINT DEFAULT 0');
CALL add_col_if_missing('ai_analysis', 'llm_calls',    'INT DEFAULT 0');
CALL add_col_if_missing('ai_analysis', 'tool_calls',   'INT DEFAULT 0');
CALL add_col_if_missing('ai_analysis', 'model',        'VARCHAR(120) NULL');
CALL add_col_if_missing('ai_analysis', 'review_revised','TINYINT DEFAULT 0');
CALL add_idx_if_missing('ai_analysis', 'idx_ai_trace',  '(`trace_id`)');
CALL add_idx_if_missing('ai_analysis', 'idx_ai_created','(`created_at`)');

-- ---------------------------------------------------------------------------
-- 2) ai_tool_log：补追溯列（原来只有摘要，无法回答"这次调用耗时多久、成功了吗"）
-- ---------------------------------------------------------------------------
CALL add_col_if_missing('ai_tool_log', 'trace_id',    'VARCHAR(40) NULL');
CALL add_col_if_missing('ai_tool_log', 'analysis_id', 'BIGINT NULL');
CALL add_col_if_missing('ai_tool_log', 'duration_ms', 'BIGINT DEFAULT 0');
CALL add_col_if_missing('ai_tool_log', 'success',     'TINYINT DEFAULT 1');
CALL add_col_if_missing('ai_tool_log', 'error_message','VARCHAR(500) NULL');
CALL add_idx_if_missing('ai_tool_log', 'idx_tool_trace', '(`trace_id`)');
CALL add_idx_if_missing('ai_tool_log', 'idx_tool_analysis', '(`analysis_id`)');

-- ---------------------------------------------------------------------------
-- 3) ai_analysis_trace：一次分析的完整决策留痕（可追溯的核心表）
--    回答"这条结论是怎么得出来的"：路由了谁、调了什么工具、用了哪个模型、
--    哪一步降级、护栏有没有报警、引用核对结果如何。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_analysis_trace (
    id             BIGINT PRIMARY KEY AUTO_INCREMENT,
    trace_id       VARCHAR(40)  NOT NULL,
    analysis_id    BIGINT       NULL,
    company_id     BIGINT       NULL,
    user_id        BIGINT       NULL,
    question       VARCHAR(1000) NULL,
    -- 路由决策：派了哪些专员、开放了哪些工具
    route_json     MEDIUMTEXT   NULL,
    -- LLM 迭代
    iterations     INT DEFAULT 0,
    llm_calls      INT DEFAULT 0,
    model          VARCHAR(120) NULL,
    model_switched VARCHAR(200) NULL COMMENT '模型降级切换记录，如 deepseek->doubao',
    -- 工具
    tool_calls     INT DEFAULT 0,
    tool_failures  INT DEFAULT 0,
    tool_detail    MEDIUMTEXT   NULL COMMENT '每个工具的入参摘要/耗时/成败',
    -- 检索
    retrieval_json MEDIUMTEXT   NULL COMMENT '召回的 chunk id 与分数，供结果复现',
    -- 质量与降级
    degrade_level  VARCHAR(16)  NOT NULL DEFAULT 'NONE' COMMENT 'NONE/PARTIAL/FULL',
    degrade_reason VARCHAR(1000) NULL,
    guardrail_flags VARCHAR(1000) NULL,
    grounding_json MEDIUMTEXT   NULL,
    review_trigger VARCHAR(200) NULL,
    review_revised TINYINT DEFAULT 0,
    -- 成本与耗时
    prompt_chars   INT DEFAULT 0,
    duration_ms    BIGINT DEFAULT 0,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_trace_id (`trace_id`),
    INDEX idx_trace_company (`company_id`),
    INDEX idx_trace_user (`user_id`),
    INDEX idx_trace_created (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- 4) ai_eval_run：评估跑批留档（原来只落文件，换机器就丢；企业要求可审计）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_eval_run (
    id             BIGINT PRIMARY KEY AUTO_INCREMENT,
    run_id         VARCHAR(40)  NOT NULL UNIQUE,
    total          INT DEFAULT 0,
    passed         INT DEFAULT 0,
    pass_rate      DOUBLE DEFAULT 0,
    -- 多维度综合评分
    score          DOUBLE DEFAULT 0,
    grade          VARCHAR(8)   NULL COMMENT 'A/B/C/D',
    gate           VARCHAR(16)  NULL COMMENT 'PASSED/FAILED',
    dimensions     MEDIUMTEXT   NULL COMMENT '各维度得分 JSON',
    model          VARCHAR(200) NULL,
    wall_clock_ms  BIGINT DEFAULT 0,
    operator_id    BIGINT       NULL,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_eval_created (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP PROCEDURE IF EXISTS add_col_if_missing;
DROP PROCEDURE IF EXISTS add_idx_if_missing;
