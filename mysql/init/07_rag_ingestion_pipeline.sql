-- Durable RAG ingestion pipeline: file retention, versioning, jobs and outbox.

DROP PROCEDURE IF EXISTS add_ingest_col;
DELIMITER $$
CREATE PROCEDURE add_ingest_col(IN tname VARCHAR(64), IN cname VARCHAR(64), IN cdef TEXT)
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

DROP PROCEDURE IF EXISTS add_ingest_idx;
DELIMITER $$
CREATE PROCEDURE add_ingest_idx(IN tname VARCHAR(64), IN iname VARCHAR(64), IN idef TEXT)
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

CALL add_ingest_col('knowledge_document', 'storage_uri',
    'VARCHAR(1000) NULL COMMENT ''原始文件相对存储路径''');
CALL add_ingest_col('knowledge_document', 'file_sha256',
    'VARCHAR(64) NULL COMMENT ''原始文件 SHA-256''');
CALL add_ingest_col('knowledge_document', 'file_size',
    'BIGINT NOT NULL DEFAULT 0');
CALL add_ingest_col('knowledge_document', 'mime_type',
    'VARCHAR(150) NULL');
CALL add_ingest_col('knowledge_document', 'document_version',
    'INT NOT NULL DEFAULT 1');
CALL add_ingest_col('knowledge_document', 'ingest_status',
    'VARCHAR(24) NOT NULL DEFAULT ''INDEXED''');
CALL add_ingest_col('knowledge_document', 'ingest_error',
    'VARCHAR(1000) NULL');
CALL add_ingest_col('knowledge_document', 'parse_quality_score',
    'DECIMAL(5,2) NULL');
CALL add_ingest_col('knowledge_document', 'parse_warning',
    'VARCHAR(1000) NULL');
CALL add_ingest_col('knowledge_document', 'chunk_count',
    'INT NOT NULL DEFAULT 0');
CALL add_ingest_col('knowledge_document', 'indexed_at',
    'DATETIME NULL');
CALL add_ingest_col('knowledge_document', 'deleted',
    'TINYINT NOT NULL DEFAULT 0');

CALL add_ingest_idx('knowledge_document', 'idx_knowledge_hash', '(`file_sha256`)');
CALL add_ingest_idx('knowledge_document', 'idx_knowledge_ingest', '(`ingest_status`,`deleted`)');

CREATE TABLE IF NOT EXISTS rag_ingest_job (
    id                 BIGINT PRIMARY KEY AUTO_INCREMENT,
    document_id        BIGINT NOT NULL,
    document_version   INT NOT NULL,
    stage              VARCHAR(24) NOT NULL DEFAULT 'VALIDATE',
    status             VARCHAR(24) NOT NULL DEFAULT 'PENDING',
    progress           INT NOT NULL DEFAULT 0,
    retry_count        INT NOT NULL DEFAULT 0,
    max_retries        INT NOT NULL DEFAULT 5,
    quality_score      DECIMAL(5,2) NULL,
    chunk_count        INT NOT NULL DEFAULT 0,
    error_message      VARCHAR(2000) NULL,
    created_by         BIGINT NULL,
    started_at         DATETIME NULL,
    heartbeat_at       DATETIME NULL,
    finished_at        DATETIME NULL,
    created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_ingest_doc_version(document_id, document_version),
    KEY idx_ingest_status(status, updated_at),
    KEY idx_ingest_document(document_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RAG 文档入库任务';

CREATE TABLE IF NOT EXISTS rag_outbox_event (
    id                 BIGINT PRIMARY KEY AUTO_INCREMENT,
    event_key          VARCHAR(160) NOT NULL,
    aggregate_type     VARCHAR(40) NOT NULL,
    aggregate_id       BIGINT NOT NULL,
    event_type         VARCHAR(40) NOT NULL,
    payload            MEDIUMTEXT NOT NULL,
    status             VARCHAR(16) NOT NULL DEFAULT 'PENDING',
    retry_count        INT NOT NULL DEFAULT 0,
    max_retries        INT NOT NULL DEFAULT 8,
    next_retry_at      DATETIME NULL,
    last_error         VARCHAR(1000) NULL,
    created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    published_at       DATETIME NULL,
    UNIQUE KEY uk_outbox_event(event_key),
    KEY idx_outbox_dispatch(status, next_retry_at, id),
    KEY idx_outbox_aggregate(aggregate_type, aggregate_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='可靠事件 Outbox';

CALL add_ingest_col('rag_chunk_state', 'document_version', 'INT NOT NULL DEFAULT 1');
CALL add_ingest_col('rag_chunk_state', 'indexed_at', 'DATETIME NULL');

DROP PROCEDURE IF EXISTS add_ingest_col;
DROP PROCEDURE IF EXISTS add_ingest_idx;
