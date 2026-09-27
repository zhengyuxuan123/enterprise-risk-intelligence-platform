"""表与列的显式声明：这里是 Python 侧唯一的 schema 真相。

只声明 agent 栈真正会用到的列。业务表全部**只读**，
唯一会写的是 ``rag_chunk_state``（语料索引的记账表）。
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Table,
    Text,
)

META = MetaData()


def _t(name: str, *cols: Column) -> Table:
    return Table(name, META, *cols)


def _pk() -> Column:
    return Column("id", BigInteger, primary_key=True)


company = _t(
    "company",
    _pk(),
    Column("company_code", String(50)),
    Column("company_name", String(200)),
    Column("industry", String(100)),
    Column("region", String(100)),
    Column("customer_level", String(30)),
    Column("company_scale", String(50)),
    Column("dept_id", BigInteger),
    Column("owner_user_id", BigInteger),
    Column("status", SmallInteger),
)

business_metric = _t(
    "business_metric",
    _pk(),
    Column("company_id", BigInteger),
    Column("metric_date", Date),
    Column("metric_code", String(60)),
    Column("metric_name", String(100)),
    Column("metric_value", Numeric(18, 4)),
    Column("unit", String(30)),
    Column("source_type", String(30)),
)

risk_event = _t(
    "risk_event",
    _pk(),
    Column("event_no", String(80)),
    Column("company_id", BigInteger),
    Column("rule_id", BigInteger),
    Column("risk_type", String(80)),
    Column("risk_title", String(200)),
    Column("risk_level", String(20)),
    Column("status", String(30)),
    Column("trigger_value", Numeric(18, 4)),
    Column("threshold_value", Numeric(18, 4)),
    Column("metric_date", Date),
    Column("assignee_user_id", BigInteger),
    Column("handle_result", Text),
    Column("review_comment", Text),
    Column("created_at", DateTime),
)

complaint = _t(
    "complaint",
    _pk(),
    Column("complaint_no", String(80)),
    Column("company_id", BigInteger),
    Column("complaint_date", Date),
    Column("customer_name", String(100)),
    Column("product_name", String(150)),
    Column("category", String(100)),
    Column("description", Text),
    Column("severity", String(10)),
    Column("repeat_flag", SmallInteger),
    Column("solve_hours", Numeric(10, 2)),
    Column("sla_exceeded", SmallInteger),
    Column("root_cause", String(100)),
    Column("churn_risk", String(20)),
    Column("status", String(30)),
)

competitor_product = _t(
    "competitor_product",
    _pk(),
    Column("company_id", BigInteger),
    Column("competitor_name", String(150)),
    Column("product_name", String(150)),
    Column("target_customer", String(100)),
    Column("price", Numeric(18, 2)),
    Column("price_unit", String(50)),
    Column("selling_point", String(500)),
    Column("weakness", String(500)),
    Column("promotion", String(300)),
    Column("delivery_cycle", String(100)),
    Column("service_commitment", String(200)),
    Column("risk_level", String(20)),
    Column("updated_date", Date),
)

knowledge_document = _t(
    "knowledge_document",
    _pk(),
    Column("company_id", BigInteger),
    Column("dept_id", BigInteger),
    Column("title", String(300)),
    Column("doc_type", String(80)),
    Column("original_filename", String(300)),
    Column("security_level", Integer),
    Column("content", Text),
    Column("uploader_user_id", BigInteger),
    Column("storage_uri", String(1000)),
    Column("file_sha256", String(64)),
    Column("file_size", BigInteger),
    Column("mime_type", String(150)),
    Column("document_version", Integer),
    Column("ingest_status", String(24)),
    Column("parse_quality_score", Numeric(5, 2)),
    Column("chunk_count", Integer),
    Column("deleted", SmallInteger),
)

risk_rule = _t(
    "risk_rule",
    _pk(),
    Column("rule_name", String(200)),
    Column("risk_type", String(80)),
    Column("metric_code", String(60)),
    Column("operator_code", String(30)),
    Column("threshold_value", Numeric(18, 4)),
    Column("risk_level", String(20)),
    Column("enabled", SmallInteger),
    Column("description", String(1000)),
)

rag_chunk_state = _t(
    "rag_chunk_state",
    _pk(),
    Column("chunk_id", String(128)),
    Column("source_type", String(32)),
    Column("source_id", String(64)),
    Column("company_id", BigInteger),
    Column("content_hash", String(32)),
    Column("status", String(16)),
    Column("retry_count", Integer),
    Column("error", String(512)),
    Column("document_version", Integer),
    Column("indexed_at", DateTime),
    Column("updated_at", DateTime),
)

# --- AI 落库表：ai_analysis / trace / conversation / message / eval / tool_log ---

ai_analysis = _t(
    "ai_analysis",
    _pk(),
    Column("company_id", BigInteger),
    Column("user_id", BigInteger),
    Column("question", Text),
    Column("answer", Text),
    Column("evidence_json", Text),
    Column("tool_trace_json", Text),
    Column("provider", String(100)),
    Column("created_at", DateTime),
    Column("conversation_id", BigInteger),
    Column("answer_json", Text),
    Column("confidence", String(20)),
    Column("grounded", SmallInteger),
    Column("trace_id", String(40)),
    Column("degrade_level", String(16)),
    Column("degrade_reason", String(500)),
    Column("duration_ms", BigInteger),
    Column("llm_calls", Integer),
    Column("tool_calls", Integer),
    Column("model", String(120)),
    Column("review_revised", SmallInteger),
)

ai_analysis_trace = _t(
    "ai_analysis_trace",
    _pk(),
    Column("trace_id", String(40)),
    Column("analysis_id", BigInteger),
    Column("company_id", BigInteger),
    Column("user_id", BigInteger),
    Column("question", String(1000)),
    Column("route_json", Text),
    Column("iterations", Integer),
    Column("llm_calls", Integer),
    Column("model", String(120)),
    Column("model_switched", String(200)),
    Column("tool_calls", Integer),
    Column("tool_failures", Integer),
    Column("tool_detail", Text),
    Column("retrieval_json", Text),
    Column("degrade_level", String(16)),
    Column("degrade_reason", String(1000)),
    Column("guardrail_flags", String(1000)),
    Column("grounding_json", Text),
    Column("review_trigger", String(200)),
    Column("review_revised", SmallInteger),
    Column("prompt_chars", Integer),
    Column("duration_ms", BigInteger),
    Column("created_at", DateTime),
)

ai_conversation = _t(
    "ai_conversation",
    _pk(),
    Column("user_id", BigInteger),
    Column("company_id", BigInteger),
    Column("title", String(200)),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
)

ai_message = _t(
    "ai_message",
    _pk(),
    Column("conversation_id", BigInteger),
    Column("role", String(20)),
    Column("content", Text),
    Column("tool_trace_json", Text),
    Column("answer_json", Text),
    Column("provider", String(100)),
    Column("created_at", DateTime),
)

#: 长期记忆（跨会话）。DDL 见 ``mysql/init/06_long_term_memory.sql``。
#: 与 ``ai_message``（会话内短期记忆）分工不同：这张表按企业 / 用户维度沉淀，
#: 换一个会话、换一个提问都能复用。``company_id`` / ``user_id`` 写 0 而非 NULL ——
#: MySQL 里 NULL 不参与唯一性判定，用 NULL 会让"同主题只有一条有效"静默失效。
ai_memory = _t(
    "ai_memory",
    _pk(),
    Column("scope", String(16)),
    Column("company_id", BigInteger),
    Column("user_id", BigInteger),
    Column("kind", String(16)),
    Column("topic", String(160)),
    Column("topic_key", String(64)),
    Column("content", String(1000)),
    Column("importance", Numeric(4, 2)),
    Column("hits", Integer),
    Column("status", String(16)),
    Column("superseded_by", BigInteger),
    Column("source_analysis_id", BigInteger),
    Column("source_trace_id", String(40)),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
    Column("last_used_at", DateTime),
)

ai_eval_run = _t(
    "ai_eval_run",
    _pk(),
    Column("run_id", String(40)),
    Column("total", Integer),
    Column("passed", Integer),
    Column("pass_rate", Numeric(10, 4)),
    Column("score", Numeric(10, 4)),
    Column("grade", String(8)),
    Column("gate", String(16)),
    Column("dimensions", Text),
    Column("model", String(200)),
    Column("wall_clock_ms", BigInteger),
    Column("operator_id", BigInteger),
    Column("created_at", DateTime),
)

ai_tool_log = _t(
    "ai_tool_log",
    _pk(),
    Column("user_id", BigInteger),
    Column("company_id", BigInteger),
    Column("tool_name", String(100)),
    Column("request_summary", String(1000)),
    Column("result_summary", String(1000)),
    Column("created_at", DateTime),
    Column("trace_id", String(40)),
    Column("analysis_id", BigInteger),
    Column("duration_ms", BigInteger),
    Column("success", SmallInteger),
    Column("error_message", String(500)),
)

ai_action_approval = _t(
    "ai_action_approval",
    _pk(),
    Column("company_id", BigInteger),
    Column("requester_user_id", BigInteger),
    Column("analysis_id", BigInteger),
    Column("action_type", String(50)),
    Column("action_args", Text),
    Column("reason", String(2000)),
    Column("risk_level", String(20)),
    Column("status", String(20)),
    Column("source", String(20)),
    Column("approver_user_id", BigInteger),
    Column("approve_comment", String(500)),
    Column("execute_result", Text),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
    Column("approved_at", DateTime),
    Column("executed_at", DateTime),
)

# --- 主动预警与审计 ------------------------------------------------

#: 事件驱动研判的配额台账。**每日配额就是按这张表计数**，
#: 所以它同时也是"这次消耗有没有发生过"的唯一凭据。
ai_proactive_alert = _t(
    "ai_proactive_alert",
    _pk(),
    Column("company_id", BigInteger),
    Column("trigger_type", String(40)),
    Column("trigger_ref", String(120)),
    Column("question", String(1000)),
    Column("risk_level", String(20)),
    Column("headline", String(500)),
    Column("answer_json", Text),
    Column("dispatched", String(20)),
    Column("approval_id", BigInteger),
    Column("created_at", DateTime),
)

#: 操作日志（审计）。通知的"必达层"就落在这里。
operation_log = _t(
    "operation_log",
    _pk(),
    Column("user_id", BigInteger),
    Column("action", String(60)),
    Column("resource_type", String(80)),
    Column("resource_id", BigInteger),
    Column("detail", String(2000)),
    Column("ip_address", String(64)),
    Column("created_at", DateTime),
)

sys_user = _t(
    "sys_user",
    _pk(),
    Column("username", String(64)),
    Column("real_name", String(64)),
    Column("dept_id", BigInteger),
    Column("status", SmallInteger),
)

# ------------------------------------------------------------------
# rag_chunk_state 的状态常量（与 Java RagChunkState 一致）
# ------------------------------------------------------------------

ST_INDEXED = "INDEXED"
ST_FAILED = "FAILED"
ST_PENDING = "PENDING"

# ------------------------------------------------------------------
# ai_memory 的枚举（与 06_long_term_memory.sql 里的默认值一致）
# ------------------------------------------------------------------

#: 企业级记忆（事实、处置经过）
MEM_SCOPE_COMPANY = "COMPANY"
#: 用户级记忆（偏好）
MEM_SCOPE_USER = "USER"

MEM_KIND_FACT = "FACT"
MEM_KIND_EPISODE = "EPISODE"
MEM_KIND_PREFERENCE = "PREFERENCE"

MEM_ACTIVE = "ACTIVE"
#: 被同一主题的新记忆取代。**只标记不删除** —— 口径变化本身是要被看见的信号。
MEM_SUPERSEDED = "SUPERSEDED"
