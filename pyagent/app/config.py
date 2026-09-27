"""配置层：与 Java 版 ``application.yml`` 一一对应的环境变量契约。

设计原则
--------
1. **环境变量名与 Java 版完全一致**，包括 `@Value` 的默认值。
   这样同一份 ``.env`` 可以同时喂给 Java 与 Python 两个实现，
   双跑对照时不会出现"差异其实是配置不同"这种查不出来的问题。
2. 键的层级也照搬 ``app.ai.*`` / ``app.rag.*`` / ``app.agent.*`` / ``app.web-search.*`` /
   ``app.web.*`` / ``app.eval.*``，便于两边逐行对照。
3. 凡是 Java 侧用 ``${ENV:default}`` 给了默认值的，这里都写同样的默认值；
   Java 侧没给默认值（必填）的，这里保持空串并交给上层给出可操作的中文原因。

关于别名
--------
有两处一个字段对应多个环境变量，用 ``AliasChoices`` 表达：

* ``ai.api_key``  ← ``AI_API_KEY`` / ``OPENAI_API_KEY``（后者仅为兼容旧部署）
* ``web_search.api_key`` ← ``APP_WEB_SEARCH_API_KEY``，留空时**按 Java 侧同样的顺序**
  回退到 ``AI_API_KEY`` / ``OPENAI_API_KEY`` / 主客户端 Key。回退逻辑写在
  :mod:`pyagent.app.web.search_service` 里，而不在这里 —— 因为它依赖主客户端是否已配置。
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV = SettingsConfigDict(
    env_file=(".env", "../.env", "backend/.env"),
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class DbSettings(BaseSettings):
    """数据库：与 Java 侧同名同默认值。"""

    model_config = _ENV

    host: str = Field("localhost", validation_alias="DB_HOST")
    port: int = Field(3306, validation_alias="DB_PORT")
    name: str = Field("risk_platform", validation_alias="DB_NAME")
    user: str = Field("root", validation_alias="DB_USER")
    password: str = Field("123456", validation_alias="DB_PASSWORD")

    @property
    def dsn(self) -> str:
        return (
            f"mysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
            "?charset=utf8mb4"
        )


class AiSettings(BaseSettings):
    """``app.ai.*``"""

    model_config = _ENV

    enabled: bool = Field(True, validation_alias="APP_AI_ENABLED")
    base_url: str = Field(
        "https://ark.cn-beijing.volces.com/api/v3", validation_alias="AI_BASE_URL"
    )
    api_key: str = Field(
        "", validation_alias=AliasChoices("AI_API_KEY", "OPENAI_API_KEY")
    )
    #! 留空 = 运行时自动从"账号实际可用的模型"里选一个（推荐），见 Java 侧注释。
    model: str = Field("", validation_alias="AI_MODEL")
    model_candidates: str = Field("", validation_alias="AI_MODEL_CANDIDATES")
    model_auto_fallback: bool = Field(True, validation_alias="AI_MODEL_AUTO_FALLBACK")
    model_discovery: bool = Field(True, validation_alias="AI_MODEL_DISCOVERY")
    auto_consume: bool = Field(False, validation_alias="APP_AI_AUTO_CONSUME")
    #! 一次调用超时 ≤ SSE 连接上限，否则必然出现"连接还在、对面已经把它掐了"。
    chat_timeout_seconds: int = Field(600, validation_alias="APP_AI_CHAT_TIMEOUT")
    #! disabled（默认）/ enabled / 空串=完全不下发该字段。
    thinking_type: str = Field("disabled", validation_alias="APP_AI_THINKING_TYPE")
    embedding_model: str = Field("", validation_alias="AI_EMBEDDING_MODEL")

    #! 注意：候选串的解析**不要**在这里做。Java 是 ``raw.split("[,\\r\\n]")``
    #    （逗号 / CR / LF 都算分隔，再 trim + 去重保序），实现放在
    #    ``ai.fallback.parse_candidates`` —— 那里有 jshell 取的 Java 基准做对账。
    #    这里只存原始串，避免同一份解析逻辑出现第二个版本。


class RagSettings(BaseSettings):
    """``app.rag.*``"""

    model_config = _ENV

    vector_weight: float = Field(0.6, validation_alias="APP_RAG_VECTOR_WEIGHT")
    embedding_enabled: bool = Field(True, validation_alias="APP_RAG_EMBEDDING_ENABLED")
    #! local（默认，零成本、不联网）/ remote / auto
    embedding_mode: str = Field("local", validation_alias="APP_RAG_EMBEDDING_MODE")
    local_embedding_dim: int = Field(512, validation_alias="APP_RAG_LOCAL_EMBEDDING_DIM")
    embedding_cache_ttl_minutes: int = Field(1440, validation_alias="APP_RAG_EMBEDDING_TTL")
    embedding_base_url: str = Field("", validation_alias="APP_EMBEDDING_BASE_URL")
    embedding_api_key: str = Field("", validation_alias="APP_EMBEDDING_API_KEY")
    embedding_model: str = Field("", validation_alias="APP_EMBEDDING_MODEL")
    embedding_dimensions: int = Field(0, validation_alias="APP_EMBEDDING_DIMENSIONS")
    embedding_timeout_seconds: int = Field(20, validation_alias="APP_EMBEDDING_TIMEOUT")

    query_rewrite: bool = Field(True, validation_alias="APP_RAG_QUERY_REWRITE")
    multi_recall: bool = Field(True, validation_alias="APP_RAG_MULTI_RECALL")
    rerank: bool = Field(True, validation_alias="APP_RAG_RERANK")
    rerank_pool: int = Field(25, validation_alias="APP_RAG_RERANK_POOL")
    rerank_weight: float = Field(0.6, validation_alias="APP_RAG_RERANK_WEIGHT")
    diversity: float = Field(0.78, validation_alias="APP_RAG_DIVERSITY")
    min_score: float = Field(6.0, validation_alias="APP_RAG_MIN_SCORE")
    relative_floor: float = Field(0.45, validation_alias="APP_RAG_RELATIVE_FLOOR")

    index_enabled: bool = Field(True, validation_alias="APP_RAG_INDEX_ENABLED")
    index_dir: str = Field("./data/rag-index", validation_alias="APP_RAG_INDEX_DIR")
    index_rebuild: bool = Field(False, validation_alias="APP_RAG_INDEX_REBUILD")
    sync_throttle_ms: int = Field(30000, validation_alias="APP_RAG_SYNC_THROTTLE_MS")

    structured_corpus: bool = Field(True, validation_alias="APP_RAG_STRUCTURED_CORPUS")
    structured_max: int = Field(12, validation_alias="APP_RAG_STRUCTURED_MAX")
    structured_guarantee: int = Field(1, validation_alias="APP_RAG_STRUCTURED_GUARANTEE")
    #! StructuredCorpus 的四个 @Value（Java 侧同名同默认值）
    structured_metrics: int = Field(8, validation_alias="APP_RAG_STRUCTURED_METRICS")
    structured_events: int = Field(8, validation_alias="APP_RAG_STRUCTURED_EVENTS")
    structured_complaints: int = Field(5, validation_alias="APP_RAG_STRUCTURED_COMPLAINTS")
    structured_metric_window: int = Field(6, validation_alias="APP_RAG_STRUCTURED_METRIC_WINDOW")

    #! 写入即索引（RagIndexingService 的 @Value）
    write_through_enabled: bool = Field(True, validation_alias="APP_RAG_WRITE_THROUGH_ENABLED")
    merge_window_ms: int = Field(3000, validation_alias="APP_RAG_WRITE_THROUGH_MERGE_WINDOW_MS")
    #! auto（默认）/ true / false —— auto = 本地向量恒算，第三方向量留到巡检批量补
    embed_on_write: str = Field("auto", validation_alias="APP_RAG_WRITE_THROUGH_EMBED_ON_WRITE")
    reconcile_enabled: bool = Field(True, validation_alias="APP_RAG_WRITE_THROUGH_RECONCILE")
    reconcile_interval_minutes: int = Field(360, validation_alias="APP_RAG_WRITE_THROUGH_RECONCILE_MINUTES")


class AgentSettings(BaseSettings):
    """``app.agent.*``"""

    model_config = _ENV

    max_iterations: int = Field(6, validation_alias="APP_AGENT_MAX_ITER")
    max_tool_calls: int = Field(20, validation_alias="APP_AGENT_MAX_TOOL_CALLS")
    default_depth: str = Field("standard", validation_alias="APP_AGENT_DEFAULT_DEPTH")
    structured_output: bool = Field(True, validation_alias="APP_AGENT_STRUCTURED")
    cache_enabled: bool = Field(True, validation_alias="APP_AGENT_CACHE")
    cache_ttl_seconds: int = Field(600, validation_alias="APP_AGENT_CACHE_TTL")
    memory_turns: int = Field(6, validation_alias="APP_AGENT_MEMORY_TURNS")
    memory_token_budget: int = Field(1200, validation_alias="APP_AGENT_MEMORY_TOKEN_BUDGET")
    memory_summary_mode: str = Field("extractive", validation_alias="APP_AGENT_MEMORY_SUMMARY_MODE")
    # ---- 长期记忆（跨会话）----
    #: 关掉即可一键回到"只有会话内记忆"的旧行为。
    long_term_memory: bool = Field(True, validation_alias="APP_AGENT_LONG_TERM_MEMORY")
    #: 是否把分析结果沉淀成长期记忆。关掉只影响写入，召回照常。
    long_term_write: bool = Field(True, validation_alias="APP_AGENT_LONG_TERM_WRITE")
    #: 注入的**独立**预算（字符）。与 memory_token_budget 分开计 ——
    #: 混在一起会让"会话上下文变长"顺带挤掉长期记忆，反之亦然，两边都说不清。
    long_term_budget: int = Field(int(600), validation_alias="APP_AGENT_LONG_TERM_BUDGET")
    #: 每次最多注入几条。
    long_term_top_k: int = Field(5, validation_alias="APP_AGENT_LONG_TERM_TOP_K")
    #: 召回得分下限（词面/向量混合分）。低于它宁可不注入，也不要用无关记忆带偏结论。
    long_term_min_score: float = Field(0.18, validation_alias="APP_AGENT_LONG_TERM_MIN_SCORE")
    #: 单企业保留的记忆条数上限，超出按"重要度 + 时间"淘汰（置 SUPERSEDED）。
    long_term_max_per_company: int = Field(500, validation_alias="APP_AGENT_LONG_TERM_MAX")
    citation_repair: bool = Field(True, validation_alias="APP_AGENT_CITATION_REPAIR")
    tool_result_max_chars: int = Field(6000, validation_alias="APP_AGENT_TOOL_MAX_CHARS")
    tool_retry: int = Field(1, validation_alias="APP_AGENT_TOOL_RETRY")
    review_revise: bool = Field(True, validation_alias="APP_AGENT_REVIEW_REVISE")
    circuit_threshold: int = Field(2, validation_alias="APP_AGENT_CIRCUIT_THRESHOLD")
    circuit_cooldown_seconds: int = Field(120, validation_alias="APP_AGENT_CIRCUIT_COOLDOWN")
    #! 必须大于一次分析的最坏耗时（实测 169s~743s），宁可长也不要短。
    sse_timeout_seconds: int = Field(900, validation_alias="APP_AGENT_SSE_TIMEOUT")
    sse_heartbeat_seconds: int = Field(15, validation_alias="APP_AGENT_SSE_HEARTBEAT")
    tool_timeout_seconds: int = Field(60, validation_alias="APP_AGENT_TOOL_TIMEOUT")
    stream_flush_chars: int = Field(400, validation_alias="APP_AGENT_STREAM_FLUSH_CHARS")
    retry_max_tokens: int = Field(8192, validation_alias="APP_AGENT_RETRY_MAX_TOKENS")
    #! 提速第一杠杆：本地毫秒级预取，省掉"模型决定调什么工具"那一整轮往返。
    prefetch_tools: bool = Field(True, validation_alias="APP_AGENT_PREFETCH_TOOLS")
    skip_structure_extract: bool = Field(True, validation_alias="APP_AGENT_SKIP_STRUCTURE_EXTRACT")
    single_flight: bool = Field(True, validation_alias="APP_AGENT_SINGLE_FLIGHT")
    single_flight_wait_seconds: int = Field(180, validation_alias="APP_AGENT_SINGLE_FLIGHT_WAIT")
    review_always_llm: bool = Field(False, validation_alias="APP_AGENT_REVIEW_ALWAYS_LLM")
    #! 编排实现。**自研 legacy 已于 2026-09-23 删除，langgraph 是唯一实现**，
    #! 所以这里不再是一个"开关"：保留字段是为了①配置项/部署脚本不用连夜改，
    #! ②让残留的 ``APP_AGENT_ORCHESTRATOR=legacy`` 得到一条明确的告警而非静默。
    #! 未知值仍回退到 langgraph —— 拼错单词不该让 AI 整体不可用。
    orchestrator: str = Field("langgraph", validation_alias="APP_AGENT_ORCHESTRATOR")
    #! LangGraph 的 SQLite 状态快照。**它不等于长期记忆**：一个是图执行状态，一个是业务事实。
    langgraph_checkpoint_path: str = Field(
        "./data/langgraph.sqlite", validation_alias="APP_AGENT_LANGGRAPH_CHECKPOINT")
    #! 状态快照开关，默认**关**。
    #! 本项目的 thread_id 就是 trace_id —— 每次分析都是一条新 thread，快照**只写不读**，
    #! 拿不到 checkpoint 真正的价值（断点续跑 / interrupt 审批），却实打实地付出写入开销：
    #! 每轮每个节点都要 put_writes 一次。实测 live 对照里 langgraph 侧耗时接近 legacy 的 3 倍，
    #! 关掉它是最直接的一刀。真要做"长任务断点续跑"再显式打开。
    langgraph_checkpoint_enabled: bool = Field(
        False, validation_alias="APP_AGENT_LANGGRAPH_CHECKPOINT_ENABLED")

    #! ---- 时间预算与超时降级（:mod:`app.agent.deadline`）----
    #! 「到点得不到结果就降级」的总开关。关掉=完全回到旧行为（到点只是不再给工具）。
    deadline_enabled: bool = Field(True, validation_alias="APP_AGENT_DEADLINE_ENABLED")
    #! 0 = 跟随深度档位自带的挂钟预算（quick 75s / standard 240s / deep 600s）；
    #! 配了正数就以它为准（秒），三个档位统一。
    deadline_seconds: int = Field(0, validation_alias="APP_AGENT_DEADLINE_SECONDS")
    #! 用掉预算的这个比例就进入「收敛」：不再给工具 + 把理由交给模型 + 只再跑一轮。
    deadline_soft_ratio: float = Field(0.7, validation_alias="APP_AGENT_DEADLINE_SOFT_RATIO")
    #! 单次模型调用超时的下限（秒）。再赶也不给低于这个值，否则请求必定空手而归。
    deadline_min_call_seconds: int = Field(15, validation_alias="APP_AGENT_DEADLINE_MIN_CALL_SECONDS")
    #! 收敛轮的 max_tokens。0 = 不显式设置（**默认不动**：收紧会截短深度档的正长文）。
    deadline_converge_tokens: int = Field(0, validation_alias="APP_AGENT_DEADLINE_CONVERGE_TOKENS")

    #! ---- 同一轮里的多个工具是否并行执行 ----
    #! 工具是 **IO 密集**（查库 + 联网检索），不是 CPU 密集，并行不需要绕开 GIL。
    #! 收益直接等于"省掉的那几次串行等待"：一次联网检索十几秒时，串着跑三个就是半分钟。
    #! 这里是两条编排**共用**的路径（``_run_tool_calls``），所以 legacy 同样受益。
    parallel_tools: bool = Field(True, validation_alias="APP_AGENT_PARALLEL_TOOLS")
    #! 并行的最大工具数。**它不是性能参数，是保护下游的闸门**：一次放行 20 个并发检索
    #! 会把本机的连接池和被调方的限流器同时打穿。
    parallel_tools_max: int = Field(4, validation_alias="APP_AGENT_PARALLEL_TOOLS_MAX")


class ProactiveSettings(BaseSettings):
    """``app.proactive.*`` —— 事件驱动的主动风险研判。

    **成本红线**：后台自动任务默认不消耗模型额度（授权闸门在
    ``LlmClient.is_auto_consume_allowed``），这里只管"要不要跑、一天跑几次"。
    """

    model_config = _ENV

    enabled: bool = Field(True, validation_alias="APP_PROACTIVE_ENABLED")
    #: 每家企业每日研判上限。"事件驱动"最容易失控：指标一抖就触发，一天能烧掉几百次调用。
    daily_limit: int = Field(5, validation_alias="APP_PROACTIVE_DAILY_LIMIT")
    #: 触发后只在这段时间内出现的高危事件才算"本次触发产生"。
    fresh_minutes: int = Field(5, validation_alias="APP_PROACTIVE_FRESH_MINUTES")
    #: 以哪个账号的身份执行自动研判（需要它对企业有可见权限）。
    system_username: str = Field("admin", validation_alias="APP_PROACTIVE_SYSTEM_USERNAME")


class NotifySettings(BaseSettings):
    """``app.notify.*`` —— 通知通道（Agent 提议的「通知责任人」落地）。"""

    model_config = _ENV

    #: 为空则只登记不投递，且回执明确写"待人工投递"，**不假装已送达**。
    webhook_url: str = Field("", validation_alias="APP_NOTIFY_WEBHOOK_URL")
    timeout_seconds: int = Field(8, validation_alias="APP_NOTIFY_TIMEOUT_SECONDS")


class McpSettings(BaseSettings):
    """``app.web-search.mcp.*`` —— MCP 是联网检索的首选通道。"""

    model_config = _ENV

    #! 默认开；真正的开关是"有没有配 command / url"（没配则自动跳过去走直连）。
    enabled: bool = Field(True, validation_alias="APP_WEB_SEARCH_MCP_ENABLED")
    transport: str = Field("", validation_alias="APP_WEB_SEARCH_MCP_TRANSPORT")
    command: str = Field("", validation_alias="APP_WEB_SEARCH_MCP_COMMAND")
    url: str = Field("", validation_alias="APP_WEB_SEARCH_MCP_URL")
    headers: str = Field("", validation_alias="APP_WEB_SEARCH_MCP_HEADERS")
    tool_name: str = Field("", validation_alias="APP_WEB_SEARCH_MCP_TOOL_NAME")
    args_template: str = Field("", validation_alias="APP_WEB_SEARCH_MCP_ARGS_TEMPLATE")
    timeout_seconds: int = Field(60, validation_alias="APP_WEB_SEARCH_MCP_TIMEOUT")


class WebSearchSettings(BaseSettings):
    """``app.web-search.*``"""

    model_config = _ENV

    enabled: bool = Field(True, validation_alias="APP_WEB_SEARCH_ENABLED")
    #! auto = MCP → 直连 → API（默认）；另有 mcp / direct / api 三个单向取值。
    mode: str = Field("auto", validation_alias="APP_WEB_SEARCH_MODE")
    #! signal（默认）/ always / never —— 见 AgentRouter 的候补开放策略。
    always_offer: str = Field("signal", validation_alias="APP_WEB_SEARCH_ALWAYS_OFFER")
    direct_engines: str = Field("bing,so360", validation_alias="APP_WEB_SEARCH_ENGINES")
    direct_timeout_seconds: int = Field(10, validation_alias="APP_WEB_SEARCH_DIRECT_TIMEOUT")
    #! 搜索引擎的跳转链（如 so.com/link?m=...）无法核对，默认还原成真实地址。
    #  代价是每条来源多一次 HTTP 往返（并行、短超时）；关掉可省这一跳，来源会保持跳转链形态。
    resolve_redirects: bool = Field(True, validation_alias="APP_WEB_SEARCH_RESOLVE_REDIRECTS")
    base_url: str = Field("", validation_alias="APP_WEB_SEARCH_BASE_URL")
    model: str = Field("", validation_alias="APP_WEB_SEARCH_MODEL")
    api_key: str = Field("", validation_alias="APP_WEB_SEARCH_API_KEY")
    max_keyword: int = Field(3, validation_alias="APP_WEB_SEARCH_MAX_KEYWORD")
    sources: str = Field("", validation_alias="APP_WEB_SEARCH_SOURCES")
    max_results: int = Field(6, validation_alias="APP_WEB_SEARCH_MAX_RESULTS")
    timeout_seconds: int = Field(60, validation_alias="APP_WEB_SEARCH_TIMEOUT")
    mcp: McpSettings = Field(default_factory=McpSettings)

    @property
    def engine_list(self) -> List[str]:
        return [x.strip() for x in (self.direct_engines or "").split(",") if x.strip()]


class WebGateSettings(BaseSettings):
    """``app.web.*`` —— 联网检索的三道闸门（成本与质量）。"""

    model_config = _ENV

    max_calls: int = Field(4, validation_alias="APP_WEB_MAX_CALLS")
    max_sources: int = Field(6, validation_alias="APP_WEB_MAX_SOURCES")
    min_relevance: float = Field(0.16, validation_alias="APP_WEB_MIN_RELEVANCE")
    min_cosine: float = Field(0.18, validation_alias="APP_WEB_MIN_COSINE")


class EvalSettings(BaseSettings):
    """``app.eval.*``"""

    model_config = _ENV

    report_dir: str = Field(".ai-eval-reports", validation_alias="APP_EVAL_REPORT_DIR")
    max_history: int = Field(60, validation_alias="APP_EVAL_MAX_HISTORY")
    gate_score: float = Field(70, validation_alias="APP_EVAL_GATE_SCORE")
    #! 用例文件默认复用 Java 版的同一份，保证两边跑的是同一套题。
    cases_file: str = Field(
        "../backend/src/main/resources/ai-eval-cases.json", validation_alias="APP_EVAL_CASES_FILE"
    )


class Settings(BaseSettings):
    """总配置。嵌套模型各自从环境变量独立装配。"""

    model_config = _ENV

    #! 本服务的监听端口：故意与 Java 的 8080 错开，便于并存与双跑对照。
    port: int = Field(8081, validation_alias="PY_AGENT_PORT")
    host: str = Field("127.0.0.1", validation_alias="PY_AGENT_HOST")
    log_level: str = Field("INFO", validation_alias="PY_AGENT_LOG_LEVEL")

    db: DbSettings = Field(default_factory=DbSettings)
    ai: AiSettings = Field(default_factory=AiSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    web: WebGateSettings = Field(default_factory=WebGateSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    proactive: ProactiveSettings = Field(default_factory=ProactiveSettings)
    notify: NotifySettings = Field(default_factory=NotifySettings)
    #: 转发层没带 ``X-User-Id`` 时的兜底。``ai_analysis.user_id`` 是 NOT NULL，
    #: 取不到用户就写不进去 —— 宁可落到配置值，也不能让整条分析记录丢失。
    default_user_id: int = Field(1, validation_alias="PY_AGENT_DEFAULT_USER_ID")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试里需要改配置时用 ``get_settings.cache_clear()``。"""
    return Settings()
