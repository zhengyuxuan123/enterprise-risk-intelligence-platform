"""``/api/ai/*`` 的业务端点（对应 Java ``AiController`` 的主干部分）。

分成两个文件只是为了让「每块落地时改哪个文件」更清楚：
``ai.py`` 管模型目录/体检/迁移自查，本文件管分析、历史、评估、运维。

三条跨端点的约定（都是从事故里长出来的，改动前请读）：

1. **SSE 的 ``done`` 事件一律取持久化实体字段**。这样「流式刚跑完」与「之后从历史里翻出来」
   看到的正文、证据、工具轨迹**逐字节一致**，前端不需要两套拼装逻辑。
   ⚠️ 命中结果缓存时不会推任何 token，所以 ``done.answer`` 必须存在，否则界面正文是空的。
2. **流式必须发心跳**。链路上任何一个"空闲即断"的中间盒子都会在长分析期间掐断连接，
   而前端只会看到一次"没有结果" —— 哪怕分析其实在后台正常跑完并落了库。
3. **耗时分钟级的接口必须走异步 + 轮询**。评估含真实联网检索，远超浏览器 30s 超时。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select, update

from ..agent.evaluation import MODE_FAST, EvaluationService
from ..agent.trace import TraceContext
from ..config import get_settings
from ..core.errors import BusinessError
from ..db import tables as T
from ..db.engine import get_db
from .schemas import ApiResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai-agent"])


# ------------------------------------------------------------------
# 请求体
# ------------------------------------------------------------------


class AiQueryRequest(BaseModel):
    """对应 Java ``AiQueryRequest``。"""

    companyId: Optional[int] = None
    question: str = ""
    topK: Optional[int] = Field(default=None, alias="topK")
    sessionId: Optional[int] = Field(default=None, alias="sessionId")
    useMultiAgent: Optional[bool] = Field(default=None, alias="useMultiAgent")
    depth: Optional[str] = None
    #: 前端「必须联网」开关。与提问里的"请联网核实"等措辞互为补充：
    #: 开关不依赖用户怎么写，词表不依赖前端是否传参。
    forceWeb: Optional[bool] = Field(default=None, alias="forceWeb")

    model_config = {"populate_by_name": True, "extra": "ignore"}


# ------------------------------------------------------------------
# 服务装配（懒加载 + 进程内单例）
# ------------------------------------------------------------------

_lock = threading.RLock()
_services: Dict[str, Any] = {}


def _svc(name: str, factory: Callable[[], Any]) -> Any:
    """进程内单例。

    索引、语料源、评估历史这些都是**有状态的**：每次请求新建会让
    「写入即索引」的记账表和模型的负缓存全部失效。
    """
    with _lock:
        if name not in _services:
            _services[name] = factory()
        return _services[name]


def _agent_service():
    def build():
        from ..agent.agent_service import AgentService
        from ..agent.evaluation import EvaluationService  # noqa: F401 - 触发装配
        from ..ai.llm_client import LlmClient
        from ..ai.tools import RiskAgentTools
        from ..rag.rag_service import RagService

        try:
            llm = LlmClient()
        except Exception as e:  # noqa: BLE001
            log.warning("[Api] 大模型底座装配失败，走本地规则: %s", e)
            llm = None
        # 注意：**不要**在这里把「没配 Key」的客户端丢掉（`if not llm.is_configured(): llm = None`）。
        # 保留它有两个必须的理由：
        # ① 使用者第一次进页面时本来就没有 Key，要靠「模型服务」卡片粘贴一把热更新 ——
        #    若这里丢了实例，那个 Post 会新建一个一次性客户端，Key 只装进它、随即被丢弃，
        #    真正的分析链路永远拿不到，界面上却显示"应用成功"。
        # ② 只有保留实例，「开关开着但没配 Key」和「总开关关掉」才是可区分的两件事
        #    （Java 侧 `llm == null` 只表示后者）。
        # 所有调用点（agent_service / review_loop / proactive / web）都自带
        # `is_configured()` 判断，传一个未配置的客户端不会被当成"可用"。
        rag = _rag_service()
        from ..agent.approvals import ActionApprovalService
        #! ``web_search`` 必须在这里传给 **工具**，不能只传给 ``AgentService``。
        #! 两者是两条独立路径：``AgentService.web`` 供服务层自用，而模型真正调用的
        #! ``web_search`` 工具读的是 ``RiskAgentTools.web_search``。
        #! 漏传时该字段为 None，工具在 ``if self.web_search is None`` 处**直接返回
        #! 「联网通道未装配」**——请求根本没发出去，界面上表现为"联网开了却没结果"，
        #! 而且换模型、开插件都不会有任何变化（曾据此误判为用户额度问题）。
        tools = RiskAgentTools(rag=rag, web_search=_web_search(),
                               approvals=ActionApprovalService().submit)
        return AgentService(llm=llm, tools=tools, rag=rag, web=_web_search())

    return _svc("agent", build)


def _rag_service():
    def build():
        from ..rag.embedding import LocalEmbedding
        from ..rag.embedding_client import EmbeddingClient
        from ..rag.indexing import RagIndexingService
        from ..rag.index_store import RagIndexStore
        from ..rag.rag_service import RagService
        from ..rag.sources import build_sources
        from ..rag.structured import StructuredCorpus

        s = get_settings()
        local = LocalEmbedding()
        embedder = EmbeddingClient(local=local)
        index = RagIndexStore(index_dir=s.rag.index_dir)
        index.init()
        # 写入即索引与检索共用同一份索引实例，分开装配会让记账表和索引对不上
        _services["indexing"] = RagIndexingService(
            index=index, sources=build_sources(), local=local, embedder=embedder)
        # 向量化客户端也登记一份：诊断接口要回显「语义检索到底走哪条通道」，
        # 另起一个实例会让它的 TTL 缓存与熔断状态跟实际检索用的那个不是同一个。
        _services["embedder"] = embedder
        return RagService(index=index, embedder=embedder, corpus=StructuredCorpus())

    return _svc("rag", build)


def _embed_client():
    """向量化客户端单例（与检索共用，诊断回显用）。"""
    _rag_service()
    return _services.get("embedder")


def _indexing_service():
    """写入即索引（与 ``_rag_service`` 共用同一个索引实例）。"""
    _rag_service()
    return _svc("indexing", lambda: None)


_heal_started = False
_heal_lock = threading.Lock()


def heal_rag_vectors_if_needed() -> Optional[str]:
    """索引"记得自己做过向量化、实际一条向量都没有" → 后台补算一次。

    只在**本地向量**（零成本、无需 Key、不联网）时自动执行；第三方 embedding
    会真花钱，那种情况只告警、把决定权留给使用者。

    为什么需要自愈而不只是报警：这个状态的成因是"写入路径以为算了向量、
    实际没算"，而增量判定用的是内容 hash —— 内容没变就永远跳过，
    于是**不会自己好**，无论重启多少次。补算的判据见
    ``RagIndexStore._vector_key_equals``：它把"键相同但没有向量"也判为需要重算。

    返回一句人话（做了/为什么没做），供启动日志与自检卡片使用。
    """
    global _heal_started
    try:
        idx = _indexing_service()
    except Exception as e:  # noqa: BLE001
        return f"跳过自愈：索引服务装配失败（{e}）"
    if idx is None or getattr(idx, "index", None) is None:
        return "跳过自愈：写入即索引未装配"
    index = idx.index
    try:
        if not index.vectors_missing():
            return None
    except Exception as e:  # noqa: BLE001
        return f"跳过自愈：读不到向量覆盖情况（{e}）"

    em = _embed_client()
    mode = getattr(em, "mode", None)
    if em is None or not getattr(em, "is_enabled", lambda: False)():
        return (f"索引向量覆盖不全（{index.stats().get('vectorRows')}"
                f"/{index.stats().get('docs')}），但 embedding 总开关是关的；"
                "语义通道需要显式开启后重建")
    if mode != "local":
        return (f"索引向量覆盖不全，但当前向量通道是 {mode}（会产生费用）；"
                "请确认后手工触发 POST /api/ai/rag/reindex")

    with _heal_lock:
        if _heal_started:
            return None
        _heal_started = True

    def work() -> None:
        try:
            t = time.time()
            n = idx.reindex(None)
            log.warning("[RagIndex] 检测到索引缺少向量，已自动补算 %s 条切片（耗时 %ss）；"
                        "现在为 %s", n, int(time.time() - t), index.stats())
        except Exception as e:  # noqa: BLE001
            log.warning("[RagIndex] 向量自愈失败（不影响关键词检索）：%s", e)

    threading.Thread(target=work, name="rag-vector-heal", daemon=True).start()
    return "索引缺少向量，已在后台按本地向量补算（零成本、不联网）"


def _web_search():
    def build():
        from ..ai.llm_client import LlmClient
        from ..web.direct_client import DirectWebSearchClient
        from ..web.mcp_client import McpWebSearchClient
        from ..web.search_service import WebSearchService

        #! 主底座客户端必须传进去。``WebSearchService`` 的方舟通道靠它推导端点与模型
        #! （``app.web-search.base-url`` / ``.model`` 通常没人单独配），少了它
        #! ``_api_configured()`` 恒为 False，**方舟通道会被整条跳过**：
        #! 现象就是"Key 明明配好了，联网却永远只在抓网页"。
        #! 底座本身没配 Key 也没关系：不带 Key 的客户端一样能给出 base-url，
        #! 而真正的可用性判断在 ``_api_configured()`` 里，不会被当成"已配置"。
        try:
            llm = LlmClient()
        except Exception as e:  # noqa: BLE001
            log.warning("[Api] 联网检索拿不到主底座客户端（方舟通道将不可用）: %s", e)
            llm = None
        return WebSearchService(direct=DirectWebSearchClient(), mcp=McpWebSearchClient(),
                                llm=llm)

    return _svc("web", build)


def _evaluation():
    def build():
        from ..agent.eval_history import EvalHistoryStore
        from ..agent.grounding import AnswerGroundingService
        from ..agent.guardrail import GuardrailService
        from ..agent.health import OperationalHealthService
        from ..agent.registry import AgentRegistry
        from ..agent.router import AgentRouter
        from ..ai.tools import RiskAgentTools

        rag = _rag_service()
        agent = _agent_service()
        return EvaluationService(
            # 同上：评估里的 web 类用例也要真跑一次工具，漏传会让它们全部落进
            # 「联网通道未装配」而判成能力失败（而不是环境跳过）。
            rag=rag, tools=RiskAgentTools(rag=rag, web_search=_web_search()),
            web_search=_web_search(), guardrail=GuardrailService(),
            grounding=AnswerGroundingService(), history=EvalHistoryStore(),
            agent_router=AgentRouter(AgentRegistry()),
            health=OperationalHealthService(),
            agent_service=agent,
            # ``llm`` 必须传：报告里的 ``model`` 字段靠它记账（``describe_model()``）。
            # 不传时恒为「本地规则（未接入外部模型）」—— 后果不只是历史表里那一列显示错，
            # 更严重的是基线对比会认为「两次跑的是同一把尺子」，而实际可能换了底座，
            # 通过率就这样被拿去横向比较了。
            llm=getattr(agent, "llm", None),
        )

    return _svc("eval", build)


# ------------------------------------------------------------------
# 异步任务（分析 / 评估）：进程内进度表
# ------------------------------------------------------------------


class _Tasks:
    """``taskId -> {status, total, done, percent, ...}``。

    刻意不做持久化：任务进度是秒级信息，重启后继续报"RUNNING"反而误导。
    """

    def __init__(self) -> None:
        self._m: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create(self, total: int = 0) -> str:
        tid = uuid.uuid4().hex[:16]
        with self._lock:
            self._m[tid] = {"status": "RUNNING", "total": total, "done": 0, "percent": 0}
        return tid

    def progress(self, tid: str, done: int, total: int) -> None:
        with self._lock:
            st = self._m.get(tid)
            if st is None:
                return
            st["done"] = done
            if total:
                st["total"] = total
            st["percent"] = int(done * 100 / total) if total else 0

    def complete(self, tid: str, payload: Any = None, key: str = "result") -> None:
        with self._lock:
            st = self._m.setdefault(tid, {})
            st.update({"status": "DONE", "percent": 100, key: payload})

    def fail(self, tid: str, message: str) -> None:
        with self._lock:
            st = self._m.setdefault(tid, {})
            st.update({"status": "FAILED", "error": message})

    def get(self, tid: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._m.get(tid) or {"status": "NOT_FOUND"})


_tasks = _Tasks()
_eval_tasks = _Tasks()


# ------------------------------------------------------------------
# 分析
# ------------------------------------------------------------------


def _top_k(r: AiQueryRequest) -> int:
    v = r.topK or 0
    return 5 if v <= 0 else min(v, 20)


@router.post("/analyze")
def analyze(r: AiQueryRequest) -> ApiResponse[Dict[str, Any]]:
    """同步分析。"""
    svc = _agent_service()
    # depth 一路从请求带到 AgentBudget：前端「快答/标准/深度」选的是它，
    # 漏传的后果不是报错，而是**静默跑成默认档**（选快答也等标准档那么久）。
    out = svc.analyze(r.companyId, r.question, _top_k(r), r.sessionId,
                      bool(r.useMultiAgent) if r.useMultiAgent is not None else True,
                      depth=r.depth, force_web=bool(r.forceWeb))
    return ApiResponse.ok(_analysis_payload(out))


@router.post("/analyze/stream")
def analyze_stream(r: AiQueryRequest) -> StreamingResponse:
    """SSE 流式分析：``token`` / ``meta`` / ``done`` / ``error`` + 心跳注释行。"""
    svc = _agent_service()
    s = get_settings()
    heartbeat_ms = max(1000, int(getattr(s.agent, "sse_heartbeat_ms", 15000) or 15000))

    def gen():
        q: "list" = []
        stop = threading.Event()

        def on_token(t: str) -> None:
            q.append(("token", t))

        def on_stage(st: Dict[str, Any]) -> None:
            q.append(("stage", st))

        def on_meta(m: Dict[str, Any]) -> None:
            # 工具取数一结束就推「风险等级 / 来源构成」，先于正文 token
            q.append(("meta", m))

        holder: Dict[str, Any] = {}

        def worker() -> None:
            try:
                holder["out"] = svc.analyze_stream(
                    r.companyId, r.question, _top_k(r), r.sessionId,
                    bool(r.useMultiAgent) if r.useMultiAgent is not None else True,
                    token_sink=on_token, stage_sink=on_stage, meta_sink=on_meta,
                    depth=r.depth, force_web=bool(r.forceWeb))
            except Exception as e:  # noqa: BLE001
                holder["err"] = e
            finally:
                stop.set()

        th = threading.Thread(target=worker, name="sse-analyze", daemon=True)
        th.start()

        def frame(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {_json(data)}\n\n"

        try:
            while not stop.is_set():
                if q:
                    event, data = q.pop(0)
                    yield frame(event, data)
                    continue
                time.sleep(0.05)
                # 心跳：注释行，浏览器原生 EventSource 会忽略，
                # 唯一作用是让链路上"空闲即断"的盒子知道这条连接还活着
                if heartbeat_ms and int(time.time() * 1000) % heartbeat_ms < 60:
                    yield f": ping {int(time.time() * 1000)}\n\n"
            while q:
                event, data = q.pop(0)
                yield frame(event, data)

            err = holder.get("err")
            if err is not None:
                yield frame("error", _friendly(err))
                return
            out = holder.get("out")
            if out is None:
                yield frame("error", "分析未产出结果")
                return
            # 一律取持久化后的字段：前端「刚跑完」与「从历史翻出来」看到的一致
            yield frame("done", _done_payload(out))
        finally:
            stop.set()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _friendly(e: Exception) -> str:
    """把底层异常翻译成人话（Key 无效 / 欠费 / 限流 / 超时）。"""
    m = str(e) or ""
    import re as _re

    mt = _re.search(r"HTTP (\d{3})", m)
    if mt:
        st = int(mt.group(1))
        if st in (400, 401, 403, 429):
            from ..ai.fallback import describe_http_error

            return describe_http_error(st, m)
    if "timeout" in m.lower():
        return "模型服务响应超时，请稍后重试。"
    return m or "分析失败"


def _json(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return "{}"


def _done_payload(out) -> Dict[str, Any]:
    """SSE ``done`` 事件：与 Java 版字段一一对应。"""
    return {
        "analysisId": out.analysis_id,
        "provider": out.model,
        "grounded": bool(out.diagnostics.get("grounded", True)),
        "confidence": out.confidence,
        "traceId": out.trace_id,
        "degradeLevel": out.degrade_level,
        "degradeReason": out.degrade_reason,
        "durationMs": out.duration_ms,
        # 命中结果缓存时不推任何 token，所以 answer 必须存在，否则界面正文是空的
        "answer": out.content or "",
        "answerJson": _json(out.answer_json or {}),
        "evidence": out.evidence or [],
        "trace": out.tool_trace or [],
        "evidenceJson": _json(out.evidence or []),
        "toolTraceJson": _json(out.tool_trace or []),
    }


def _analysis_payload(out) -> Dict[str, Any]:
    d = _done_payload(out)
    d["question"] = None
    return d


@router.post("/analyze/async")
def analyze_async(r: AiQueryRequest) -> ApiResponse[Dict[str, Any]]:
    """异步提交：立即返回 taskId，分析在后台线程进行。"""
    svc = _agent_service()
    tid = _tasks.create()

    def work() -> None:
        try:
            out = svc.analyze(r.companyId, r.question, _top_k(r), r.sessionId,
                              bool(r.useMultiAgent) if r.useMultiAgent is not None else True,
                              depth=r.depth, force_web=bool(r.forceWeb))
            _tasks.complete(tid, out.analysis_id, "analysisId")
        except Exception as e:  # noqa: BLE001
            _tasks.fail(tid, str(e))

    threading.Thread(target=work, name="analyze-async", daemon=True).start()
    return ApiResponse.ok({"taskId": tid, "status": "RUNNING"})


@router.get("/async/{taskId}")
def async_status(taskId: str) -> ApiResponse[Dict[str, Any]]:
    """异步轮询：完成时附带分析结果。"""
    st = _tasks.get(taskId)
    if st.get("status") == "DONE" and st.get("analysisId"):
        st = dict(st)
        #! 同样要过 DTO：前端拿到后直接当 detail 用（record.value = st.analysis），
        #! 给 snake_case 会让异步过来的结果也只显示纯文本。
        #? 保留 None 语义：{} 是"真值"，会让前端的 if(st.analysis) 拿到一条空记录。
        row = _load_analysis(st["analysisId"])
        st["analysis"] = _analysis_dto(row) if row else None
    return ApiResponse.ok(st)


def _load_analysis(aid: Any) -> Optional[Dict[str, Any]]:
    """取原始行（**snake_case**，给内部使用：血缘/导出直接读 answer/evidence_json）。"""
    try:
        row = get_db().fetch_one(select(T.ai_analysis).where(T.ai_analysis.c.id == int(aid)))
        return dict(row) if row else None
    except Exception:  # noqa: BLE001
        return None


#: ai_analysis 的列名 → API 契约的 camelCase。
#! 前端统一按 camelCase 读（``answerJson`` / ``companyId`` / ``durationMs``…），
#! 而这里原来是直接把数据库行 dict 出去 —— 于是结构化区块会**静默消失**：
#! 点「查看」只看到纯文本正文（来源分层、引用核对、证据表、原始 JSON 全空），
#! 历史表还有三列（时间 / 企业ID / 结果摘要）整列为空。**而页面看上去"有正文、很正常"**，
#! 属于本项目记录过的「静默失败」家族（键名不符 → 整块变空）。
#! Java 侧当年返回的是实体 JSON（camelCase），迁到 Python 时丢掉了这层映射。
_ANALYSIS_CAMEL = {
    "company_id": "companyId", "user_id": "userId",
    "answer_json": "answerJson", "evidence_json": "evidenceJson",
    "tool_trace_json": "toolTraceJson", "conversation_id": "conversationId",
    "trace_id": "traceId", "degrade_level": "degradeLevel",
    "degrade_reason": "degradeReason", "duration_ms": "durationMs",
    "llm_calls": "llmCalls", "tool_calls": "toolCalls",
    "created_at": "createdAt", "review_revised": "reviewRevised",
}


def _fmt_dt(v: Any) -> Any:
    """datetime → ``YYYY-MM-DD HH:MM:SS``（前端直接显示，带 T 的 ISO 串在表格里很刺眼）。"""
    if v is None:
        return None
    s = str(v)
    return s.replace("T", " ")[:19] if len(s) >= 19 else s


def _analysis_dto(row: Any, answer_chars: int = 0) -> Dict[str, Any]:
    """数据库行 → 前端契约的 camelCase。

    ``answer_chars > 0`` 时截断正文：列表页只要一行摘要，
    50 条完整正文（每条数千字）纯属白传，而「结果摘要」列本来也只会显示一行。
    """
    if not row:
        return {}
    out: Dict[str, Any] = {}
    for k, v in dict(row).items():
        key = _ANALYSIS_CAMEL.get(k, k)
        if k == "created_at":
            out[key] = _fmt_dt(v)
        elif k == "answer" and answer_chars:
            text = v or ""
            out[key] = text if len(text) <= answer_chars else text[:answer_chars] + "…"
        else:
            out[key] = v
    return out


# ------------------------------------------------------------------
# 历史
# ------------------------------------------------------------------


@router.get("/history")
def history(companyId: Optional[int] = Query(default=None),
            limit: int = Query(default=50)) -> ApiResponse[List[Dict[str, Any]]]:
    """历史分析列表。

    正文只取前若干字做「结果摘要」：列表页渲染一行，前端也不需要第二条完整正文。
    """
    stmt = select(T.ai_analysis.c.id, T.ai_analysis.c.company_id, T.ai_analysis.c.question,
                  T.ai_analysis.c.confidence, T.ai_analysis.c.degrade_level,
                  T.ai_analysis.c.duration_ms, T.ai_analysis.c.model, T.ai_analysis.c.provider,
                  T.ai_analysis.c.trace_id, T.ai_analysis.c.created_at,
                  func.left(T.ai_analysis.c.answer, 160).label("answer"))
    if companyId is not None:
        stmt = stmt.where(T.ai_analysis.c.company_id == companyId)
    rows = get_db().fetch_all(stmt.order_by(desc(T.ai_analysis.c.id)).limit(max(1, limit)))
    return ApiResponse.ok([_analysis_dto(r) for r in rows])


@router.get("/history/{aid}")
def history_detail(aid: int) -> ApiResponse[Optional[Dict[str, Any]]]:
    """单条分析详情（含原始答案 JSON / 原始证据 JSON）。

    **必须过 ``_analysis_dto``**：前端按 camelCase 读 ``answerJson``，
    直接返回数据库行会让「来源分层 / 引用核对 / 证据表 / 原始 JSON」四块静默消失。
    """
    row = _load_analysis(aid)
    if row is None:
        raise BusinessError(f"分析记录不存在：{aid}")
    return ApiResponse.ok(_analysis_dto(row))


@router.delete("/history/{aid}")
def delete_history_item(aid: int) -> ApiResponse[int]:
    db = get_db()
    db.execute(T.ai_analysis.delete().where(T.ai_analysis.c.id == aid))
    return ApiResponse.ok(aid)


@router.delete("/history")
def clear_history(companyId: Optional[int] = Query(default=None)) -> ApiResponse[int]:
    """批量清空；传 companyId 只清该企业。"""
    db = get_db()
    stmt = T.ai_analysis.delete()
    if companyId is not None:
        stmt = stmt.where(T.ai_analysis.c.company_id == companyId)
    cur = db.execute(stmt)
    return ApiResponse.ok(int(getattr(cur, "rowcount", 0) or 0))


@router.get("/conversations")
def conversations(companyId: Optional[int] = Query(default=None)) -> ApiResponse[List[Dict[str, Any]]]:
    stmt = select(T.ai_conversation)
    if companyId is not None:
        stmt = stmt.where(T.ai_conversation.c.company_id == companyId)
    rows = get_db().fetch_all(stmt.order_by(desc(T.ai_conversation.c.id)).limit(50))
    return ApiResponse.ok([dict(r) for r in rows])


@router.delete("/conversations/{cid}")
def delete_conversation(cid: int) -> ApiResponse[int]:
    db = get_db()
    db.execute(T.ai_message.delete().where(T.ai_message.c.conversation_id == cid))
    db.execute(T.ai_conversation.delete().where(T.ai_conversation.c.id == cid))
    return ApiResponse.ok(cid)


# ------------------------------------------------------------------
# 评估
# ------------------------------------------------------------------


@router.get("/evaluate")
def evaluate(companyId: Optional[int] = Query(default=None),
             mode: str = Query(default=MODE_FAST)) -> ApiResponse[Dict[str, Any]]:
    """同步评估（供脚本用；耗时可达分钟级，前端请走 ``/evaluate/async``）。"""
    rep = _evaluation().evaluate(companyId or 1, mode=mode)
    return ApiResponse.ok(rep.to_dict())


@router.post("/evaluate/async")
def evaluate_async(companyId: Optional[int] = Query(default=None),
                   mode: str = Query(default=MODE_FAST)) -> ApiResponse[Dict[str, Any]]:
    """推荐入口：立即返回任务 ID，后台并行跑完，前端轮询进度。"""
    ev = _evaluation()
    total = ev.case_count(mode)
    tid = _eval_tasks.create(total)

    def work() -> None:
        try:
            rep = ev.evaluate(companyId or 1,
                              lambda done, tot: _eval_tasks.progress(tid, done, tot), mode)
            _eval_tasks.complete(tid, rep.to_dict(), "report")
        except Exception as e:  # noqa: BLE001
            _eval_tasks.fail(tid, str(e))

    threading.Thread(target=work, name="evaluate-async", daemon=True).start()
    return ApiResponse.ok({"taskId": tid, "status": "RUNNING", "total": total})


@router.get("/evaluate/async/{taskId}")
def evaluate_async_status(taskId: str) -> ApiResponse[Dict[str, Any]]:
    return ApiResponse.ok(_eval_tasks.get(taskId))


@router.get("/evaluate/reports")
def evaluate_reports() -> ApiResponse[List[Dict[str, Any]]]:
    from ..agent.eval_history import EvalHistoryStore

    return ApiResponse.ok(EvalHistoryStore().history())


@router.get("/evaluate/reports/{rid}")
def evaluate_report(rid: Optional[str] = None) -> ApiResponse[Dict[str, Any]]:
    """指定报告的完整内容；无 id 则返回最近一份。"""
    from ..agent.eval_history import EvalHistoryStore

    st = EvalHistoryStore()
    m = st.latest_report() if not rid else st.report(rid)
    return ApiResponse.ok(m or {})


@router.delete("/evaluate/reports/{rid}")
def delete_evaluate_report(rid: str) -> ApiResponse[bool]:
    from ..agent.eval_history import EvalHistoryStore

    return ApiResponse.ok(EvalHistoryStore().delete(rid))


@router.get("/evaluate/health")
def evaluate_health(hours: int = Query(default=24)) -> ApiResponse[Dict[str, Any]]:
    """运行时健康聚合（评估 cost 维度的数据来源，零 token）。"""
    from ..agent.health import OperationalHealthService

    return ApiResponse.ok(OperationalHealthService().health(hours))


# ------------------------------------------------------------------
# 检索运维
# ------------------------------------------------------------------


@router.get("/rag/status")
def rag_status() -> ApiResponse[Dict[str, Any]]:
    """写入即索引的运行状态（``RagIndexingService.status()``）。

    注意它**不是**索引库自身的统计：Java 的 ``/rag/status`` 返回的是
    ``ragIndexing.status()``（enabled / pending / byType / index 之类），
    索引库统计只是其中的 ``index`` 子对象。早先这里直接返回 ``index.stats()``，
    键集合与 Java 完全不重叠，前端那一块会静默变空。
    """
    idx_service = _indexing_service()
    if idx_service is None:
        return ApiResponse.ok({"enabled": False, "issue": "写入即索引服务未装配"})
    return ApiResponse.ok(idx_service.status())


@router.post("/rag/reindex")
def rag_reindex(companyId: Optional[int] = Query(default=None)) -> ApiResponse[Dict[str, Any]]:
    """全量回填。``companyId`` 为空 = 所有企业 + 全局语料（与 Java 同语义）。"""
    idx_service = _indexing_service()
    if idx_service is None:
        return ApiResponse.fail(500, "写入即索引服务未装配")
    t = time.time()
    n = idx_service.reindex(companyId)
    return ApiResponse.ok({"chunks": n, "companyId": companyId,
                           "costMs": int((time.time() - t) * 1000)})


class RagEventRequest(BaseModel):
    sourceType: str
    sourceId: Optional[int] = None
    companyId: Optional[int] = None
    kind: str = "UPSERT"
    documentVersion: Optional[int] = None


@router.post("/rag/event")
def rag_event(r: RagEventRequest) -> ApiResponse[Dict[str, Any]]:
    """可靠单记录入库入口；成功返回前，切片和状态表都已写完。"""
    from ..rag.indexing import CorpusChangeEvent, Kind
    idx_service = _indexing_service()
    if idx_service is None:
        return ApiResponse.fail(500, "写入即索引服务未装配")
    try:
        kind = Kind((r.kind or "UPSERT").upper())
        result = idx_service.apply_now(CorpusChangeEvent(r.sourceType, r.sourceId, r.companyId, kind))
        if kind == Kind.UPSERT and result.get("chunks", 0) <= 0:
            raise RuntimeError("业务文档未生成任何可检索切片")
        if r.sourceType == "knowledge" and r.documentVersion is not None and kind == Kind.UPSERT:
            get_db().execute(update(T.rag_chunk_state)
                             .where(T.rag_chunk_state.c.source_type == "knowledge")
                             .where(T.rag_chunk_state.c.source_id == str(r.sourceId))
                             .values(document_version=r.documentVersion, indexed_at=datetime.now()))
        result["documentVersion"] = r.documentVersion
        return ApiResponse.ok(result)
    except Exception as e:  # noqa: BLE001
        return ApiResponse.fail(500, f"索引事件处理失败: {e}")


@router.post("/rag/flush")
def rag_flush() -> ApiResponse[Dict[str, Any]]:
    idx_service = _indexing_service()
    if idx_service is None:
        return ApiResponse.fail(500, "写入即索引服务未装配")
    t = time.time()
    processed = idx_service.flush()
    return ApiResponse.ok({"processed": processed,
                           "costMs": int((time.time() - t) * 1000),
                           "status": idx_service.status()})


@router.post("/rag/reconcile")
def rag_reconcile() -> ApiResponse[Dict[str, Any]]:
    return ApiResponse.ok(_indexing_service().reconcile())


@router.get("/rag/probe")
def rag_probe(q: str = Query(default=""), companyId: int = Query(default=1),
              topK: int = Query(default=5)) -> ApiResponse[Dict[str, Any]]:
    """零 token 验收神器：直接看这次检索走了哪几条路径、召回了什么。"""
    res = _rag_service().search_detailed(companyId, q, topK)
    return ApiResponse.ok({
        "diagnostics": res.diagnostics,
        "hits": [{"ref": h.ref(), "title": h.title, "score": h.score,
                  "rawScore": getattr(h, "raw_score", None)} for h in res.hits],
    })


# ------------------------------------------------------------------
# 长期记忆（跨会话）—— 全部零 token，不消耗任何模型额度
# ------------------------------------------------------------------


def _long_term():
    from ..agent.long_term_memory import LongTermMemory
    return LongTermMemory()


@router.get("/memory")
def memory_list(companyId: Optional[int] = Query(default=None),
                userId: Optional[int] = Query(default=None),
                includeSuperseded: bool = Query(default=False),
                limit: int = Query(default=200)) -> ApiResponse[Dict[str, Any]]:
    """长期记忆清单 + 统计。

    ``includeSuperseded=true`` 时把"被取代的历史口径"也列出来 ——
    口径漂移（上次 8.3%、这次 12%）本身就是要被看见的信息，默认折叠不是丢弃。
    """
    svc = _long_term()
    #! 注意参数名：HTTP 层是 camelCase（companyId），服务层是 snake_case（company_id）。
    #! 直接把 HTTP 参数名透传给服务会 500（TypeError: unexpected keyword argument），
    #! 而报错信息只指到这一行、不指到"名字写法不一致"。
    return ApiResponse.ok({
        "stats": svc.stats(company_id=companyId),
        "items": svc.list_rows(company_id=companyId, user_id=userId,
                               include_superseded=includeSuperseded, limit=limit),
    })


@router.get("/web/probe")
def web_probe(q: str = Query(default=""), live: bool = Query(default=False),
              topN: int = Query(default=3)) -> ApiResponse[Dict[str, Any]]:
    """联网检索自检。**默认零成本**（``live=0``）：只汇报通道配置与上一次失败原因。

    ``live=1`` 才会真的发一次检索——这次调用会消耗额度，所以必须显式要求：
    体检页常驻刷新时不该悄悄烧钱，而在「联网为什么没结果」时又必须能看到真实原因。
    """
    svc = _web_search()
    return ApiResponse.ok(svc.probe(q, live=live, top_n=topN))


@router.get("/memory/probe")
def memory_probe(q: str = Query(default=""), companyId: int = Query(default=1),
                 userId: int = Query(default=0), topK: int = Query(default=0),
                 minScore: float = Query(default=-1)) -> ApiResponse[Dict[str, Any]]:
    """召回预演：这次会注入哪几条记忆、各自几分、拼出来的上下文长什么样。

    与 ``/rag/probe`` 对称：只看不写，**零 token**，用来回答"为什么这次没想起那件事"。
    """
    svc = _long_term()
    hits = svc.recall(companyId, userId, q, top_k=(topK or None),
                      min_score=(None if minScore < 0 else minScore))
    ctx = svc.build_context(hits)
    return ApiResponse.ok({
        "hits": [h.to_dict() for h in hits],
        "context": ctx,
        "contextChars": len(ctx),
        "diagnostics": svc.diagnostics(hits, ctx),
    })


@router.delete("/memory/{mid}")
def memory_delete(mid: int) -> ApiResponse[int]:
    """删除一条记忆。

    这里是**物理删除**而不是置为失效：用户说"忘掉这条"时必须真的忘掉
    （合规诉求），置为 SUPERSEDED 会把内容继续留在库里。
    """
    return ApiResponse.ok(_long_term().delete(mid))


@router.post("/memory/clear")
def memory_clear(companyId: Optional[int] = Query(default=None),
                 userId: Optional[int] = Query(default=None)) -> ApiResponse[int]:
    """清空记忆。两个条件都不给等于"清空全库"，这个动作必须被拒绝。"""
    if companyId is None and userId is None:
        raise BusinessError("清空长期记忆必须指定 companyId 或 userId")
    return ApiResponse.ok(_long_term().clear(companyId, userId))


@router.post("/memory/rebuild")
def memory_rebuild(companyId: int = Query(default=1),
                   limit: int = Query(default=50)) -> ApiResponse[Dict[str, Any]]:
    """从历史分析回填长期记忆（零 token）。

    功能上线时企业往往已经积累了几百条历史分析，不回填就只能从今天开始记，
    而长期记忆最有价值的恰恰是"以前发生过什么"。回填与实时写入用的是同一套抽取规则。
    """
    return ApiResponse.ok(_long_term().rebuild_from_history(companyId, limit))


# ------------------------------------------------------------------
# 追溯
# ------------------------------------------------------------------


@router.get("/analysis/{aid}/trace")
def analysis_trace(aid: int) -> ApiResponse[Dict[str, Any]]:
    """决策留痕：路由、工具、模型、降级、护栏、引用核对。"""
    from ..agent.trace_query import TraceQueryService

    return ApiResponse.ok(TraceQueryService().trace(aid))


@router.get("/analysis/{aid}/lineage")
def analysis_lineage(aid: int) -> ApiResponse[Dict[str, Any]]:
    """血缘：引用 → 语料块 → 源表记录，并标出是否被正文引用。

    「证据召回了但正文没提」是漏引用，「正文提了但证据里没有」是悬空引用，
    两类都要能被查出来 —— 所以这里的 ``citedInAnswer`` 是逐条标的。
    """
    from ..agent.grounding import AnswerGroundingService
    from ..agent.trace_query import TraceQueryService

    out = TraceQueryService().lineage(aid)
    if not out.get("found"):
        raise BusinessError(f"分析记录不存在：{aid}")
    row = _load_analysis(aid) or {}
    check = AnswerGroundingService().verify(row.get("answer") or "",
                                            _safe_json(row.get("evidence_json")))
    out["evidence"] = _safe_json(row.get("evidence_json"))
    out["grounding"] = check.to_dict()
    return ApiResponse.ok(out)


@router.get("/analysis/{aid}/export")
def analysis_export(aid: int, format: str = Query(default="pdf")) -> Any:
    """导出分析报告（``format=docx`` 出 Word，其余出 PDF）。

    报告一旦离开系统就会在邮件和群里流转，所以**来源清单与引用核对摘要是强制项**，
    不能为了排版好看省掉。
    """
    from fastapi.responses import Response

    from ..agent.report_export import ReportExportService

    rep = ReportExportService().export(aid, format)
    return Response(
        content=rep.bytes,
        media_type=rep.content_type,
        headers={"Content-Disposition":
                 "attachment; filename*=UTF-8''" + quote(rep.filename)})


# ------------------------------------------------------------------
# 主动预警（事件驱动研判）
# ------------------------------------------------------------------


@router.get("/proactive/recent")
def proactive_recent(companyId: Optional[int] = Query(default=None),
                     limit: int = Query(default=20)) -> ApiResponse[List[Dict[str, Any]]]:
    """某企业的预警历史。路径与 Java 一致 —— Spring 是按路径转发的。"""
    from ..agent.proactive import ProactiveRiskService

    if companyId is None:
        raise BusinessError("缺少 companyId")
    return ApiResponse.ok(ProactiveRiskService().recent(companyId, limit))


class ProactiveRequest(BaseModel):
    companyId: Optional[int] = None
    triggerType: str = "MANUAL"
    triggerRef: Optional[str] = None


@router.post("/proactive/evaluate")
def proactive_evaluate(r: ProactiveRequest) -> ApiResponse[Dict[str, Any]]:
    """手动触发一次主动研判。

    ⚠️ 默认会被**授权闸门**挡下：后台自动调用不消耗模型额度。
    要真的跑，先在页面上打开【后台自动研判】或设 ``APP_AI_AUTO_CONSUME=true``。
    """
    from ..agent.approvals import ActionApprovalService
    from ..agent.proactive import ProactiveRiskService

    svc = ProactiveRiskService(agent_service=_agent_service(),
                               approvals=ActionApprovalService().submit)
    return ApiResponse.ok(svc.evaluate(r.companyId, r.triggerType, r.triggerRef))


@router.post("/proactive/evaluate-metric/{metric_id}")
def proactive_evaluate_metric(metric_id: int) -> ApiResponse[Dict[str, Any]]:
    """MQ 链路入口：指标越阈值被判为高风险后调用。"""
    from ..agent.approvals import ActionApprovalService
    from ..agent.proactive import ProactiveRiskService

    svc = ProactiveRiskService(agent_service=_agent_service(),
                               approvals=ActionApprovalService().submit)
    return ApiResponse.ok(svc.evaluate_by_metric(metric_id))


class NotifyRequest(BaseModel):
    eventId: Optional[int] = None
    assigneeUserId: Optional[int] = None
    message: Optional[str] = None


@router.post("/notify/owner")
def notify_owner(r: NotifyRequest) -> ApiResponse[Dict[str, Any]]:
    """通知风险事件责任人。

    没配 webhook 时回执为「已登记待人工投递」，**不假装已送达**。
    """
    from ..agent.notification import NotificationService

    return ApiResponse.ok(NotificationService().notify_owner(
        r.eventId, r.assigneeUserId, r.message).to_dict())


def _safe_json(v: Any) -> Any:
    if not v:
        return []
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except Exception:  # noqa: BLE001
        return []


# ------------------------------------------------------------------
# 待审批动作
# ------------------------------------------------------------------


@router.get("/actions")
def actions(companyId: Optional[int] = Query(default=None)) -> ApiResponse[List[Dict[str, Any]]]:
    from ..agent.approvals import ActionApprovalService

    return ApiResponse.ok(ActionApprovalService().pending(companyId))


@router.get("/actions/all")
def actions_all() -> ApiResponse[List[Dict[str, Any]]]:
    from ..agent.approvals import ActionApprovalService

    return ApiResponse.ok(ActionApprovalService().pending())


class DecideRequest(BaseModel):
    approve: bool = False
    comment: Optional[str] = None
    approverUserId: Optional[int] = Field(default=None, alias="approverUserId")

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/actions/{aid}/decide")
def decide_action(aid: int, r: DecideRequest) -> ApiResponse[Dict[str, Any]]:
    from ..agent.approvals import ActionApprovalService

    ok = ActionApprovalService().decide(aid, r.approve, r.approverUserId, r.comment)
    return ApiResponse.ok({"id": aid, "approved": r.approve, "updated": ok})


# ------------------------------------------------------------------
# 模型运行时开关
# ------------------------------------------------------------------


class AutoConsumeRequest(BaseModel):
    enabled: bool = False


@router.post("/auto-consume")
def set_auto_consume(r: AutoConsumeRequest) -> ApiResponse[Dict[str, Any]]:
    """热开关：后台自动任务是否允许消耗模型额度。

    用户明确要求「没有经过我的运行，不能使用我的模型」，所以默认是关的。
    """
    svc = _agent_service()
    llm = getattr(svc, "llm", None)
    m: Dict[str, Any] = {}
    if llm is None:
        m.update(ok=False, issue="未启用外部大模型（APP_AI_ENABLED=false），后台本来也不会调用模型。")
        return ApiResponse.ok(m)
    llm.set_auto_consume(r.enabled)
    m.update(ok=True, autoConsume=llm.is_auto_consume_allowed())
    # 文案与 Java 一致：把「这个开关到底管住什么」讲清楚，
    # 免得使用者以为关了它页面上的分析也会跟着不能跑。
    m["note"] = (
        "已允许后台自动研判调用模型：MQ 触发的主动预警会真实消耗额度（仍受每企业每日上限约束）。"
        if r.enabled else
        "已禁止后台自动研判调用模型：只有你在页面上主动发起的分析才会消耗额度。")
    return ApiResponse.ok(m)


@router.get("/trace/{trace_id}")
def trace_detail(trace_id: str) -> ApiResponse[Dict[str, Any]]:
    """按追溯号查留痕：跨系统排查时这是唯一的锚。"""
    row = get_db().fetch_one(
        select(T.ai_analysis_trace).where(T.ai_analysis_trace.c.trace_id == trace_id))
    if row is None:
        raise BusinessError(f"追溯号不存在：{trace_id}")
    return ApiResponse.ok(dict(row))


# ------------------------------------------------------------------
# 运行时改配置（只保存在内存，重启回落到环境变量）
# ------------------------------------------------------------------


class KeyRequest(BaseModel):
    key: Optional[str] = None
    baseUrl: Optional[str] = None
    model: Optional[str] = None
    embeddingBaseUrl: Optional[str] = None
    embeddingModel: Optional[str] = None
    embeddingKey: Optional[str] = None


@router.post("/key")
def update_key(r: KeyRequest) -> ApiResponse[Dict[str, Any]]:
    """换厂商时 base-url / 模型名 / Key 是**一组**，只改其中一个必然 401。

    改系统环境变量对已运行的进程无效，使用者极容易卡在这里，所以给一个运行时入口。
    只保存在内存，重启后回落到环境变量。
    """
    svc = _agent_service()
    llm = getattr(svc, "llm", None)
    m: Dict[str, Any] = {}
    if llm is None:
        m.update(ok=False, issue="未启用外部大模型（APP_AI_ENABLED=false），无需配置 Key。")
        return ApiResponse.ok(m)

    def blank(v: Optional[str]) -> bool:
        return v is None or not v.strip()

    touch_emb = any(not blank(x) for x in (r.embeddingBaseUrl, r.embeddingModel, r.embeddingKey))
    touch_llm = any(not blank(x) for x in (r.baseUrl, r.model, r.key))
    if not touch_llm and not touch_emb:
        m.update(ok=False, issue="未提供任何要修改的内容：请至少填写 baseUrl、model、key 之一"
                                 "（向量化则是 embeddingBaseUrl / embeddingModel / embeddingKey）。")
        return ApiResponse.ok(m)

    if touch_emb:
        apply_emb = getattr(llm, "apply_embedding_config", None)
        issue = apply_emb(r.embeddingBaseUrl, r.embeddingModel, r.embeddingKey) \
            if apply_emb else "当前客户端不支持运行时改向量化配置"
        m.update(embeddingOk=issue is None)
        if issue:
            m["embeddingIssue"] = issue

    issue = llm.apply_runtime_config(r.baseUrl, r.model, r.key) if touch_llm else None
    m.update(ok=issue is None,
             baseUrl=llm.get_base_url(), model=llm.get_model(),
             embeddingModel=llm.get_embedding_model(),
             keyHint=llm.get_key_hint(), keySource=llm.get_key_source_name(),
             configSource=llm.get_config_source(), capabilities=llm.capabilities())
    if issue:
        m["issue"] = issue
    # 与 Java 一致：自检响应里顺带把「可用模型列表 / 自动降级记录 / 联网通道」一起带回，
    # 省得前端为了刷新这几块再各发一次请求（前端就是拿它当 diag 用的）。
    from .ai import model_info, web_search_info

    m["models"] = model_info(False)
    m["webSearch"] = web_search_info()
    return ApiResponse.ok(m)


@router.post("/models/adopt")
def adopt_model() -> ApiResponse[Dict[str, Any]]:
    """一键「自动选一个能用的模型」：逐个实测，选中第一个真能调通的并切过去。

    会真的发一次极小请求（1 token），所以只在**使用者主动触发**时发生，不做后台自动调用。

    返回里**必须带上完整的模型信息**（``m.putAll(modelInfo(false))``，与 Java 一致）：
    前端拿到响应会直接 `applyModelsMeta(它)` 覆盖「账号可用模型」下拉的数据源，
    只回 ``ok/current`` 会让下拉在下一次刷新前变成空列表 —— 点一下按钮，选项全没了。
    """
    svc = _agent_service()
    llm = getattr(svc, "llm", None)
    m: Dict[str, Any] = {}
    if llm is None:
        m.update(ok=False, issue="未启用外部大模型（APP_AI_ENABLED=false）。")
        return ApiResponse.ok(m)
    issue = llm.adopt_working_model()
    m.update(ok=issue is None)
    if issue:
        m["issue"] = issue
    from .ai import model_info

    m.update(model_info(False))
    return ApiResponse.ok(m)
