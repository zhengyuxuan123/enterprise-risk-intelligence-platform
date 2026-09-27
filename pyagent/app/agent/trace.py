"""全链路留痕（对应 Java ``TraceContext`` + ``TraceRecorder``）。

为什么需要 traceId
------------------
一次分析会经过「护栏 → 路由 → 预取 → 工具循环 → 成稿 → 引用核对 → 留痕」七道环节，
任何一道出问题都要能回答「这次到底跑到哪一步、卡在哪个工具、换了哪个模型」。
没有统一追溯号时，日志、响应头、数据库留痕三者对不上，等于没有可追溯性。

.. warning::
   **ThreadLocal / contextvars 不跨线程池**。工具与检索跑在线程池里，
   必须用 :meth:`TraceContext.attach` 把 traceId 带过去，
   否则响应头里的追溯号与落库留痕对不上（=可追溯失效）。
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import insert, select

from ..db import tables as T
from ..db.engine import Database, get_db

log = logging.getLogger(__name__)

_TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("pyagent_trace_id", default="")
#: 追溯号是**请求带进来的**还是**本次新开的**。
#: 必须分清楚：调用方（网关/前端）在 X-Trace-Id 里带的号要一路复用，
#: 而本进程自己 new() 出来的号只能属于这一次分析 —— 否则同一线程里的
#: 第二次分析会沿用上一次的号，两条不同分析在界面上显示成同一个追溯号。
_INHERITED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "pyagent_trace_inherited", default=False)

HEADER = "X-Trace-Id"


class TraceContext:
    """请求级追溯号的存取。"""

    @staticmethod
    def current() -> str:
        v = _TRACE_ID.get()
        return v or ""

    @staticmethod
    def is_inherited() -> bool:
        """当前追溯号是否由调用方（请求头）指定。"""
        return bool(_INHERITED.get())

    @staticmethod
    def new() -> str:
        v = uuid.uuid4().hex[:32]
        _TRACE_ID.set(v)
        _INHERITED.set(False)
        return v

    @staticmethod
    def set(trace_id: Optional[str]) -> str:
        v = (trace_id or "").strip() or uuid.uuid4().hex[:32]
        _TRACE_ID.set(v)
        _INHERITED.set(True)
        return v

    @staticmethod
    def clear() -> None:
        _TRACE_ID.set("")
        _INHERITED.set(False)

    @staticmethod
    def token() -> Any:
        """返回可用于跨线程传播的 contextvars token。"""
        return _TRACE_ID.get()

    @staticmethod
    def attach(value: str) -> None:
        """在线程池任务里恢复追溯号（**必须显式调用**）。"""
        _TRACE_ID.set(value or "")


@dataclass
class ToolRun:
    """一次工具执行的留痕。"""

    name: str
    ok: bool
    duration_ms: int
    error: Optional[str] = None
    source_type: str = "internal"
    prefetch: bool = False
    result_chars: int = 0


@dataclass
class TraceRecorder:
    """一次分析的留痕收集器。最后 :meth:`flush` 落 ``ai_analysis_trace``。"""

    trace_id: str = ""
    company_id: Optional[int] = None
    user_id: Optional[int] = None
    question: Optional[str] = None
    db: Optional[Database] = None

    iterations: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    model: Optional[str] = None
    model_switched: Optional[str] = None
    degrade_level: Optional[str] = None
    degrade_reason: Optional[str] = None
    prompt_chars: int = 0
    review_trigger: Optional[str] = None
    review_revised: bool = False
    finish_reason: Optional[str] = None

    tool_runs: List[ToolRun] = field(default_factory=list)
    route: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    guardrail_flags: List[str] = field(default_factory=list)
    grounding: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=lambda: _now())
    #: 这次的结论是从哪次分析缓存来的（``None`` = 本次真跑出来的）
    cache_hit: Optional[int] = None

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # -- 记录 --------------------------------------------------------

    def note_tool(self, run: ToolRun) -> None:
        with self._lock:
            self.tool_runs.append(run)
            self.tool_calls += 1
            if not run.ok:
                self.tool_failures += 1

    def note_llm(self, model: Optional[str] = None, finish_reason: Optional[str] = None) -> None:
        with self._lock:
            self.llm_calls += 1
            if model:
                self.model = model
            if finish_reason:
                self.finish_reason = finish_reason

    def note_switch(self, reason: Optional[str]) -> None:
        with self._lock:
            self.model_switched = reason

    def note_route(self, route: Dict[str, Any]) -> None:
        with self._lock:
            self.route = route or {}

    def note_retrieval(self, diag: Optional[Dict[str, Any]]) -> None:
        with self._lock:
            if diag:
                self.retrieval = diag

    def note_guardrail(self, flags: Optional[List[str]]) -> None:
        with self._lock:
            self.guardrail_flags = list(flags or [])

    def note_cache_hit(self, source_analysis_id: Optional[int]) -> None:
        """这次提问命中了结果缓存，结论来自 ``source_analysis_id`` 那次分析。

        写进 ``route`` 是为了不新增表列也能被审计页看到：
        「这条结论是第 N 次算出后被复用的」本身就是追溯时最想知道的事。
        """
        with self._lock:
            self.cache_hit = source_analysis_id
            self.route = dict(self.route or {})
            self.route["cachedFrom"] = source_analysis_id

    def note_grounding(self, result: Dict[str, Any]) -> None:
        with self._lock:
            self.grounding = result or {}

    def note_review(self, trigger: Optional[str], revised: bool) -> None:
        with self._lock:
            self.review_trigger = trigger
            self.review_revised = self.review_revised or bool(revised)

    def note_degrade(self, level: Optional[str], reason: Optional[str]) -> None:
        with self._lock:
            self.degrade_level = level
            self.degrade_reason = reason

    def duration_ms(self) -> int:
        return int((_now() - self.started_at) * 1000)

    # -- 落库 --------------------------------------------------------

    def flush(self, analysis_id: Optional[int] = None) -> Optional[int]:
        """写入 ``ai_analysis_trace``。失败只记日志，绝不影响主流程。"""
        db = self.db or get_db()
        try:
            row = {
                "trace_id": self.trace_id,
                "analysis_id": analysis_id,
                "company_id": self.company_id,
                "user_id": self.user_id,
                "question": (self.question or "")[:1000],
                "route_json": _json(self.route),
                "iterations": self.iterations,
                "llm_calls": self.llm_calls,
                "model": (self.model or "")[:120],
                "model_switched": (self.model_switched or "")[:200] or None,
                "tool_calls": self.tool_calls,
                "tool_failures": self.tool_failures,
                "tool_detail": _json([{
                    "name": r.name, "ok": r.ok, "ms": r.duration_ms,
                    "error": r.error, "sourceType": r.source_type,
                    "prefetch": r.prefetch, "chars": r.result_chars,
                } for r in self.tool_runs]),
                "retrieval_json": _json(self.retrieval),
                "degrade_level": self.degrade_level,
                "degrade_reason": (self.degrade_reason or "")[:1000] or None,
                "guardrail_flags": ",".join(self.guardrail_flags)[:1000] or None,
                "grounding_json": _json(self.grounding),
                "review_trigger": (self.review_trigger or "")[:200] or None,
                "review_revised": 1 if self.review_revised else 0,
                "prompt_chars": self.prompt_chars,
                "duration_ms": self.duration_ms(),
                "created_at": datetime.now(),
            }
            return db.insert_id(insert(T.ai_analysis_trace).values(**row))
        except Exception as e:  # noqa: BLE001
            log.warning("留痕写入失败（已忽略）：%s", e)
            return None

    def summary(self) -> Dict[str, Any]:
        """给前端「这次分析是怎么跑出来的」看的结构。"""
        return {
            "traceId": self.trace_id,
            "iterations": self.iterations,
            "llmCalls": self.llm_calls,
            "toolCalls": self.tool_calls,
            "toolFailures": self.tool_failures,
            "model": self.model,
            "modelSwitched": self.model_switched,
            "degradeLevel": self.degrade_level,
            "degradeReason": self.degrade_reason,
            "guardrailFlags": self.guardrail_flags,
            "durationMs": self.duration_ms(),
            "cachedFrom": self.cache_hit,
            "tools": [{"name": r.name, "ok": r.ok, "ms": r.duration_ms,
                       "sourceType": r.source_type, "prefetch": r.prefetch}
                      for r in self.tool_runs],
        }


class TraceQueryService:
    """血缘查询：引用 → 语料块 → 源表记录（对应 Java ``TraceQueryService``）。"""

    def __init__(self, db: Optional[Database] = None) -> None:
        self.db = db or get_db()

    def by_trace_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
        rows = self.db.fetch_all(
            select(T.ai_analysis_trace).where(T.ai_analysis_trace.c.trace_id == trace_id).limit(1))
        return rows[0] if rows else None

    def by_analysis_id(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        rows = self.db.fetch_all(
            select(T.ai_analysis_trace).where(T.ai_analysis_trace.c.analysis_id == analysis_id).limit(1))
        return rows[0] if rows else None

    def lineage(self, analysis_id: int) -> Dict[str, Any]:
        """引用 → 语料块 → 源表记录，并标是否被正文引用。"""
        from sqlalchemy import text

        rows = self.db.fetch_all(
            select(T.ai_analysis).where(T.ai_analysis.c.id == analysis_id).limit(1))
        if not rows:
            return {"found": False, "analysisId": analysis_id}
        a = rows[0]
        answer = str(a.get("answer") or "")
        evidence = _safe_json(a.get("evidence_json")) or []
        items: List[Dict[str, Any]] = []
        for ev in evidence if isinstance(evidence, list) else []:
            if not isinstance(ev, dict):
                continue
            srcs = ev.get("sources")
            if not isinstance(srcs, list):
                continue
            for s in srcs:
                if not isinstance(s, dict):
                    continue
                ref = s.get("sourceRef") or (str(s.get("documentId")) if s.get("documentId") else None)
                idx = s.get("index")
                used = _ref_used_in_answer(answer, ref, idx, ev.get("sourceType"))
                items.append({
                    "ref": ref,
                    "index": idx,
                    "title": s.get("title"),
                    "sourceType": ev.get("sourceType"),
                    "title2": None,
                    "used": used,
                })
        return {
            "found": True,
            "analysisId": analysis_id,
            "traceId": a.get("trace_id"),
            "degradeLevel": a.get("degrade_level"),
            "items": items,
        }


def _ref_used_in_answer(answer: str, ref: Optional[str], index: Any, source_type: Optional[str]) -> bool:
    """判断这条引用是否真的出现在正文里——「召回但没引用」是评估里最该看见的信号。"""
    if not answer:
        return False
    if source_type == "web" and index is not None:
        return f"[{index}]" in answer
    if ref:
        return f"来源ID={ref}" in answer or f"来源ID = {ref}" in answer
    return False


def _now() -> float:
    import time

    return time.time()


def _json(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _safe_json(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError):
        return None
