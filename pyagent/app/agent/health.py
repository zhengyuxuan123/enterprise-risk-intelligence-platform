"""运行时健康聚合（对应 Java ``OperationalHealthService``）。

评估的 ``cost`` 维度**不来自跑批**，而来自线上留痕 —— 这是六维里唯一反映真实表现的一维：
跑批再绿，线上天天降级也没用。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy import func, select

from ..db import tables as T
from ..db.engine import Database, get_db

log = logging.getLogger(__name__)

DEFAULT_HOURS = 24


def _ladder(v: float, good: float, ok: float, bad: float) -> float:
    """阶梯打分：优秀档 100 / 良好 85 / 及格 70 / 超出 50。"""
    if v <= good:
        return 100.0
    if v <= ok:
        return 85.0
    if v <= bad:
        return 70.0
    return 50.0


def _r2(v: float) -> float:
    from .evaluation import round2  # 局部导入：避免与 evaluation 形成循环依赖
    return round2(v)


class OperationalHealthService:
    """按时间窗口聚合 ``ai_analysis_trace``，给出降级率 / 调用次数 / 工具失败率。"""

    def __init__(self, db: Optional[Database] = None) -> None:
        self.db = db or get_db()

    def health(self, hours: int = DEFAULT_HOURS) -> Dict[str, Any]:
        h = hours if hours and hours > 0 else DEFAULT_HOURS
        m: Dict[str, Any] = {"windowHours": h}
        try:
            since = datetime.now() - timedelta(hours=h)
            tr = T.ai_analysis_trace
            q = select(
                func.count(tr.c.id).label("total"),
                func.sum(func.if_(tr.c.degrade_level != "NONE", 1, 0)).label("degraded"),
                func.sum(func.if_(tr.c.degrade_level == "FULL", 1, 0)).label("degradedFull"),
                func.sum(func.if_(tr.c.tool_failures > 0, 1, 0)).label("withToolFailure"),
                func.sum(tr.c.tool_failures).label("toolFailures"),
                func.sum(tr.c.tool_calls).label("toolCalls"),
                func.sum(tr.c.llm_calls).label("llmCalls"),
                func.avg(tr.c.duration_ms).label("avgDurationMs"),
                func.max(tr.c.duration_ms).label("maxDurationMs"),
            ).where(tr.c.created_at >= since)
            row = self.db.fetch_one(q) or {}
            total = int(row.get("total") or 0)
            m["samples"] = total
            if total == 0:
                m["available"] = False
                m["note"] = "窗口内没有分析记录（尚未产生留痕数据）"
                return m

            degraded = float(row.get("degraded") or 0)
            degraded_full = float(row.get("degradedFull") or 0)
            with_tool_failure = float(row.get("withToolFailure") or 0)
            tool_failures = float(row.get("toolFailures") or 0)
            tool_calls = float(row.get("toolCalls") or 0)
            llm_calls = float(row.get("llmCalls") or 0)
            avg_ms = float(row.get("avgDurationMs") or 0)

            m["available"] = True
            m["degradeRate"] = _r2(degraded * 100.0 / total)
            m["fullDegradeRate"] = _r2(degraded_full * 100.0 / total)
            m["analysisWithToolFailureRate"] = _r2(with_tool_failure * 100.0 / total)
            m["toolFailureRate"] = _r2(tool_failures * 100.0 / tool_calls) if tool_calls > 0 else 0.0
            m["avgLlmCalls"] = _r2(llm_calls / total)
            m["avgDurationMs"] = round(avg_ms)
            m["maxDurationMs"] = int(row.get("maxDurationMs") or 0)
            m["score"] = _r2(
                _ladder(degraded * 100.0 / total, 2, 5, 10) * 0.4
                + _ladder(llm_calls / total, 3, 5, 7) * 0.3
                + _ladder((tool_failures * 100.0 / tool_calls) if tool_calls > 0 else 0.0, 2, 5, 10) * 0.3
            )
            return m
        except Exception as e:  # noqa: BLE001 - 健康度是加分项，不能因此让诊断端点 500
            log.warning("[Health] 运行时健康聚合失败: %s", e)
            m["available"] = False
            m["error"] = str(e)
            return m
