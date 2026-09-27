"""结构化业务数据 → 可召回的文本切片（对应 Java ``StructuredCorpus``）。

为什么需要它
------------
原来的 RAG 只召回 ``knowledge_document``。可企业里最值钱的判断依据恰恰是结构化的：
近几期指标走势、未闭环的风险事件、投诉的类别分布。过去这些内容只能靠模型
「想到才去调工具」取，一旦它没想起来就整段缺席——而这类数据恰恰最该出现在上下文里。

现在它们以切片形式进入同一套四阶段流水线，享受同样的查询改写、多路召回与精排。
模型即使不调工具，关键异常也会出现在证据里；引用时写成「来源ID=metric:12」，
与文档来源「来源ID=5」一样可被确定性核对。

切片是**实时拼装**的（指标表通常每企业几百行，一次渲染不到毫秒），
不做增量落表，免得引入"数据与切片不一致"的维护负担。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional

from sqlalchemy import desc, select

from ..db import tables as T
from ..db.engine import Database, get_db
from .corpus import date, num, nz, trim

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """一条结构化切片。``source_id`` 同时用作引用标识与去重键。"""

    source_id: str
    title: str
    text: str


def _pct_change(cur: Optional[Decimal], prev: Optional[Decimal]) -> Optional[float]:
    """环比百分比。Java 是 ``cur.subtract(pv).divide(pv, 4, HALF_UP) * 100``。

    ``divide(pv, 4, HALF_UP)`` 的 4 是**小数位数**、舍入是 HALF_UP——
    Python 的 ``/`` 是无限精度后再量化，两者在边界值上会差 1 个最小单位，
    所以这里显式量化到 4 位再转 float，保持与 Java 一致。
    """
    if cur is None or prev is None:
        return None
    try:
        pv = Decimal(str(prev))
        if pv == 0:
            return None
        return float((Decimal(str(cur)) - pv).divide(pv, 4, ROUND_HALF_UP) * Decimal(100))
    except Exception:  # noqa: BLE001 - 数值异常不该让切片生成失败
        return None


class StructuredCorpus:
    """把结构化业务数据渲染成切片。三块互不牵连：任何一块查库失败都不能让整个检索挂掉。"""

    def __init__(self, db: Optional[Database] = None, limits: Optional[dict] = None) -> None:
        self.db = db or get_db()
        from ..config import get_settings

        r = get_settings().rag
        lm = limits or {}
        self.metric_limit = int(lm.get("metrics", r.structured_metrics))
        self.event_limit = int(lm.get("events", r.structured_events))
        self.complaint_limit = int(lm.get("complaints", r.structured_complaints))
        self.metric_window = int(lm.get("metric_window", r.structured_metric_window))

    # -- 入口 --------------------------------------------------------

    def chunks(self, company_id: Optional[int]) -> List[Chunk]:
        if company_id is None:
            return []
        out: List[Chunk] = []
        for name, fn in (
            ("指标", self.metric_chunks),
            ("风险事件", self.event_chunks),
            ("投诉", self.complaint_chunks),
        ):
            try:
                out.extend(fn(company_id))
            except Exception as e:  # noqa: BLE001
                log.warning("结构化语料：%s切片生成失败（已跳过）%s", name, e)
        return out

    # -- 指标：按 metricCode 聚合，最新值 + 环比 + 近期走势 ------------

    def metric_chunks(self, cid: int) -> List[Chunk]:
        scan = max(20, self.metric_limit * self.metric_window)
        rows = self.db.fetch_all(
            select(T.business_metric)
            .where(T.business_metric.c.company_id == cid)
            .order_by(desc(T.business_metric.c.metric_date))
            .limit(scan)
        )
        if not rows:
            return []

        groups: dict[str, list] = {}
        for m in rows:
            code = m.get("metric_code")
            key = code if (code and not code.isspace()) else (m.get("metric_name") or "unknown")
            groups.setdefault(key, []).append(m)

        out: List[Chunk] = []
        n = 0
        for key, ms in groups.items():
            if n >= self.metric_limit:
                break
            n += 1
            latest = ms[0]
            name = self._name_of(latest)
            body = (
                f"经营指标【{name}】最新一期 {date(latest.get('metric_date'))} 为 "
                f"{num(latest.get('metric_value'))}{self._unit(latest)}"
            )
            if len(ms) >= 2:
                prev = ms[1]
                chg = _pct_change(latest.get("metric_value"), prev.get("metric_value"))
                if chg is not None:
                    body += (
                        f"，较上一期（{date(prev.get('metric_date'))} "
                        f"{num(prev.get('metric_value'))}{self._unit(prev)}）"
                        f"{'上升' if chg >= 0 else '下降'} {abs(chg):.1f}%"
                    )
            if len(ms) > 2:
                k = min(self.metric_window, len(ms))
                body += f"。近 {k} 期走势（由旧到新）："
                body += "、".join(
                    f"{date(ms[i].get('metric_date'))} {num(ms[i].get('metric_value'))}"
                    for i in range(k - 1, -1, -1)
                )
            body += "。"
            out.append(Chunk(f"metric:{latest['id']}", f"经营指标 · {name}", body))
        return out

    # -- 风险事件 ----------------------------------------------------

    def event_chunks(self, cid: int) -> List[Chunk]:
        rows = self.db.fetch_all(
            select(T.risk_event)
            .where(T.risk_event.c.company_id == cid)
            .order_by(desc(T.risk_event.c.created_at))
            .limit(max(1, self.event_limit))
        )
        out: List[Chunk] = []
        for r in rows:
            title = nz(r.get("risk_title"))
            body = (
                f"风险事件【{title}】：类型 {nz(r.get('risk_type'))}"
                f"，等级 {nz(r.get('risk_level'))}"
                f"，状态 {nz(r.get('status'))}"
            )
            if r.get("trigger_value") is not None or r.get("threshold_value") is not None:
                body += f"，触发值 {num(r.get('trigger_value'))}（阈值 {num(r.get('threshold_value'))}）"
            if r.get("metric_date") is not None:
                body += f"，数据日期 {date(r.get('metric_date'))}"
            hr = r.get("handle_result")
            if hr and not hr.isspace():
                body += f"。处置情况：{hr.strip()}"
            rc = r.get("review_comment")
            if rc and not rc.isspace():
                body += f"。复核意见：{rc.strip()}"
            body += "。"
            out.append(Chunk(f"event:{r['id']}", f"风险事件 · {title}", body))
        return out

    # -- 投诉：按类别聚合 --------------------------------------------

    def complaint_chunks(self, cid: int) -> List[Chunk]:
        rows = self.db.fetch_all(
            select(T.complaint)
            .where(T.complaint.c.company_id == cid)
            .order_by(desc(T.complaint.c.complaint_date))
            .limit(200)
        )
        if not rows:
            return []

        groups: dict[str, list] = {}
        for c in rows:
            groups.setdefault(nz(c.get("category")), []).append(c)

        out: List[Chunk] = []
        n = 0
        for cat, cs in groups.items():
            if n >= self.complaint_limit:
                break
            n += 1
            repeat = sum(1 for c in cs if (c.get("repeat_flag") or 0) == 1)
            sla = sum(1 for c in cs if (c.get("sla_exceeded") or 0) == 1)
            churn = sum(1 for c in cs if (nz(c.get("churn_risk")).upper() == "HIGH"))
            body = f"客户投诉【类别 {cat}】近 {len(cs)} 条"
            if cs[0].get("complaint_date") is not None:
                body += f"，最新 {date(cs[0].get('complaint_date'))}"
            body += f"。其中重复投诉 {repeat} 条、超 SLA {sla} 条、高流失风险 {churn} 条"
            sample = None
            for c in cs:
                d = c.get("description")
                if d and not d.isspace():
                    sample = d.strip()
                    break
            if sample is not None:
                body += f"。典型问题：{trim(sample, 120)}"
            rc = cs[0].get("root_cause")
            if rc and not rc.isspace():
                body += f"。根因：{rc.strip()}"
            body += "。"
            out.append(Chunk(f"complaint:{cs[0]['id']}", f"客户投诉 · {cat}", body))
        return out

    # -- 小工具 ------------------------------------------------------

    @staticmethod
    def _name_of(m: dict) -> str:
        mn = m.get("metric_name")
        return mn.strip() if (mn and not mn.isspace()) else nz(m.get("metric_code"))

    @staticmethod
    def _unit(m: dict) -> str:
        u = m.get("unit")
        return u.strip() if (u and not u.isspace()) else ""
