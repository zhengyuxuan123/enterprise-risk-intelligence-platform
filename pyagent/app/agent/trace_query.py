"""追溯查询 —— 对应 Java ``TraceQueryService``。

两类查询：

* :meth:`TraceQueryService.trace` —— 决策留痕：路由、工具、模型、降级、护栏、引用核对；
* :meth:`TraceQueryService.lineage` —— 数据血缘：结论里每个引用 → 语料块 → 源表/源记录，
  并标明这条证据是否被正文真的引用过。

血缘里最容易踩的一件事：**正文的引用标识（ref）与索引里的 chunk_id 不是同一套编码**。
知识库 ref 是文档号（``7``），chunk_id 是 ``k:7#0``（带分片序号）；
结构化切片 ref 是 ``metric:12``，chunk_id 是 ``m:12``；聚合切片是 ``agg:{企业}:metric:16``。
所以必须按规则还原后再查，查不到就如实标"未找到"，不猜、不硬凑。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select

from ..db import get_db
from ..db import tables as T

log = logging.getLogger(__name__)

#: 各类语料在索引里的 chunk_id 前缀（与 ``CorpusSource`` 的 id 规则一致）。
CHUNK_PREFIX: Dict[str, str] = {
    "knowledge": "k",
    "metric": "m",
    "event": "e",
    "complaint": "c",
    "competitor": "p",
    "company": "co",
}


def _str(o: Any) -> str:
    return "" if o is None else str(o)


def _read_json(s: Any) -> Any:
    """留痕里的 JSON 列：能解析就解析，解析不了就原样返回字符串（不丢信息）。"""
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    text = str(s).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return text


class TraceQueryService:
    """把「一次分析是怎么得出结论的」查得出来、讲得清楚。"""

    def __init__(self, db=None) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # 自检
    # ------------------------------------------------------------------

    def recent_count(self, hours: int) -> int:
        """最近 N 小时的留痕条数；表不可用时返回 ``-1``（自检端点据此判定留痕能力缺失）。"""
        try:
            since = datetime.now() - timedelta(hours=hours if hours > 0 else 24)
            n = self._db().scalar(
                select(func.count(T.ai_analysis_trace.c.id)).where(
                    T.ai_analysis_trace.c.created_at >= since),
                default=0)
        except Exception:  # noqa: BLE001
            return -1
        return int(n or 0)

    # ------------------------------------------------------------------
    # 决策留痕
    # ------------------------------------------------------------------

    def trace(self, analysis_id: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        a = self._load_analysis(analysis_id)
        if a is None:
            out["found"] = False
            return out
        out["found"] = True
        for k in ("id", "trace_id", "question", "provider", "model", "confidence",
                  "grounded", "degrade_level", "degrade_reason", "duration_ms",
                  "llm_calls", "tool_calls", "created_at"):
            if k in a:
                out[_OUT_KEY.get(k, k)] = a[k]
        out["analysisId"] = a.get("id")

        try:
            rows = self._db().fetch_all(
                select(T.ai_analysis_trace).where(
                    T.ai_analysis_trace.c.analysis_id == analysis_id))
            if not rows:
                # 兼容旧数据：留痕表上线之前的历史分析没有明细
                out["detail"] = None
                out["detailNote"] = "该分析发生在决策留痕功能上线之前，仅有汇总字段"
                return out
            t = rows[0]
            out["detail"] = {
                "traceId": t.get("trace_id"),
                "route": _read_json(t.get("route_json")),
                "iterations": t.get("iterations"),
                "llmCalls": t.get("llm_calls"),
                "model": t.get("model"),
                "modelSwitched": t.get("model_switched"),
                "toolCalls": t.get("tool_calls"),
                "toolFailures": t.get("tool_failures"),
                "toolDetail": _read_json(t.get("tool_detail")),
                "retrieval": _read_json(t.get("retrieval_json")),
                "degradeLevel": t.get("degrade_level"),
                "degradeReason": t.get("degrade_reason"),
                "guardrailFlags": t.get("guardrail_flags"),
                "grounding": _read_json(t.get("grounding_json")),
                "reviewTrigger": t.get("review_trigger"),
                "reviewRevised": t.get("review_revised"),
                "promptChars": t.get("prompt_chars"),
                "durationMs": t.get("duration_ms"),
            }
        except Exception as e:  # noqa: BLE001
            log.warning("[Trace] 留痕查询失败: %s", e)
            out["detailError"] = str(e)
        return out

    # ------------------------------------------------------------------
    # 数据血缘
    # ------------------------------------------------------------------

    def lineage(self, analysis_id: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        a = self._load_analysis(analysis_id)
        if a is None:
            out["found"] = False
            return out
        answer = a.get("answer") or ""
        out["found"] = True
        out["analysisId"] = analysis_id
        out["traceId"] = a.get("trace_id")

        items, refs = self._collect_refs(a.get("evidence_json"))

        resolved: Dict[str, Optional[dict]] = {}
        for it in items:
            st = self._resolve_chunk(it["ref"], it["type"])
            if st is not None:
                resolved[it["ref"]] = st

        cited = 0
        for it in items:
            ref = it["ref"]
            used = bool(ref) and (("来源ID=" + ref) in answer or ref in answer)
            it["citedInAnswer"] = used
            if used:
                cited += 1
            st = resolved.get(ref)
            if st is not None:
                it["indexed"] = True
                it["sourceType"] = st.get("source_type")
                it["sourceId"] = st.get("source_id")
                it["chunkId"] = st.get("chunk_id")
                it["chunkStatus"] = st.get("status")
            else:
                it["indexed"] = False
                # 网页来源本就不在语料库里，不算异常
                it["note"] = ("外部网页来源，不入库" if it["type"] == "web"
                              else "未在语料索引中找到对应块")

        # 被正文引用的排前面：复盘时最该看的是"结论到底靠什么撑着"
        items.sort(key=lambda x: not x["citedInAnswer"])
        out["items"] = items
        out["total"] = len(items)
        out["cited"] = cited
        out["uncited"] = len(items) - cited
        out["answerChars"] = len(answer)
        return out

    # ------------------------------------------------------------------

    def _resolve_chunk(self, ref: Optional[str], type_: str) -> Optional[dict]:
        """把正文里的引用标识还原成索引里的语料块。匹配不上返回 None（调用方标"未找到"）。"""
        if not ref or not ref.strip() or type_ == "web":
            return None
        try:
            db = self._db()
            tbl = T.rag_chunk_state
            hit = db.fetch_one(select(tbl).where(tbl.c.chunk_id == ref).limit(1))
            if hit:
                return hit

            # 知识库：ref 是文档号，索引里带分片序号（k:7#0…）
            if type_ == "knowledge" and ref.isdigit():
                hit = db.fetch_one(
                    select(tbl).where(tbl.c.chunk_id.like(f"k:{ref}#%")).limit(1))
                if hit:
                    return hit

            c = ref.find(":")
            if c > 0:
                t, sid = ref[:c], ref[c + 1:]
                p = CHUNK_PREFIX.get(t)
                if p is not None:
                    hit = db.fetch_one(
                        select(tbl).where(tbl.c.chunk_id == f"{p}:{sid}").limit(1))
                    if hit:
                        return hit
                # 聚合切片：agg:{企业}:metric:16
                hit = db.fetch_one(
                    select(tbl).where(tbl.c.source_type == "aggregate")
                    .where(tbl.c.chunk_id.like(f"%{ref}")).limit(1))
                if hit:
                    return hit
        except Exception as e:  # noqa: BLE001
            log.warning("[Lineage] 语料块解析失败 ref=%s: %s", ref, e)
        return None

    def _collect_refs(self, evidence_json: Any):
        """从证据 JSON 里抽出所有可被引用的标识（内部 ref / 文档 id / 网页 URL）。"""
        items: List[Dict[str, Any]] = []
        refs: List[str] = []
        evs = _read_json(evidence_json)
        if not isinstance(evs, list):
            return items, refs
        for ev in evs:
            if not isinstance(ev, dict):
                continue
            src = ev.get("sources")
            if not isinstance(src, list):
                continue
            for s in src:
                if not isinstance(s, dict):
                    continue
                ref = _str(s.get("ref")).strip()
                doc_id = _str(s.get("documentId")).strip()
                url = _str(s.get("url")).strip()
                title = _str(s.get("title"))
                item: Dict[str, Any] = {}
                if ref:
                    item.update(ref=ref,
                                type="web" if _str(ev.get("sourceType")) == "web" else "internal",
                                title=title)
                    if ref not in refs:
                        refs.append(ref)
                elif doc_id:
                    # 知识库引用标识就是文档号（正文写作「来源ID=7」），分片序号在索引里才带
                    item.update(ref=doc_id, type="knowledge", title=title)
                    if doc_id not in refs:
                        refs.append(doc_id)
                elif url:
                    item.update(ref=url, type="web", title=title)
                else:
                    continue
                item["tool"] = _str(ev.get("tool"))
                items.append(item)
        return items, refs

    def _load_analysis(self, analysis_id: int) -> Optional[dict]:
        return self._db().fetch_one(
            select(T.ai_analysis).where(T.ai_analysis.c.id == analysis_id))

    def _db(self):
        if self.db is None:
            self.db = get_db()
        return self.db


#: 留痕汇总字段：DB 列名 → 对外 camelCase
_OUT_KEY: Dict[str, str] = {
    "id": "analysisId",
    "trace_id": "traceId",
    "degrade_level": "degradeLevel",
    "degrade_reason": "degradeReason",
    "duration_ms": "durationMs",
    "llm_calls": "llmCalls",
    "tool_calls": "toolCalls",
    "created_at": "createdAt",
}
