"""长期记忆（跨会话）—— 存取、冲突消解与召回。

和 ``ConversationMemory``（会话内短期记忆）的区别
--------------------------------------------------
``ai_message`` 只在**同一个会话**里回看最近几轮，换个会话就失忆；
本模块把"这家企业曾经发生过什么"沉淀到 ``ai_memory``，跨会话、跨提问都能复用。
两者独立计预算、独立诊断，互不挤占。

四道设计闸门
------------
1. **零 token**：写入靠规则抽取（见 :mod:`memory_extract`），召回靠本地词面 + 本地哈希向量。
   记忆是"省钱的东西"，它自己先烧一次模型额度就本末倒置了。
2. **冲突消解，不是覆盖**：同一主题出现新口径时，旧条目置 ``SUPERSEDED`` 并指向新条目。
   "上次说 8.3%、这次说 12%"必须留下痕迹 —— 那是口径漂移的早期信号。
3. **召回侧再按主题去重**：写入路径的并发（或多进程）理论上可能留下两条同主题的有效记忆，
   召回到两条互相矛盾的记忆比召回不到更糟，所以排序后同 ``topic_key`` 只保留最新一条。
4. **记忆不是证据**：它不参与正文的 ``[n]`` / ``来源ID=x`` 编号，只作为一致性对照注入；
   提示词同时写明"历史数字不得作为本次结论依据"（见 ``guardrail.build_system_prompt``）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import desc, insert, select, update

from ..config import get_settings
from ..core.logging import get_logger
from ..db import tables as T
from ..db.engine import Database, get_db
from ..rag.embedding import LocalEmbedding, cosine
from ..rag.text import tokenize
from .memory_extract import Candidate, content_fingerprint

log = get_logger("pyagent.agent.long_term_memory")

#: 词面覆盖 / 向量余弦 / 重要度 的权重。词面最高 —— 记忆是短的，字面重叠就是最强的相关信号。
W_LEXICAL = 0.55
W_VECTOR = 0.30
W_IMPORTANCE = 0.15

#: 时间衰减半衰期（天）。记忆会过期，但不要"到期即失效" —— 老的处置经过仍有参照价值。
HALF_LIFE_DAYS = 90.0

_EMBED_DIM = 512


@dataclass
class MemoryHit:
    """一条被召回的记忆。"""

    id: int
    scope: str
    kind: str
    topic: str
    content: str
    importance: float
    score: float = 0.0
    lexical: float = 0.0
    vector: float = 0.0
    age_days: float = 0.0
    source_analysis_id: Optional[int] = None
    updated_at: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "kind": self.kind,
            "topic": self.topic,
            "content": self.content,
            "importance": round(self.importance, 2),
            "score": round(self.score, 4),
            "lexical": round(self.lexical, 4),
            "vector": round(self.vector, 4),
            "ageDays": round(self.age_days, 1),
            "sourceAnalysisId": self.source_analysis_id,
        }


@dataclass
class WriteResult:
    written: int = 0
    superseded: int = 0
    reinforced: int = 0
    pruned: int = 0
    ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"written": self.written, "superseded": self.superseded,
                "reinforced": self.reinforced, "pruned": self.pruned, "ids": self.ids}


class LongTermMemory:
    """长期记忆服务。无状态（配置从 settings 读），可当单例。"""

    def __init__(self, db: Optional[Database] = None) -> None:
        self.db = db or get_db()
        self._emb = LocalEmbedding(_EMBED_DIM)

    def _cfg(self):
        return get_settings().agent

    def enabled(self) -> bool:
        return bool(self._cfg().long_term_memory)

    def write_enabled(self) -> bool:
        return bool(self._cfg().long_term_memory) and bool(self._cfg().long_term_write)

    # ------------------------------------------------------------------ 写入

    def write(self, candidates: Iterable[Candidate], force: bool = False) -> WriteResult:
        """落库 + 冲突消解。同一主题（``topic_key``）出现新内容时取代旧条目。

        任何一步失败都不外抛：记忆是**增强**能力，写不进去最多是"这次没记住"，
        绝不能让一次已经跑完的分析因为记忆而失败。

        :param force: 跳过"写入开关"检查（回填历史时用）。**总开关 ``long_term_memory``
            仍然生效** —— 一键关掉长期记忆就该真的什么都不写。
        """
        res = WriteResult()
        if not self.enabled():
            return res
        if not force and not self.write_enabled():
            return res
        for c in candidates:
            try:
                self._write_one(c, res)
            except Exception as e:  # noqa: BLE001
                log.warning("[Memory] 记忆写入失败（已忽略）：%s", e)
        try:
            res.pruned = self._prune(candidates)
        except Exception as e:  # noqa: BLE001
            log.warning("[Memory] 记忆容量淘汰失败（已忽略）：%s", e)
        if res.written or res.superseded or res.reinforced:
            log.info("[Memory] 长期记忆写入：新增 %d，取代 %d，强化 %d，淘汰 %d",
                     res.written, res.superseded, res.reinforced, res.pruned)
        return res

    def _write_one(self, c: Candidate, res: WriteResult) -> None:
        # 幂等闸：同一次分析对同一主题只贡献一条记忆。
        # 没有它，重复回填（或同一次分析被重跑）会在同主题上反复"覆盖"，
        # 记忆里堆出一串**本来不存在的口径变化** —— 那是假的漂移信号，
        # 比没有信号更坏：它会让人以为指标在反复波动。
        if self._already_contributed(c):
            res.reinforced += 1
            return

        old = self._active_row(c)
        if old is not None:
            if self._same_content(old, c):
                # 同一件事又发生了一次：只提升重要度（说明它反复出现），不新增行。
                imp = max(float(old.get("importance") or 0.5), float(c.importance))
                self.db.execute(
                    update(T.ai_memory).where(T.ai_memory.c.id == int(old["id"]))
                    .values(importance=imp, updated_at=datetime.now()))
                res.reinforced += 1
                return
            # 口径变了：新行先插入，再把旧行标记为被它取代（顺序反了会出现"没有有效记忆"的空窗）
            new_id = self._insert(c)
            if new_id:
                self.db.execute(
                    update(T.ai_memory).where(T.ai_memory.c.id == int(old["id"]))
                    .values(status=T.MEM_SUPERSEDED, superseded_by=new_id,
                            updated_at=datetime.now()))
                res.written += 1
                res.superseded += 1
                res.ids.append(new_id)
            return
        new_id = self._insert(c)
        if new_id:
            res.written += 1
            res.ids.append(new_id)

    def _already_contributed(self, c: Candidate) -> bool:
        """这条记忆是不是已经由"同一次分析"贡献过了（任意状态都算）。"""
        if c.source_analysis_id is None:
            return False
        row = self.db.fetch_one(
            select(T.ai_memory.c.id)
            .where(T.ai_memory.c.scope == c.scope)
            .where(T.ai_memory.c.company_id == int(c.company_id or 0))
            .where(T.ai_memory.c.user_id == int(c.user_id or 0))
            .where(T.ai_memory.c.kind == c.kind)
            .where(T.ai_memory.c.topic_key == c.topic_key)
            .where(T.ai_memory.c.source_analysis_id == int(c.source_analysis_id))
            .limit(1))
        return row is not None

    def _insert(self, c: Candidate) -> Optional[int]:
        now = datetime.now()
        return self.db.insert_id(insert(T.ai_memory).values(
            scope=c.scope, company_id=int(c.company_id or 0), user_id=int(c.user_id or 0),
            kind=c.kind, topic=(c.topic or "")[:160], topic_key=c.topic_key,
            content=c.content[:1000], importance=float(c.importance), hits=0,
            status=T.MEM_ACTIVE, source_analysis_id=c.source_analysis_id,
            source_trace_id=c.source_trace_id, created_at=now, updated_at=now,
        ))

    def _active_row(self, c: Candidate) -> Optional[Dict[str, Any]]:
        return self.db.fetch_one(
            select(T.ai_memory)
            .where(T.ai_memory.c.scope == c.scope)
            .where(T.ai_memory.c.company_id == int(c.company_id or 0))
            .where(T.ai_memory.c.user_id == int(c.user_id or 0))
            .where(T.ai_memory.c.kind == c.kind)
            .where(T.ai_memory.c.topic_key == c.topic_key)
            .where(T.ai_memory.c.status == T.MEM_ACTIVE)
            .order_by(desc(T.ai_memory.c.id)).limit(1))

    @staticmethod
    def _same_content(old: Dict[str, Any], c: Candidate) -> bool:
        """同主题之下"是不是同一件事"。

        用 :func:`content_fingerprint`（抹标点与日期、**保留数字**）：
        "流失率 8.3%" 与 "流失率 12%" 必须判为**两件事** —— 那是口径变了，
        要覆盖旧条目并留档；判成"又说了一遍"会让口径漂移被静默抹平。
        """
        return content_fingerprint(str(old.get("content") or "")) == content_fingerprint(c.content)

    def _prune(self, candidates: Iterable[Candidate]) -> int:
        """单企业容量上限：超出后把"最不重要 + 最旧"的置为 SUPERSEDED。

        不物理删除：记忆是审计材料的一部分，删了就没法回答"这条结论当时参考过什么"。
        """
        limit = int(self._cfg().long_term_max_per_company)
        if limit <= 0:
            return 0
        cids = {int(c.company_id or 0) for c in candidates}
        pruned = 0
        for cid in cids:
            rows = self.db.fetch_all(
                select(T.ai_memory.c.id)
                .where(T.ai_memory.c.company_id == cid)
                .where(T.ai_memory.c.status == T.MEM_ACTIVE)
                .order_by(desc(T.ai_memory.c.importance), desc(T.ai_memory.c.updated_at)))
            extra = rows[limit:]
            if not extra:
                continue
            for r in extra:
                self.db.execute(
                    update(T.ai_memory).where(T.ai_memory.c.id == int(r["id"]))
                    .values(status=T.MEM_SUPERSEDED, superseded_by=None,
                            updated_at=datetime.now()))
                pruned += 1
        return pruned

    # ------------------------------------------------------------------ 召回

    def recall(self, company_id: Optional[int], user_id: Optional[int],
               question: Optional[str], top_k: Optional[int] = None,
               min_score: Optional[float] = None) -> List[MemoryHit]:
        """按企业 + 用户召回相关记忆。失败返回空列表（记忆挂了不该挡住分析）。"""
        if not self.enabled() or not (question or "").strip():
            return []
        k = int(top_k if top_k is not None else self._cfg().long_term_top_k)
        floor = float(min_score if min_score is not None else self._cfg().long_term_min_score)
        if k <= 0:
            return []
        rows = self._candidate_rows(company_id, user_id)
        if not rows:
            return []
        q_terms = set(tokenize(question))
        q_vec = self._emb.embed(question)
        now = datetime.now()
        hits: List[MemoryHit] = []
        for r in rows:
            content = str(r.get("content") or "")
            h = self._score_row(r, content, q_terms, q_vec, now)
            if h.score >= floor:
                hits.append(h)
        hits.sort(key=lambda x: (-x.score, -(x.id or 0)))
        hits = self._dedupe_by_topic(hits)[:k]
        return hits

    def _candidate_rows(self, company_id: Optional[int],
                        user_id: Optional[int]) -> List[Dict[str, Any]]:
        """候选池 = 本企业记忆 + 本用户偏好。

        两个作用域混在一起排序，是为了让"这家企业的事实"和"这个人要我怎么写"
        在同一轮里竞争注入名额 —— 谁的得分高谁上，而不是固定各占一半。
        """
        cid = int(company_id or 0)
        uid = int(user_id or 0)
        stmt = select(T.ai_memory).where(T.ai_memory.c.status == T.MEM_ACTIVE)
        if cid > 0 and uid > 0:
            stmt = stmt.where(
                ((T.ai_memory.c.scope == T.MEM_SCOPE_COMPANY) & (T.ai_memory.c.company_id == cid))
                | ((T.ai_memory.c.scope == T.MEM_SCOPE_USER) & (T.ai_memory.c.user_id == uid))
            )
        elif cid > 0:
            stmt = stmt.where(T.ai_memory.c.scope == T.MEM_SCOPE_COMPANY).where(
                T.ai_memory.c.company_id == cid)
        elif uid > 0:
            stmt = stmt.where(T.ai_memory.c.scope == T.MEM_SCOPE_USER).where(
                T.ai_memory.c.user_id == uid)
        else:
            return []
        return self.db.fetch_all(stmt.order_by(desc(T.ai_memory.c.updated_at)).limit(500))

    def _score_row(self, row: Dict[str, Any], content: str, q_terms: set,
                   q_vec, now: datetime) -> MemoryHit:
        c_terms = set(tokenize(content))
        lexical = 0.0
        if q_terms:
            lexical = len(q_terms & c_terms) / float(len(q_terms))
        vector = cosine(q_vec, self._emb.embed(content))
        importance = float(row.get("importance") or 0.5)
        base = W_LEXICAL * lexical + W_VECTOR * vector + W_IMPORTANCE * importance
        age_days = _age_days(row.get("updated_at"), now)
        # 时间衰减只做温和压制（半衰期 90 天，最低保留七成）：老记忆仍可能正是要找的那条。
        decay = 0.7 + 0.3 * math.pow(0.5, age_days / HALF_LIFE_DAYS)
        return MemoryHit(
            id=int(row.get("id") or 0),
            scope=str(row.get("scope") or T.MEM_SCOPE_COMPANY),
            kind=str(row.get("kind") or T.MEM_KIND_FACT),
            topic=str(row.get("topic") or ""),
            content=content,
            importance=importance,
            score=base * decay,
            lexical=lexical,
            vector=vector,
            age_days=age_days,
            source_analysis_id=row.get("source_analysis_id"),
            updated_at=row.get("updated_at"),
        )

    @staticmethod
    def _dedupe_by_topic(hits: List[MemoryHit]) -> List[MemoryHit]:
        """同主题只留一条（已按分数降序）。"""
        seen = set()
        out: List[MemoryHit] = []
        for h in hits:
            key = (h.kind, h.topic)
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out

    def mark_used(self, hits: Iterable[MemoryHit]) -> None:
        """记一次命中：``hits`` 计数是复盘"这条记忆到底有没有被用上"的唯一凭据。"""
        now = datetime.now()
        for h in hits:
            try:
                self.db.execute(
                    update(T.ai_memory).where(T.ai_memory.c.id == int(h.id))
                    .values(hits=T.ai_memory.c.hits + 1, last_used_at=now))
            except Exception as e:  # noqa: BLE001
                log.debug("[Memory] 命中计数失败（已忽略）：%s", e)

    # ------------------------------------------------------------------ 注入

    def build_context(self, hits: List[MemoryHit], budget: Optional[int] = None,
                      with_ids: bool = True) -> str:
        """把命中的记忆拼成可注入的一段文本（超预算丢最弱的）。

        ``with_ids`` 打开时每条带 ``记忆#id``：结论被质疑时能顺着它查出处的分析记录。
        **它刻意不用 ``[n]`` 或 ``来源ID=x`` 这两种编号** —— 那两个编号属于证据，
        记忆一旦占了号，引用核对就会把"记忆里的旧数字"当成"本次证据"。
        """
        if not hits:
            return ""
        cap = int(budget if budget is not None else self._cfg().long_term_budget)
        header = (
            "【长期记忆｜历史快照，仅供一致性对照】\n"
            "用法：只用来判断“这次的说法与以前是否一致”；其中的数字是**历史值**，\n"
            "必须重新核实后才可使用，不得当作本次结论的依据，也不要为它标注来源编号。\n"
        )
        lines: List[str] = []
        used = len(header)
        for h in hits:
            tag = "用户偏好" if h.scope == T.MEM_SCOPE_USER else "企业记忆"
            suffix = f"（记忆#{h.id}）" if with_ids else ""
            line = f"- [{tag}] {h.content}{suffix}\n"
            if used + len(line) > cap:
                break
            lines.append(line)
            used += len(line)
        if not lines:
            return ""
        return header + "".join(lines)

    def diagnostics(self, hits: List[MemoryHit], injected: str) -> Dict[str, Any]:
        """进 ``answer_json.memory.longTerm``：这次用了哪几条、为什么用它们。"""
        return {
            "enabled": self.enabled(),
            #! 把生效阈值一起回传：界面上说"命中 0 条"时，用户需要知道是"确实没有"还是
            #! "被阈值挡住了"，否则会误判成功能没生效。
            "topK": int(self._cfg().long_term_top_k),
            "minScore": round(float(self._cfg().long_term_min_score), 4),
            "hits": len(hits),
            "injectedChars": len(injected or ""),
            "items": [{
                "id": h.id, "kind": h.kind, "scope": h.scope,
                "topic": h.topic,
                #! content 是给前端「本次对照了哪些历史记忆」看的：只给 topic
                #! 用户看不到到底对照了什么，"与以前是否一致"就无法人工复核。
                "content": h.content,
                "score": round(h.score, 4),
                "lexical": round(h.lexical, 4), "vector": round(h.vector, 4),
                "ageDays": round(h.age_days, 1),
                "sourceAnalysisId": h.source_analysis_id,
            } for h in hits],
        }

    # ------------------------------------------------------------------ 运维

    def list_rows(self, company_id: Optional[int] = None, user_id: Optional[int] = None,
                  include_superseded: bool = False, limit: int = 200) -> List[Dict[str, Any]]:
        stmt = select(T.ai_memory)
        if not include_superseded:
            stmt = stmt.where(T.ai_memory.c.status == T.MEM_ACTIVE)
        if company_id is not None:
            stmt = stmt.where(T.ai_memory.c.company_id == int(company_id))
        if user_id is not None:
            stmt = stmt.where(T.ai_memory.c.user_id == int(user_id))
        rows = self.db.fetch_all(stmt.order_by(desc(T.ai_memory.c.updated_at)).limit(int(limit)))
        out = []
        for r in rows:
            out.append({
                "id": int(r.get("id") or 0),
                "scope": r.get("scope"),
                "companyId": r.get("company_id"),
                "userId": r.get("user_id"),
                "kind": r.get("kind"),
                "topic": r.get("topic"),
                "content": r.get("content"),
                "importance": float(r.get("importance") or 0.5),
                "hits": int(r.get("hits") or 0),
                "status": r.get("status"),
                "supersededBy": r.get("superseded_by"),
                "sourceAnalysisId": r.get("source_analysis_id"),
                "sourceTraceId": r.get("source_trace_id"),
                "createdAt": _dt(r.get("created_at")),
                "updatedAt": _dt(r.get("updated_at")),
            })
        return out

    def stats(self, company_id: Optional[int] = None) -> Dict[str, Any]:
        rows = self.list_rows(company_id=company_id, include_superseded=True, limit=100000)
        active = [r for r in rows if r["status"] == T.MEM_ACTIVE]
        by_kind: Dict[str, int] = {}
        for r in active:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        return {
            "enabled": self.enabled(),
            "writeEnabled": self.write_enabled(),
            "active": len(active),
            "superseded": len(rows) - len(active),
            "byKind": by_kind,
            "budget": int(self._cfg().long_term_budget),
            "topK": int(self._cfg().long_term_top_k),
        }

    def delete(self, memory_id: int) -> int:
        """物理删除单条。用户明确要求"忘掉这条"时必须真的忘掉（合规要求）。"""
        return self.db.execute(T.ai_memory.delete().where(T.ai_memory.c.id == int(memory_id)))

    def clear(self, company_id: Optional[int] = None, user_id: Optional[int] = None) -> int:
        stmt = T.ai_memory.delete()
        cond = []
        if company_id is not None:
            cond.append(T.ai_memory.c.company_id == int(company_id))
        if user_id is not None:
            cond.append(T.ai_memory.c.user_id == int(user_id))
        if cond:
            stmt = stmt.where(cond[0])
            for c in cond[1:]:
                stmt = stmt.where(c)
        return self.db.execute(stmt)

    def rebuild_from_history(self, company_id: int, limit: int = 50) -> Dict[str, Any]:
        """从历史分析回填长期记忆（**零 token**）。

        为什么需要回填：功能上线时企业往往已经积累了几百条历史分析。
        不回填的话，"长期记忆"要从今天才生效，而它最有价值的恰恰是"以前发生过什么"。
        回填走的是与实时写入**完全相同**的抽取规则，所以回填出来的东西
        与"当初就开着"是一致的；顺序按分析 id 升序，保证同主题覆盖的结果也是同一个。
        """
        from .memory_extract import MemoryExtractor

        rows = self.db.fetch_all(
            select(T.ai_analysis)
            .where(T.ai_analysis.c.company_id == int(company_id))
            .order_by(T.ai_analysis.c.id).limit(int(limit)))
        scanned = len(rows)
        if not scanned:
            return {"scanned": 0, "written": 0, "superseded": 0, "reinforced": 0}

        ex = MemoryExtractor()
        agg = WriteResult()
        for r in rows:
            aj = _json_of(r.get("answer_json"))
            try:
                cands = ex.extract(
                    company_id=company_id,
                    user_id=int(r.get("user_id") or 0),
                    question=r.get("question"),
                    answer_json=aj,
                    answer_text=r.get("answer"),
                    risk_level=str((aj or {}).get("risk_level") or ""),
                    analysis_id=int(r.get("id") or 0),
                    trace_id=r.get("trace_id"),
                    degrade_level=r.get("degrade_level"),
                )
            except Exception as e:  # noqa: BLE001 - 单条失败不影响其余
                log.warning("[Memory] 回填第 %s 条抽取失败（已跳过）：%s", r.get("id"), e)
                continue
            if not cands:
                continue
            wr = self.write(cands, force=True)
            agg.written += wr.written
            agg.superseded += wr.superseded
            agg.reinforced += wr.reinforced
        out = {"scanned": scanned, "written": agg.written,
               "superseded": agg.superseded, "reinforced": agg.reinforced}
        log.info("[Memory] 回填完成：扫描 %d 条分析，新增记忆 %d，取代 %d，强化 %d",
                 scanned, agg.written, agg.superseded, agg.reinforced)
        return out


# ---------------------------------------------------------------------- 工具


def _age_days(value: Any, now: datetime) -> float:
    if isinstance(value, datetime):
        delta = now - value
        return max(0.0, delta.total_seconds() / 86400.0)
    return 0.0


def _dt(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return None if value is None else str(value)


def _json_of(raw: Any) -> Optional[Dict[str, Any]]:
    """JSON 列解析：从库里读出来是 str，内存里传进来可能是 dict。"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(str(raw))
        return v if isinstance(v, dict) else None
    except (TypeError, ValueError):
        return None
