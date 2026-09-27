"""多轮对话记忆（对应 Java ``ConversationMemory``）。

为什么需要它
------------
用户第二轮问「那投诉呢」时，没有上下文的模型只能看到「那投诉呢」五个字。
把前几轮的问答带进去，回答才能接得上。

约束
----
* ``memory_turns``：最多回看几轮（默认 6）。
* ``memory_token_budget``：注入的上下文总长上限（默认 1200 字符量级），
  超了就压缩——**长上下文不会让回答更好，只会让它更贵、更慢、更容易跑题**。
* ``memory_summary_mode``：``extractive``（默认，本地抽取式压缩，零 token）
  / ``abstractive``（交给模型做摘要，要花 token）。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import asc, desc, insert, select

from ..config import get_settings
from ..db import tables as T
from ..db.engine import Database, get_db
from .review_loop import ReviewLoopService as _ReviewLoop

log = logging.getLogger(__name__)

_SENTENCE = re.compile(r"[^。！？；\n]+[。！？；]?")


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


class ConversationMemory:
    """会话与消息的读写 + 上下文压缩。"""

    def __init__(self, db: Optional[Database] = None) -> None:
        s = get_settings().agent
        self.db = db or get_db()
        self.turns = int(s.memory_turns)
        self.budget = int(s.memory_token_budget)
        self.summary_mode = (s.memory_summary_mode or "extractive").strip().lower()

    # -- 会话 --------------------------------------------------------

    def ensure_conversation(self, user_id: Optional[int], company_id: Optional[int],
                            conversation_id: Optional[int], title: Optional[str] = None) -> Optional[int]:
        """有就复用，没有就建一条。失败返回 ``None``（记忆不是主流程，挂了就当没有）。"""
        if conversation_id:
            return int(conversation_id)
        if user_id is None:
            return None
        try:
            from datetime import datetime

            return self.db.insert_id(insert(T.ai_conversation).values(
                user_id=user_id, company_id=company_id,
                title=(title or "新的风险分析会话")[:200], created_at=datetime.now(),
                updated_at=datetime.now()))
        except Exception as e:  # noqa: BLE001
            log.warning("创建会话失败（本次不使用记忆）：%s", e)
            return None

    def recent(self, conversation_id: Optional[int], limit: Optional[int] = None) -> List[Turn]:
        n = int(limit or self.turns)
        if not conversation_id or n <= 0:
            return []
        rows = self.db.fetch_all(
            select(T.ai_message)
            .where(T.ai_message.c.conversation_id == conversation_id)
            .order_by(desc(T.ai_message.c.id)).limit(n * 2))
        rows.reverse()
        out: List[Turn] = []
        for r in rows:
            c = str(r.get("content") or "").strip()
            if c:
                out.append(Turn(str(r.get("role") or "user"), c))
        return out[-n * 2:]

    def append(self, conversation_id: Optional[int], role: str, content: str,
               answer_json: Optional[str] = None, provider: Optional[str] = None,
               tool_trace_json: Optional[str] = None) -> None:
        if not conversation_id or not content:
            return
        from datetime import datetime

        self.db.execute(insert(T.ai_message).values(
            conversation_id=conversation_id, role=role,
            content=content[:60000], tool_trace_json=tool_trace_json,
            answer_json=answer_json, provider=provider, created_at=datetime.now()))

    # -- 上下文 ------------------------------------------------------

    def build_context(self, conversation_id: Optional[int]) -> str:
        """把历史问答压成一段可注入的上下文。超预算就截断。"""
        turns = self.recent(conversation_id)
        if not turns:
            return ""
        parts: List[str] = []
        for t in turns:
            who = "用户" if t.role == "user" else "助手"
            parts.append(f"{who}：{self._compress(t.content)}")
        ctx = "\n".join(parts)
        if len(ctx) <= self.budget:
            return ctx
        # 超预算：优先保留最近几轮（越近越相关）
        kept: List[str] = []
        used = 0
        for line in reversed(parts):
            if used + len(line) > self.budget:
                break
            kept.append(line)
            used += len(line)
        kept.reverse()
        return "\n".join(kept)

    def diagnostics(self, conversation_id: Optional[int]) -> Dict[str, Any]:
        """记忆窗口的诊断（对应 Java ``HistoryWindow.diagnostics()``）。

        它进 ``answer_json.memory``，用途只有一个：某次结论为什么变了，事后能回溯。
        所以键名与 Java 逐字对齐（rawTokens / keptTokens / turns /
        compressedItems / compressed），量纲也照搬「中文 1 字≈0.75 token」。
        """
        turns = self.recent(conversation_id) if conversation_id else []
        if not turns:
            return {"rawTokens": 0, "keptTokens": 0, "turns": 0,
                    "compressedItems": 0, "compressed": False}
        raw = sum(self._estimate_tokens(t.content or "") for t in turns)
        parts = [f"{'用户' if t.role == 'user' else '助手'}：{self._compress(t.content)}"
                 for t in turns]
        kept = sum(self._estimate_tokens(p) for p in parts)
        compressed_items = sum(1 for t in turns if len(t.content or "") > 200)
        # 超预算才真的丢内容；keptTokens 报的是"最终注入的量"，不是原始累计
        if kept > self.budget:
            kept = self.budget
        return {
            "rawTokens": raw,
            "keptTokens": kept,
            "turns": len(turns),
            "compressedItems": compressed_items,
            "compressed": compressed_items > 0,
        }

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """中文 1 字≈0.75 token、英数 4 字符≈1 token（与 Java 同一个量纲）。"""
        if not text:
            return 0
        han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        return int(math.ceil(han * 0.75 + (len(text) - han) / 4.0))

    def _compress(self, content: str) -> str:
        """抽取式压缩（零 token）：只留含数字/风险关键词的句子。

        为什么不用模型摘要：一次分析本来就要调好几次模型，
        再为"回忆上一轮说了什么"多花一次调用，成本与延迟都不划算；
        而历史问答里真正有用的往往就是那几句带数字和判断的。
        """
        if len(content) <= 200:
            return content
        keep: List[str] = []
        for m in _SENTENCE.finditer(content):
            s = m.group(0).strip()
            if not s:
                continue
            if re.search(r"\d", s) or re.search(r"(风险|异常|下降|上升|投诉|超|流失|建议|结论)", s):
                keep.append(s)
            if sum(len(x) for x in keep) >= 200:
                break
        return "".join(keep) if keep else content[:200] + "…"


    # ------------------------------------------------------------------
    # 历史相似分析（原 ReviewLoopService.retrieve，归位到记忆模块）
    # ------------------------------------------------------------------

    def similar_analyses(self, company_id: Optional[int], question: Optional[str],
                         top_k: int = 3) -> List[Dict[str, Any]]:
        """取历史上相似的分析结论，作为这一轮的参考（避免前后矛盾）。"""
        if not company_id or not question:
            return []
        rows = self.db.fetch_all(
            select(T.ai_analysis.c.id, T.ai_analysis.c.question, T.ai_analysis.c.answer,
                   T.ai_analysis.c.confidence)
            .where(T.ai_analysis.c.company_id == company_id)
            .order_by(desc(T.ai_analysis.c.id)).limit(50))
        scored: List[tuple] = []
        q_terms = {c for c in re.split(r"[\s，。、？?]+", question or "") if len(c) >= 2}
        for r in rows:
            a = str(r.get("answer") or "")
            q = str(r.get("question") or "")
            hit = sum(1 for t in q_terms if t in q or t in a)
            if hit:
                scored.append((hit, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"id": r.get("id"), "question": r.get("question"),
                 "excerpt": str(r.get("answer") or "")[:300],
                 "confidence": r.get("confidence")} for _, r in scored[:top_k]]

    @staticmethod
    def append_review(draft: str, opinion: str) -> str:
        """把复核意见**附加**到草稿上（本地确定性修订，不调模型）。

        默认不开「让模型改一遍」：一次修订要重发整段正文，成本高，
        而大部分复核意见（补充依据 / 降低确定性）用追加一段来说明就够了。
        """
        if not opinion:
            return draft
        return (draft or "").rstrip() + "\n\n【复核补充】" + opinion.strip()


class ReviewLoopService:
    """**已废弃**：这个类名字叫复核回路，实际做的是"翻历史相似分析"，
    与 Java ``ReviewLoopService``（成稿复核 + 回环修订）完全不是一回事。

    真正的实现在 :mod:`app.agent.review_loop`。这里保留一个薄壳，
    把两个方法迁回它们该待的地方，避免老引用报错。
    """

    def __init__(self, db: Optional[Database] = None, llm=None) -> None:
        self.memory = ConversationMemory(db)
        self.review = _ReviewLoop(llm=llm)

    #: 迁移到 :meth:`ConversationMemory.similar_analyses`
    def retrieve(self, company_id: Optional[int], question: Optional[str],
                 top_k: int = 3) -> List[Dict[str, Any]]:
        return self.memory.similar_analyses(company_id, question, top_k)

    #: 迁移到 :meth:`app.agent.review_loop.ReviewLoopService.revise`
    def revise(self, draft: str, opinion: str, question: Optional[str] = None) -> str:
        fixed = self.review.revise(draft, opinion, question or "")
        return fixed if fixed is not None else self.memory.append_review(draft, opinion)
