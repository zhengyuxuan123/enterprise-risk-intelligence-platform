"""成稿复核回环（**只做复核，不做编排**）—— 对应 Java ``ReviewLoopService``。

Java 里这个类原名 ``MultiAgentOrchestrator``，但它其实从不负责"编排"——
真正的角色分工在 ``AgentRegistry`` / ``AgentRouter``（零 token 规则路由）。
名字带着 Orchestrator 又不做编排，是后来接手的人最容易踩的坑，因此改名。

它只做两件事：

* **检索员简报**：开启复核时为分析师预召回顾据（仅一次本地检索，零 token）；
* **复核员**：对成稿做 groundedness 复核，发现硬伤时带回环修订。

**成本闸门**：确定性核对（引用悬空 / 覆盖率过低）判不出问题时，
默认**不再调用模型**做复核——绝大多数成稿根本不需要多付这一次推理。
需要每次都让模型过一遍的场景，把 ``APP_AGENT_REVIEW_ALWAYS_LLM`` 打开。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..config import get_settings

log = logging.getLogger(__name__)

#: 复核意见里出现这些措辞，说明复核员认定成稿有实质问题。
SEVERE_PATTERN = re.compile(
    "编造|虚构|捏造|未提供的数据|无数据支撑|缺少依据|证据不足|数据不一致|与证据不符|凭空|未联网核实.*断言")


@dataclass
class ReviewResult:
    """复核结果。``severe=True`` 表示成稿存在必须修订的硬伤。"""

    text: str = ""
    severe: bool = False
    issues: List[str] = field(default_factory=list)
    trigger: Optional[str] = None

    def to_dict(self) -> dict:
        return {"text": self.text, "severe": self.severe,
                "issues": list(self.issues), "trigger": self.trigger}


def _pct(v: float) -> str:
    # Java 是 String.format(Locale.ROOT, "%.0f%%", v*100)
    return "%.0f%%" % (v * 100.0)


class ReviewLoopService:
    def __init__(self, llm=None, always_llm: Optional[bool] = None) -> None:
        self.llm = llm
        self._always = always_llm

    # ------------------------------------------------------------------
    # 检索员
    # ------------------------------------------------------------------

    def retrieve(self, company_id: int, question: str, top_k: int, rag) -> str:
        """召回知识库证据并产出简报（零 token）。"""
        hits = rag.search(company_id, question, top_k) if rag is not None else []
        if not hits:
            return "（检索员：知识库未检索到直接相关的文档，建议结合结构化业务数据判断）"
        b = ["（检索员简报，知识库召回 %d 篇）：" % len(hits)]
        for i, h in enumerate(hits):
            line = "  %d. %s（相关度 %.1f）：%s" % (i + 1, h.title, h.score, h.snippet)
            # 召回里可能混有结构化切片（metric:id / event:id / complaint:id），
            # 标注出来，分析师才知道这类证据可以写成「来源ID=metric:12」引用。
            #! ``document_id`` 是 dataclass **字段**（int/None），不是方法。写成 ``h.document_id()``
            #! 会抛 ``'int' object is not callable``，而外层 try/except 只打一条 warning 就吞掉 ——
            #! 结果是「检索员简报」这一整段永远走不到，多智能体召回静默失效。
            if h.ref() and h.document_id is None:
                line += "（来源ID=%s）" % h.ref()
            b.append(line)
        return "\n".join(b)

    # ------------------------------------------------------------------
    # 复核员
    # ------------------------------------------------------------------

    def review(self, draft_answer: str, evidence_summary: str) -> str:
        """对成稿与证据做复核，返回复核意见（可能为空）。"""
        from .guardrail import GuardrailService

        if self.llm is None or not self.llm.is_configured():
            return "（复核员：本地模式未启用外部模型，跳过自动复核；请确保关键结论有数据支撑。）"
        system = ("你是风控分析复核员。只依据给定证据评估分析师成稿：判断是否编造了未提供的数据、"
                  "是否遗漏重大风险、高影响动作是否提示了人工审批。用 2-4 条简练的中文给出复核意见；"
                  "若无明显问题，仅回复「复核通过」。" + GuardrailService.DOMAIN_KNOWLEDGE)
        user = ("【证据摘要】\n" + evidence_summary + "\n\n【分析师成稿】\n" + draft_answer
                + "\n\n请输出复核意见：")
        try:
            review = self.llm.complete(system, user, 0.1)
            return review.strip() if review and review.strip() else ""
        except Exception as e:  # noqa: BLE001
            log.warning("[Review] 复核调用失败: %s", e)
            return "（复核员：自动复核调用失败，已跳过）"

    def review_structured(self, draft: str, evidence_summary: str, ck: Any) -> ReviewResult:
        """结构化复核。

        **省钱要点**：严重性优先用已有的确定性信号判定 ——
        ``AnswerGroundingService.verify`` 是纯字符串比对、不消耗一个 token。
        确定性信号判不出问题且未开 ``review_always_llm`` 时，直接放行，不再付一次模型推理。
        """
        issues: List[str] = []
        trigger: Optional[str] = None

        if ck is not None:
            if getattr(ck, "web_dangling", None):
                issues.append("存在 %d 处网页引用在证据中找不到对应来源（悬空引用）"
                              % len(ck.web_dangling))
                trigger = "确定性：悬空引用"
            if getattr(ck, "kb_dangling", None):
                issues.append("存在 %d 处知识库「来源ID=」未命中实际召回"
                              % len(ck.kb_dangling))
                if trigger is None:
                    trigger = "确定性：来源ID未命中"
            cov = float(getattr(ck, "citation_coverage", 1.0) or 0.0)
            if cov > 0 and cov < 0.5:
                issues.append("引用覆盖率仅 " + _pct(cov))
                if trigger is None:
                    trigger = "确定性：引用覆盖率过低"
            for s in getattr(ck, "issues", None) or []:
                if s and str(s).strip():
                    issues.append(str(s))
        severe = trigger is not None

        # 确定性信号没抓到问题 → 默认不再烧一次模型；只有显式开启才每次复核
        if trigger is None and not self.always_llm():
            return ReviewResult("", False, issues, "确定性核对通过，已免模型复核")

        text = self.review(draft, evidence_summary) or ""
        if text.strip() and SEVERE_PATTERN.search(text):
            severe = True
            if trigger is None:
                trigger = "复核员措辞判定"
        return ReviewResult(text, severe, issues, trigger)

    def revise(self, draft: str, review: str, question: str) -> Optional[str]:
        """按复核意见修订成稿。

        只做一次、不重跑工具——重跑 ReAct 会把取数成本再付一遍，
        而复核指出的绝大多数问题（悬空引用、无据断言）根本不需要新数据就能修。

        :return: 修订后的成稿；无法修订（模型不可用 / 输出过短）时返回 None，调用方保留原文
        """
        from .guardrail import GuardrailService

        if self.llm is None or not self.llm.is_configured():
            return None
        if not draft or not draft.strip():
            return None
        system = ("你是风控分析师，正在根据复核意见修订自己的成稿。必须遵守：\n"
                  "1. 只修订复核意见指出的问题，不得推翻已有且正确的结论；\n"
                  "2. 严禁新增任何证据中不存在的数据、链接或来源编号；\n"
                  "3. 保留原有引用标注（[n] 与「来源ID=x」）与章节结构；\n"
                  "4. 复核指出的内容若证据里确实没有，就删掉该句或改成「证据未覆盖」，绝对不许编造；\n"
                  "5. 直接输出修订后的完整成稿，不要解释改了什么。"
                  + GuardrailService.DOMAIN_KNOWLEDGE)
        user = ("【原始问题】\n" + question + "\n\n【你的成稿】\n" + draft
                + "\n\n【复核意见】\n" + review + "\n\n请输出修订后的完整成稿：")
        try:
            r = self.llm.complete(system, user, 0.2)
            if not r or not r.strip():
                return None
            r = r.strip()
            # 输出异常短说明模型没理解任务，宁可用原文也不要一个残缺版本
            return None if len(r) < min(60, len(draft) * 0.4) else r
        except Exception as e:  # noqa: BLE001
            log.warning("[Review] 修订调用失败: %s", e)
            return None

    def always_llm(self) -> bool:
        if self._always is not None:
            return bool(self._always)
        try:
            return bool(get_settings().agent.review_always_llm)
        except Exception:  # noqa: BLE001
            return False
