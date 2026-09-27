"""Agent 编排主链路（对应 Java ``AgentService``）。

一次分析的完整链路
------------------
::

    护栏输入检查 → 路由（选专员 + 裁工具）→ 预取证椐 → 工具循环（ReAct）
    → 成稿 → 空正文自愈 → 引用核对与修补 → 护栏输出检查 → 落库 → 留痕

三条不可回退的行为约定
----------------------
1. **空正文绝不当结论**。推理型模型可能思考 token 吃满预算（``finish_reason=length``）
   而返回空正文；这里三级兜底：加预算重试 → 本地确定性报告 → 记 PARTIAL。
2. **写动作只提议不执行**。工具层 :mod:`app.agent.approvals` 只入审批队列。
3. **来源分层写死在 system prompt**：一、结论(内部) / 二、结论(外部) /
   三、建议动作 / 四、不确定性。未联网时第二节必须原样写「本次未联网核实」。

成本红线
--------
``app.ai.auto-consume`` 默认 False：后台自动任务不消耗额度，
只有用户主动点的分析才调模型。这条不要动。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from ..ai.fallback import parse_candidates
from ..ai.llm_client import ChatMessage, ChatResult, LlmClient, ToolCall, llm_call_timeout
from ..ai.tool_context import ToolContext
from ..ai.tools import RiskAgentTools, ToolSpec
from ..config import get_settings
from ..core.errors import BusinessError
from ..core.security import CurrentUser
from ..db import tables as T
from ..db.engine import Database, get_db
from ..rag.rag_service import RagService
from ..rag.web_cleaner import clean as clean_query
from ..web.search_service import WebSearchService
from .approvals import ActionApprovalService
from .budget import AgentBudget, plan_of
from .deadline import Deadline
from .grounding import AnswerGroundingService, Check
from .guardrail import GuardrailService
from .langgraph_impl import (
    AGENT_TEMPERATURE,
    ORCH_LANGGRAPH,
    ORCHESTRATORS,
    ModelChain,
    run as lg_run,
)
from .long_term_memory import LongTermMemory, MemoryHit
from .memory import ConversationMemory
from .memory_extract import MemoryExtractor
from .registry import AgentRegistry
from .report_schema import extract as extract_report
from .report_schema import render_markdown as render_report
from .resilience import Resilience
from .review_loop import ReviewLoopService
from .router import AgentRouter, describe_participants
from .tool_node import build_tool_node
from .trace import ToolRun, TraceContext, TraceRecorder

log = logging.getLogger(__name__)

DEGRADE_NONE = "NONE"
DEGRADE_PARTIAL = "PARTIAL"
DEGRADE_FULL = "FULL"

#: 小节标题两侧可能的 Markdown / 列表装饰。切标题之前必须剥掉 ——
#: 模型很爱把模板里的「一、结论（基于内部知识库与经营数据）」写成
#: 「**一、结论（基于内部知识库与经营数据）**」，本地兜底报告又写成
#: 「## 一、结论（基于内部数据）」。按裸前缀匹配的旧实现一律切不出来，
#: 后果是 **正文有 1800 字、四个小节全空**：接口 200、正文也在，
#: 只有界面上那一格是白的 —— 属于最难被发现的一类失败。
_HEADING_DECOR = " \t\u3000*#>-+·•|_"

#: 每个小节标题里必须出现的关键词。只认序号是不够的：正文里「一、」开头的
#: 句子（如「一、客户流失率处于高位」）会把小节提前截断，所以再用关键词兜一层。
#: 「四」原先因为写死成「必须含、结论」，而模板标题是「四、不确定性 / 需人工确认」，
#: 于是**不确定性这一格从来就没被填过**。
_SECTION_KEYS: Dict[str, tuple] = {
    "一": ("结论", "内部"),
    "二": ("结论", "外部"),
    "三": ("建议", "动作"),
    "四": ("不确定", "需人工", "人工确认", "结论"),
}


@dataclass
class AgentOutcome:
    """一次分析的产出（对应 Java ``AgentOutcome``）。"""

    analysis_id: Optional[int] = None
    trace_id: str = ""
    content: str = ""
    answer_json: Optional[Dict[str, Any]] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    tool_trace: List[str] = field(default_factory=list)
    confidence: str = "中"
    degrade_level: str = DEGRADE_NONE
    degrade_reason: Optional[str] = None
    duration_ms: int = 0
    model: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class AgentService:
    """编排服务。所有依赖都可注入；``llm`` 为 ``None`` 时走本地规则兜底。"""

    def __init__(
        self,
        llm: Optional[LlmClient] = None,
        tools: Optional[RiskAgentTools] = None,
        rag: Optional[RagService] = None,
        web: Optional[WebSearchService] = None,
        db: Optional[Database] = None,
        guardrail: Optional[GuardrailService] = None,
        router: Optional[AgentRouter] = None,
        registry: Optional[AgentRegistry] = None,
        approvals: Optional[ActionApprovalService] = None,
        grounding: Optional[AnswerGroundingService] = None,
        memory: Optional[ConversationMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        review_loop: Optional[ReviewLoopService] = None,
        cache: Optional[Dict[str, Any]] = None,
        resilience: Optional[Resilience] = None,
        scope: Optional[Callable[[Optional[int]], bool]] = None,
        orchestrator: Optional[str] = None,
        lg_model_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        s = get_settings()
        self.db = db or get_db()
        self.llm = llm
        self.rag = rag
        self.web = web
        self.tools = tools
        self.guardrail = guardrail or GuardrailService()
        self.registry = registry or AgentRegistry()
        self.router = router or AgentRouter(self.registry, s.web_search.always_offer)
        self.approvals = approvals or ActionApprovalService(self.db)
        self.grounding = grounding or AnswerGroundingService()
        self.memory = memory or ConversationMemory(self.db)
        #: 长期记忆（跨会话）。与 self.memory（会话内短期）独立计预算、独立诊断。
        self.long_term = long_term or LongTermMemory(self.db)
        self._memory_extractor = MemoryExtractor()
        self.review_loop = review_loop or ReviewLoopService(llm=llm)
        self.cache: Dict[str, Any] = cache if cache is not None else {}
        self.resilience = resilience or Resilience(
            threshold=s.agent.circuit_threshold,
            cooldown_seconds=s.agent.circuit_cooldown_seconds,
            single_flight=s.agent.single_flight,
            wait_seconds=s.agent.single_flight_wait_seconds,
        )
        #: 数据权限判定回调；``None`` 表示不做越权检查（批处理 / 评估脚本）
        self.scope = scope
        self.max_iterations = int(s.agent.max_iterations)
        self.tool_timeout = int(s.agent.tool_timeout_seconds)
        self.tool_retry = int(s.agent.tool_retry)
        self.retry_max_tokens = int(s.agent.retry_max_tokens)
        self.prefetch_enabled = bool(s.agent.prefetch_tools)
        #: 是否让模型在正文之后交一份**结构化回执**（``APP_AGENT_STRUCTURED``）。
        #: 开 = 界面四格用模型自己交出的字段；关 = 退回"从正文切"（旧行为）。
        #: 这个配置项以前**定义了却没人读**，等于开关失灵，这里把它接上。
        self.structured_output = bool(s.agent.structured_output)
        self.citation_repair = bool(s.agent.citation_repair)
        self.review_revise = bool(s.agent.review_revise)
        self.cache_enabled = bool(s.agent.cache_enabled)
        self.cache_ttl = int(s.agent.cache_ttl_seconds)
        #: 「到点得不到结果就降级」的总开关（``APP_AGENT_DEADLINE_ENABLED``）。
        #: 关掉之后 :class:`Deadline` 恒为 RUNNING，全链路回到旧行为。
        self.deadline_enabled = bool(s.agent.deadline_enabled)
        #: 覆盖档位自带挂钟预算的秒数；0 = 跟随档位。
        self.deadline_seconds = int(s.agent.deadline_seconds or 0)
        #: 同一轮多个工具是否并行执行。两条编排共用这条路径，所以都受益。
        self.parallel_tools = bool(getattr(s.agent, "parallel_tools", True))
        self.parallel_tools_max = max(1, int(getattr(s.agent, "parallel_tools_max", 4) or 4))
        #: 编排实现。自研 legacy 已删除，这里**只认 langgraph**；
        #: 未知值（拼错单词、旧环境变量残留）一律回退到它 —— 不该让 AI 整体不可用，
        #: 但要吭一声，否则"配了 legacy 却还在跑"会被当成生效而没人发现。
        _orch = (orchestrator if orchestrator is not None
                 else str(s.agent.orchestrator or "")).strip().lower()
        self.orchestrator = _orch if _orch in ORCHESTRATORS else ORCH_LANGGRAPH
        if _orch and _orch not in ORCHESTRATORS:
            log.warning("[Agent] APP_AGENT_ORCHESTRATOR=%r 已不存在（自研编排已删除），"
                        "按 langgraph 处理", _orch)
        #: 仅供测试注入假的 LangGraph 模型，生产路径恒为 None（走 build_chain）
        self._lg_model_factory = lg_model_factory
        self._lock = threading.RLock()

    # -- 对外 --------------------------------------------------------

    def analyze(self, cid: Optional[int], question: Optional[str], top_k: int = 5,
                session_id: Optional[int] = None, use_multi_agent: bool = True,
                user_id: Optional[int] = None,
                depth: Optional[str] = None,
                force_web: bool = False) -> AgentOutcome:
        key = self._single_flight_key(user_id, cid, question, depth, force_web)
        return self.resilience.guards(key, cid, user_id,
                                      lambda: self._run(cid, question, top_k, session_id,
                                                        use_multi_agent, user_id,
                                                        None, None, None,
                                                        depth=depth, force_web=force_web))

    def analyze_stream(self, cid: Optional[int], question: Optional[str], top_k: int = 5,
                       session_id: Optional[int] = None, use_multi_agent: bool = True,
                       user_id: Optional[int] = None,
                       token_sink: Optional[Callable[[str], None]] = None,
                       stage_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
                       meta_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
                       depth: Optional[str] = None,
                       force_web: bool = False) -> AgentOutcome:
        """流式分析。``stage_sink`` 推阶段、``meta_sink`` 推统计证据（不等正文）。"""
        key = self._single_flight_key(user_id, cid, question, depth, force_web)
        return self.resilience.guards(key, cid, user_id,
                                      lambda: self._run(cid, question, top_k, session_id,
                                                        use_multi_agent, user_id,
                                                        token_sink, stage_sink, meta_sink,
                                                        depth=depth, force_web=force_web))

    # -- 主链路 ------------------------------------------------------

    def _run(self, cid, question, top_k, session_id, use_multi_agent, user_id,
             token_sink, stage_sink, meta_sink,
             depth=None, force_web: bool = False) -> AgentOutcome:
        started = time.time()
        # ai_analysis.user_id 是 NOT NULL：拿不到用户会让整条记录写不进去，
        # 而"写不进去"在界面上的表现只是"历史里没有这条"，极难联想到是缺了用户 id。
        if user_id is None:
            user_id = CurrentUser.user_id()
        # 调用方（网关/前端）在 X-Trace-Id 里带了号就一路复用，否则每次分析新开一个。
        trace_id = (TraceContext.current() if TraceContext.is_inherited()
                    else TraceContext.new())
        rec = TraceRecorder(trace_id=trace_id, company_id=cid, user_id=user_id,
                            question=question, db=self.db)
        out = AgentOutcome(trace_id=trace_id)

        self._stage(stage_sink, "护栏检查")
        try:
            self.guardrail.input_check(question)
        except Exception as e:  # noqa: BLE001
            out.degrade_level = DEGRADE_FULL
            out.degrade_reason = f"输入未通过安全护栏：{e}"
            out.content = f"该问题已被安全护栏拦截：{e}"
            out.confidence = "低"
            rec.note_degrade(DEGRADE_FULL, out.degrade_reason)
            rec.flush(None)
            return self._finish(out, rec, started)

        # -- 数据权限：AI 不能绕过当前用户的数据权限 --
        # 这道闸必须在**任何**取数与模型调用之前，否则越权问题会一路走到模型里才发现。
        if self.scope is not None and not self.scope(cid):
            out.degrade_level = DEGRADE_FULL
            out.degrade_reason = "AI不能绕过当前用户数据权限"
            out.content = "该问题已被安全护栏拦截：AI不能绕过当前用户数据权限"
            out.confidence = "低"
            rec.note_degrade(DEGRADE_FULL, out.degrade_reason)
            rec.flush(None)
            return self._finish(out, rec, started)

        # -- 企业存在性：不存在就必须明确说"不存在"，不能让模型对着空档案编 --
        # 这也是零 token 验收护栏分支的抓手：请求 999999，放行→「企业不存在」，拦截→「已被安全护栏拦截」。
        company = self._load_company(cid) if cid is not None else None
        if cid is not None and company is None:
            # Java 是 ``throw new BusinessException("企业不存在")`` → HTTP 400。
            # 这里原先返回 200 + 一句降级正文：接口不报错，但前端拿不到"这是个错误"的信号，
            # 于是会把「企业不存在」当成一次**成功的分析**展示给用户。
            out.degrade_level = DEGRADE_FULL
            out.degrade_reason = f"企业不存在：{cid}"
            rec.note_degrade(DEGRADE_FULL, out.degrade_reason)
            rec.flush(None)
            raise BusinessError(f"企业不存在：{cid}", 400)

        # 工具上下文只依赖请求本身，提前建好：命中缓存时也要用它推 meta 与重跑引用核对。
        ctx = ToolContext(company_id=cid, top_k=top_k, user_id=user_id, question=question,
                          max_web_calls=get_settings().web.max_calls)

        # -- 缓存 --
        cache_key = self._cache_key(user_id, cid, question, use_multi_agent, top_k,
                                    depth, force_web)
        if self.cache_enabled:
            hit = self._cache_get(cache_key)
            if hit is not None:
                return self._serve_cached(hit, cid, user_id, question, session_id, ctx,
                                          rec, token_sink, stage_sink, meta_sink, started)

        # 深度档位必须来自**本次请求**，而不是配置默认值。
        # 早先这里固定读 default_depth，导致前端「快答 / 标准 / 深度」三个按钮形同虚设：
        # 选「快答」照样跑标准档（多花两轮模型调用、正文上限翻倍、等更久），
        # 评估里的 liveDepth 也被一并吞掉 —— 那 3 条 quick 用例拿标准档去撞
        # expectMaxLlmCalls=1，必然假失败。
        budget = AgentBudget.start(depth or get_settings().agent.default_depth)

        # -- 时间预算：到点得不到结果就降级 --
        # 起点直接取 budget 的（同一个钟）：各起一个钟会出现"预算说没到点、
        # deadline 说到点了"，而降级理由一旦自相矛盾，排障第一反应就是不信它。
        deadline = (Deadline.for_plan(budget.plan.time_budget_ms,
                                      started_at_ms=budget.started_at_ms)
                    if self.deadline_enabled else Deadline(0))
        # 挂到工具上下文上：慢工具（尤其联网）要在执行前就能看到还剩多少时间。
        setattr(ctx, "deadline", deadline)

        # -- 路由：零 token 规则路由，按「证据来自哪里」切分 --
        self._stage(stage_sink, "路由")
        route = self.router.route(question,
                                  allow_web=self.web is not None and self.web.is_enabled(),
                                  force_web_requested=force_web)
        #! 这里原先是 getattr(route, "participants", []) —— Route 上根本没有 participants
        #! 字段（叫 agents），于是留痕里的参与名单**恒为空列表**，而报告/审计页看起来
        #! 一切正常（那一格本来就是空的）。改用 describe_participants 后既有名单也有判据。
        rec.note_route({"participants": describe_participants(route.agents, route),
                        "openedTools": route.opened_tools,
                        "needsWeb": route.needs_web,
                        "forceWeb": route.force_web,
                        "multiAgent": use_multi_agent})

        specs = self._pick_specs(route, budget)
        hint = self._tool_hint(specs, budget, force_web=route.force_web)

        system_prompt = self.guardrail.build_system_prompt(
            company, use_multi_agent, force_web=route.force_web)
        rec.prompt_chars = len(system_prompt) + len(question or "") + len(hint)

        messages: List[ChatMessage] = [ChatMessage.system(system_prompt)]
        ctx_brief = ""
        if session_id:
            ctx_brief = self.memory.build_context(session_id)
            if ctx_brief:
                messages.append(ChatMessage.system("以下是同一会话中此前的问答摘要，仅供保持上下文一致：\n" + ctx_brief))

        # -- 长期记忆（跨会话）：只作为"一致性对照"注入，不是证据 --
        # 独立预算、独立诊断。它**不进 evidence 列表**，因此不会占用 [n] / 来源ID 编号 ——
        # 一旦记忆参与编号，引用核对就会把"记忆里的旧数字"验成本次证据，
        # 那等于给"编造数字"开了一条合法通道。
        lt_hits: List[MemoryHit] = []
        lt_ctx = ""
        try:
            lt_hits = self.long_term.recall(cid, user_id, question)
            lt_ctx = self.long_term.build_context(lt_hits)
        except Exception as e:  # noqa: BLE001 - 记忆是增强项，召回失败不能挡住分析
            log.warning("[Agent] 长期记忆召回失败（已忽略）：%s", e)
        if lt_ctx:
            messages.append(ChatMessage.system(lt_ctx))
            try:
                self.long_term.mark_used(lt_hits)
            except Exception as e:  # noqa: BLE001
                log.debug("[Agent] 记忆命中计数失败（已忽略）：%s", e)

        user_content = (question or "") + hint
        # 多智能体：检索员先召回知识库证据，作为上下文提供给分析师（零 token）
        if use_multi_agent and self.rag is not None:
            try:
                brief = self.review_loop.retrieve(cid, question, top_k, self.rag)
                if brief:
                    user_content += "\n【检索员简报】\n" + brief
            except Exception as e:  # noqa: BLE001 - 简报失败不能挡住主链路
                log.warning("[Agent] 检索员简报失败: %s", e)
        messages.append(ChatMessage.user(user_content))

        # （ctx 已在上面提前构造：工具上下文只依赖请求本身，不依赖这一段的中间变量）

        # -- 预取：本地毫秒级，省掉"模型决定调什么工具"那一整轮往返 --
        evidence: List[Dict[str, Any]] = []
        tool_trace: List[str] = []
        model_specs = specs
        if self.prefetch_enabled and self.tools is not None:
            self._stage(stage_sink, "预取证椐")
            pre = self._prefetch(
                specs, route, question, budget, ctx, rec, tool_trace,
                limit_override=(len(specs) if token_sink is not None else None),
            )
            evidence.extend(pre)
            if meta_sink:
                self._emit_meta(meta_sink, ctx, evidence)

            # Prefetch only saves a model round-trip when the model can see the
            # fetched evidence. The old path collected it for auditing but left
            # it out of the prompt, so the model called the same tools again.
            if pre:
                messages.append(ChatMessage.system(
                    "以下证据已由平台在本轮预取，可直接据此成稿；不要重复调用已完成的工具：\n"
                    + self._evidence_summary(pre, tool_trace)
                ))

                # For SSE requests, prefetch covers every selected tool. Once all
                # tools produced usable evidence, remove tool declarations so the
                # first model round is guaranteed to be the final, streamable text.
                succeeded = self._successful_prefetch_tools(pre)
                if token_sink is not None and all(s.name in succeeded for s in specs):
                    model_specs = []

        # -- 工具循环 --
        model_streamed = False

        def stream_final_token(piece: str) -> None:
            """Forward the model's final-answer round and remember that it streamed."""
            nonlocal model_streamed
            if not piece or token_sink is None:
                return
            model_streamed = True
            token_sink(piece)

        if self.llm is not None and self.llm.is_configured() and not deadline.expired:
            self._stage(stage_sink, "模型推理")
            # LangGraph only forwards the round that cannot call tools (or buffers
            # a round until it knows no tool was requested), so this is verified
            # as final-answer text rather than an intermediate reasoning draft.
            loop = self._dispatch_loop(
                messages, model_specs, ctx, budget, evidence, tool_trace, rec,
                trace_id, token_sink=stream_final_token if token_sink else None,
                deadline=deadline,
            )
            # Java 的 ``provider`` 是 ``ai.name()`` = ``"openai-compatible:" + model``。
            # 前端「模型服务」卡片按这个串显示来源，少了前缀会显示成"未知底座"。
            out.model = "openai-compatible:" + self._effective_model(loop, self.llm)
        else:
            loop = None
            if deadline.expired:
                # 预算在模型介入之前就见底了（通常是慢工具/Web 检索吃掉的）。
                # 这时**不能**再开一轮注定来不及的模型调用 —— 直接走零 token 兜底。
                out.degrade_level = DEGRADE_PARTIAL
                out.degrade_reason = deadline.expire_reason()
                log.warning("[Agent] 时间预算在模型推理前即耗尽：%s", deadline.to_dict())

        # -- 成稿 --
        draft = ""
        if loop is not None and loop.content:
            draft = loop.content
        if not draft.strip() and not deadline.expired:
            self._stage(stage_sink, "空正文自愈")
            draft = self._retry_final_answer(messages, ctx, evidence, rec, deadline)
        if not draft.strip():
            self._stage(stage_sink, "本地确定性报告")
            draft = self._local_report(company, question, evidence, ctx)
            out.degrade_level = DEGRADE_PARTIAL
            # 到点的降级理由优先：它比"模型未产出正文"准确得多 ——
            # 后者会让人以为模型坏了，实际是我们主动不再等它。
            out.degrade_reason = (deadline.expire_reason() if deadline.expired
                                  else rec.degrade_reason or "模型未产出正文，已生成本地确定性报告")
        rec.note_degrade(out.degrade_level, out.degrade_reason)

        # -- 结构化回执：用模型自己交出的四节结构，替掉"从正文切"--
        # 必须在引用核对**之前**剥掉回执块：回执里也会出现 [n] / 来源ID=x 的字样，
        # 带着它核对等于把"模型抄了一遍的引用"计成正文引用，引用数凭空翻倍。
        report = None
        if self.structured_output and draft.strip():
            report, draft = extract_report(draft)
            if report is not None and not draft.strip():
                # 模型只交了结构、没写正文（少见但会发生）：把结构渲染回四节正文，
                # 用户拿到的仍然是一份完整报告，而不是一句"分析失败"。
                draft = render_report(report)

        # -- 引用核对与修补 --
        self._stage(stage_sink, "引用核对")
        check = self.grounding.verify(draft, evidence)
        repair = None
        if self.citation_repair and not check.ok:
            repair = self.grounding.repair(draft, evidence)
            draft = repair.answer
            check = self.grounding.verify(draft, evidence)
        rec.note_grounding(check.to_dict())

        # -- 多智能体：复核员环节（带回环 —— 发现硬伤就改，而不是只在文末记一笔）--
        review_revised = False
        rr = None
        if use_multi_agent:
            rr = self.review_loop.review_structured(
                draft, self._evidence_summary(evidence, tool_trace), check)
            rec.note_review(rr.trigger, False)
            # 到点就不再"改稿"：那是一次额外的模型调用，而正文已经成型 ——
            # 为了一段"复核意见"再花掉几十秒，正是用户等不起的那部分。
            if (rr.severe and self.review_revise and out.degrade_level == DEGRADE_NONE
                    and self.llm is not None and self.llm.is_configured()
                    and not deadline.expired):
                fixed = self.review_loop.revise(draft, rr.text, question or "")
                if fixed and fixed.strip():
                    draft = fixed
                    review_revised = True
                    rec.note_review(rr.trigger, True)
                    # 修订后引用可能位移，必须重新核对一次
                    check = self.grounding.verify(draft, evidence)
                    rec.note_grounding(check.to_dict())
            if rr.text and rr.text.strip():
                tag = "【复核意见】（已据此修订成稿）" if review_revised else "【复核意见】"
                # 「复核通过」是简短语，紧贴标题即可；真正的意见另起一行才好读
                sep = "" if "复核通过" in rr.text else "\n"
                draft = (draft or "").rstrip() + "\n\n" + tag + sep + rr.text.strip()

        # -- 护栏输出检查 --
        flags = self.guardrail.output_check(draft)
        rec.note_guardrail(flags)
        if flags and self.guardrail.is_severe(flags):
            draft = (draft or "").rstrip() + "\n\n【安全提示】" + "；".join(flags)
            out.confidence = "低"

        # -- 证据拆分：只把正文引用过的来源交给前端 --
        web_used, web_unused, all_web = self._split_web_sources(draft, evidence)
        kb_sources = self._collect_knowledge(evidence)

        out.content = draft
        out.evidence = evidence
        out.tool_trace = tool_trace
        # -- 结构化答案：键集合与 Java ``AgentService`` 逐项对齐 --
        # 这一组键是**前端契约**：分工卡片读 agents/agent_tools/agent_routing，
        # 风险等级卡片读 risk_level，"需人工复核"标记读 needs_human_review。
        # 早先只填了 14 个键（Java 27 个），转发之后这些卡片会整块空白 ——
        # 接口照样 200、正文照样有字，属于最不容易被发现的那类迁移失败。
        # 四节内容**优先取模型交出的结构化回执**，拿不到才退回切分。
        # 切分是移植期 Java 的做法，认不出加粗标题 / "1. 结论" / "### 一、总结"
        # 这类写法，而且失败是**静默**的（正文在、接口 200，只有那一格是白的）。
        # 回执里某个字段为空也不硬覆盖 —— 退回切分的结果，宁可少一点也不要空一格。
        if report is not None:
            conclusion_internal = report.conclusion_internal or self._section(draft, "一")
            conclusion_web = report.conclusion_web or self._section(draft, "二")
            uncertainties = report.uncertainties_text() or self._section(draft, "四")
            recs_internal = report.recs_as_dicts(True) or self._recs(draft, evidence, True)
            recs_web = report.recs_as_dicts(False) or self._recs(draft, evidence, False)
            structure_source = "structured-receipt"
        else:
            recs_internal = self._recs(draft, evidence, True)
            recs_web = self._recs(draft, evidence, False)
            conclusion_internal = self._section(draft, "一")
            conclusion_web = self._section(draft, "二")
            uncertainties = self._section(draft, "四")
            structure_source = "markdown-split"
        # 风险等级：规则判定优先（它来自真实数据，比模型自评可信）；
        # 规则没判出来时才用模型交的值，且必须落在白名单里 —— 模型自创一个
        # "CRITICAL" 会让前端那张卡片显示不出来。
        risk_level = ctx.rule_risk_level()
        if not risk_level and report is not None:
            cand = (report.risk_level or "").strip().upper()
            if cand in ("HIGH", "MEDIUM", "LOW", "UNKNOWN"):
                risk_level = cand
        flat_recs = [str(x.get("action") or "") for x in recs_internal] + \
                    [str(x.get("action") or "") for x in recs_web]
        degraded = out.degrade_level != DEGRADE_NONE

        aj: Dict[str, Any] = {
            "headline": ((report.headline.strip() if report is not None else "")
                         or self._headline(draft)),
            "conclusion_internal": conclusion_internal,
            "conclusion_web": conclusion_web,
            "recommendations_internal": recs_internal,
            "recommendations_web": recs_web,
            "source_split": {"internal": len(kb_sources), "web": len(web_used),
                             "webUnused": len(web_unused)},
            "grounding_check": check.to_dict(),
            "grounding_summary": self.grounding.summarize(check),
            "web_sources": web_used,
            "web_sources_unused": web_unused,
            "web_trails": ctx.web_trails(),
            "kb_sources": kb_sources,
            "degrade_level": out.degrade_level,
            "degrade_reason": out.degrade_reason,
            # ↓↓↓ 与 Java 对齐的补充键 ↓↓↓
            "conclusion": conclusion_internal or conclusion_web,  # 向后兼容：旧前端读它
            "recommendations": [x for x in flat_recs if x],
            "risk_level": risk_level or "UNKNOWN",
            #: 四节内容的来源。**审计价值**：它回答"这次界面上的结论是模型交的，
            #: 还是我们从正文里切出来的" —— 两者不一致时这是第一手的判据。
            "structure_source": structure_source,
            "uncertainties": uncertainties,
            "needs_human_review": bool(degraded or not check.ok),
            "degraded": degraded,
            "evidence": evidence,
            "citation_required": bool(web_used),
            "agents": describe_participants(route.agents, route),
            "agent_tools": list(route.tools or []),
            "agent_routing": "on" if use_multi_agent else "off",
        }
        if degraded and out.degrade_reason:
            aj["degraded_reason"] = out.degrade_reason
        if ctx.rag_diagnostics:
            aj["rag"] = ctx.rag_diagnostics
        try:
            mem_diag = self.memory.diagnostics(session_id)
            # 长期记忆与短期记忆在同一格里各占一个键：界面上一眼能分清
            # "这是本轮会话带进来的上下文"还是"这是跨会话沉淀下来的东西"。
            try:
                mem_diag["longTerm"] = self.long_term.diagnostics(lt_hits, lt_ctx)
            except Exception as e:  # noqa: BLE001
                log.debug("[Agent] 长期记忆诊断失败（已忽略）：%s", e)
            aj["memory"] = mem_diag
        except Exception:  # noqa: BLE001 - 诊断是加分项，不能因此让整轮分析失败
            aj["memory"] = {"turns": 0, "compressed": False}
        if repair is not None and getattr(repair, "repairs", None):
            reps = repair.repairs
            aj["grounding_repair"] = reps
            aj["grounding_repaired_count"] = len(reps)
        if rr is not None and getattr(rr, "trigger", None):
            aj["review_trigger"] = rr.trigger
        if review_revised:
            aj["review_revised"] = True
        out.answer_json = aj
        out.diagnostics = {
            "traceId": trace_id,
            "route": rec.route,
            "retrieval": (ctx.rag_diagnostics or {}),
            "webTrails": ctx.web_trails(),
            "stats": ctx.stats_snapshot(),
            "riskLevel": ctx.rule_risk_level(),
            "guardrailFlags": flags,
            # 「到点降级」的三态与余量。排障时第一眼要知道
            # "这次 PARTIAL 是环境慢，还是我们主动掐的"。
            "deadline": deadline.to_dict(),
        }

        # Local fallback and providers without a streaming implementation still
        # need visible output. A genuinely streamed model answer must not be sent
        # again here; ``done.answer`` remains the authoritative post-processed text.
        if token_sink and not model_streamed:
            self._stage(stage_sink, "输出最终报告")
            self._push_content(draft, token_sink)

        # Finalize timing before persistence. Persisting first wrote the dataclass
        # default (0) to every ai_analysis.duration_ms row.
        self._finish(out, rec, started)

        # -- 落库 + 留痕 --
        out.analysis_id = self._persist(out, cid, user_id, question, session_id, check)
        rec.flush(out.analysis_id)
        # -- 沉淀长期记忆 --
        # 放在落库之后，是为了让每条记忆都带得上 ``source_analysis_id``：
        # "这条记忆凭什么存在"必须能一路查到当初那次分析，否则遗忘与纠错都无从下手。
        write_info = self._remember(cid, user_id, question, out)
        if write_info:
            out.diagnostics["memoryWrite"] = write_info
        if self.cache_enabled:
            self._cache_put(cache_key, out)
        return out

    # -- 工具循环 ----------------------------------------------------

    @dataclass
    class _LoopResult:
        content: str = ""

    # -- 编排入口：LangGraph 是唯一实现 -------------------------------------- #
    #
    # 自研 ReAct 循环（``_tool_loop``）已于 2026-09-23 删除，理由见 langgraph_impl 的
    # 模块文档。保留备份在 D:\backup\pyagent-legacy-orchestrator-20260923\。
    #
    # 删除前的验收：把全量 pytest 强制切到 langgraph 编排，**564 条全绿**
    # （这套用例原本就是为 legacy 写的全链路用例）。也就是说新编排在功能等价性上
    # 已经不需要 legacy 兜底，继续并存只会让每条改动都要维护两份实现。

    def _dispatch_loop(self, messages, specs, ctx, budget, evidence, tool_trace, rec,
                       trace_id: str = "", token_sink=None, deadline=None):
        """编排入口。**只有 LangGraph 一条实现**，这个形状留着是为了调用点稳定。

        以前这里按 ``APP_AGENT_ORCHESTRATOR`` 在两套实现里挑一套。删除 legacy 之后
        它变成纯粹的转发，但不建议直接改成调用点写 ``_langgraph_loop`` ——
        流式那一路（``analyze_stream``）也要过这里，入口统一才不会漏接招牌参数。
        """
        return self._langgraph_loop(messages, specs, ctx, budget, evidence, tool_trace, rec,
                                    trace_id, token_sink, deadline)

    @staticmethod
    def _effective_model(loop, llm: Optional[LlmClient] = None) -> str:
        """取本次真正生效的模型名（langgraph 分支可能是轮换后的候选）。"""
        used = getattr(loop, "model", None)
        if used:
            return used
        if llm is not None:
            try:
                return llm.get_model() or ""
            except Exception:  # noqa: BLE001 - provider display must not break analysis
                return ""
        return ""

    def _langgraph_loop(self, messages, specs, ctx, budget, evidence, tool_trace, rec,
                        trace_id: str = "", token_sink=None, deadline=None):
        """LangGraph 版工具循环（**唯一实现**）。

        职责是"把回合交给 StateGraph 的条件边驱动"，而工具执行、预算闸门、
        证据池、留痕全部走 ``_run_tool_calls`` —— **不允许**为 LangGraph 单开一条
        工具执行路径，否则权限校验与证据溯源会被绕过。

        ``token_sink`` 走**成稿那一轮**边生成边推；中间轮先攒着，本轮结束再看要不要补推
        """
        call_cache: Dict[str, str] = {}
        tool_usage: Dict[str, int] = {}
        last_model = [self.llm.get_model() if self.llm is not None else ""]
        # 预算未接通（或假客户端没这个属性）时按"不收紧"处理。
        _default_call_timeout = float(getattr(self.llm, "chat_timeout_seconds", 0) or 0)

        # 硬截止：进图之前就已经没时间了 —— 一次调用都不发。
        if deadline is not None and deadline.expired:
            log.warning("[Agent] 时间预算耗尽，LangGraph 不再进图：%s", deadline.to_dict())
            return self._LoopResult("")

        #: 工具节点：**官方 ToolNode（或并行变体）+ wrap_tool_call 挂闸门**。
        #: 以前这里是一个自研闭包 ``dispatch``，工具身份靠字符串、参数不过 Pydantic、
        #: 错误处理自己写。换成官方节点后，闸门/留痕/预算挂在它公开的钩子上，
        #: 执行仍然走 ``execute_with_meta``（权限闸门在里面）。
        tool_node = build_tool_node(
            ctx=ctx, tools=self.tools, budget=budget, deadline=deadline,
            recorder=rec, tool_usage=tool_usage, call_cache=call_cache,
            evidence=evidence, tool_trace=tool_trace,
            parallel=bool(self.parallel_tools),
            parallel_max=max(2, int(self.parallel_tools_max or 2)),
        )

        def stop_check():
            """图每轮开跑前问一次：还让不让继续。

            返回非 None 即收敛（不带工具 + 本轮结束）。与自研分支同一套判据：
            预算先判（工具配额 / 挂钟），再判时间预算。
            """
            why = budget.tool_restriction_reason()
            if why:
                return why
            if deadline is None:
                return None
            if deadline.expired:
                return deadline.expire_reason()
            if deadline.converging:
                return deadline.converge_hint()
            return None

        # 进来时就已经在收敛区：把理由直接塞进消息里。
        # ⚠️ 已知差异：图跑到一半才到点的那一次，LangGraph 只做"不给工具"，
        #    来不及把提示语插进已经在跑的消息序列 —— 自研分支是每轮都能插的。
        #    后果只是"模型不知道为什么被收手"，正文照出，可接受。
        if deadline is not None and (deadline.converging or deadline.expired):
            messages = list(messages) + [ChatMessage.system(deadline.converge_hint())]

        chain = None
        if self._lg_model_factory is not None:
            # 测试注入：模型名仍从配置取（factory 只决定"模型名 → 模型对象"怎么建），
            # 配置里没写模型时给个占位名，避免 models 为空导致整条链表不可用。
            _s = get_settings()
            models = [m for m in [_s.ai.model, *parse_candidates(_s.ai.model_candidates)] if m]
            chain = ModelChain(
                base_url=_s.ai.base_url, api_key=_s.ai.api_key,
                model=models[0] if models else lg.INJECTED_MODEL,
                candidates=models[1:],
                model_factory=self._lg_model_factory,
                # 与自研 client 同源：测试注入的假工厂不看这个参数，但别让默认漂移。
                thinking=_s.ai.thinking_type or "disabled",
            )
        try:
            content, used = lg_run(
                messages=messages,
                specs=specs,
                tool_node=tool_node,
                max_rounds=max(1, self.max_iterations),
                thread_id=trace_id or "default",
                model_chain=chain,
                stop_check=stop_check,
                token_hook=token_sink,
                #! 每轮重新开始算：预算是在流逝的，第 3 轮能用的时间必然少于第 1 轮。
                #! 传进去的是**函数**而不是一个算好的数，让它每次都重新求值。
                timeout_fn=(None if deadline is None else
                            lambda: deadline.call_timeout(_default_call_timeout)),
                #! 默认关：thread_id 用的是每次唯一的 trace_id，快照只写不读。
                #! 开了就是白付 I/O —— 实测这一项能占到端到端耗时的相当一部分。
                use_checkpoint=bool(get_settings().agent.langgraph_checkpoint_enabled),
            )
        except Exception as e:  # noqa: BLE001 - 编排层失败要降级到已有兜底，不能让请求 500
            log.warning("[Agent] LangGraph 编排失败，转空正文兜底：%s", e)
            rec.note_degrade(DEGRADE_PARTIAL, f"LangGraph 编排失败：{e}")
            return self._LoopResult("")
        if used:
            last_model[0] = used
            rec.note_llm(used, None)
            rec.iterations += 1
        res = self._LoopResult(content or "")
        res.model = last_model[0]
        return res

    def _run_tool_calls(self, calls, ctx, budget, call_cache, tool_usage,
                        evidence, tool_trace, rec, deadline: Optional[Deadline] = None) -> None:
        """执行一轮里的全部工具调用。

        **预算判定在本方法里串行完成，工具执行可以并行**（见 :meth:`_execute_planned`）。
        这条顺序不能反：预算闸门如果被并发地穿透，"最多 N 次"就不再是上限。
        """
        items = list(calls or [])
        if not items:
            return

        planned: List[tuple] = []
        for tc in items:
            started = int(time.time() * 1000)
            denied = budget.try_consume(tc.name, tool_usage.get(tc.name or "", 0), False)
            if denied is None and deadline is not None and deadline.expired:
                # 到点后工具也不再跑：最慢的就是联网检索，让它跑完只会更迟。
                # 拒绝理由照样回灌给模型 —— 它得知道"不是工具坏了，是没时间了"。
                denied = ("本次分析的时间预算已用尽，该工具未执行。"
                          "请立即基于上面已经取到的证据给出最终结论，"
                          "需要外部信息佐证的在「四、不确定性」里写明「本次未联网核实」。")
            planned.append((tc, denied, started))

        executed = self._execute_planned(planned, ctx)

        # 后处理**按原始调用顺序**在主线程串行做：tool_trace、留痕、证据池都是有序数据，
        # 按完成顺序写会让同一批输入的审计日志长得不一样 —— 那等于丢掉了可追溯性换来的收益。
        for (tc, denied, started), (text, ok, src_type, res) in zip(planned, executed):
            key = self._call_key(tc)
            if ok and tc.name:
                tool_usage[tc.name] = tool_usage.get(tc.name, 0) + 1
                # 循环里取到的数也要进证据池，否则"证据溯源"只剩预取那一批
                if res is not None:
                    evidence.append(self._evidence_entry(tc.name, res, 1, False))
            call_cache[key] = text
            ms = int(time.time() * 1000) - started
            rec.note_tool(ToolRun(tc.name or "", ok, ms, None if ok else text[:200],
                                  src_type if ok else "internal", False, len(text)))
            tool_trace.append(f"{tc.name}({ms}ms)")

    # -- 工具执行：预算已在 _run_tool_calls 里扣完，这里只负责"跑" ------ #

    def _execute_one(self, tc, denied, ctx):
        """单个工具：被拒就返回理由，否则真跑。返回 ``(text, ok, src_type, res)``。

        抽出来是为了让串行路径和并行路径**共用同一段语义**：以前 ``src_type`` 依赖
        ``if denied`` 分支不赋值、再由调用方的三元表达式绕过，看着省事，
        一旦有人改动分支顺序就会 ``UnboundLocalError``。
        """
        if denied:
            return denied, False, "internal", None
        return self._invoke(tc, ctx)

    def _execute_planned(self, planned: List[tuple], ctx) -> List[tuple]:
        """执行一批**已扣过预算**的工具。返回与 ``planned`` 等长、等序的结果列表。

        只在「真的有 ≥2 个要跑的工具」时才并行 —— 只有一个工具还开线程池，
        付出的是调度 overhead，收获是 0。
        """
        pendable = [i for i, p in enumerate(planned) if not p[1]]
        if len(pendable) < 2 or not self.parallel_tools:
            return [self._execute_one(tc, denied, ctx) for tc, denied, _ in planned]

        import contextvars  # 延迟导入：只有真正并行时才需要

        from concurrent.futures import ThreadPoolExecutor

        out: List[Optional[tuple]] = [None] * len(planned)
        workers = min(max(2, self.parallel_tools_max), len(pendable))

        def job(idx: int, tc) -> None:
            out[idx] = self._execute_one(tc, None, ctx)

        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agt-tool") as ex:
                futs = []
                for idx in pendable:
                    #! **必须显式把上下文带进子线程**：``TraceContext`` 是 contextvars，
                    #! 线程池只是 ``threading.Thread`` 的包装，不会自动继承。
                    #! 漏了它，子线程里拼的那次工具调用会没有 trace id，留痕断在这一层。
                    futs.append(ex.submit(contextvars.copy_context().run, job, idx,
                                          planned[idx][0]))
                for f in futs:
                    f.result()
        except Exception as e:  # noqa: BLE001 - 并行失败要退回串行，不能整轮工具全丢
            log.warning("[Agent] 工具并行执行失败，回退串行：%s", e)
            return [self._execute_one(tc, denied, ctx) for tc, denied, _ in planned]

        # 兜底：任何没写回的位置（理论上不该有）按串行补跑一次
        return [r if r is not None else self._execute_one(tc, denied, ctx)
                for (tc, denied, _), r in zip(planned, out)]

    def _invoke(self, tc: ToolCall, ctx: ToolContext):
        """返回 ``(text, ok, sourceType, ToolResult|None)``。"""
        if self.tools is None:
            return '{"error":"工具层未装配"}', False, "internal", None
        last_err = None
        for attempt in range(self.tool_retry + 1):
            try:
                res = self.tools.execute_with_meta(tc.name, tc.arguments_json, ctx)
                # 结果回灌给证据池（供引用核对与前端来源卡使用）
                self._absorb(res, tc, ctx)
                return res.text, True, res.source_type, res
            except Exception as e:  # noqa: BLE001
                last_err = e
        return '{"error":"工具执行失败: ' + str(last_err) + '"}', False, "internal", None

    def _absorb(self, res, tc: ToolCall, ctx: ToolContext) -> None:
        """把工具产出的结构化来源并入证据池（供引用核对使用）。"""
        if not res.sources:
            return
        pool = getattr(ctx, "_evidence_pool", None)
        if pool is None:
            pool = []
            setattr(ctx, "_evidence_pool", pool)
        pool.append({"tool": tc.name, "sourceType": res.source_type, "sources": res.sources})

    # -- 预取 --------------------------------------------------------

    @staticmethod
    def _evidence_entry(tool: str, res: Any, attempts: int = 1, cached: bool = False) -> Dict[str, Any]:
        """一条证据条目。**实现在 :func:`app.agent.tool_node.evidence_entry`**——
        工具节点与预取路径必须记出**同一形状**的条目，否则前端「证据溯源」
        会因为"这一条是从哪条路径来的"而长得不一样。

        早先这里只在 ``res.sources`` 非空时才记一条，且只写 ``sources`` 不写 ``preview``：
        于是模型在循环里调的 ``get_metrics`` / ``get_risk_events`` 之类**根本不进证据池**，
        前端「证据溯源」只剩知识库一路，而 Java 侧是每次工具调用一条。
        条数对不上还是小事，真正的问题是"这次到底取了哪些数"变得不可追溯。
        """
        from .tool_node import evidence_entry

        return evidence_entry(tool, res, attempts, cached)

    def _prefetch(self, specs, route, question, budget, ctx, rec, tool_trace,
                  limit_override: Optional[int] = None) -> List[Dict[str, Any]]:
        """本地毫秒级预取：省掉"模型决定调什么工具"那一整轮往返。"""
        if self.tools is None:
            return []
        deadline: Optional[Deadline] = getattr(ctx, "deadline", None)
        names = self._prefetch_names(route, specs)
        plan = budget.plan
        limit = getattr(plan, "prefetch_limit", 2)
        if limit_override is not None:
            limit = max(int(limit), int(limit_override))
        if (getattr(route, "force_web", False) or getattr(route, "needs_web", False)) and "web_search" in names:
            limit = max(int(limit), names.index("web_search") + 1)
        out: List[Dict[str, Any]] = []
        for name in names[: max(0, int(limit))]:
            if deadline is not None and deadline.expired:
                # 预取是"顺手多拿一点证据"，不是必需项：没时间就别拿了。
                # 它一旦把预算吃光，后面连一次成稿调用都发不出去，反而更糟。
                log.warning("[Agent] 时间预算耗尽，预取证椐提前收工：%s", deadline.to_dict())
                break
            args = self._prefetch_args(name, question, getattr(ctx, "top_k", 5))
            started = int(time.time() * 1000)
            try:
                res = self.tools.execute_with_meta(name, args, ctx)
                ms = int(time.time() * 1000) - started
                rec.note_tool(ToolRun(name, True, ms, None, res.source_type, True, len(res.text)))
                tool_trace.append(f"{name}(预取 {ms}ms)")
                out.append(self._evidence_entry(name, res, 1, False))
            except Exception as e:  # noqa: BLE001
                rec.note_tool(ToolRun(name, False, int(time.time() * 1000) - started,
                                      str(e)[:200], "internal", True, 0))
        return out

    @staticmethod
    def _successful_prefetch_tools(evidence: List[Dict[str, Any]]) -> Set[str]:
        """Return tools whose prefetched result is usable for direct drafting.

        Internal tools are useful when they produced a preview. Web search is
        stricter: it must contain at least one structured source, otherwise the
        model keeps the web_search tool and may retry with a better query.
        """
        out: Set[str] = set()
        for item in evidence or []:
            name = str(item.get("tool") or "")
            if not name or not str(item.get("preview") or "").strip():
                continue
            if name == "web_search" and not item.get("sources"):
                continue
            out.add(name)
        return out

    @staticmethod
    def _prefetch_names(route, specs) -> List[str]:
        opened = getattr(route, "opened_tools", {}) or {}
        ordered: List[str] = []
        for group in opened.values():
            for t in group or []:
                if t and t not in ordered:
                    ordered.append(t)
        # 知识库检索总在最前：它是本地零成本、命中率最高的一路
        if "search_knowledge" in ordered:
            ordered.remove("search_knowledge")
            ordered.insert(0, "search_knowledge")
        elif specs:
            ordered.insert(0, "search_knowledge")
        return ordered

    def _prefetch_args(self, name: str, question: Optional[str], top_k: int = 5) -> str:
        q = (question or "").strip()
        cq = clean_query(q)
        term = cq.query or q
        requested = max(1, int(top_k or 5))
        if name == "search_knowledge":
            return json.dumps({"query": term[:60], "topK": min(10, max(5, requested))},
                              ensure_ascii=False)
        if name == "web_search":
            return json.dumps({"query": term[:60], "topN": 5}, ensure_ascii=False)
        detail_limit = min(30, max(15, requested * 3))
        if name == "get_risk_events":
            return json.dumps({"limit": detail_limit}, ensure_ascii=False)
        if name == "get_metrics":
            return json.dumps({"limit": detail_limit}, ensure_ascii=False)
        if name == "get_complaints":
            return json.dumps({"limit": detail_limit}, ensure_ascii=False)
        if name == "get_competitors":
            return json.dumps({"limit": detail_limit}, ensure_ascii=False)
        return "{}"

    # -- 空正文自愈 --------------------------------------------------

    def _retry_final_answer(self, messages, ctx, evidence, rec, deadline=None) -> str:
        """三级兜底的第一级：非流式 + 不带工具 + 给足预算重试一次。

        推理型模型常常"想了很久却没输出"——思考 token 把默认预算吃满了，
        于是正文为空而 ``finish_reason=length``。再给一次机会、这次不带工具，通常就成了。

        到点则**直接放弃这一级**：它是一次完整的模型调用，
        在预算已经见底时发它，等于为了"再试一次"让用户多等一整轮。
        """
        if self.llm is None or not self.llm.is_configured():
            return ""
        if deadline is not None and deadline.expired:
            return ""
        msgs = list(messages)
        msgs.append(ChatMessage.system(
            "请直接输出最终分析正文，不要再调用任何工具。按「一、结论…二、结论…三、建议动作…四、不确定性」四节输出。"))
        max_tokens = (deadline.converge_max_tokens_for(self.retry_max_tokens)
                      if deadline is not None else self.retry_max_tokens)
        try:
            with llm_call_timeout(
                    deadline.call_timeout(float(getattr(self.llm, "chat_timeout_seconds", 0) or 0))
                    if deadline is not None else None):
                r = self.llm.chat_with_tools([], msgs, AGENT_TEMPERATURE, max_tokens)
            rec.note_llm(self.llm.get_model(), r.finish_reason)
            if r.content and r.content.strip():
                return r.content
            if r.finish_reason:
                rec.note_degrade(DEGRADE_PARTIAL, f"重试仍为空正文（finish_reason={r.finish_reason}）")
        except Exception as e:  # noqa: BLE001
            rec.note_degrade(DEGRADE_PARTIAL, f"重试失败：{e}")
        return ""

    def _local_report(self, company, question, evidence, ctx) -> str:
        """三级兜底的最后一级：**本地确定性报告**。

        它不编造结论，只把已经取到的证据按四节格式摆出来，并明确标注"未调用模型"。
        用户拿到的是一份能用的东西，而不是一句"分析失败"。
        """
        name = (company or {}).get("company_name") if isinstance(company, dict) else None
        parts = [f"# 风险分析（本地确定性报告）\n",
                 f"企业：{name or '未知'}　问题：{(question or '').strip()}\n"]
        kb = [s for e in evidence if e.get("sourceType") == "knowledge" for s in e.get("sources", [])]
        web = [s for e in evidence if e.get("sourceType") == "web" for s in e.get("sources", [])]
        internal = [s for e in evidence if e.get("sourceType") == "internal" for s in e.get("sources", [])]
        stats = ctx.stats_snapshot()

        parts.append("\n## 一、结论（基于内部数据）")
        if kb or internal:
            for s in (kb + internal)[:10]:
                ref = s.get("sourceRef") or s.get("documentId")
                parts.append(f"- {s.get('title')}（来源ID={ref}）")
        else:
            parts.append("- 本次未召回内部证据。")
        if stats:
            parts.append("- 已取到的统计：" + "，".join(f"{k}={v}" for k, v in stats.items()))

        parts.append("\n## 二、结论（基于外部公开资料）")
        if web:
            for s in web[:10]:
                parts.append(f"- [{s.get('index')}] {s.get('title')}（{s.get('url')}）")
        else:
            parts.append("- 本次未联网核实")

        parts.append("\n## 三、建议动作")
        parts.append("（一）基于内部数据的动作")
        if kb or internal:
            parts.append("- 按上述内部证据逐条核对，优先处理高等级风险与重复投诉。")
        else:
            parts.append("- 内部证据不足，建议先补充数据或换用更具体的检索词。")
        parts.append("（二）基于外部资料的动作")
        parts.append("- 本次未联网核实" if not web else "- 参考上述外部来源，结合本企业情况评估。")

        parts.append("\n## 四、不确定性")
        parts.append("- 本报告由本地规则生成，**未调用外部大模型**：结论为证据归纳，不含模型归因。")
        parts.append("- 所有引用均来自本次实际召回的证据，可逐条核对。")
        return "\n".join(parts)

    # -- 来源拆分 ----------------------------------------------------

    @staticmethod
    def _evidence_summary(evidence: List[Dict[str, Any]], trace: List[str]) -> str:
        """Bounded evidence context used by drafting and review.

        Structured tools usually have no ``sources`` list; their useful data is
        in ``preview``. Omitting it meant the model only saw "0 sources" after
        metrics and complaint tools had successfully returned many rows.
        """
        lines: List[str] = []
        remaining = 14_000
        for e in evidence or []:
            if not isinstance(e, dict):
                continue
            srcs = e.get("sources")
            lines.append("- 工具 %s（%s）：%s 条来源"
                         % (e.get("tool") or "", e.get("sourceType") or "",
                            len(srcs) if isinstance(srcs, list) else 0))
            if isinstance(srcs, list):
                for o in srcs[:8]:
                    if not isinstance(o, dict):
                        continue
                    label = o.get("title") or o.get("url") or o.get("sourceRef") or ""
                    lines.append("    · %s" % str(label)[:120])
            preview = str(e.get("preview") or "").strip()
            if preview and remaining > 0:
                piece = preview[: min(2400, remaining)]
                lines.append("  工具返回内容：\n" + piece)
                remaining -= len(piece)
        for t in (trace or [])[:20]:
            lines.append("- " + str(t)[:200])
        return "\n".join(lines) or "（本次未召回任何证据）"

    @staticmethod
    def _split_web_sources(draft: str, evidence):
        """前端只展示**正文引用过**的来源；未引用的单独折叠，便于核对"召回但没用"。"""
        all_web = [s for e in evidence if e.get("sourceType") == "web" for s in e.get("sources", [])]
        used, unused = [], []
        for s in all_web:
            idx = s.get("index")
            token = f"[{idx}]"
            (used if token in (draft or "") else unused).append(s)
        return used, unused, all_web

    @staticmethod
    def _collect_knowledge(evidence):
        out = []
        for e in evidence or []:
            if e.get("sourceType") != "knowledge":
                continue
            for s in e.get("sources", []):
                out.append({"index": s.get("index"),
                            "sourceRef": s.get("sourceRef") or s.get("documentId"),
                            "title": s.get("title"), "score": s.get("score")})
        return out

    # -- 落库 --------------------------------------------------------

    def _serve_cached(self, hit: "AgentOutcome", cid, user_id, question, session_id, ctx,
                      rec, token_sink, stage_sink, meta_sink, started) -> "AgentOutcome":
        """命中结果缓存：结论照搬上次，但**这次提问自己要有痕、要有追溯号、要推流**。

        原先这里 ``return hit`` 一行把第一次的 outcome 原样吐回去，代价是三件事：

        1. **一个 token 都不推**。界面在这一整段里完全没有"正在输出"的反馈，
           而 SSE 一旦因为链路抖动丢了 ``done`` 事件，用户看到的就是「第二次没有结果」
           ——而后端以为自己成功了，日志里什么都查不到。
        2. **traceId / analysisId 复用上一次**。两次提问在审计页里塌成一条，
           点追溯看到的工具调用也是上一次的：这等于**追溯失效**，不是省事。
        3. **不重新落库**，历史列表里不会出现"你又问了一次"这件事，
           连"这个问题被反复问过、该写进知识库了"这类信号也一起丢了。

        所以这里刻意**重成本为零地把该有的都补齐**：重跑一次确定性的引用核对、
        写出一条新的分析记录，再把正文按块推给流式通道。
        """
        self._stage(stage_sink, "命中结果缓存")

        # answer_json 要深拷贝：下面会往里写"本次是缓存命中"的标记，
        # 直接改会污染缓存里那份对象（下一次命中就会被标记两次）。
        try:
            aj = copy.deepcopy(hit.answer_json) if isinstance(hit.answer_json, dict) else {}
        except Exception:  # noqa: BLE001
            aj = {}

        out = AgentOutcome(
            trace_id=rec.trace_id,
            content=hit.content or "",
            answer_json=aj,
            evidence=[dict(e) for e in (hit.evidence or [])],
            tool_trace=list(hit.tool_trace or []),
            confidence=hit.confidence or "中",
            degrade_level=hit.degrade_level or DEGRADE_NONE,
            degrade_reason=hit.degrade_reason,
            model=hit.model,
        )
        out.diagnostics["cached"] = True
        out.diagnostics["cachedFrom"] = hit.analysis_id
        out.diagnostics["cachedStage"] = "命中结果缓存"
        aj["cached"] = True
        aj["cachedFrom"] = hit.analysis_id
        if isinstance(aj.get("diagnostics"), dict):
            aj["diagnostics"]["cached"] = True

        # meta：不等正文就把「风险等级 / 来源构成」给出去（首次适量的重复不限流量）
        try:
            self._emit_meta(meta_sink, ctx, out.evidence)
        except Exception:  # noqa: BLE001
            pass

        # 引用核对重跑一次：零 token 的确定性校验，不能因为"是缓存"就跳过，
        # 否则 ai_analysis.grounded 这一列对不同来源的记录口径不一致。
        check = None
        try:
            if self.grounding is not None:
                check = self.grounding.verify(out.content, out.evidence)
        except Exception as e:  # noqa: BLE001
            log.debug("[Agent] 缓存命中后重跑引用核对失败（按已核对处理）：%s", e)
        if check is None:
            check = Check()

        # 留痕 + 正文推流
        try:
            out.analysis_id = self._persist(out, cid, user_id, question, session_id, check)
        except Exception as e:  # noqa: BLE001 - 写库失败不能让这次分析变成"没有结果"
            log.warning("[Agent] 缓存命中后写分析记录失败（继续返回结论）：%s", e)
        self._push_content(out.content, token_sink)

        rec.note_cache_hit(hit.analysis_id)
        rec.flush(out.analysis_id)
        return self._finish(out, rec, started)

    @staticmethod
    def _push_content(content: Optional[str], token_sink) -> None:
        """把命中的缓存结果按块推给流式通道。

        为什么不整块一次推：界面（``stores/aiTask.js``）对 token 做了 80ms 批量 flush，
        一次塞几万字会让首屏长时间空白；按块推则表现与真实流式一致。
        """
        if not token_sink or not content:
            return
        step = 180
        for i in range(0, len(content), step):
            try:
                token_sink(content[i:i + step])
            except Exception:  # noqa: BLE001 - 推流失败不能影响主流程
                return

    def _persist(self, out, cid, user_id, question, session_id, check) -> Optional[int]:
        import json as _json

        from sqlalchemy import insert

        row = {
            "company_id": cid,
            "user_id": user_id,
            "question": question or "",
            "answer": out.content or "",
            "evidence_json": _json.dumps(out.evidence, ensure_ascii=False),
            "tool_trace_json": _json.dumps(out.tool_trace, ensure_ascii=False),
            # Java 落库的是 ``ai.name()``；这里若写死 "pyagent"，
            # 历史列表里两版记录会显示成两种"模型服务"，没法横向比。
            "provider": out.model or "local-rule",
            "created_at": datetime.now(),
            "conversation_id": session_id,
            "answer_json": _json.dumps(out.answer_json or {}, ensure_ascii=False),
            "confidence": out.confidence,
            "grounded": 1 if check.ok else 0,
            "trace_id": out.trace_id,
            "degrade_level": out.degrade_level,
            "degrade_reason": (out.degrade_reason or "")[:500] or None,
            "duration_ms": out.duration_ms,
            "llm_calls": 0,
            "tool_calls": len(out.tool_trace),
            "model": (out.model or "")[:120] or None,
            "review_revised": 0,
        }
        return self.db.insert_id(insert(T.ai_analysis).values(**row))

    def _remember(self, cid, user_id, question, out) -> Optional[Dict[str, Any]]:
        """把这次分析沉淀成长期记忆。

        **零 token**：抽取全在本地规则里做完（见 :mod:`memory_extract`），
        不为"记住"多付一次推理。失败只记日志 —— 记忆写不进去最多是这次没记住，
        绝不能让一次已经跑完的分析因为记忆而报错。
        """
        try:
            if not self.long_term.write_enabled():
                return None
            aj = out.answer_json or {}
            cands = self._memory_extractor.extract(
                company_id=cid, user_id=user_id, question=question,
                answer_json=aj, answer_text=out.content,
                risk_level=str(aj.get("risk_level") or ""),
                analysis_id=out.analysis_id, trace_id=out.trace_id,
                degrade_level=out.degrade_level)
            if not cands:
                return None
            return self.long_term.write(cands).to_dict()
        except Exception as e:  # noqa: BLE001
            log.warning("[Agent] 长期记忆写入失败（已忽略）：%s", e)
            return None

    # -- 辅助 --------------------------------------------------------

    def _load_company(self, cid: Optional[int]) -> Optional[Dict[str, Any]]:
        if cid is None:
            return None
        from sqlalchemy import select

        rows = self.db.fetch_all(select(T.company).where(T.company.c.id == cid).limit(1))
        return rows[0] if rows else None

    def _pick_specs(self, route, budget) -> List[ToolSpec]:
        """按路由结果裁剪工具白名单——「Agent 少」不等于「工具全开」。"""
        if self.tools is None:
            return []
        opened = getattr(route, "opened_tools", {}) or {}
        allowed: List[str] = []
        for group in opened.values():
            for t in group or []:
                if t and t not in allowed:
                    allowed.append(t)
        if not allowed:
            return self.tools.specs()
        return [s for s in self.tools.specs() if s.name in allowed]

    @staticmethod
    def _tool_hint(specs, budget, force_web: bool = False) -> str:
        """把「本次开放了哪些工具」写进用户消息，让模型不必猜。"""
        if not specs:
            return "\n\n（本次不开放任何工具，请直接基于已有信息作答。）"
        names = "、".join(s.name for s in specs)
        base = (f"\n\n【本次可用的工具】{names}。"
                "需要外部或最新信息时用 web_search，内部证据用 search_knowledge；"
                "写动作只能「提议」，不要说成已执行。")
        #! 【2026-09-22 修】``budget.answer_max_chars`` 的字段注释写着
        #! 「写进 system 提示约束模型，不是事后截断」，但全项目**从未使用过它**。
        #! 后果：用户选「快答」（该档预算 700 字）时模型毫不知情，照样写 1900 字 ——
        #! 档位只约束了平台侧的轮数与配额，唯独没约束最耗时的那个变量：生成多少字。
        plan = getattr(budget, "plan", None)
        if plan is not None:
            base += (f"\n【正文预算】{plan.answer_max_chars} 字以内（本次为「{plan.depth.key}」档，"
                     "超出会显著拉长等待时间）。四个小节都要写全，但每节保持精炼。")
        if force_web:
            # 白名单里"有"这个工具和模型"会调"这个工具是两回事，
            # 这里把要求提到用户消息里再说一遍（system 与 user 双侧都讲，实测才稳）。
            base += ("\n【用户明确要求联网】请先调用 web_search 获取外部公开资料，"
                     "再结合内部证据成稿；不得在未调用的情况下直接写「本次未联网核实」。")
        return base

    @staticmethod
    def _call_key(tc: ToolCall) -> str:
        return f"{tc.name}|{(tc.arguments_json or '{}').strip()}"

    @staticmethod
    def _single_flight_key(user_id, cid, question, depth=None, force_web=False) -> str:
        h = hashlib.md5(
            f"{question or ''}|{depth or ''}|{int(bool(force_web))}".encode("utf-8")
        ).hexdigest()[:12]
        return f"ai:{user_id}:{cid}:{h}"

    def _cache_key(self, user_id, cid, question, use_multi_agent, top_k,
                   depth=None, force_web=False) -> str:
        #! depth 必须进键。漏掉它会出现「先用标准档问过、再选快答却秒回标准档结果」，
        #! 而用户以为自己换的是档位 —— 缓存串味比缓存未命中难查得多。
        #! force_web 同理：勾了「必须联网」却命中一条没联网的缓存，等于开关失效。
        raw = (f"{user_id}|{cid}|{question}|{use_multi_agent}|{top_k}"
               f"|{depth or ''}|{int(bool(force_web))}")
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[AgentOutcome]:
        with self._lock:
            hit = self.cache.get(key)
        if hit is None:
            return None
        expire_at, value = hit
        if time.time() > expire_at:
            with self._lock:
                self.cache.pop(key, None)
            return None
        return value

    def _cache_put(self, key: str, value: AgentOutcome) -> None:
        with self._lock:
            self.cache[key] = (time.time() + self.cache_ttl, value)

    @staticmethod
    def _stage(sink, stage: str) -> None:
        if sink:
            try:
                sink({"stage": stage})
            except Exception:  # noqa: BLE001 - 推流失败不能影响主流程
                pass

    @staticmethod
    def _push(sink, text: str) -> None:
        if sink and text:
            try:
                sink(text)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _emit_meta(sink, ctx: ToolContext, evidence) -> None:
        """把「风险等级 / 来源构成」先推给前端，不等正文。"""
        level = ctx.rule_risk_level()
        web = sum(len(e.get("sources", [])) for e in evidence if e.get("sourceType") == "web")
        kb = sum(len(e.get("sources", [])) for e in evidence if e.get("sourceType") == "knowledge")
        try:
            sink({"riskLevel": level, "webSources": web, "kbSources": kb,
                  "stats": ctx.stats_snapshot()})
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _headline(draft: str) -> str:
        for line in (draft or "").splitlines():
            t = line.strip().lstrip("#").strip()
            if t:
                return t[:120]
        return ""

    @staticmethod
    def _norm_heading(line: str) -> str:
        """归一化一行标题：剥掉两侧的 Markdown 装饰与空白。

        ``**一、结论（…）**`` → ``一、结论（…）``；``## 四、不确定性`` → ``四、不确定性``。
        只用于**判定**，不改动正文本身（正文照原样进 JSON，前端才有原始可读文本）。
        """
        return (line or "").strip().strip(_HEADING_DECOR).strip()

    @staticmethod
    def _section_of(line: str) -> str:
        """若这一行是四节标题之一，返回其序号（一/二/三/四），否则返回空串。

        「标题行」的定义很窄：以序号开头 + 命中该小节关键词 + 足够短（标题不会写成长句）。
        这三条同时成立才认，避免把正文里「一、xxx」开头的句子当成新的小节起点。
        """
        s = AgentService._norm_heading(line)
        if not s or len(s) > 60:
            return ""
        num = s[0]
        if num not in _SECTION_KEYS:
            return ""
        return num if any(k in s[1:] for k in _SECTION_KEYS[num]) else ""

    @staticmethod
    def _section(draft: str, marker: str) -> str:
        """从正文里切出 ``marker`` 对应的小节正文（标题行本身不进正文）。

        切不出来就返回空串 —— 它只负责"取"，不负责"编"。
        """
        lines = (draft or "").splitlines()
        buf: List[str] = []
        on = False
        for ln in lines:
            sec = AgentService._section_of(ln)
            if sec:
                if sec == marker:
                    on = True
                    continue
                if on:
                    break  # 撞到下一节标题即收工
                continue
            if on:
                buf.append(ln)
        # 小节之间常用 `---` 分隔线，它不属于任何一节的正文，别带进结论里
        while buf and AgentService._norm_heading(buf[-1]) in ("", "---", "***", "___", "—"):
            buf.pop()
        return "\n".join(buf).strip()

    @staticmethod
    def _recs(draft: str, evidence, internal: bool) -> List[Dict[str, Any]]:
        """从建议动作小节里抽条目；抽不到就返回空，绝不凭空编建议。"""
        out: List[Dict[str, Any]] = []
        on = False
        want = "（一）" if internal else "（二）"
        for ln in (draft or "").splitlines():
            s = AgentService._norm_heading(ln)
            if s.startswith(want):
                if on:
                    break
                on = True
                continue
            if on and (s.startswith("（一）") or s.startswith("（二）")
                       or AgentService._section_of(s)):
                break
            if on and s:
                out.append({"action": s.lstrip("-·0123456789. ").strip(), "basis": "", "refs": []})
        return out[:10]

    @staticmethod
    def _finish(out: AgentOutcome, rec: TraceRecorder, started: float) -> AgentOutcome:
        out.duration_ms = int((time.time() - started) * 1000)
        out.diagnostics["durationMs"] = out.duration_ms
        out.diagnostics["trace"] = rec.summary()
        return out


_ = uuid  # 预留：运行期可为一次分析生成稳定的幂等键
