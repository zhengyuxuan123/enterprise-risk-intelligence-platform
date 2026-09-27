# -*- coding: utf-8 -*-
"""阶段 11：时间预算与超时降级（**全程离线、零 token**）。

「指定时间得不到结果就降级」听着像一句口号，落地要钉住三件事：

1. **硬截止真的一个调用都不发**。到点之后最贵的行为就是"又发了一次注定来不及的
   模型调用" —— 它既烧额度，又把用户按在等待里。所以断言的是 ``llm.calls == []``，
   而不是"耗时变短了"（后者在 CI 上根本测不准）。
2. **软截止把理由真的送到模型面前**。旧代码算出了「预算已用尽，请立即成稿」却只用它
   决定不带工具，**文案被丢掉**，模型根本不知道该收敛，于是接着空转。
   这里断言提示语**进了 messages**，且收敛后不再有第二轮。
3. **降级不抛异常**。用户拿到的是本地确定性报告 + PARTIAL + 明确理由，不是 500。
"""

from __future__ import annotations

import ast
import pathlib
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from app.agent import agent_service as as_mod
from app.agent.agent_service import AgentService
from app.agent.budget import AgentBudget
from app.agent.deadline import CONVERGING, EXPIRED, RUNNING, Deadline
from app.ai.llm_client import (
    ChatMessage,
    ChatResult,
    ToolCall,
    current_call_timeout,
    llm_call_timeout,
)
from app.ai.tool_context import ToolContext
from app.ai.tools import ToolResult, ToolSpec
from fake_langchain import adapt_factory
from app.agent.trace import TraceRecorder

ANSWER = ("一、结论（基于内部数据）\n- 无内部证据。\n"
          "二、结论（基于外部公开资料）\n- 本次未联网核实\n"
          "三、建议动作\n（一）内部\n- 无\n（二）外部\n- 无\n"
          "四、不确定性\n- 无")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _deadline(elapsed_ms: int, budget_ms: int = 240_000, **kw) -> Deadline:
    """造一个"已经跑了 elapsed_ms 毫秒"的 deadline。"""
    return Deadline(budget_ms, started_at_ms=_now_ms() - elapsed_ms, **kw)


# --------------------------------------------------------------------------- #
# 假模型：除了应答，还要记录"调用时生效的单次超时"与"看到的消息"
# --------------------------------------------------------------------------- #


class FakeLlm:
    def __init__(self, results: Optional[List[ChatResult]] = None,
                 sticky: Optional[ChatResult] = None) -> None:
        self.queue: List[ChatResult] = list(results or [])
        self.sticky = sticky
        self.calls: List[Dict[str, Any]] = []
        self.seen: List[str] = []
        self.timeouts: List[Optional[float]] = []
        self.completes: List[str] = []
        self.chat_timeout_seconds = 600

    def chat_with_tools(self, tools, messages, temperature=0.2, max_tokens=0) -> ChatResult:
        self.calls.append({"tools": [t.name for t in (tools or [])], "n": len(messages)})
        self.seen.append("\n".join((m.content or "") for m in messages))
        self.timeouts.append(current_call_timeout())
        if self.queue:
            return self.queue.pop(0)
        return self.sticky if self.sticky is not None else ChatResult(content=ANSWER,
                                                                      finish_reason="stop")

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """复核员改稿会走这里；记到单独的列表，避免污染"第几次工具对话"的断言。"""
        self.completes.append(system)
        return ""

    def is_configured(self) -> bool:
        return True

    def get_model(self) -> str:
        return "fake-model"


class FakeTools:
    """只数执行次数：断言"到点后工具没被真的跑一遍"。"""

    def __init__(self) -> None:
        self.executed: List[str] = []

    def execute_with_meta(self, name: str, args: str, ctx) -> ToolResult:
        self.executed.append(name)
        return ToolResult(text='{"ok":true}', sources=[], source_type="internal")


def _svc(llm=None, tools=None, **kw) -> AgentService:
    """构造服务，并把同一个 llm 接到 LangGraph 分支上。

    少了 ``lg_model_factory``，LangGraph 会自己去构造**真实的** ChatOpenAI：
    用例于是开始真发请求（实测收到过方舟的 429 SetLimitExceeded），
    而且它以为在测降级、实际在测网络。护栏见 ``conftest._no_real_llm_client``。
    """
    if llm is not None:
        kw.setdefault("lg_model_factory", adapt_factory(llm))
    svc = AgentService(llm=llm, tools=tools, **kw)
    svc.cache_enabled = False
    svc.cache.clear()
    return svc


def _ctx() -> ToolContext:
    return ToolContext(company_id=None, top_k=5, user_id=1, question="问")


def _rec() -> TraceRecorder:
    return TraceRecorder(trace_id="t1", company_id=None, user_id=1, question="问", db=None)


SPEC = ToolSpec("search_knowledge", "知识库检索", '{"type":"object","properties":{}}')


# --------------------------------------------------------------------------- #
# Deadline 三态
# --------------------------------------------------------------------------- #


class TestDeadlineStates:
    def test_three_states_advance_by_elapsed(self):
        assert _deadline(1_000).state() == RUNNING
        assert _deadline(200_000).state() == CONVERGING      # soft = 240s * 0.7 = 168s
        assert _deadline(250_000).state() == EXPIRED

    def test_zero_budget_means_no_interference(self):
        """关掉（或没配预算）时，deadline 必须完全隐形 —— 否则等于偷偷改了旧行为。"""
        d = Deadline(0)
        assert d.state() == RUNNING
        assert not d.expired and not d.converging
        assert d.remaining_ms() is None
        assert d.call_timeout(600) is None
        assert d.to_dict()["enabled"] is False

    def test_expired_is_absorbing(self):
        """时间只往前走：一旦 EXPIRED 不会因任何计算抖动退回 CONVERGING。"""
        d = _deadline(250_000)
        assert d.expired
        assert not d.converging

    def test_call_timeout_clamps_to_remaining(self):
        d = _deadline(210_000)          # 还剩 30s
        assert 29 <= (d.call_timeout(600) or 0) <= 31

    def test_call_timeout_has_a_floor(self):
        """只剩 2 秒时给 15 秒而不是 2 秒：2 秒的请求必定空手而归，还白搭一次往返。"""
        d = _deadline(238_000, min_call_seconds=15)
        assert d.call_timeout(600) == pytest.approx(15, abs=0.5)

    def test_converge_hint_tells_model_to_stop_using_tools(self):
        hint = _deadline(200_000).converge_hint()
        assert "时间预算" in hint and "不要" in hint or "停止" in hint
        assert "四节" in hint or "一、结论" in hint

    def test_expire_reason_is_actionable(self):
        reason = _deadline(250_000).expire_reason()
        assert "240" in reason and "本地确定性报告" in reason


class TestDeadlineForPlan:
    @staticmethod
    def _settings(**agent_kw) -> SimpleNamespace:
        base = {"deadline_enabled": True, "deadline_seconds": 0, "deadline_soft_ratio": 0.7,
                "deadline_min_call_seconds": 15, "deadline_converge_tokens": 0}
        base.update(agent_kw)
        return SimpleNamespace(agent=SimpleNamespace(**base))

    def test_follows_depth_plan_by_default(self):
        d = Deadline.for_plan(240_000, settings=self._settings())
        assert d.budget_ms == 240_000

    def test_explicit_seconds_overrides_plan(self):
        d = Deadline.for_plan(600_000, settings=self._settings(deadline_seconds=90))
        assert d.budget_seconds == 90

    def test_switch_off_gives_a_noop_deadline(self):
        d = Deadline.for_plan(600_000, settings=self._settings(deadline_enabled=False))
        assert d.enabled is False and d.state() == RUNNING

    def test_soft_ratio_is_configurable(self):
        d = Deadline.for_plan(240_000, settings=self._settings(deadline_soft_ratio=0.5))
        assert d.soft_ms == 120_000


# --------------------------------------------------------------------------- #
# 单次调用超时：必须按执行上下文隔离，不能写到客户端实例上
# --------------------------------------------------------------------------- #


class TestCallTimeoutOverride:
    def test_restored_after_exit(self):
        assert current_call_timeout() is None
        with llm_call_timeout(12.5):
            assert current_call_timeout() == 12.5
        assert current_call_timeout() is None

    def test_none_or_zero_means_no_override(self):
        with llm_call_timeout(None):
            assert current_call_timeout() is None
        with llm_call_timeout(0):
            assert current_call_timeout() is None

    def test_does_not_leak_into_other_threads(self):
        """客户端是进程内单例：若把超时写在实例上，A 的预算会掐死并发的 B。"""
        seen: List[Optional[float]] = []
        with llm_call_timeout(7.0):
            t = threading.Thread(target=lambda: seen.append(current_call_timeout()))
            t.start()
            t.join(2)
        assert seen == [None]


# --------------------------------------------------------------------------- #
# 工具循环：硬截止 / 软截止
# --------------------------------------------------------------------------- #


def _run_loop(svc: AgentService, deadline=None, sticky=None):
    """跑一次编排循环。

    2026-09-23：自研 ``_tool_loop`` 已删除，这里改走 ``_langgraph_loop``。
    同一份 ``FakeLlm`` 通过 ``adapt_factory`` 驱动它 —— 见 ``fake_langchain.py``：
    不接这个工厂的话 LangGraph 分支会去构造**真实的** ChatOpenAI。
    """
    msgs = [ChatMessage.system("sys"), ChatMessage.user("问")]
    return svc._langgraph_loop(msgs, [SPEC], _ctx(), AgentBudget.start("standard"), [], [],
                               _rec(), deadline=deadline)


class TestToolLoopDeadline:
    def test_hard_deadline_sends_no_request_at_all(self):
        """核心断言：到点之后一次模型调用都不许发。"""
        llm = FakeLlm()
        svc = _svc(llm)
        out = _run_loop(svc, deadline=_deadline(250_000))
        assert llm.calls == []
        assert out.content == ""

    def test_soft_deadline_drops_tools_and_hands_the_reason_to_model(self):
        llm = FakeLlm()
        svc = _svc(llm)
        _run_loop(svc, deadline=_deadline(200_000))
        assert len(llm.calls) == 1
        assert llm.calls[0]["tools"] == []          # 收敛轮不带工具
        assert "时间预算提醒" in llm.seen[0]         # 理由真的送到了模型面前

    def test_soft_deadline_stops_after_one_round_even_if_answer_is_empty(self):
        """旧行为是空正文就继续下一轮 —— 收敛后必须收工。"""
        llm = FakeLlm(sticky=ChatResult(content="", finish_reason="stop"))
        svc = _svc(llm)
        _run_loop(svc, deadline=_deadline(200_000))
        assert len(llm.calls) == 1

    def test_langgraph_does_not_spin_on_empty_answer(self):
        """空正文时不再盲目空转 —— 这是相对自研循环的一处**改进**，不是退化。

        自研 ``_tool_loop`` 的逻辑是"拿到空正文就下一轮重来"，最多跑满
        ``max_iterations``（6 轮里 5 轮很可能同样空），每一轮都是一次真实付费的模型调用。
        LangGraph 的图语义则是``没有工具调用即收敛``：本轮空，图直接走到 END。

        空正文的**自愈责任因此上移**到 ``_retry_final_answer`` —— 它不带工具、
        给足预算再问一次。那条路径由
        ``test_stage5::test_empty_answer_then_valid_answer_is_self_healed`` 覆盖
        （删 legacy 时该用例在 LangGraph 编排下是绿的）。

        换句话说：自愈还在，只是不再靠"把同一件事重复做六遍"来兜底。
        """
        llm = FakeLlm(sticky=ChatResult(content="", finish_reason="stop"))
        svc = _svc(llm)
        svc.max_iterations = 6
        _run_loop(svc, deadline=_deadline(1_000))   # 远未到点
        assert len(llm.calls) == 1, "空正文应当就地收敛，由上层自愈接管"

    def test_each_round_timeout_is_tightened_to_what_is_left(self):
        llm = FakeLlm()
        svc = _svc(llm)
        _run_loop(svc, deadline=_deadline(210_000))     # 剩 30s
        assert llm.timeouts and 29 <= (llm.timeouts[0] or 0) <= 31

    def test_no_deadline_leaves_call_timeout_untouched(self):
        llm = FakeLlm()
        svc = _svc(llm)
        _run_loop(svc, deadline=None)
        assert llm.timeouts == [None]


class TestToolsBlockedWhenExpired:
    def test_expired_deadline_skips_execution_and_explains_why(self):
        """到点后最慢的是联网检索；不跑，还要把"为什么没有这条证据"告诉模型。"""
        tools = FakeTools()
        svc = _svc(FakeLlm(), tools=tools)
        cache: Dict[str, str] = {}
        usage: Dict[str, int] = {}
        ev: List[Dict[str, Any]] = []
        trace: List[str] = []
        svc._run_tool_calls(
            [ToolCall.of("1", "web_search", "{}")],
            _ctx(), AgentBudget.start("standard"), cache, usage, ev, trace, _rec(),
            deadline=_deadline(250_000))
        assert tools.executed == []
        key = next(iter(cache))
        assert "时间预算已用尽" in cache[key]

    def test_not_expired_still_runs_the_tool(self):
        tools = FakeTools()
        svc = _svc(FakeLlm(), tools=tools)
        cache: Dict[str, str] = {}
        svc._run_tool_calls(
            [ToolCall.of("1", "web_search", "{}")],
            _ctx(), AgentBudget.start("standard"), cache, {}, [], [], _rec(),
            deadline=_deadline(1_000))
        assert tools.executed == ["web_search"]


# --------------------------------------------------------------------------- #
# 两条编排都要接：只接一条 = 换个开关行为就变了
# --------------------------------------------------------------------------- #


def test_dispatch_loop_forwards_deadline_to_the_orchestrator():
    """入口必须把 ``deadline`` 传下去 —— 少传就等于"到点降级"整套失效。

    以前这条断言"两条编排分支都要被分派到"（legacy 已删），现在改为断言唯一主线。
    """
    src = pathlib.Path(as_mod.__file__).read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_dispatch_loop")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_langgraph_loop"]
    assert calls, "编排入口必须转发到 LangGraph 实现"
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "_tool_loop" for n in ast.walk(fn)), \
        "自研 _tool_loop 已删除，不该再出现在 _dispatch_loop 里"
    for c in calls:
        by_keyword = any(kw.arg == "deadline" for kw in c.keywords)
        assert by_keyword or len(c.args) >= 9, \
            f"{c.func.attr} 没有把 deadline 传下去"


def test_langgraph_branch_skips_graph_when_expired():
    """进图之前就没时间了 → 直接空手返回（连 langgraph 都不必安装即可断言）。"""
    llm = FakeLlm()
    svc = _svc(llm, orchestrator="langgraph")
    out = svc._dispatch_loop([ChatMessage.system("s")], [SPEC], _ctx(),
                             AgentBudget.start("standard"), [], [], _rec(),
                             deadline=_deadline(250_000))
    assert out.content == ""
    assert llm.calls == []


# --------------------------------------------------------------------------- #
# 端到端：降级交付，不是报错
# --------------------------------------------------------------------------- #


def _patch_expired(monkeypatch, budget_ms: int = 240_000, elapsed_ms: int = 0):
    """让 ``_run`` 拿到的 deadline 处于指定进度（不碰真实时钟）。"""
    def fake_for_plan(cls, *_a, **_kw):
        return Deadline(budget_ms, started_at_ms=_now_ms() - elapsed_ms)

    monkeypatch.setattr(Deadline, "for_plan", classmethod(fake_for_plan))


class TestAnalyzeDegrades:
    def test_time_is_up_delivers_local_report_without_any_llm_call(self, monkeypatch):
        _patch_expired(monkeypatch, elapsed_ms=250_000)
        llm = FakeLlm()
        svc = _svc(llm)
        out = svc.analyze(1, "这家企业最近有什么风险", 5, user_id=1)
        assert llm.calls == [], "到点之后不该再发起任何模型调用"
        assert out.degrade_level == "PARTIAL"
        assert "超过预算" in (out.degrade_reason or "")
        assert "本地确定性报告" in out.content
        assert out.diagnostics["deadline"]["state"] == EXPIRED

    def test_normal_case_is_untouched_when_deadline_disabled(self, monkeypatch):
        _patch_expired(monkeypatch, budget_ms=0)
        llm = FakeLlm()
        svc = _svc(llm)
        out = svc.analyze(1, "这家企业最近有什么风险", 5, user_id=1)
        assert out.degrade_level == "NONE"
        assert "一、结论" in out.content
        assert llm.calls, "没到点就该照常调用模型"

    def test_soft_deadline_still_delivers_a_model_answer(self, monkeypatch):
        """软截止不是降级：模型那一次调用照发，只是不许再调工具。"""
        _patch_expired(monkeypatch, elapsed_ms=200_000)
        llm = FakeLlm()
        svc = _svc(llm)
        out = svc.analyze(1, "这家企业最近有什么风险", 5, user_id=1)
        assert out.degrade_level == "NONE", "收敛轮拿到正文就不该判降级"
        assert llm.calls, "收敛轮这一次调用照发"
        assert llm.calls[0]["tools"] == [], "收敛轮不许再带工具"
