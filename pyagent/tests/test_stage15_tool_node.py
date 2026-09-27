"""阶段 15：工具节点 = 官方 ``ToolNode`` + ``wrap_tool_call`` 挂闸门。

钉的是四件事——换外观**不能**换行为：

1. 执行仍然只有 ``execute_with_meta`` 一个出口（权限闸门在里面）；
2. 预算 / 到期闸门还在，且**理由是回灌给模型**的（不是静默跳过）；
3. 留痕与证据池照旧进账（``sources`` 不能因为"LangChain 工具只返回文本"而丢掉）；
4. 并行变体与官方串行节点**行为等价**：顺序、条数、内容都一样。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from app.ai.messages import tool_call
from app.ai.tool_context import ToolContext
from app.ai.tools import ToolResult
from app.agent.tool_node import (
    ParallelToolNode,
    ToolGate,
    build_tool_node,
    ensure_runtime,
    evidence_entry,
    make_tool_call_wrapper,
)


# --------------------------------------------------------------------------- #
# 桩
# --------------------------------------------------------------------------- #

class _FakeTools:
    """``RiskAgentTools`` 的最小替身，只提供工具节点需要的 ``as_langchain_tools``。"""

    def __init__(self, text: str = '{"ok":1}') -> None:
        self.seen: List[tuple] = []
        self.text = text
        self.on_result_calls: List[str] = []

    def as_langchain_tools(self, ctx, on_result=None) -> list:
        from langchain_core.tools import StructuredTool

        from app.ai.tool_schemas import args_model

        def _mk(name: str):
            def _invoke(**kw: Any) -> str:
                args = {k: v for k, v in kw.items() if v is not None}
                self.seen.append((name, args))
                res = ToolResult(text=self.text, source_type="internal",
                                 sources=[{"title": "证据一", "sourceRef": "7"}])
                if on_result is not None:
                    on_result(name, res)
                    self.on_result_calls.append(name)
                return res.text

            return StructuredTool.from_function(func=_invoke, name=name,
                                                description=name, args_schema=args_model(name))

        return [_mk("get_metrics"), _mk("get_complaints")]


class _Budget:
    def __init__(self, deny: Optional[str] = None) -> None:
        self.deny = deny
        self.asked: List[str] = []

    def try_consume(self, name: str, used: int, prefetch: bool) -> Optional[str]:
        self.asked.append(name)
        return self.deny


class _Deadline:
    def __init__(self, expired: bool = False) -> None:
        self.expired = expired


class _Rec:
    def __init__(self) -> None:
        self.tools: List[str] = []

    def note_tool(self, run) -> None:
        self.tools.append(run.name)


def _ctx() -> ToolContext:
    return ToolContext(company_id=1, top_k=5, user_id=1, question="毛利率为什么下滑",
                       max_web_calls=4)


def _state(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    from langchain_core.messages import AIMessage

    return {"messages": [AIMessage(content="", tool_calls=calls)], "rounds": 0}


def _calls() -> List[Dict[str, Any]]:
    return [tool_call("c1", "get_metrics", {"limit": 3}),
            tool_call("c2", "get_complaints", {"limit": 2})]


# --------------------------------------------------------------------------- #
# 1. 执行只有 execute_with_meta 一个出口
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("parallel", [False, True])
def test_tool_execution_goes_through_the_single_dispatch(parallel):
    """两条路径都必须真的把工具跑起来，参数原样传到执行层。"""
    tools = _FakeTools()
    node = build_tool_node(ctx=_ctx(), tools=tools, parallel=parallel)
    out = node(_state(_calls()))
    msgs = out["messages"]
    assert len(msgs) == 2
    assert [m.content for m in msgs] == ['{"ok":1}', '{"ok":1}']
    assert sorted(tools.seen) == [("get_complaints", {"limit": 2}), ("get_metrics", {"limit": 3})]


@pytest.mark.parametrize("parallel", [False, True])
def test_results_are_written_back_in_call_order(parallel):
    """结果必须**按调用顺序**写回。

    审计日志与证据池都是有序数据：按完成顺序写会让同一批输入的日志长得不一样——
    那等于拿可追溯性去换并行省下的几百毫秒。
    """
    node = build_tool_node(ctx=_ctx(), tools=_FakeTools(), parallel=parallel)
    out = node(_state(_calls()))
    assert [m.tool_call_id for m in out["messages"]] == ["c1", "c2"]


def test_parallel_and_serial_produce_the_same_messages():
    """并行变体与官方串行节点必须**行为等价**（这是并行变体敢存在的唯一理由）。"""
    a = build_tool_node(ctx=_ctx(), tools=_FakeTools(), parallel=False)(_state(_calls()))
    b = build_tool_node(ctx=_ctx(), tools=_FakeTools(), parallel=True)(_state(_calls()))
    assert [(m.tool_call_id, m.name, m.content) for m in a["messages"]] == \
           [(m.tool_call_id, m.name, m.content) for m in b["messages"]]


# --------------------------------------------------------------------------- #
# 2. 闸门：预算 / 到期
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("parallel", [False, True])
def test_budget_denial_is_fed_back_to_the_model(parallel):
    """预算拒绝**不能**静默跳过：模型必须看到理由，否则它以为工具坏了会一直重试。"""
    tools = _FakeTools()
    node = build_tool_node(ctx=_ctx(), tools=tools, parallel=parallel,
                           budget=_Budget(deny="本次分析的工具调用配额已用尽"))
    out = node(_state(_calls()))
    assert all("配额已用尽" in m.content for m in out["messages"])
    assert tools.seen == [], "被拒的工具不许真的执行"


@pytest.mark.parametrize("parallel", [False, True])
def test_expired_deadline_skips_execution_and_explains(parallel):
    """到点了工具不执行，且理由要写清"没时间了"而不是"工具坏了"。"""
    tools = _FakeTools()
    node = build_tool_node(ctx=_ctx(), tools=tools, parallel=parallel,
                           deadline=_Deadline(expired=True))
    out = node(_state(_calls()))
    assert tools.seen == []
    assert all("时间预算已用尽" in m.content for m in out["messages"])


def test_gate_plans_serially_before_any_execution():
    """配额判定必须在执行**之前**串行做完——并行下并发扣会超发。"""
    budget = _Budget()
    gate = ToolGate(budget=budget, deadline=_Deadline(False), tool_usage={}, call_cache={})
    gate.plan(_calls())
    assert budget.asked == ["get_metrics", "get_complaints"]


def test_gate_cache_hit_short_circuits():
    """同参数重复调用直接回缓存：省的是一次真实的联网/取数往返。"""
    cache: Dict[str, str] = {}
    gate = ToolGate(tool_usage={}, call_cache=cache)
    calls = _calls()
    gate.plan(calls)  # 未命中
    cache[ToolGate.call_key(calls[0])] = "缓存里的旧答案"
    decisions = gate.plan(calls)
    assert decisions["c1"]["cached"] == "缓存里的旧答案"
    assert decisions["c2"]["cached"] is None


# --------------------------------------------------------------------------- #
# 3. 留痕 / 证据池
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("parallel", [False, True])
def test_evidence_pool_and_trace_are_still_recorded(parallel):
    """``ToolResult.sources`` 必须进证据池。

    LangChain 工具只能返回**文本**，而来源（sourceRef / 引用号）住在 ``ToolResult`` 里；
    不挂 ``on_result`` 回调，"证据溯源"会只剩预取那一批。
    """
    evidence: List[Dict[str, Any]] = []
    trace: List[str] = []
    rec = _Rec()
    node = build_tool_node(ctx=_ctx(), tools=_FakeTools(), parallel=parallel,
                           recorder=rec, evidence=evidence, tool_trace=trace)
    node(_state(_calls()))
    assert len(evidence) == 2, "每次工具调用一条，不是只在 sources 非空时才记"
    # 并行下证据池的**写入顺序**不定（执行顺序不定），所以按集合断言；
    # 顺序有要求的是 ToolMessage 列表（见上面那条"按调用顺序写回"）。
    assert {e["tool"] for e in evidence} == {"get_metrics", "get_complaints"}
    assert all(e["sources"] == [{"title": "证据一", "sourceRef": "7"}] for e in evidence)
    assert all(e["preview"].startswith('{"ok":1}') for e in evidence)
    assert len(trace) == 2 and all(t.endswith("ms)") for t in trace)
    assert sorted(rec.tools) == ["get_complaints", "get_metrics"]


def test_second_identical_call_is_served_from_cache():
    """同参数第二次调用不再执行工具（缓存命中），但**仍然留痕**。"""
    tools = _FakeTools()
    cache: Dict[str, str] = {}
    usage: Dict[str, int] = {}
    rec = _Rec()
    node = build_tool_node(ctx=_ctx(), tools=tools, parallel=False, recorder=rec,
                           call_cache=cache, tool_usage=usage)
    state = _state(_calls())
    node(state)
    assert len(tools.seen) == 2
    node(state)
    assert len(tools.seen) == 2, "缓存命中就不该再打一次工具"
    assert rec.tools == ["get_metrics", "get_complaints", "get_metrics", "get_complaints"]


def test_unwired_tool_layer_reports_instead_of_crashing():
    """工具层未装配时，图退化成"没有工具可调用"，而不是整次分析崩掉。"""
    node = build_tool_node(ctx=_ctx(), tools=None)
    out = node(_state(_calls()))
    assert all("工具层未装配" in m.content for m in out["messages"])


# --------------------------------------------------------------------------- #
# 4. wrap_tool_call 是官方钩子
# --------------------------------------------------------------------------- #

def test_official_toolnode_accepts_our_wrapper():
    """闸门挂在 **ToolNode 公开的可配置项**上，不是 monkey patch。

    这条是防"哪天升级 langgraph 把钩子改名了，闸门静默失效"——
    那种失效的表现是"工具忽然没有配额限制"，极难归因。
    """
    from langgraph.prebuilt import ToolNode

    from app.ai.tool_schemas import args_model

    tools = _FakeTools().as_langchain_tools(_ctx())
    calls_seen: List[str] = []
    wrap = make_tool_call_wrapper(on_done=lambda n, ms, ok, t: calls_seen.append(n))
    node = ToolNode(tools, wrap_tool_call=wrap)
    # 官方 ToolNode 的 ``_func`` 需要一个没有默认值的 ``runtime``，
    # 图执行时由 LangGraph 注入；直接调用要自己补（见 ``ensure_runtime``）。
    out = node.invoke(_state(_calls()), config=ensure_runtime(None))
    assert [m.tool_call_id for m in out["messages"]] == ["c1", "c2"]
    assert calls_seen, "wrap_tool_call 没被调用 = 钩子没生效"


def test_wrapper_turns_tool_exception_into_a_message():
    """工具抛异常也要把理由给模型，不能让整轮工具调用消失。"""

    class _Boom:
        def as_langchain_tools(self, ctx, on_result=None) -> list:
            from langchain_core.tools import StructuredTool

            from app.ai.tool_schemas import args_model

            def _invoke(**kw: Any) -> str:
                raise RuntimeError("数据库连接中断")

            return [StructuredTool.from_function(func=_invoke, name="get_metrics",
                                                 description="d", args_schema=args_model("get_metrics"))]

    node = build_tool_node(ctx=_ctx(), tools=_Boom(), parallel=False)
    out = node(_state([tool_call("c1", "get_metrics", {})]))
    assert "工具执行失败" in out["messages"][0].content
    assert "数据库连接中断" in out["messages"][0].content


def test_parallel_node_falls_back_to_serial_when_pool_fails(monkeypatch):
    """线程池起不来时退回串行，而不是整轮工具全丢。"""
    import app.agent.tool_node as tn

    monkeypatch.setattr(tn, "ThreadPoolExecutor", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    node = tn.ParallelToolNode(_FakeTools().as_langchain_tools(_ctx()))
    out = node(_state(_calls()))
    assert len(out["messages"]) == 2


# --------------------------------------------------------------------------- #
# 5. 证据条目形状：两条路径必须一致
# --------------------------------------------------------------------------- #

def test_evidence_entry_shape_matches_the_prefetch_path():
    """工具节点与预取路径记出**同一形状**的证据条目。

    前端「证据溯源」不区分来源路径；形状一旦分叉，界面会出现"这一条没有 preview"之类的洞。
    """
    from app.agent.agent_service import AgentService

    res = ToolResult(text="x" * 3000, source_type="knowledge",
                     sources=[{"title": "t", "sourceRef": "7"}])
    a = evidence_entry("search_knowledge", res)
    b = AgentService._evidence_entry("search_knowledge", res)
    assert a == b
    assert a["preview"] == "x" * 2000, "证据正文应保留足够细节且有明确上限"
    assert a["chars"] == 3000
    assert json.dumps(a, ensure_ascii=False)  # 可序列化（要落 answer_json）
