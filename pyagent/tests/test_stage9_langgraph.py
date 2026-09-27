# -*- coding: utf-8 -*-
"""LangGraph 编排分支的回归测试（**全程离线、零 token**）。

覆盖三件事：
1. **并存不串线** —— 配置怎么切就把请求分派到哪条实现，未知值退回 legacy。
2. **单一数据源** —— 工具 schema 与自研 ``RiskAgentTools.specs()`` 一致，不能两边各写一份。
3. **轮换降级不丢** —— ChatOpenAI 本身没有模型轮换，429 要换下一个候选，401 不许换。
"""

from __future__ import annotations

import json

import pytest

from app.agent import langgraph_impl as lg
from app.ai.llm_client import ChatMessage, ToolCall
from app.ai.tools import ToolSpec
from app.config import get_settings

pytest.importorskip("langgraph", reason="langgraph 未安装")
pytest.importorskip("langchain_core", reason="langchain-core 未安装")


# --------------------------------------------------------------------------- #
# 假的 LangChain 对话模型
# --------------------------------------------------------------------------- #

class _FakeStatusError(Exception):
    """模拟 langchain/openai 抛出的 HTTP 错误（带 status_code）。"""

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeModel:
    """按脚本推进的假模型。

    ``script`` 是一串动作：``("ai", content)`` 直接回正文；``("tool", name, args)`` 回一次工具调用。
    同一轮要多次尝试不同模型时，用 ``fail_first`` 让**第一个**模型先失败。
    """

    def __init__(self, model: str, script: list, calls: list) -> None:
        self.model = model
        #! **绝不能 list(script)**：每次 build 都拷贝一份，脚本就永远消耗不完，
        #! 现象是"每轮都在调同一个工具、最后 content 为空"——假模型自己的 bug，别去改实现。
        self._script = script
        self._calls = calls

    def bind_tools(self, tools):  # noqa: D102
        self._tools = tools
        return self

    def stream(self, messages):  # noqa: D102
        """模拟真实模型的 SSE 流式逐片段返回。

        ⚠️ 同一个 id 是必须的：LangChain 聚合 chunk 时会校验 id，
        id 不一致会在 `full + piece` 处抛错 —— 那是**测试桩**的问题，别去改实现。
        """
        from langchain_core.messages import AIMessageChunk

        msg = self.invoke(messages)
        cid = f"chunk-{self.model}"
        if getattr(msg, "tool_calls", None):
            yield AIMessageChunk(content="", tool_calls=msg.tool_calls, id=cid)
            return
        text = msg.content or ""
        for i in range(0, max(len(text), 1), 3):
            yield AIMessageChunk(content=text[i:i + 3], id=cid)

    def invoke(self, messages):  # noqa: D102
        from langchain_core.messages import AIMessage

        self._calls.append((self.model, len(messages)))
        if not self._script:
            return AIMessage(content="收尾。")
        step = self._script.pop(0)
        if step[0] == "ai":
            return AIMessage(content=step[1])
        if step[0] == "tool":
            return AIMessage(content="", tool_calls=[{
                "id": f"call_{len(self._calls)}",
                "type": "function",
                "name": step[1],
                "args": step[2] or {},
            }])
        if step[0] == "raise":
            raise step[1]
        if step[0] == "both":  # 同一轮里既有草稿正文、又要调工具（模拟流式事故的来源）
            return AIMessage(content=step[1], tool_calls=[{
                "id": f"call_{len(self._calls)}",
                "type": "function",
                "name": step[2],
                "args": {},
            }])
        raise AssertionError(f"未知脚本动作：{step}")


def _factory(script, calls, fail_models=()):
    """构造 ModelChain 需要的 model_factory。"""

    def build(model: str):
        if model in fail_models:
            return _FakeModel(model, [("raise", _FakeStatusError(429, "SetLimitExceeded"))], calls)
        return _FakeModel(model, script, calls)

    return build


# --------------------------------------------------------------------------- #
# 1. 配置切换：legacy / langgraph 并存
# --------------------------------------------------------------------------- #

def _build_service(monkeypatch, orchestrator: str):
    monkeypatch.setenv("APP_AGENT_ORCHESTRATOR", orchestrator)
    #! get_settings 是 lru_cache：不清缓存的话 setenv 对已缓存实例没有任何作用，
    #! 会得到一个"明明配了却不生效"的假失败（本项目已在 Key 变量上栽过同类的坑）。
    get_settings.cache_clear()
    from app.agent.agent_service import AgentService

    monkeypatch.setattr("app.agent.agent_service.get_db", lambda: object())
    monkeypatch.setattr("app.agent.agent_service.GuardrailService", lambda: object())
    monkeypatch.setattr("app.agent.agent_service.ActionApprovalService", lambda _db: object())
    monkeypatch.setattr("app.agent.agent_service.AnswerGroundingService", lambda: object())
    monkeypatch.setattr("app.agent.agent_service.ConversationMemory", lambda _db: object())
    monkeypatch.setattr("app.agent.agent_service.LongTermMemory", lambda _db: object())
    monkeypatch.setattr("app.agent.agent_service.MemoryExtractor", lambda: object())
    monkeypatch.setattr("app.agent.agent_service.ReviewLoopService", lambda **kw: object())
    monkeypatch.setattr("app.agent.agent_service.Resilience", lambda **kw: object())
    return AgentService()


def test_default_orchestrator_is_langgraph(monkeypatch):
    """LangGraph 是默认、**也是唯一**的编排实现（自研 legacy 已删除）。"""
    monkeypatch.delenv("APP_AGENT_ORCHESTRATOR", raising=False)
    from app.config import AgentSettings

    assert AgentSettings().orchestrator == "langgraph"


def test_legacy_value_falls_back_to_langgraph_and_warns(monkeypatch, caplog):
    """历史环境变量/部署脚本里残留 ``legacy`` 时必须能用，但**必须告警**。

    宁可按唯一实现继续跑，也不要因为一行旧配置让 AI 整体不可用；
    可是悄悄处理的话，运维会以为自己配的是另一套编排 —— 这类"配了却没生效"
    在本项目已经因为 Key 变量栽过一次，不能再来一次。
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="app.agent.agent_service"):
        svc = _build_service(monkeypatch, "legacy")
    assert svc.orchestrator == "langgraph"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("legacy" in m for m in msgs), f"残留的 legacy 配置没有告警：{msgs}"


def test_orchestrator_switch(monkeypatch):
    svc = _build_service(monkeypatch, "langgraph")
    assert svc.orchestrator == "langgraph"


def test_unknown_orchestrator_falls_back_to_langgraph(monkeypatch):
    """拼错单词不该让 AI 整体不可用。"""
    svc = _build_service(monkeypatch, "LanggraphV2")
    assert svc.orchestrator == "langgraph"


def test_only_one_orchestrator_is_registered():
    """位编排常量的**唯一性**本身要被钉住。

    以前 ``ORCHESTRATORS`` 有两个值，切换是不可信能力的来源；现在只剩一个，
    这条断言防止有人"顺手"把 legacy 加回来却没有对应的实现。
    """
    from app.agent.agent_service import AgentService
    from app.agent.langgraph_impl import ORCHESTRATORS

    assert ORCHESTRATORS == ("langgraph",)
    assert not hasattr(AgentService, "_tool_loop"), \
        "自研工具循环已删除；若要恢复它，必须同时恢复 deadline 接线与单次超时收紧"


def test_dispatch_delegates_to_langgraph(monkeypatch):
    """入口转发到 LangGraph —— 入口保持单点，流式那一路才不会漏接参数。"""
    import inspect

    from app.agent import agent_service as asm

    src = inspect.getsource(asm.AgentService._dispatch_loop)
    assert "_langgraph_loop" in src
    assert "_tool_loop" not in src


# --------------------------------------------------------------------------- #
# 2. 单一数据源：工具 schema 来自 specs()
# --------------------------------------------------------------------------- #

def test_to_openai_tools_matches_specs():
    specs = [
        ToolSpec("get_metrics", "查指标", '{"type":"object","properties":{"limit":{"type":"integer"}}}'),
        ToolSpec("web_search", "联网", "{}"),
    ]
    tools = lg.to_openai_tools(specs)
    assert [t["function"]["name"] for t in tools] == ["get_metrics", "web_search"]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["parameters"]["properties"]["limit"]["type"] == "integer"
    # 空 schema 也要补出 type=object，否则部分服务商报 400
    assert tools[1]["function"]["parameters"]["type"] == "object"


def test_to_openai_tools_survives_broken_schema():
    """schema 坏了不能拖垮整次分析：ToolSpec 构造时已兜底成空 object。"""
    tools = lg.to_openai_tools([ToolSpec("x", "d", "{not-json")])
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert tools[0]["function"]["parameters"]["properties"] == {}


def test_message_roundtrip():
    msgs = [
        ChatMessage.system("你是风控总监"),
        ChatMessage.user("毛利率为什么下滑"),
    ]
    lc_msgs = lg.to_langchain_messages(msgs)
    assert [type(m).__name__ for m in lc_msgs] == ["SystemMessage", "HumanMessage"]

    ai = _FakeModel("m", [], []).bind_tools([])
    from langchain_core.messages import AIMessage

    ai_msg = AIMessage(content="", tool_calls=[{
        "id": "c1", "type": "function", "name": "get_metrics", "args": {"limit": 3},
    }])
    calls = lg.lc_to_tool_calls(ai_msg)
    assert len(calls) == 1
    assert calls[0].name == "get_metrics"
    assert json.loads(calls[0].arguments_json) == {"limit": 3}


# --------------------------------------------------------------------------- #
# 3. 模型轮换：429 才换，401 不换
# --------------------------------------------------------------------------- #

def test_model_chain_rotates_on_429():
    calls: list = []
    chain = lg.ModelChain(
        base_url="http://127.0.0.1:9099/v1", api_key="sk-x", model="model-a",
        candidates=["model-b"], tries=3, model_factory=_factory(
            [("ai", "来自备用模型的答案")], calls, fail_models=("model-a",)),
    )
    from langchain_core.messages import HumanMessage

    ai, used = chain.invoke([HumanMessage(content="hi")], None)
    assert used == "model-b"
    assert ai.content == "来自备用模型的答案"
    assert [a.ok for a in chain.attempts] == [False, True]


def test_model_chain_does_not_rotate_on_401():
    """Key/账号错了换一百个模型也没用 —— 还会把真实原因（401）掩盖成"模型不可用"。"""
    calls: list = []
    chain = lg.ModelChain(
        base_url="http://127.0.0.1:9099/v1", api_key="sk-bad", model="model-a",
        candidates=["model-b"], tries=3,
        model_factory=lambda m: _FakeModel(m, [("raise", _FakeStatusError(401, "invalid key"))], calls),
    )
    from langchain_core.messages import HumanMessage

    with pytest.raises(_FakeStatusError):
        chain.invoke([HumanMessage(content="hi")], None)
    assert len(chain.attempts) == 1
    assert chain.attempts[0].model == "model-a"


def test_model_chain_gives_up_after_tries():
    calls: list = []
    chain = lg.ModelChain(
        base_url="http://x", api_key="k", model="m1", candidates=["m2", "m3", "m4"],
        tries=2, model_factory=lambda m: _FakeModel(m, [("raise", _FakeStatusError(429, "limit"))], calls),
    )
    from langchain_core.messages import HumanMessage

    with pytest.raises(_FakeStatusError):
        chain.invoke([HumanMessage(content="hi")], None)
    assert [a.model for a in chain.attempts] == ["m1", "m2"]  # 上限 2 次


# --------------------------------------------------------------------------- #
# 4. 图：工具轮执行 + 最后一轮收敛
# --------------------------------------------------------------------------- #

def test_graph_runs_tools_then_converges(monkeypatch):
    calls: list = []
    seen_calls: list = []

    def dispatch(tool_calls, _msgs):
        for tc in tool_calls:
            seen_calls.append((tc.name, json.loads(tc.arguments_json or "{}")))
        return {tc.id or "": '{"ok":true,"rows":3}' for tc in tool_calls}

    chain = lg.ModelChain(
        base_url="", api_key="", model="m1", candidates=[], tries=1,
        model_factory=_factory([("tool", "get_metrics", {"limit": 3}), ("ai", "毛利率下滑 2pt")], calls),
    )
    content, used = lg.run(
        messages=[ChatMessage.system("sys"), ChatMessage.user("毛利率为什么下滑")],
        specs=[ToolSpec("get_metrics", "查指标", '{"type":"object","properties":{"limit":{"type":"integer"}}}')],
        dispatch_tools=dispatch,
        max_rounds=3,
        thread_id="ut-graph-1",
        model_chain=chain,
        use_checkpoint=False,
    )
    assert content == "毛利率下滑 2pt"
    assert used == "m1"
    assert seen_calls == [("get_metrics", {"limit": 3})]
    # 至少两轮：一轮工具 + 一轮成稿
    assert len(calls) >= 2


def test_graph_stops_after_max_rounds_when_model_never_converges():
    """模型一直要调工具时，图必须在 max_rounds 处**明确结束**。

    结束时 content 为空，这**不是缺陷**——上层 `_run` 有"空正文自愈 → 本地确定性报告"
    两级兜底。真正要防的是 LangGraph 抛 GraphRecursionError 变成 HTTP 500。
    """
    calls: list = []

    def dispatch(tool_calls, _msgs):
        return {tc.id or "": "{}" for tc in tool_calls}

    # 脚本里全是工具调用，永不收敛
    chain = lg.ModelChain(
        base_url="", api_key="", model="m1", candidates=[], tries=1,
        model_factory=_factory([("tool", "get_metrics", {}), ("tool", "get_metrics", {}),
                                ("tool", "get_metrics", {}), ("tool", "get_metrics", {})], calls),
    )
    content, _used = lg.run(
        messages=[ChatMessage.user("q")],
        specs=[ToolSpec("get_metrics", "查指标", "{}")],
        dispatch_tools=dispatch,
        max_rounds=2,
        thread_id="ut-graph-3",
        model_chain=chain,
        use_checkpoint=False,
    )
    # 轮数上限生效：既不无限循环，也不抛 GraphRecursionError
    assert len(calls) == 2
    assert content == ""


def test_budget_gate_drops_tools_and_converges():
    """预算闸门触发时：① 不再下发工具 ② 图在有限轮内结束 ③ content 交给上层自愈。

    content 为空**不是缺陷**：上层 `_run` 有"空正文自愈 → 本地确定性报告"两级兜底。
    真正要防的是预算失效导致深度档无限调用工具。
    """
    calls: list = []
    binds: list = []

    class SpyModel(_FakeModel):
        def bind_tools(self, tools):  # noqa: D102
            binds.append(len(tools or []))
            return self

    script = [("tool", "get_metrics", {}), ("ai", "收尾")]
    factory = lambda model: SpyModel(model, script, calls)  # noqa: E731

    chain = lg.ModelChain(base_url="", api_key="", model="m1", candidates=[], tries=1,
                          model_factory=factory)
    content, _used = lg.run(
        messages=[ChatMessage.user("q")],
        specs=[ToolSpec("get_metrics", "查指标", "{}")],
        dispatch_tools=lambda tcs, _m: {tc.id or "": "{}" for tc in tcs or []},
        max_rounds=4,
        thread_id="ut-graph-2",
        model_chain=chain,
        use_checkpoint=False,
        stop_check=lambda: "工具配额已用尽",
    )
    # 预算一直不允许 → 一次工具都没下发，图也立刻结束
    assert binds == 0 or binds == []
    assert len(calls) == 1
    assert content == ""


# --------------------------------------------------------------------------- #
# 4.5 逐 token 流式：只有「正文」能流向用户
# --------------------------------------------------------------------------- #

_FINAL_TEXT = "毛利率下滑 2pt，主因是原材料涨价"


def _stream_chain(script, calls):
    return lg.ModelChain(base_url="", api_key="", model="m1", candidates=[], tries=1,
                         model_factory=_factory(script, calls))


def test_token_hook_receives_exactly_the_answer():
    """推给用户的字节必须**正好等于**正文，不多不少。"""
    calls: list = []
    pushed: list = []
    seen: list = []

    def dispatch(tool_calls, _msgs):
        seen.extend(tc.name for tc in tool_calls)
        return {tc.id or "": '{"ok":true}' for tc in tool_calls}

    content, _used = lg.run(
        messages=[ChatMessage.user("毛利率为什么下滑")],
        specs=[ToolSpec("get_metrics", "查指标", "{}")],
        dispatch_tools=dispatch,
        max_rounds=3,
        thread_id="ut-graph-stream-1",
        model_chain=_stream_chain([("tool", "get_metrics", {}), ("ai", _FINAL_TEXT)], calls),
        use_checkpoint=False,
        token_hook=pushed.append,
    )
    assert content == _FINAL_TEXT
    assert seen == ["get_metrics"]                 # 工具照常执行，没被流式干扰
    assert "".join(pushed) == _FINAL_TEXT          # 内容一致
    assert len(pushed) > 1                         # 且是分片的，不是整块塞给用户


def test_no_tool_round_forwards_tokens_before_model_stream_finishes():
    """没有工具声明时必须边生成边推，不能在整轮结束后补推。"""
    from langchain_core.messages import AIMessageChunk

    events: list[str] = []

    class NoToolModel:
        def stream(self, _messages):
            events.append("yield-1")
            yield AIMessageChunk(content="第一段", id="live")
            events.append("yield-2")
            yield AIMessageChunk(content="第二段", id="live")
            events.append("stream-finished")

    chain = lg.ModelChain(
        base_url="", api_key="", model="m1", candidates=[], tries=1,
        model_factory=lambda _model: NoToolModel(),
    )
    content, _used = lg.run(
        messages=[ChatMessage.user("q")],
        specs=[],
        max_rounds=2,
        thread_id="ut-graph-live-no-tools",
        model_chain=chain,
        use_checkpoint=False,
        token_hook=lambda token: events.append("hook:" + token),
    )

    assert content == "第一段第二段"
    assert events == [
        "yield-1", "hook:第一段",
        "yield-2", "hook:第二段",
        "stream-finished",
    ]


def test_token_hook_drops_the_draft_of_a_tool_round():
    """模型"先写两句再去调工具"时，那两句草稿**不许**进正文。

    这是流式最容易出的事故：界面上多出来一段模型自言自语，用户会把它当成结论的一部分。
    """
    calls: list = []
    pushed: list = []

    def dispatch(tool_calls, _msgs):
        return {tc.id or "": "{}" for tc in tool_calls}

    content, _used = lg.run(
        messages=[ChatMessage.user("q")],
        specs=[ToolSpec("get_metrics", "查指标", "{}")],
        dispatch_tools=dispatch,
        max_rounds=3,
        thread_id="ut-graph-stream-2",
        # 第一轮：既有草稿正文、又要调工具
        model_chain=_stream_chain(
            [("both", "我先看一下指标数据", "get_metrics"), ("ai", "毛利率下滑 2pt")], calls),
        use_checkpoint=False,
        token_hook=pushed.append,
    )
    assert content == "毛利率下滑 2pt"
    assert "我先看一下指标数据" not in "".join(pushed)
    assert "".join(pushed) == "毛利率下滑 2pt"


def test_streamed_chunk_keeps_tool_calls():
    """流式拼回来的消息必须仍是可识别的工具调用。

    ``AIMessageChunk`` **不是** ``AIMessage`` 的实例：不转的话 isinstance 判定会把它
    降级成纯文本消息，tool_calls 当场丢失 —— 图会在工具还没执行时就收敛出结论。
    """
    from langchain_core.messages import AIMessage, HumanMessage

    calls: list = []
    ai, _used = _stream_chain([("tool", "get_metrics", {"limit": 3})], calls).invoke(
        [HumanMessage(content="q")], None, token_hook=lambda t: None)
    assert isinstance(ai, AIMessage)
    calls_back = lg.lc_to_tool_calls(ai)
    assert [c.name for c in calls_back] == ["get_metrics"]
    assert json.loads(calls_back[0].arguments_json) == {"limit": 3}


def test_token_hook_failure_does_not_break_inference():
    """SSE 连接断了，模型也该把结论跑完并落库 —— 半途腰斩会让这一整次提问消失。"""
    calls: list = []

    def boom(_t: str) -> None:
        raise RuntimeError("SSE 连接已断开")

    def dispatch(tool_calls, _msgs):
        return {tc.id or "": "{}" for tc in tool_calls}

    content, _used = lg.run(
        messages=[ChatMessage.user("q")],
        specs=[ToolSpec("get_metrics", "查指标", "{}")],
        dispatch_tools=dispatch,
        max_rounds=2,
        thread_id="ut-graph-stream-3",
        model_chain=_stream_chain([("ai", _FINAL_TEXT)], calls),
        use_checkpoint=False,
        token_hook=boom,
    )
    assert content == _FINAL_TEXT


# --------------------------------------------------------------------------- #
# 5. 集成：LangGraph 分支必须复用同一套工具执行链路
# --------------------------------------------------------------------------- #

class _RecStub:
    """``TraceRecorder`` 的最小替身（只用到这三个方法），避免单测连库。"""

    def __init__(self) -> None:
        self.tools: list = []
        self.degrades: list = []
        self.llms: list = []
        self.iterations = 0

    def note_tool(self, run) -> None:  # noqa: D102
        self.tools.append(getattr(run, "name", ""))

    def note_degrade(self, level, reason) -> None:  # noqa: D102
        self.degrades.append(reason)

    def note_llm(self, model, finish) -> None:  # noqa: D102
        self.llms.append(model)


class _ToolsStub:
    """``RiskAgentTools`` 的最小替身：只为证明 **调用确实穿过了 `execute_with_meta`**。

    工具节点现在走官方 ``ToolNode``，所以替身也得提供 ``as_langchain_tools``——
    但**里面仍然只有一个出口**：``execute_with_meta``。
    这条正是"换外观不能换执行通道"的守卫：要是哪天有人给 LangGraph 开了第二条
    通道，``seen`` 会是空的，这条测试当场变红。
    """

    def __init__(self) -> None:
        self.seen: list = []

    def execute_with_meta(self, name, arguments_json, ctx) -> ToolResult:
        self.seen.append((name, json.loads(arguments_json or "{}"), int(ctx.company_id or 0)))
        return ToolResult(text='{"ok":true,"rows":3}', sources=[], source_type="internal")

    def as_langchain_tools(self, ctx, on_result=None) -> list:
        from langchain_core.tools import StructuredTool

        from app.ai.tool_schemas import args_model

        def _invoke(**kwargs) -> str:
            # 与真实 ``RiskAgentTools._lc_invoke`` 同口径：滤掉 schema 补出来的 None
            args = {k: v for k, v in kwargs.items() if v is not None}
            res = self.execute_with_meta("get_metrics", json.dumps(args, ensure_ascii=False), ctx)
            if on_result is not None:
                on_result("get_metrics", res)
            return res.text

        return [StructuredTool.from_function(func=_invoke, name="get_metrics",
                                             description="查指标",
                                             args_schema=args_model("get_metrics"))]


def test_langgraph_loop_shares_the_same_tool_pipeline(monkeypatch):
    """最关键的一条：LangGraph **不许**单开工具执行路径。

    工具执行必须走 ``RiskAgentTools.execute_with_meta``——数据权限闸门就住在那里，
    一旦图为了"方便"绕过它，Agent 就获得了超越当前用户的取数能力。
    """
    from app.agent.budget import AgentBudget
    from app.ai.tool_context import ToolContext
    from app.ai.tools import ToolResult

    calls: list = []
    svc = _build_service(monkeypatch, "langgraph")
    tools = _ToolsStub()
    svc.tools = tools
    svc._lg_model_factory = _factory(
        [("tool", "get_metrics", {"limit": 3}), ("ai", "毛利率下滑 2pt")], calls)

    rec = _RecStub()
    evidence: list = []
    trace: list = []
    out = svc._langgraph_loop(
        messages=[ChatMessage.user("毛利率为什么下滑")],
        specs=[ToolSpec("get_metrics", "查指标", '{"type":"object","properties":{"limit":{"type":"integer"}}}')],
        ctx=ToolContext(company_id=1, top_k=5, user_id=1, question="毛利率为什么下滑", max_web_calls=4),
        budget=AgentBudget.start("standard"),
        evidence=evidence,
        tool_trace=trace,
        rec=rec,
        trace_id="ut-lg-loop",
    )
    assert out.content == "毛利率下滑 2pt"
    # 工具确实被打到了（没有绕过 execute_with_meta）
    assert tools.seen and tools.seen[0][0] == "get_metrics"
    assert tools.seen[0][1] == {"limit": 3}
    # 与自研同口径：证据池、工具留痕、使用的模型名都要进账
    assert rec.tools == ["get_metrics"]
    assert trace and trace[0].startswith("get_metrics(")
    # 模型名走的是"本次实际生效的模型"（可能与配置不同：轮换降级后的候选）
    assert out.model and rec.llms == [out.model]


def test_available_does_not_throw():
    ok, why = lg.available()
    assert isinstance(ok, bool)
    assert isinstance(why, str)


def test_settings_expose_new_fields():
    s = get_settings()
    assert hasattr(s.agent, "orchestrator")
    assert s.agent.langgraph_checkpoint_path.endswith(".sqlite")


# --------------------------------------------------------------------------- #
# 6. 采样参数必须与自研循环一致
# --------------------------------------------------------------------------- #

def test_model_chain_defaults_to_shared_temperature():
    """``ChatOpenAI`` 默认 temperature 是 1.0，不显式传就会和自研循环的 0.2 差一个量级。"""
    chain = lg.ModelChain(base_url="", api_key="", model="m", candidates=[])
    assert chain.temperature == lg.AGENT_TEMPERATURE
    assert chain.temperature == 0.2


def test_legacy_loop_uses_the_same_temperature_constant():
    """AST 扫描：自研循环的温度参数必须引用同一常量，不许写死成字面量。

    这个缺陷（一边 0.2、一边默认 1.0）在日志里只表现为「模型有点飘」，
    排查时很难想到采样参数，所以直接把它钉成源码级契约。
    """
    import ast
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "app" / "agent" / "agent_service.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if getattr(f, "attr", None) != "chat_with_tools":
            continue
        # 形如 chat_with_tools(specs, messages, <temperature>, max_tokens)
        if len(node.args) < 3:
            continue
        hits.append(node.args[2])
    assert hits, "没找到 chat_with_tools 调用点，契约失效了（可能已重命名）"
    for arg in hits:
        assert isinstance(arg, ast.Name) and arg.id == "AGENT_TEMPERATURE", (
            "自研循环的温度必须引用 AGENT_TEMPERATURE，不能写死字面量 —— "
            "否则它会与 LangGraph 分支静默漂移"
        )
