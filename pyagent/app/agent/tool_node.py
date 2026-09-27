"""工具执行节点：**LangGraph 官方 ``ToolNode``** + 我们的闸门。

为什么不再是自研 ``dispatch_tools``
-----------------------------------
上一版这里是一个自研闭包 ``(tool_calls, messages) -> {call_id: 文本}``，
由 LangGraph 的 ``tools_node`` 手动把结果拼成 ``ToolMessage``。它的毛病和
``app/ai/messages.py`` 里那套自研 DTO 是同一类：

1. **工具身份靠字符串**：模型说要调 ``get_metrics``，我们拿这个名字去查一张 dict，
   查不到才报错。官方 ``ToolNode`` 拿的是 ``BaseTool`` 对象，
   ``tools_by_name`` 在建节点时就校验过，且**参数会过一遍 Pydantic**。
2. **错误处理各写各的**：工具抛异常、工具不存在、参数不合法，三处兜底文案不一样。
   官方 ``ToolNode`` 有统一的 ``handle_tool_errors``。
3. **进不了生态**：``create_react_agent`` / 各种 prebuilt 组件只吃 ``BaseTool``；
   用自研闭包就永远只能自己拼图。

现在改成：**节点就是官方 ``ToolNode``**，我们的闸门/留痕/预算挂在它**官方提供的
``wrap_tool_call`` 钩子**上（不是 monkey patch，是它公开的可配置项）。

闸门一个都不能少
----------------
换外观**不等于**换执行通道。四件事必须还在，且有用例钉住：

* **权限闸门**：在 :meth:`RiskAgentTools.execute_with_meta` 里，``BaseTool`` 只是它的外壳。
* **预算闸门**：:class:`ToolGate` 在**执行前串行**判定（并行下绝不能并发扣配额）。
* **到期闸门**：到点了工具不执行，且**把理由回灌给模型**（"不是工具坏了，是没时间了"）。
* **留痕 / 证据池**：走 ``on_result`` 回调 —— ``ToolResult`` 里有 ``sources``，
  而 LangChain 工具只能返回文本，不挂回调就会丢掉来源。

并行
----
官方 ``ToolNode`` 是**串行**执行的。同轮多个工具并行是我们自己做的性能优化
（联网检索最慢，串行会把整份时间预算吃在等待上），所以保留一个 :class:`ParallelToolNode`：
它**复用同一批 ``BaseTool`` 与同一个 ``wrap_tool_call``**，只是把执行阶段铺到线程池，
结果**按原始调用顺序**写回（审计日志必须确定）。两条路径的行为等价性由测试钉住。
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from ..ai.messages import args_of, id_of, name_of, tool_calls_of

log = logging.getLogger(__name__)

# Keep enough tool output for the drafting model to see actual metrics,
# complaints and knowledge snippets. The old 200-character preview often kept
# only the heading and first row, so a successful tool call still produced a
# thin final report. The prompt builder applies an additional total cap.
EVIDENCE_PREVIEW_CHARS = 2000


# --------------------------------------------------------------------------- #
# 证据条目
# --------------------------------------------------------------------------- #

def evidence_entry(tool: str, res: Any, attempts: int = 1, cached: bool = False) -> Dict[str, Any]:
    """一条证据条目。**单一实现**——``AgentService._evidence_entry`` 委托到这里。

    早先这里只在 ``res.sources`` 非空时才记一条，且只写 ``sources`` 不写 ``preview``：
    于是模型在循环里调的 ``get_metrics`` / ``get_risk_events`` 之类**根本不进证据池**，
    前端「证据溯源」只剩知识库一路，而 Java 侧是每次工具调用一条。
    条数对不上还是小事，真正的问题是"这次到底取了哪些数"变得不可追溯。
    """
    text = getattr(res, "text", None) or ""
    ev: Dict[str, Any] = {
        "tool": tool,
        "sourceType": getattr(res, "source_type", None) or "internal",
        "preview": text[:EVIDENCE_PREVIEW_CHARS],
        "chars": len(text),
        "attempts": attempts,
        "cached": bool(cached),
    }
    sources = getattr(res, "sources", None)
    if sources:
        ev["sources"] = sources
    return ev


# --------------------------------------------------------------------------- #
# 闸门：只做"让不让跑"，不做执行
# --------------------------------------------------------------------------- #

class ToolGate:
    """执行前的**串行**判定：预算 / 到期 / 缓存。

    .. WARNING::
        :meth:`plan` 必须在主线程**串行**调一次。配额是共享状态，
        放到并行分支里扣会出现"两个线程同时判定通过、实际超发"——
        那类 bug 只在压力上来的偶发超时里露面，极难复现。
    """

    def __init__(
        self,
        *,
        budget: Any = None,
        deadline: Any = None,
        tool_usage: Optional[Dict[str, int]] = None,
        call_cache: Optional[Dict[str, str]] = None,
    ) -> None:
        self.budget = budget
        self.deadline = deadline
        self.usage = tool_usage if tool_usage is not None else {}
        self.cache = call_cache if call_cache is not None else {}

    @staticmethod
    def call_key(call: Any) -> str:
        """与自研循环同一套：``name|arguments``。改了它缓存就全 miss。"""
        return f"{name_of(call)}|{json.dumps(args_of(call), ensure_ascii=False, sort_keys=True)}"

    def plan(self, calls: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
        """对一批 tool_call 做一次串行判定。

        返回 ``{call_id: {"denied": 理由|None, "cached": 文本|None}}``。
        """
        out: Dict[str, Dict[str, Any]] = {}
        for c in calls or []:
            name = name_of(c)
            cid = id_of(c)
            denied: Optional[str] = None
            cached: Optional[str] = None
            if self.budget is not None:
                denied = self.budget.try_consume(name, self.usage.get(name or "", 0), False)
            if denied is None and self.deadline is not None and getattr(self.deadline, "expired", False):
                #! 到点后工具也不再跑：最慢的就是联网检索，让它跑完只会更迟。
                #! 拒绝理由照样回灌给模型 —— 它得知道"不是工具坏了，是没时间了"。
                denied = ("本次分析的时间预算已用尽，该工具未执行。"
                          "请立即基于上面已经取到的证据给出最终结论，"
                          "需要外部信息佐证的在「四、不确定性」里写明「本次未联网核实」。")
            if denied is None:
                cached = self.cache.get(self.call_key(c))
            out[cid] = {"denied": denied, "cached": cached}
        return out


# --------------------------------------------------------------------------- #
# wrap_tool_call：官方钩子，挂预算/到期/异常兜底
# --------------------------------------------------------------------------- #

def make_tool_call_wrapper(
    *,
    gate: Optional[ToolGate] = None,
    decisions: Any = None,
    on_done: Optional[Callable[[str, int, bool, str], None]] = None,
) -> Callable[[Any, Callable[[Any], Any]], Any]:
    """构造 ``ToolNode(wrap_tool_call=...)``。

    :param decisions: :meth:`ToolGate.plan` 的结果（``call_id → {denied, cached}``），
        也可以是**返回它的可调用对象** —— 节点每轮都会重新判定，
        传快照会让第二轮读到空表（表现是所有工具都变成"允许且未缓存"，静默不报错）。
    :param on_done: ``(工具名, 耗时ms, 成功与否, 结果文本)``，用于留痕。
    """
    def _wrap(request: Any, execute: Callable[[Any], Any]) -> Any:
        table = decisions() if callable(decisions) else (decisions or {})
        call = request.tool_call
        name = name_of(call)
        cid = id_of(call)
        d = table.get(cid) or {}

        denied = d.get("denied")
        if denied:
            if on_done is not None:
                on_done(name, 0, False, denied)
            return ToolMessage(content=denied, tool_call_id=cid, name=name)
        cached = d.get("cached")
        if cached is not None:
            # 命中缓存：文本直接回灌，但**仍然要留痕**（审计看的是"模型问过什么"）
            if on_done is not None:
                on_done(name, 0, True, cached)
            return ToolMessage(content=cached, tool_call_id=cid, name=name)

        started = int(time.time() * 1000)
        try:
            msg = execute(request)
            if on_done is not None:
                on_done(name, int(time.time() * 1000) - started, True,
                        str(getattr(msg, "content", "") or ""))
            return msg
        except Exception as e:  # noqa: BLE001 - 工具炸了也要把理由给模型，不能让整轮消失
            ms = int(time.time() * 1000) - started
            text = '{"error":"工具执行失败: ' + str(e)[:200] + '"}'
            if on_done is not None:
                on_done(name, ms, False, text)
            return ToolMessage(content=text, tool_call_id=cid, name=name, status="error")

    return _wrap


# --------------------------------------------------------------------------- #
# 并行变体：与 ToolNode 行为一致，只是把执行阶段铺到线程池
# --------------------------------------------------------------------------- #

class ParallelToolNode:
    """``ToolNode`` 的并行变体：**同一批 BaseTool + 同一个 wrap_tool_call**。

    对外行为与官方 ``ToolNode`` 一致（入 state、出 ``{"messages": [ToolMessage...]}``），
    可以当 drop-in 替换。若将来 LangGraph 原生支持并行执行，直接删掉这个类。
    """

    #: 与 ``AgentService.parallel_tools_max`` 对齐的上限（由构造参数覆盖）。
    MAX_WORKERS = 8

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        wrap_tool_call: Optional[Callable[..., Any]] = None,
        max_workers: int = MAX_WORKERS,
        name: str = "tools",
    ) -> None:
        self.tools_by_name: Dict[str, BaseTool] = {t.name: t for t in tools or []}
        self._wrap = wrap_tool_call
        self.max_workers = max(1, int(max_workers or 1))
        self.name = name

    #: ``config`` 必须类型化为 ``RunnableConfig | None``：LangGraph 在 ``add_node`` 时
    #: 会检查节点签名，写成 ``Any`` 会告警（某些版本直接拒绝注册）。
    def __call__(self, state: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        return self.invoke(state, config)

    def invoke(self, state: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """与官方 ``ToolNode`` 同名的入口：两条路径在调用方看来是同一个东西。"""
        msgs = state.get("messages") if isinstance(state, dict) else state
        ai = (msgs or [])[-1] if msgs else None
        calls = tool_calls_of(ai)
        if not calls:
            return {"messages": []}
        #! 只有一个工具还开线程池，付出的是调度 overhead，收获是 0。
        if len(calls) == 1:
            return {"messages": [self._run_one(calls[0], state, config)]}
        out: List[Any] = [None] * len(calls)
        try:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(calls)),
                                    thread_name_prefix="agt-tool") as ex:
                futs = []
                for idx, c in enumerate(calls):
                    #! **必须显式把上下文带进子线程**：``TraceContext`` 是 contextvars，
                    #! 线程池不会自动继承，漏了它留痕会断在这一层（没有 trace id）。
                    futs.append(ex.submit(contextvars.copy_context().run,
                                          self._run_one_and_store, out, idx, c, state, config))
                for f in futs:
                    f.result()
        except Exception as e:  # noqa: BLE001 - 并行失败退回串行，不能整轮工具全丢
            log.warning("[Agent] 工具并行执行失败，回退串行：%s", e)
            return {"messages": [self._run_one(c, state, config) for c in calls]}
        # 兜底：任何没写回的位置（理论上不该有）按串行补跑一次
        return {"messages": [r if r is not None else self._run_one(c, state, config)
                             for c, r in zip(calls, out)]}

    def _run_one_and_store(self, sink: List[Any], idx: int, call: Any,
                           state: Any, config: Any) -> None:
        sink[idx] = self._run_one(call, state, config)

    def _run_one(self, call: Any, state: Any, config: Any = None) -> Any:
        name = name_of(call)
        cid = id_of(call)
        tool = self.tools_by_name.get(name)

        def execute(_req: Any) -> ToolMessage:
            if tool is None:
                raise KeyError(f"未登记的工具：{name}")
            return ToolMessage(content=str(tool.invoke(args_of(call), config=config)),
                               tool_call_id=cid, name=name)

        if self._wrap is None:
            return execute(None)
        # 鸭子类型的 request：我们自己的 wrapper 只读 tool_call / tool / state。
        return self._wrap(_SimpleRequest(tool_call=call, tool=tool, state=state), execute)


class _SimpleRequest:
    """``langgraph.prebuilt.tool_node.ToolCallRequest`` 的最小替身。

    不走真的 ``ToolCallRequest`` 是因为它需要 ``ToolRuntime``（带 config / store / state），
    构造方式在不同 langgraph 版本间不稳定；而我们的 wrapper 只读这三个字段。
    """

    __slots__ = ("tool_call", "tool", "state", "runtime")

    def __init__(self, tool_call: Any, tool: Any, state: Any) -> None:
        self.tool_call = tool_call
        self.tool = tool
        self.state = state
        self.runtime = None


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #

def ensure_runtime(config: Any) -> Any:
    """给 config 补上 LangGraph 的 ``Runtime``。

    官方 ``ToolNode`` 的 ``_func`` 有一个**没有默认值**的 ``runtime`` 参数，
    值来自 ``config["configurable"][CONFIG_KEY_RUNTIME]``。
    图执行时 LangGraph 会注入它，但**直接调用节点不会** —— 结果是
    ``ValueError: Missing required config key 'N/A' for 'tools'``，
    一个完全看不出与"缺 runtime"有关的报错。

    补上它之后，节点既能挂进图，也能被单独调用（单元测试、临时脚本）。
    """
    try:  # 私有模块，路径可能变 —— 取不到就退回已知常量
        from langgraph._internal._runnable import CONFIG_KEY_RUNTIME as _KEY
    except Exception:  # noqa: BLE001
        _KEY = "__pregel_runtime"
    cfg = dict(config or {})
    conf = dict(cfg.get("configurable") or {})
    if conf.get(_KEY) is None:
        try:
            from langgraph.runtime import Runtime

            conf[_KEY] = Runtime()
        except Exception:  # noqa: BLE001 - 拿不到就原样传：挂在图里时本来就不需要我们补
            return cfg or None
    cfg["configurable"] = conf
    return cfg


def build_tool_node(
    *,
    ctx: Any,
    tools: Any,
    budget: Any = None,
    deadline: Any = None,
    recorder: Any = None,
    tool_usage: Optional[Dict[str, int]] = None,
    call_cache: Optional[Dict[str, str]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    tool_trace: Optional[List[str]] = None,
    parallel: bool = True,
    parallel_max: int = ParallelToolNode.MAX_WORKERS,
) -> Any:
    """造一个 LangGraph 工具节点。**真实链路走这里**。

    :param tools: :class:`~app.ai.tools.RiskAgentTools`（``None`` 时返回一个"工具层未装配"
        节点，让图退化成"没有工具可调用"，而不是崩掉）。
    :param recorder: :class:`~app.agent.trace.TraceRecorder`。
    :param evidence: 证据池（前端「证据溯源」）。
    :param tool_trace: 人类可读的工具轨迹。
    :param parallel: 同轮多个工具并行。关掉就是**纯官方串行 ToolNode**。
    :return: 一个节点可调用对象 ``(state, config) -> {"messages": [ToolMessage...]}``。
    """
    from langgraph.prebuilt import ToolNode

    from ..agent.trace import ToolRun

    usage = tool_usage if tool_usage is not None else {}
    cache = call_cache if call_cache is not None else {}
    gate = ToolGate(budget=budget, deadline=deadline, tool_usage=usage, call_cache=cache)
    #: 本轮判定表。节点是被图串行调用的（一张图一次只会走一个 tools 节点），
    #: 用闭包变量而非构造时快照——每轮都会重新判定。
    holder: Dict[str, Any] = {"decisions": {}}
    #: 工具名 → 最近一次 ``source_type``：留痕要带来源，而 wrapper 只拿得到文本。
    seen_type: Dict[str, str] = {}

    def on_result(name: str, res: Any) -> None:
        """执行**成功后**的回灌：证据池 + 配额。失败路径拿不到 ``res``。"""
        if name:
            seen_type[name] = getattr(res, "source_type", None) or "internal"
            usage[name] = usage.get(name, 0) + 1
        if evidence is not None:
            evidence.append(evidence_entry(name, res, 1, False))

    def on_done(name: str, ms: int, ok: bool, text: str) -> None:
        """留痕。``tool_trace`` / ``recorder`` 都是**有序**数据，必须按调用顺序写。"""
        if tool_trace is not None:
            tool_trace.append(f"{name}({ms}ms)")
        if recorder is not None:
            try:
                recorder.note_tool(ToolRun(name or "", ok, ms, None if ok else text[:200],
                                           seen_type.get(name, "internal"), False, len(text)))
            except Exception as e:  # noqa: BLE001 - 留痕失败不能让整次分析挂掉
                log.debug("[Agent] 工具留痕失败（已忽略）：%s", e)

    if tools is None:
        def _unwired(state: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
            msgs = state.get("messages") if isinstance(state, dict) else (state or [])
            calls = tool_calls_of(msgs[-1] if msgs else None)
            return {"messages": [ToolMessage(content='{"error":"工具层未装配"}',
                                             tool_call_id=id_of(c), name=name_of(c),
                                             status="error") for c in calls]}

        return _unwired

    lc_tools = tools.as_langchain_tools(ctx, on_result=on_result)
    wrap = make_tool_call_wrapper(gate=gate, decisions=lambda: holder["decisions"],
                                  on_done=on_done)
    inner: Any = (ParallelToolNode(lc_tools, wrap_tool_call=wrap, max_workers=parallel_max)
                  if parallel else ToolNode(lc_tools, wrap_tool_call=wrap))

    def _entry(state: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        msgs = state.get("messages") if isinstance(state, dict) else (state or [])
        calls = tool_calls_of(msgs[-1] if msgs else None)
        #! **先串行判定**：配额是共享状态，并行下并发扣会超发。
        holder["decisions"] = gate.plan(calls)
        #! 官方 ``ToolNode`` **不是 callable**（它是 Runnable，只有 ``invoke``），
        #! 而并行变体两者都行。统一走 ``invoke``，两条路径在调用方看来没区别。
        out = inner.invoke(state, ensure_runtime(config))
        # 缓存回填放在**节点出口串行做**：``on_result`` 拿不到原始 tool_call，
        # 在那里拼 key 会得到 ``name|{}``，永远 miss（静默失效，最难查的那种）。
        produced = out.get("messages") if isinstance(out, dict) else (out or [])
        for c, tm in zip(calls, produced):
            key = gate.call_key(c)
            if key not in cache:
                cache[key] = str(getattr(tm, "content", "") or "")
        return out

    return _entry


__all__ = [
    "ParallelToolNode",
    "ToolGate",
    "build_tool_node",
    "ensure_runtime",
    "evidence_entry",
    "make_tool_call_wrapper",
]
