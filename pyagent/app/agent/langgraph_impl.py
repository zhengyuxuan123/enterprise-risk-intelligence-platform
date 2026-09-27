"""LangGraph 编排实现 —— **本项目的唯一 Agent 编排**。

自研 legacy（2026-09-23 删除）
------------------------------
在此之前这里是"并存可切换"的两套实现：

```ini
APP_AGENT_ORCHESTRATOR=legacy      # 默认，走自研 _tool_loop
APP_AGENT_ORCHESTRATOR=langgraph   # 走本模块
```

并存是**过渡手段**，不是目标：它的价值在于让同一批用例能分别跑两侧、
差异可解释。真到了删的那天，判据不是"新旧之争"，而是这三条：

1. **功能等价**：把全量 pytest 强制切到 langgraph 编排，**564 条全绿**
   （这套用例原本就是为 legacy 写的全链路用例，覆盖路由 / 工具 / 引用核对 /
   护栏 / 缓存 / 降级 / 审计）。
2. **性能归因已闭环**：此前"慢 3 倍"的根因是漏发 ``thinking``（服务端默认开思考），
   已修复并被端到端测试钉住（见 ``_thinking_kwargs`` 的说明）。
3. **没有收益的复杂度**：继续并存意味着每次改编排都要维护两条路径，
   且任何"只在默认编排下成立"的行为都会变成隐藏分歧。

备份：`D:\\backup\\pyagent-legacy-orchestrator-20260923\\`（含删除前的
``agent_service.py`` / ``langgraph_impl.py`` / ``config.py``）。

设计约束（改动前必读）
----------------------
1. **工具只能是 `RiskAgentTools`**。它的 ``execute_with_meta`` 里挂着 companyId 校验与
   ``scope()`` 数据权限闸门，LangGraph 的 ToolNode 若绕过它，等于给 Agent 开了越权通道。
2. **工具 Schema 单一数据源** = ``RiskAgentTools.specs()``。这里只是把它转成 OpenAI 原生
   dict 交给 ``bind_tools``（已验证 langchain-openai 接受该格式），**不再手写第二份**。
3. **模型轮换降级不能丢**。``ChatOpenAI`` 原生没有"429 就换下一个候选"的能力，而方舟上
   429 常常是「账号 + 模型」二维问题、换个模型立刻可用。所以这里保留 ``app.ai.fallback``
   的 ``is_model_unavailable`` 判据，自己实现轮换。
4. **依赖缺失必须显式失败**。以前 import 失败只是 ``available() == False``，
   因为那时还有 legacy 顶着；**现在它是唯一实现**，缺依赖就是服务不可用，
   必须在启动/自检阶段明确报错，不许静默降级成"没有 AI"。
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

from ..ai.fallback import parse_candidates
from ..ai.llm_client import llm_call_timeout
from ..ai.messages import (
    ChatMessage,
    ToolCall,
    args_of,
    id_of,
    name_of,
    text_of,
    tool_calls_of,
)
from ..ai.tool_context import ToolContext
from ..ai.tools import ToolSpec
from ..config import get_settings

log = logging.getLogger(__name__)

ORCH_LANGGRAPH = "langgraph"
#: 自研 legacy 编排已于 2026-09-23 删除，这里**只剩一个有效值**。
ORCHESTRATORS = (ORCH_LANGGRAPH,)
#: 测试注入假模型、而配置里又没写 AI_MODEL 时的占位名（不是真实模型）。
INJECTED_MODEL = "injected"

# checkpoint 的 SQLite 连接是**跨进程共享**的，放进 ExitStack 保持整个进程生命周期，
# 不要每次分析开关一次（频繁开关 SQLite 会放大延迟，也会让快照看起来"丢"）。
_STACK: Optional[ExitStack] = None
_SAVER: Any = None
_SAVER_PATH: str = ""


try:  # langgraph >=0.2 的 messages reducer：按 id 去重合并，而不是整表覆盖
    from langgraph.graph.message import add_messages as _add_messages
except Exception:  # noqa: BLE001 - 极低版本兜底：退化为"后来的整表替换"

    def _add_messages(left: List[Any], right: List[Any]) -> List[Any]:  # type: ignore[misc]
        return list(right)


class LGState(TypedDict):
    """LangGraph 编排状态。

    !!! **必须定义在模块顶层**：LangGraph 用 ``get_type_hints`` 解析它，
    而该函数解析不了**函数内部**定义时的局部符号（``Annotated`` 会 NameError）。
    这是本模块唯一一处"看着可以放进去但其实不行"的位置。
    """

    messages: Annotated[List[Any], _add_messages]
    rounds: int


class LangGraphUnavailable(RuntimeError):
    """LangGraph 未安装或不可用时抛出（携带可自助解决的动作）。"""


#: 消息文本化现在由 :mod:`app.ai.messages` 提供（消息层只有一份实现）。
#: 这里 re-export 是为了既有导入点（``lg.text_of``）与测试不用改。
text_of = text_of  # noqa: F811  - 显式 re-export，见上行注释


def available() -> Tuple[bool, str]:
    """LangGraph 运行时是否可用。返回 ``(可用, 说明)``，永不抛异常。"""
    try:  # noqa: SIM105
        import langgraph  # noqa: F401
        from langgraph.graph import StateGraph  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"langgraph 未安装或不可用：{e}（pip install langgraph langchain-openai）"
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"langchain-openai 未安装或不可用：{e}（pip install langchain-openai）"
    return True, ""


# --------------------------------------------------------------------------- #
# 单一数据源：specs → OpenAI 原生 tool dict
# --------------------------------------------------------------------------- #

def _convert_langchain_tool(tool: Any) -> Optional[Dict[str, Any]]:
    """LangChain 原生工具（``StructuredTool`` / ``BaseTool``）→ OpenAI tool dict。

    走 langchain 自己的转换函数，而不是这边再拼一份：拼错了模型看到的参数
    就和真正能执行的不一致，属于最难查的那类错（模型一直传错参，
    看起来却像"模型笨"）。
    """
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool
    except Exception:  # noqa: BLE001 - 依赖缺失时交给调用方走 ToolSpec 分支
        return None
    try:
        return convert_to_openai_tool(tool)
    except Exception as e:  # noqa: BLE001
        log.warning("[LangGraph] LangChain 工具 %r 转换失败，已跳过：%s",
                    getattr(tool, "name", tool), e)
        return None


def to_openai_tools(specs: Sequence[Any]) -> List[Dict[str, Any]]:
    """把工具转成 OpenAI 原生格式的 tool 字典列表。

    两种输入都收：

    * 自研 ``ToolSpec``（schema 来自 :mod:`app.ai.tool_schemas` 的 Pydantic 模型）；
    * **LangChain 原生工具对象**（``RiskAgentTools.as_langchain_tools()`` 的产物，
      可被 ``ToolNode`` / 各类 Agent 框架直接消费）。

    **唯一来源就是工具自身的声明**：不在 LangGraph 侧再维护一份 schema，
    否则两边漂移后，模型看到的工具与真正能执行的工具会不一致。
    """
    #! ``ToolSpec.parameters`` 已经是**解析好的** dict（构造时就 json.loads 过了，
    #! 坏 JSON 会在那里兜底），这里不要再假定它是字符串。
    tools: List[Dict[str, Any]] = []
    for s in specs or []:
        if not hasattr(s, "parameters"):
            # 不是 ToolSpec —— 当作 LangChain 原生工具处理
            converted = _convert_langchain_tool(s)
            if converted:
                tools.append(converted)
            continue
        raw = getattr(s, "parameters", None)
        if isinstance(raw, str):
            try:
                parameters = dict(json.loads(raw))
            except Exception:  # noqa: BLE001
                parameters = {}
        elif isinstance(raw, dict):
            parameters = dict(raw)
        else:
            parameters = {}
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        tools.append({
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description or "",
                "parameters": parameters,
            },
        })
    return tools


def to_langchain_messages(messages: Sequence[Any]) -> List[Any]:
    """→ LangChain 消息列表。**内部表示本来就是 LangChain 消息，所以这一步是直通**。

    保留这个函数是为了兼容三种历史输入：
    LangChain 消息（直通）、OpenAI 线格式 dict（``{"role": ..., "content": ...}``）、
    以及早期自研 DTO 的鸭子类型（有 ``.role`` / ``.content``）。
    一旦三种输入都消失，这个函数就可以整体删掉。
    """
    from langchain_core.messages import BaseMessage

    from ..ai.messages import make

    out: List[Any] = []
    for m in messages or []:
        if isinstance(m, BaseMessage):
            out.append(m)
            continue
        if isinstance(m, dict):
            payload = dict(m)
            role = str(payload.pop("role", "") or "")
            fn = payload.pop("function", None)  # 旧格式：工具结果叫 function
            if fn and "name" not in payload:
                payload["name"] = fn if isinstance(fn, str) else (fn.get("name") if isinstance(fn, dict) else None)
            calls = payload.pop("tool_calls", None)
            out.append(make(role, payload.get("content"), name=payload.get("name"),
                            tool_call_id=payload.get("tool_call_id"),
                            tool_calls=tool_calls_of({"tool_calls": calls}) if calls else None))
            continue
        role = getattr(m, "role", "") or ""
        out.append(make(role, getattr(m, "content", ""),
                        name=getattr(m, "name", None),
                        tool_call_id=getattr(m, "tool_call_id", None),
                        tool_calls=getattr(m, "tool_calls", None)))
    return out


def lc_to_tool_calls(msg: Any) -> List[ToolCall]:
    """从任意"像 AIMessage 的东西"里取 tool_calls，统一成 :class:`ToolCall`。

    :class:`ToolCall` 是 ``dict`` 子类（LangChain 形状 ``{id,name,args,type}``），
    同时保留 ``.name`` / ``.arguments_json`` 两个移植期属性名。
    """
    return [ToolCall.of(id_of(c), name_of(c), args_of(c)) for c in tool_calls_of(msg)]


# --------------------------------------------------------------------------- #
# 模型链：保留"按模型轮换"的降级能力
# --------------------------------------------------------------------------- #

@dataclass
class ModelAttempt:
    model: str
    ok: bool
    error: str = ""


#: **两条编排共用的采样温度**。改这个值等于同时改 legacy 与 langgraph，不要只改一边。
#:
#: 值必须与 LangGraph 侧的调用点一致（见 :meth:`ModelChain._new_client`）。
#: 曾经两侧不一致：自研 client 用 0.2、这里的 ``ChatOpenAI`` 没设 temperature 落到
#: SDK 默认 1.0，结果同一批用例在新编排下出现「该调的工具没调」「成稿偏短」两类退化，
#: 而报错信息完全指向模型、很难怀疑到采样参数。
AGENT_TEMPERATURE = 0.2


class ModelChain:
    """主模型 + 候选轮换。

    与自研 ``LlmClient`` 同一套判据（``app.ai.fallback.is_model_unavailable``）：
    **401/403/欠费不换模型**（换了也失败，还会掩盖真实原因），**404/429/5xx 才换**。
    """

    def __init__(self, base_url: str, api_key: str, model: str, candidates: Sequence[str],
                 timeout_seconds: int = 600, model_factory: Optional[Callable[..., Any]] = None,
                 tries: int = 3, temperature: float = AGENT_TEMPERATURE,
                 thinking: str = "disabled", stream_usage: bool = False) -> None:
        from ..ai.fallback import is_model_unavailable  # 延迟导入：避免循环依赖

        self._is_unavailable = is_model_unavailable
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout_seconds
        self.temperature = temperature
        #: 思考模式开关。**默认必须与自研 client 一致**（见 :meth:`_build` 的说明）。
        self.thinking = (thinking or "").strip()
        #: 是否让 LangChain 在流式时附加 ``stream_options.include_usage``。
        self.stream_usage = bool(stream_usage)
        self.models: List[str] = []
        seen = set()
        for m in [model, *(candidates or [])]:
            mm = (m or "").strip()
            if mm and mm not in seen:
                seen.add(mm)
                self.models.append(mm)
        self.tries = max(1, int(tries))
        self.attempts: List[ModelAttempt] = []
        self._factory = model_factory
        # 同一个 ModelChain 实例内部是串行的（一轮一轮跑），所以这里的缓存不需要加锁；
        # 并发时重复构造也只是白做一次构造，不影响正确性。
        self._llm_cache: Dict[str, Any] = {}
        self._bound_cache: Dict[str, Any] = {}

    # -- 内部 ------------------------------------------------------------ #

    def _build(self, model: str) -> Any:
        """构造模型客户端。**同一实例按 model 缓存** —— 每轮重建会连带重建 HTTP 连接池。"""
        if self._factory is not None:
            return self._factory(model)
        cached = self._llm_cache.get(model)
        if cached is not None:
            return cached
        llm = self._new_client(model)
        self._llm_cache[model] = llm
        return llm

    def _new_client(self, model: str) -> Any:
        from langchain_openai import ChatOpenAI

        kwargs: Dict[str, Any] = dict(
            model=model,
            api_key=self.api_key or "sk-not-set",
            base_url=self.base_url,
            timeout=self.timeout,
            #! 必须显式传：ChatOpenAI 默认是 1.0，自研循环用的是 0.2。
            #! 漏传的后果不是"略有差异"，而是模型明显更容易跳过工具调用、成稿更短。
            temperature=self.temperature,
            max_retries=0,  # 重试由本类统一编排，避免框架与应用两层重试叠加
        )

        #! **★ 必须下发 ``thinking``，这是本项目最大的一处两侧不对齐。**
        #! 自研 client 每次请求都显式带 ``thinking: {"type": "disabled"}``，
        #! 而 ChatOpenAI 原生不知道这个约定 —— 不下发时服务端按模型默认走，
        #! 对 seed / v4 这类支持推理的模型意味着**默认打开思考**：
        #!   · 每条请求要额外生成大量 reasoning token，耗时成倍放大；
        #!   · 表象极具有迷惑性：轮数更少却慢 2~6 倍（EVAL-904：legacy 2 轮 39.8s
        #!     vs langgraph 1 轮 259.8s），怎么查都不像"参数问题"。
        #! 所以这里按自研 client 的同一套语义补齐：enabled / disabled 下发，空串则完全不带。
        if self.thinking in ("enabled", "disabled"):
            kwargs.update(self._thinking_kwargs({"thinking": {"type": self.thinking}}))

        if not self.stream_usage:
            # LangChain 默认会在流式请求里附加 stream_options.include_usage，
            # 让服务端在流末尾多算一次 usage。本项目只取正文，关掉它。
            kwargs["stream_usage"] = False
        return ChatOpenAI(**kwargs)

    def _thinking_kwargs(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """选择 ``thinking`` 的传参入口：**构造后回读**，而不是构造前猜。

        !!! 这里曾踩过一个极隐蔽的坑，改动前务必读完：

        最早的版本用 ``"extra_body" in inspect.signature(ChatOpenAI.__init__)``
        来判断支持哪种入口。**这个探测恒为 False** —— langchain 走 pydantic v2，
        ``__init__`` 签名是动态生成的，里面根本没有这两个字段名。于是代码每次
        都落到 ``model_kwargs`` 分支，而 langchain-openai 1.6.x 会把
        ``model_kwargs`` 原样展开喂给 ``Completions.create()``，结果是：

            TypeError: Completions.create() got an unexpected keyword argument 'thinking'

        **请求一个字节都没发出去**。更糟的是它骗过了上一轮的验证 ——
        ``_payload_diff.py`` 只走到 body *构造* 层（``_get_request_payload``），
        那一层看着完全正确，于是我们据此判定"两侧已对齐"。直到把全量 pytest
        强制切到 langgraph 编排，这条才是真的炸出来。

        教训：**不要靠签名/元数据猜测能力，要构造出来回读它到底生效没有**。
        """

        def _try(key: str) -> Optional[Any]:
            from langchain_openai import ChatOpenAI

            try:
                llm = ChatOpenAI(**dict(kwargs_probe, **{key: extra}))
            except Exception:  # noqa: BLE001 - 字段名不被接受时 pydantic 会抛
                return None
            # 回读：值真的挂在实例上才说明这个入口有效
            return llm if getattr(llm, key, None) is not None else None

        kwargs_probe: Dict[str, Any] = dict(
            model="probe", api_key="sk-not-set", base_url=self.base_url,
            timeout=1, temperature=self.temperature, max_retries=0,
        )
        if _try("extra_body") is not None:
            return {"extra_body": extra}
        if _try("model_kwargs") is not None:
            # 老版本入口。注意：在已验证不支持 extra_body 的新版上走这里会直接炸，
            # 所以上面的判定顺序不能反。
            return {"model_kwargs": extra}
        log.warning("[LangGraph] ChatOpenAI 两种扩展字段入口都不可用，"
                    "本次不下发 thinking —— 可能比自研 client 慢很多")
        return {}

    @property
    def primary(self) -> str:
        return self.models[0] if self.models else ""

    @staticmethod
    def _chunk_to_ai(chunk: Any) -> Any:
        """流式产物的最后一个 chunk → 完整 ``AIMessage``。

        !!! 不能直接把 ``AIMessageChunk`` 交给下游：它不是 ``AIMessage`` 的实例，
        任何 ``isinstance(x, AIMessage)`` 判定都会把它降级成纯文本消息，
        **tool_calls 当场丢失** —— 图会在 ``should_continue`` 里看不到工具调用而直接收敛，
        表现是"模型明明要查工具，却什么都没查就出结论了"。
        """
        from langchain_core.messages import AIMessage

        if isinstance(chunk, AIMessage):
            return chunk
        content = text_of(getattr(chunk, "content", "") or "")
        try:
            calls = list(getattr(chunk, "tool_calls", None) or [])
        except Exception:  # noqa: BLE001
            calls = []
        return AIMessage(content=content, tool_calls=calls,
                         id=getattr(chunk, "id", None) or None)

    @staticmethod
    def _chunk_has_tools(chunk: Any) -> bool:
        """chunk 是否带了工具调用（无论完整还是残缺片段）。"""
        try:
            if getattr(chunk, "tool_calls", None):
                return True
            return bool(getattr(chunk, "tool_call_chunks", None))
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _chunks_to_ai(chunks: List[Any]) -> Any:
        """一批流式片段 → 完整 ``AIMessage``。

        **不能**直接用 ``直接 for c in chunks: full = full + c``：
        LangChain 的 chunk 合并是**逐片新建对象并 merge 内部列表**（content blocks、
        ``tool_call_chunks``、``additional_kwargs`` …），所以 n 片就是 O(n²)。
        实测 600 片逐片合并 12.6ms、一次性处理 3.6ms，正文越长差距越夸张。

        分两条路径：

        * **纯文本（成稿轮，绝大多数情况）** —— 只攒字符串，O(n)。
        * **带工具调用** —— 片段数很少（就那几个 call），走原来的对象合并以保证
          ``tool_call_chunks`` 按 index 拼接的语义不出错。
        """
        from langchain_core.messages import AIMessage

        if not chunks:
            raise RuntimeError("模型流式返回为空（未收到任何 chunk）")
        if any(ModelChain._chunk_has_tools(c) for c in chunks):
            full = chunks[0]
            for piece in chunks[1:]:
                full = full + piece
            return ModelChain._chunk_to_ai(full)
        text = "".join(text_of(getattr(c, "content", "") or "") for c in chunks)
        return AIMessage(content=text, id=getattr(chunks[-1], "id", None) or None)

    @staticmethod
    def _stream_collect(bound: Any, lc_messages: List[Any],
                        token_hook: Callable[[str], None]) -> Any:
        """流式取回并逐片外推，最后拼成一条完整的 AIMessage。

        ``token_hook`` 抛异常**不中断推理**：它下游站的是 SSE 连接，
        写失败说明连接断了，让模型继续跑完并落库比"半途腰斩"更有价值。
        """
        chunks: List[Any] = []
        for piece in bound.stream(lc_messages):
            text = getattr(piece, "content", None)
            if isinstance(text, str) and text and token_hook is not None:
                try:
                    token_hook(text)
                except Exception as e:  # noqa: BLE001
                    log.debug("[LangGraph] token 推送失败（已忽略）：%s", e)
            chunks.append(piece)
        return ModelChain._chunks_to_ai(chunks)

    def invoke(self, lc_messages: List[Any], tools: Optional[List[Dict[str, Any]]] = None,
               token_hook: Optional[Callable[[str], None]] = None,
               timeout_fn: Optional[Callable[[], Optional[float]]] = None
               ) -> Tuple[Any, str]:
        """返回 ``(AIMessage, 实际生效的模型名)``。全败时抛出最后一个异常。

        :param token_hook: 给了它就走**流式**取回（边取边推），否则整块取回。
            流式失败会自动回退到 ``invoke`` —— 有的兼容端点不支持 SSE，
            而"能不能流式"不该决定"能不能出结论"。

            ⚠️ 流式取回中途失败时可能已经推出去了一些 token。**没关系**：本项目的 SSE 契约是
            「``token`` 是预告，``done`` 才是真相」——``done`` 一律取持久化实体，
            终端用户看到的最终正文不会被这些碎片影响。

        :param timeout_fn: 每次尝试前问一次"这次的单次超时该是多少"。
            **必须按次重新求值**，不能在外面算好传个死值进来：预算是在流逝的，
            第 3 次重试时能用的时间一定比第 1 次少。

            ⚠️ 超时走 ``contextvars`` 而不是实例属性：模型客户端是进程内单例，
            写在实例上会让 A 的时间预算掐死并发的 B。
        """
        last_exc: Optional[BaseException] = None
        for i, model in enumerate(self.models[: self.tries]):
            started = time.time()
            try:
                with llm_call_timeout(timeout_fn() if timeout_fn is not None else None):
                    llm = self._build(model)
                    bound = llm.bind_tools(tools) if tools else llm
                    if token_hook is not None and hasattr(bound, "stream"):
                        try:
                            res = self._stream_collect(bound, lc_messages, token_hook)
                            self.attempts.append(ModelAttempt(model=model, ok=True))
                            log.debug("[LangGraph] 模型 %s 流式调用成功（%.2fs）",
                                      model, time.time() - started)
                            return res, model
                        except Exception as se:  # noqa: BLE001 - 流式不可用就整体退化
                            log.warning("[LangGraph] %s 流式取回失败，回退整块调用：%s",
                                        model, str(se)[:160])
                    res = bound.invoke(lc_messages)
                    self.attempts.append(ModelAttempt(model=model, ok=True))
                    log.debug("[LangGraph] 模型 %s 调用成功（%.2fs）", model, time.time() - started)
                    return res, model
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
                retryable = self._is_unavailable(int(status) if status else 0, str(e))
                self.attempts.append(ModelAttempt(model=model, ok=False, error=str(e)[:200]))
                last_exc = e
                log.warning("[LangGraph] 模型 %s 调用失败（%.2fs，status=%s，可换模型=%s）：%s",
                            model, time.time() - started, status, retryable, str(e)[:200])
                if not retryable:
                    raise
                # 轮换：换下一个模型再试
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LangGraph 模型链为空：请检查 AI_MODEL / AI_MODEL_CANDIDATES")


def build_chain(model_factory: Optional[Callable[..., Any]] = None, tries: int = 3) -> ModelChain:
    """从全局配置构造模型链（含候选解析）。

    ``thinking`` 必须**跟随自研 client 的同一份配置**（``ai.thinking_type``），
    不能在这一侧写死 —— 两边一旦不一致，同一模型会跑出完全不同的生成路径。
    """
    s = get_settings()
    candidates = parse_candidates(s.ai.model_candidates)
    return ModelChain(
        base_url=s.ai.base_url,
        api_key=s.ai.api_key,
        model=s.ai.model or "",
        candidates=candidates,
        timeout_seconds=s.ai.chat_timeout_seconds,
        model_factory=model_factory,
        tries=tries,
        thinking=s.ai.thinking_type or "disabled",
    )


# --------------------------------------------------------------------------- #
# checkpoint（SQLite 快照）
# --------------------------------------------------------------------------- #

def get_checkpointer() -> Any:
    """SQLite checkpoint 单例。**它不等于长期记忆** —— 一个是图执行状态，一个是业务事实。"""
    global _STACK, _SAVER, _SAVER_PATH  # noqa: PLW0603
    if _SAVER is not None:
        return _SAVER
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = get_settings().agent.langgraph_checkpoint_path or "./data/langgraph.sqlite"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cm = SqliteSaver.from_conn_string(path)
    stack = ExitStack()
    saver = stack.enter_context(cm)
    _STACK, _SAVER, _SAVER_PATH = stack, saver, path
    log.info("[LangGraph] checkpoint 已挂载：%s", path)
    return saver


def reset_checkpointer() -> None:
    """测试用：释放 checkpoint 资源。"""
    global _STACK, _SAVER, _SAVER_PATH  # noqa: PLW0603
    if _STACK is not None:
        try:
            _STACK.close()
        except Exception as e:  # noqa: BLE001
            log.debug("[LangGraph] checkpoint 释放失败（已忽略）：%s", e)
    _STACK, _SAVER, _SAVER_PATH = None, None, ""


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #

def build_app(
    *,
    specs: Sequence[ToolSpec],
    dispatch_tools: Optional[Callable[[List[ToolCall], List[Any]], Dict[str, str]]] = None,
    tool_node: Any = None,
    max_rounds: int = 6,
    model_chain: Optional[ModelChain] = None,
    use_checkpoint: bool = True,
    stop_check: Optional[Callable[[], Optional[str]]] = None,
    token_hook: Optional[Callable[[str], None]] = None,
    timeout_fn: Optional[Callable[[], Optional[float]]] = None,
) -> Tuple[Any, ModelChain]:
    """构造可执行的 LangGraph 应用。

    :param tool_node: **首选**。一个现成的工具节点（见
        :func:`app.agent.tool_node.build_tool_node`）——官方 ``ToolNode`` 或其并行变体，
        闸门/留痕/预算挂在它的 ``wrap_tool_call`` 上。
    :param dispatch_tools: **兼容路径**。``(本轮 tool_calls, 此前的 LangChain 消息) ->
        {call_id: 结果文本}``，内部走 ``RiskAgentTools.execute_with_meta``。
        新代码请传 ``tool_node``；这个参数只留给既有的测试桩与临时脚本。
    :param stop_check: 每轮开始前调用，返回非 None 则立即收敛（预算耗尽等）。
    :param token_hook: 成稿轮的逐 token 出口（接 SSE）。**只有最后一轮会推** ——
        中间轮模型在写"我该调哪个工具"之类的草稿，把它当正文推到界面上就是幻觉。
    """
    from langchain_core.messages import BaseMessage, ToolMessage
    from langgraph.graph import END, StateGraph

    tools_payload = to_openai_tools(specs)
    chain = model_chain or build_chain()

    def agent_node(state: LGState) -> Dict[str, Any]:
        rounds = int(state.get("rounds") or 0)
        restriction = stop_check() if stop_check else None
        #! 与自研同构：最后一轮（或预算触发收敛）**不带工具**，逼模型吐正文，
        #! 否则它会继续"再查一次"，最后一轮拿不到成稿就只能走本地兜底。
        last_round = restriction is not None or rounds >= max(1, max_rounds) - 1
        payload = None if last_round else tools_payload

        # token 出口按"这一轮模型拿不拿得到工具"分两种处置：
        # * **被逼成稿的那一轮**（payload 为空，模型拿不到工具）——产出必然是正文，放心直推。
        #   这是最常见的路径（轮数用尽 / 预算触发收敛），也是流式收益真正生效的地方。
        # * 其余轮 —— 先攒在 ``pending``，等本轮结束再定夺：没要工具说明模型提前成稿了，
        #   补推；要了工具就把攒下的东西丢掉。模型偶尔会"先写两句再决定调工具"，那两句话
        #   是它的草稿，推给用户等于把幻觉混进结论 —— 这远比晚半秒看到字严重。
        pending: List[str] = []
        hook: Optional[Callable[[str], None]] = None
        if token_hook is not None:
            # An empty tool payload is already a final-answer round: there is no
            # possible tool call whose draft must be hidden. Buffering this case
            # delayed every upstream chunk until the model had completely
            # finished, which made a valid SSE response look non-streaming.
            hook = token_hook if last_round or not payload else pending.append

        ai, _model = chain.invoke(state["messages"], payload, token_hook=hook,
                                  timeout_fn=timeout_fn)
        if not isinstance(ai, BaseMessage):  # 防御：少数实现返回 str
            ai = AIMessage(content=str(ai))
        if pending and token_hook is not None and not getattr(ai, "tool_calls", None):
            for piece in pending:
                try:
                    token_hook(piece)
                except Exception as e:  # noqa: BLE001 - 连接断了就别再推了，但结论要留着
                    log.debug("[LangGraph] 补推 token 失败（已忽略）：%s", e)
                    break
        return {"messages": [ai], "rounds": rounds + 1}

    def legacy_tools_node(state: LGState) -> Dict[str, Any]:
        """``dispatch_tools`` 兼容路径。**真实链路请传 ``tool_node``**。

        它把结果按 ``{call_id: 文本}`` 拼回去，语义与官方节点等价，
        但工具身份是字符串、参数不过 Pydantic、错误处理自己写——留着只是为了让
        既有测试桩与临时脚本还能跑，不是推荐写法。
        """
        if dispatch_tools is None:
            return {"messages": [], "rounds": int(state.get("rounds") or 0)}
        msgs = state["messages"]
        ai = msgs[-1] if msgs else None
        calls = lc_to_tool_calls(ai)
        results = dispatch_tools(calls, msgs) or {}
        out: List[Any] = []
        for c in getattr(ai, "tool_calls", None) or []:
            cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
            out.append(ToolMessage(content=results.get(cid or "", ""), tool_call_id=cid or ""))
        return {"messages": out, "rounds": int(state.get("rounds") or 0)}

    def should_continue(state: LGState) -> str:
        msgs = state["messages"]
        ai = msgs[-1] if msgs else None
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            return END
        if int(state.get("rounds") or 0) >= max(1, max_rounds):
            return END
        restriction = stop_check() if stop_check else None
        if restriction:
            return END
        return "tools"

    g = StateGraph(LGState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node if tool_node is not None else legacy_tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    app = g.compile(checkpointer=get_checkpointer() if use_checkpoint else None)
    return app, chain


def run(
    *,
    messages: Sequence[Any],
    specs: Sequence[ToolSpec],
    dispatch_tools: Optional[Callable[[List[ToolCall], List[Any]], Dict[str, str]]] = None,
    tool_node: Any = None,
    max_rounds: int = 6,
    thread_id: str = "",
    model_chain: Optional[ModelChain] = None,
    use_checkpoint: bool = True,
    stop_check: Optional[Callable[[], Optional[str]]] = None,
    token_hook: Optional[Callable[[str], None]] = None,
    timeout_fn: Optional[Callable[[], Optional[float]]] = None,
) -> Tuple[str, str]:
    """执行一次 LangGraph 编排。返回 ``(正文, 生效模型名)``。

    :param token_hook: 逐 token 出口，只在**成稿轮**被调用（见 :func:`build_app`）。
    :param timeout_fn: 每轮模型调用前重新求值的单次超时。delete legacy 之前，
        「按剩余预算收紧单次超时」只在自研循环里有；LangGraph 侧缺了这一条，
        一次卡住的请求就能吃掉整份时间预算。这里补齐，**并且必须逐次求值**。
    """
    ok, why = available()
    if not ok:
        raise LangGraphUnavailable(why)

    app, chain = build_app(
        specs=specs,
        dispatch_tools=dispatch_tools,
        tool_node=tool_node,
        max_rounds=max_rounds,
        model_chain=model_chain,
        use_checkpoint=use_checkpoint,
        stop_check=stop_check,
        token_hook=token_hook,
        timeout_fn=timeout_fn,
    )
    config = {"configurable": {"thread_id": thread_id or "default"}}
    #! checkpoint 存在时首次 invoke 会带上历史 thread 的消息；我们的语义是
    #! "每次分析就是一条新 thread"，thread_id 用 trace_id（天然唯一），不会串会话。
    final = app.invoke(
        {"messages": to_langchain_messages(messages), "rounds": 0},
        config=config,
    )
    content = ""
    for m in reversed(list(final.get("messages") or [])):
        if type(m).__name__ in ("AIMessage", "AIMessageChunk"):
            content = text_of(getattr(m, "content", "") or "")
            break
    used = chain.attempts[-1].model if chain.attempts else chain.primary
    return content, used


__all__ = [
    "ORCH_LANGGRAPH",
    "ORCHESTRATORS",
    "INJECTED_MODEL",
    "LangGraphUnavailable",
    "ModelChain",
    "ModelAttempt",
    "available",
    "build_app",
    "build_chain",
    "get_checkpointer",
    "lc_to_tool_calls",
    "reset_checkpointer",
    "run",
    "text_of",
    "to_langchain_messages",
    "to_openai_tools",
]
