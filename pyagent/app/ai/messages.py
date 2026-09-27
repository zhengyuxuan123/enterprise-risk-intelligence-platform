"""消息层：**LangChain 消息类型是唯一内部表示**。

为什么要有这一层
----------------
移植期这里有一套自研 DTO（``ChatMessage`` / ``ToolCall`` / ``ChatResult``），
形状是照着 Java 侧的 DTO 抄的（``role`` 字符串 + ``to_payload()``）。它有三个硬伤：

1. **进不了生态**：``bind_tools`` / ``ToolNode`` / checkpoint / 任何 LangChain 组件
   都只认 :class:`~langchain_core.messages.BaseMessage`；用自研 DTO 就得在边界上不断手写转换，
   每加一个生态组件就多一处漂移的可能（``to_langchain_messages`` 就是这么来的）。
2. **身份要靠约定而非类型**：``role == "assistant"`` 是字符串比较，写错不报错，
   只是消息被静默当成 user（``to_langchain_messages`` 里就留着这么一条"宁可读错也不要丢"的兜底）。
3. **两套 tool_call 形状**：自研是 ``{id, type, function:{name, arguments}}``（OpenAI 线格式），
   LangChain 是 ``{id, name, args, type}``。两边来回转，参数是 JSON 字符串还是 dict 全靠记忆。

现在改成：**内存里只存在 LangChain 消息对象**；OpenAI 线格式只在
:func:`to_payload` 这一处产生（那才是真正的边界）。

兼容策略
--------
``ChatMessage`` / ``ToolCall`` 保留为**门面**——调用它们拿到的是**真的 LangChain 对象**，
所以既有调用点不用改，但内部表示已经是生态的。新代码请直接写
``SystemMessage(...)`` / ``HumanMessage(...)`` / ``tool_call(...)``。

.. WARNING::
    ``ToolCall`` 虽然是 ``dict`` 子类，**不要依赖"取回来的 tool_call 一定是 ToolCall 实例"**——
    LangChain 在构造 ``AIMessage`` 时会把 ``tool_calls`` 规范化成纯 ``dict``（pydantic 会复制）。
    要读字段一律走 :func:`name_of` / :func:`args_of` / :func:`args_json_of` / :func:`id_of`。
    这条曾经会表现为"模型怎么都不传参"之类的假象。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

__all__ = [
    "AIMessage",
    "BaseMessage",
    "ChatMessage",
    "HumanMessage",
    "SystemMessage",
    "ToolCall",
    "ToolMessage",
    "args_json_of",
    "args_of",
    "as_message",
    "assistant",
    "id_of",
    "make",
    "name_of",
    "system",
    "text_of",
    "tool_call",
    "tool_calls_of",
    "tool_result",
    "to_payload",
    "to_payloads",
    "user",
]


# --------------------------------------------------------------------------- #
# 工厂：新代码请用这些
# --------------------------------------------------------------------------- #

def system(content: str) -> SystemMessage:
    return SystemMessage(content=content or "")


def user(content: str) -> HumanMessage:
    return HumanMessage(content=content or "")


def assistant(
    content: Optional[str] = None,
    tool_calls: Optional[Sequence[Any]] = None,
    **kw: Any,
) -> AIMessage:
    """构造 assistant 消息。

    .. NOTE::
        带 ``tool_calls`` 时 ``content`` 传 ``""`` 而不是 ``None``：
        部分服务商会因为"有 content 又带 tool_calls"报错，
        而自研 client 的约定也是 assistant 带 tool_calls 时序列化省略 content。
    """
    #! ``tool_calls`` **不能传 None**：pydantic 校验会直接拒绝（"Input should be a valid list"），
    #! 而"没有工具调用"的自然写法就是 None —— 这里统一成空列表。
    return AIMessage(content=content or "", tool_calls=list(_norm_calls(tool_calls)), **kw)


def tool_result(content: str, *, tool_call_id: str, name: Optional[str] = None) -> ToolMessage:
    return ToolMessage(content=content or "", tool_call_id=tool_call_id or "", name=name)


#: LangChain ``message.type`` → OpenAI ``role``。
ROLE_OF = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}


def make(role: str, content: Optional[str] = None, **kw: Any) -> BaseMessage:
    """按角色名造消息。**未知角色当 user**——宁可读错也不要把消息丢掉。"""
    r = (role or "").strip().lower()
    if r == "system":
        return system(content or "")
    if r in ("tool", "function"):
        return tool_result(content or "", tool_call_id=kw.get("tool_call_id") or "",
                           name=kw.get("name"))
    if r in ("assistant", "ai"):
        return assistant(content, kw.get("tool_calls"))
    return user(content or "")


# --------------------------------------------------------------------------- #
# tool_call：LangChain 风格 ``{id, name, args, type}``
# --------------------------------------------------------------------------- #

def _as_args(arguments: Any) -> Dict[str, Any]:
    """参数既收 dict 也收 JSON 字符串——模型给的是字符串，程序内部用 dict。"""
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str):
        s = arguments.strip()
        if not s:
            return {}
        try:
            v = json.loads(s)
        except Exception:  # noqa: BLE001 - 参数坏了也要让模型看到"调用失败"而不是 400
            return {}
        return dict(v) if isinstance(v, Mapping) else {}
    return {}


def tool_call(call_id: str, name: str, arguments: Any = None) -> "ToolCall":
    """构造一条 LangChain 风格的 tool_call。"""
    return ToolCall(id=call_id or "", name=name or "", args=_as_args(arguments), type="tool_call")


def name_of(call: Any) -> str:
    if isinstance(call, Mapping):
        return str(call.get("name") or "")
    return str(getattr(call, "name", "") or "")


def id_of(call: Any) -> str:
    if isinstance(call, Mapping):
        return str(call.get("id") or "")
    return str(getattr(call, "id", "") or "")


def args_of(call: Any) -> Dict[str, Any]:
    """取出参数 dict。OpenAI 线格式（``function.arguments`` 字符串）也认。"""
    if isinstance(call, Mapping):
        if "args" in call:
            a = call.get("args")
            return dict(a) if isinstance(a, Mapping) else _as_args(a)
        fn = call.get("function")
        if isinstance(fn, Mapping):
            return _as_args(fn.get("arguments"))
        return {}
    return _as_args(getattr(call, "arguments_json", None))


def args_json_of(call: Any) -> str:
    """参数 → JSON 字符串（给自研 HTTP client / 工具执行层用）。"""
    return json.dumps(args_of(call), ensure_ascii=False)


def tool_calls_of(msg: Any) -> List[Dict[str, Any]]:
    """从任意"像 AIMessage 的东西"里取 tool_calls，统一成 LangChain 形状。"""
    if msg is None:
        return []
    raw = msg.get("tool_calls") if isinstance(msg, Mapping) else getattr(msg, "tool_calls", None)
    out: List[Dict[str, Any]] = []
    for c in raw or []:
        if isinstance(c, Mapping):
            out.append({
                "id": id_of(c),
                "name": name_of(c),
                "args": args_of(c),
                "type": c.get("type") or "tool_call",
            })
        else:
            out.append({"id": id_of(c), "name": name_of(c),
                        "args": args_of(c), "type": "tool_call"})
    return out


def _norm_calls(raw: Any) -> List[Dict[str, Any]]:
    """把任意形状的 tool_calls 归一化成 LangChain 形状（``tool_calls_of`` 的宽松版）。"""
    if not raw:
        return []
    if isinstance(raw, Mapping):  # 单条而不是列表
        raw = [raw]
    return tool_calls_of({"tool_calls": raw})


# --------------------------------------------------------------------------- #
# 兼容门面：内部表示已经是 LangChain 对象，这里只保留旧调用形状
# --------------------------------------------------------------------------- #

class ToolCall(dict):
    """一条 tool_call。**是 dict 子类**——可直接进 ``AIMessage(tool_calls=[...])``。

    额外提供移植期的属性名（``.name`` / ``.arguments_json`` / ``.function`` / ``.to_payload()``），
    免得老调用点大面积改动。见模块 docstring 里的警告：取回来的不一定是这个类。
    """

    @property
    def name(self) -> str:
        return str(self.get("name") or "")

    @property
    def arguments_json(self) -> str:
        return json.dumps(self.get("args") or {}, ensure_ascii=False)

    @property
    def function(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments_json}

    def to_payload(self) -> Dict[str, Any]:
        """→ OpenAI 线格式。"""
        return {"id": str(self.get("id") or ""), "type": "function", "function": self.function}

    def __getattr__(self, item: str) -> Any:
        """属性访问转发到 dict 的 key。

        ``dict`` 子类默认**没有**这个行为，而移植期代码写的是 ``tc.id`` / ``tc.name``。
        不加它会静默返回 ``None``（``getattr(tc, "id", None)`` 那种写法更是直接吞掉），
        加上之后至少会在 key 不存在时抛 ``AttributeError`` —— 比空串好排查。
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item) from None

    @staticmethod
    def of(call_id: str, name: str, arguments: Any = None) -> "ToolCall":
        return tool_call(call_id, name, arguments)


class ChatMessage:
    """兼容门面：**调用它的静态方法拿到的是真的 LangChain 消息对象**。

    保留的唯一理由是既有调用点已经写成了 ``ChatMessage.system(...)``。
    新代码请直接用 :class:`SystemMessage` / :class:`HumanMessage` / :func:`tool_result`。
    """

    system = staticmethod(system)
    user = staticmethod(user)
    assistant = staticmethod(assistant)

    @staticmethod
    def tool(call_id: str, name: str, content: str) -> ToolMessage:
        """:meth:`ChatMessage.tool` 的移植期参数顺序是 ``(call_id, name, content)``。"""
        return tool_result(content, tool_call_id=call_id, name=name)

    def __new__(cls, role: str, content: Optional[str] = None, **kw: Any) -> BaseMessage:  # noqa: D102
        return make(role, content, **kw)


# --------------------------------------------------------------------------- #
# 序列化 / 归一化
# --------------------------------------------------------------------------- #

def text_of(content: Any) -> str:
    """消息正文 → 纯文本。

    ``AIMessage.content`` 在多模态响应里是 **块列表**（``[{"type":"text","text":…}]``），
    直接 ``str()`` 会把整个结构拼进正文。这里只取文本块。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, Mapping):
                if b.get("type") == "text" or ("text" in b and b.get("type") is None):
                    parts.append(str(b.get("text") or ""))
        return "".join(parts)
    return str(content)


def to_payload(msg: Any) -> Dict[str, Any]:
    """LangChain 消息 → OpenAI chat 请求体里的一条。**线格式只在这里产生**。"""
    if isinstance(msg, Mapping):  # 已经是 payload（防御：调用方可能直接传 dict）
        return dict(msg)
    if not isinstance(msg, BaseMessage):
        return {"role": "user", "content": text_of(msg)}

    m: Dict[str, Any] = {"role": ROLE_OF.get(getattr(msg, "type", ""), "user")}
    if isinstance(msg, ToolMessage):
        m["content"] = text_of(msg.content)
        if msg.tool_call_id:
            m["tool_call_id"] = msg.tool_call_id
        if getattr(msg, "name", None):
            m["name"] = msg.name
        return m

    content = text_of(getattr(msg, "content", ""))
    calls = tool_calls_of(msg)
    #! 带 tool_calls 时省略 content：部分服务商会因为"有 content 又带 tool_calls"报错，
    #! 而自研 client 的约定也是这样（移植期就是这么序列化的，不能悄悄改）。
    if calls:
        m["tool_calls"] = [{
            "id": id_of(c),
            "type": "function",
            "function": {"name": name_of(c), "arguments": args_json_of(c)},
        } for c in calls]
        if content:
            m["content"] = content
        return m
    if content or isinstance(msg, (SystemMessage, HumanMessage)):
        m["content"] = content
    return m


def to_payloads(messages: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    return [to_payload(m) for m in messages or []]


def as_message(x: Any) -> AIMessage:
    """把"一次模型调用的结果"归一化为 :class:`AIMessage`。

    吃三种输入：``AIMessage``（直通）、``ChatResult``（移植期结果 DTO）、``str``（当正文）。
    """
    if isinstance(x, AIMessage):
        return x
    if isinstance(x, BaseMessage):
        return AIMessage(content=text_of(getattr(x, "content", "")),
                         tool_calls=tool_calls_of(x) or None)
    if isinstance(x, str):
        return AIMessage(content=x)
    # ChatResult 及任何"有 content/tool_calls 的东西"
    meta: Dict[str, Any] = {}
    fr = getattr(x, "finish_reason", None)
    if fr:
        meta["finish_reason"] = fr
    rc = getattr(x, "reasoning_chars", 0)
    if rc:
        meta["reasoning_chars"] = rc
    return AIMessage(
        content=text_of(getattr(x, "content", "")),
        tool_calls=tool_calls_of(x) or [],
        response_metadata=meta or {},
    )
