"""把自研 ``FakeLlm`` 适配成 LangChain 的模型接口。

为什么需要它
------------
``AgentService`` 有两条编排：legacy 走 ``self.llm.chat_with_tools``（自研 ``ChatResult``
语义），langgraph 走 ``ModelChain`` → LangChain 的 ``AIMessage`` 语义。

测试里注入的 ``FakeLlm`` **只对 legacy 生效**。langgraph 分支没拿到
``lg_model_factory`` 时会自己去构造**真实的 ChatOpenAI** —— 后果有两个，
而且都很糟：

1. **测试在偷偷烧真实额度**。把全量 pytest 强制切到 langgraph 编排后，立刻收到
   方舟返回的 ``429 SetLimitExceeded · Your account [2132187327]``。
   这不是"环境不支持"，这就是**真的发出去的请求**。
2. **这条用例根本没有形成对照**。它以为在测降级自愈，实际在测网络。

本适配器让**同一份 FakeLlm 同时驱动两条编排**：队列、计数、sticky 语义完全共用，
于是"同一批用例两侧分别跑"才真正成立。

用法：

```python
llm = FakeLlm([ChatResult(content="", finish_reason="length")])
svc = AgentService(llm=llm, lg_model_factory=adapt_factory(llm))
```
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


class _Named:
    """给工具的 OpenAI dict 补一个 ``.name``。

    ``FakeLlm.chat_with_tools`` 里写的是 ``[t.name for t in tools]`` —— 它面向
    ``ToolSpec``。而 LangGraph 侧经 ``to_openai_tools`` 拿到的是原生 dict，
    没有 ``.name``，不包一层会直接 ``AttributeError``。
    """

    __slots__ = ("name", "raw")

    def __init__(self, raw: Any) -> None:
        if isinstance(raw, dict):
            fn = raw.get("function") or {}
            self.name = str(fn.get("name") or raw.get("name") or "")
        else:
            self.name = str(getattr(raw, "name", "") or "")
        self.raw = raw


class FakeLangChainModel:
    """LangChain 模型外观，内部转发到 ``FakeLlm``。"""

    def __init__(self, fake: Any, model: str = "fake-model") -> None:
        self._fake = fake
        self.model = model
        self._tools: List[Any] = []

    # -- Runnable 外观 -------------------------------------------------- #

    def bind_tools(self, tools: Any, **_kw: Any) -> "FakeLangChainModel":
        real = [t["tool"] if isinstance(t, dict) and "tool" in t else t
                for t in (tools or [])]
        self._tools = [_Named(t) for t in real]
        return self

    def invoke(self, messages: Any, **_kw: Any) -> Any:
        return self._to_ai(self._ask(messages))

    def stream(self, messages: Any, **_kw: Any) -> Iterable[Any]:
        from langchain_core.messages import AIMessageChunk

        msg = self._to_ai(self._ask(messages))
        # ⚠️ id 必须前后一致：LangChain 聚合 chunk 会校验 id，
        # 不一致会在 ``full + piece`` 处抛错 —— 那是**桩**的问题，不要去改实现。
        cid = f"chunk-{self.model}"
        if getattr(msg, "tool_calls", None):
            yield AIMessageChunk(content="", tool_calls=msg.tool_calls, id=cid)
            return
        text = msg.content or ""
        step = 24
        if not text:
            yield AIMessageChunk(content="", id=cid)
            return
        for i in range(0, len(text), step):
            yield AIMessageChunk(content=text[i:i + step], id=cid)

    # -- 内部 ------------------------------------------------------------ #

    def _ask(self, messages: Any) -> Any:
        """向自研 FakeLlm 提问。参数顺序与其 ``chat_with_tools`` 一致。"""
        return self._fake.chat_with_tools(self._tools, list(messages or []), 0.2, 0)

    @staticmethod
    def _to_ai(res: Any) -> Any:
        """``ChatResult`` → ``AIMessage``。"""
        from langchain_core.messages import AIMessage

        content = getattr(res, "content", "") or ""
        calls: List[Dict[str, Any]] = []
        for i, tc in enumerate(getattr(res, "tool_calls", None) or []):
            args: Dict[str, Any] = {}
            raw = getattr(tc, "arguments_json", "") or "{}"
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    args = loaded
            except Exception:  # noqa: BLE001 - 坏参数不该拖垮整条链路
                args = {}
            calls.append({
                "id": getattr(tc, "id", "") or f"call_{i}",
                "name": getattr(tc, "name", "") or "",
                "args": args,
                "type": "function",
            })
        return AIMessage(content=content, tool_calls=calls)


class Recorder:
    """可选：记录 LangGraph 侧每次调用看到了什么工具。"""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []


def adapt_factory(fake: Any, model: str = "fake-model") -> Any:
    """返回 ``ModelChain`` 需要的 ``model_factory``。

    同一个 model 名返回同一个实例：脚本化用例依赖"调用 N 次"的顺序语义，
    每次新建会把队列错位。
    """
    cache: Dict[str, FakeLangChainModel] = {}

    def build(name: str) -> FakeLangChainModel:
        key = name or model
        inst = cache.get(key)
        if inst is None:
            inst = FakeLangChainModel(fake, model=key)
            cache[key] = inst
        return inst

    return build
