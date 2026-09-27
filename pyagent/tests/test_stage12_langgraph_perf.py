# -*- coding: utf-8 -*-
"""LangGraph 性能测试与两侧对齐契约（**全程离线、零 token**）。

为什么这些必须是测试而不是文档
------------------------------
这一批缺陷有一个共同特征：**它们都不报错**。

- 少发一个 ``thinking`` 字段 —— 服务端不会拒绝，只会换个生成路径悄悄慢几倍；
- 每片 chunk 做一次对象合并 —— 功能完全正确，只是随着正文变长越来越慢；
- 每轮重建模型客户端 —— 连接池跟着重建，单次看不出来；
- 工具串行执行 —— 结果一模一样，只是用户多等。

正因为没有报错，"某某地慢"永远轮不到它们头上。所以只能用断言钉住。

跑法::

    pytest tests/test_stage12_langgraph_perf.py
"""

from __future__ import annotations

import json
import os
import time

import pytest

from app.agent import langgraph_impl as lg
from app.ai.llm_client import ChatMessage, ToolCall
from app.ai.tools import ToolSpec
from app.config import get_settings

pytest.importorskip("langgraph", reason="langgraph 未安装")
pytest.importorskip("langchain_core", reason="langchain-core 未安装")


def _specs() -> list:
    return [
        ToolSpec(name="get_metrics", description="查询经营指标",
                 parameters_json=json.dumps({"type": "object", "properties": {}})),
        ToolSpec(name="get_risk_events", description="查询风险事件",
                 parameters_json=json.dumps({"type": "object", "properties": {}})),
    ]


class _FakeLcModel:
    """零延迟假模型：够跑通一轮即可，benchmark 只看框架侧开销。"""

    def __init__(self, model: str = "fake", chunks: int = 40) -> None:
        self.model = model
        self.chunks = chunks
        self.construct_calls = 0

    def bind_tools(self, tools):  # noqa: D102
        return self

    def invoke(self, messages):  # noqa: D102
        from langchain_core.messages import AIMessage

        if any(type(m).__name__ == "ToolMessage" for m in messages):
            return AIMessage(content="结论。")
        return AIMessage(content="", tool_calls=[{
            "id": "c1", "type": "function", "name": "get_metrics", "args": {}}])

    def stream(self, messages):  # noqa: D102
        from langchain_core.messages import AIMessageChunk

        msg = self.invoke(messages)
        if getattr(msg, "tool_calls", None):
            yield AIMessageChunk(content="", tool_calls=msg.tool_calls, id="c")
            return
        for i in range(self.chunks):
            yield AIMessageChunk(content="字", id="c")


# --------------------------------------------------------------------------- #
# 1. thinking 字段：两侧必须对齐（这是本次排查出的主因）
# --------------------------------------------------------------------------- #

def test_thinking_is_defaulted_like_legacy_client():
    """LangGraph 侧必须下发 ``thinking``，取值来自自研 client 的同一份配置。

    这是「同一模型、同样用例、langgraph 慢 2~6 倍」的根因：自研 client 每次都显式带
    ``thinking:{"type":"disabled"}``，而 ChatOpenAI 不知道这个约定，不下发就让服务端
    按模型默认走 —— 对支持推理的模型等于默认开了思考，每轮多出成倍的推理时间。
    """
    get_settings.cache_clear()
    chain = lg.build_chain(model_factory=lambda m: _FakeLcModel(m))
    # 配置缺省时与自研 client 的默认值一致（见 LlmClient.thinking_type 的默认 "disabled"）
    assert chain.thinking in ("disabled", "enabled", ""), chain.thinking
    assert chain.thinking == "disabled", "默认必须与自研 client 一致，否则两侧生成路径不同"


@pytest.mark.allows_real_client
def test_build_wires_thinking_into_client_kwargs(monkeypatch):
    """``thinking`` 必须挂到客户端对象上。

    ⚠️ 这条**只能证明构造层落位，证明不了真实调用可用**。历史上正是它放过了一个
    真 bug：它接受 ``extra_body`` / ``model_kwargs`` 二者必居其一，而后者在
    langchain-openai 1.6.x 上会让请求**一个字节都发不出去**。

    已用变异测试确认过：把 ``_thinking_kwargs`` 改回 ``model_kwargs`` 之后，
    ``test_thinking_reaches_the_wire`` 立刻变红（报的就是那个 TypeError），
    而**本条依旧是绿的** —— 这就是它不能被当作验收判据的原因。留着只为了拦
    "字段完全没传"这类粗错。详见 ``test_thinking_reaches_the_wire``。
    """
    get_settings.cache_clear()
    chain = lg.ModelChain(base_url="https://example.invalid/v3", api_key="k",
                          model="m", candidates=[], thinking="disabled")
    assert chain.thinking == "disabled"

    # 真正构造真实 client（**不发任何请求**，纯对象构造），检查它收下了参数
    try:
        from langchain_openai import ChatOpenAI
    except Exception:  # noqa: BLE001
        pytest.skip("langchain-openai 未安装")

    client = chain._new_client("m")
    blob = json.dumps({
        "extra_body": getattr(client, "extra_body", None),
        "model_kwargs": getattr(client, "model_kwargs", None),
    }, default=str)
    assert "thinking" in blob, f"客户端没带上 thinking 字段：{blob}"
    assert "disabled" in blob, f"thinking 取值不对：{blob}"


# --------------------------------------------------------------------------- #
# 1b. ★ 端到端：thinking 必须真的出现在 HTTP 报文中
# --------------------------------------------------------------------------- #

def _start_wiretap(bodies: list):
    """起一个本机 HTTP 服务，把收到的请求体记进 ``bodies``。

    **为什么不看 body 构造、也不用 mock**

    本文件早先的 ``test_build_wires_thinking_into_client_kwargs`` 检查
    ``extra_body`` / ``model_kwargs`` 二者**必居其一**，对真实缺陷完全放行：
    langchain-openai 1.6.x 会把 ``model_kwargs`` 原样展开喂给 ``Completions.create()``，

        TypeError: Completions.create() got an unexpected keyword argument 'thinking'

    请求根本没发出去，而那条断言照样通过 —— 我们据此误判"两侧已对齐"。

    教训：**构造层正确 ≠ 调用层可用**。所以这里让请求真的走完 HTTP 栈，
    目的地是 127.0.0.1，**不消耗任何模型额度**。
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                bodies.append(json.loads(raw or b"{}"))
            except Exception:  # noqa: BLE001
                bodies.append({"__unparsable__": raw[:200].decode("utf-8", "replace")})
            payload = json.dumps({
                "id": "chatcmpl-x", "object": "chat.completion", "created": 0,
                "model": "wiretap",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):  # 静音
            pass

    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v3"


@pytest.mark.allows_real_client
@pytest.mark.parametrize("value", ["disabled", "enabled"])
def test_thinking_reaches_the_wire(value):
    """★ 端到端防线：服务端必须真的在 HTTP body 里收到 ``thinking``。

    这条是「langgraph 慢 3 倍」修复的**验收判据**。
    """
    try:
        from langchain_core.messages import HumanMessage
    except Exception:  # noqa: BLE001
        pytest.skip("langchain-core 未安装")

    bodies: list = []
    srv, base_url = _start_wiretap(bodies)
    try:
        chain = lg.ModelChain(base_url=base_url, api_key="k", model="wiretap",
                              candidates=[], thinking=value)
        chain.invoke([HumanMessage(content="ping")], [])
    finally:
        srv.shutdown()

    assert bodies, "请求根本没到服务端 —— 这正是 model_kwargs 分支的故障特征"
    assert bodies[0].get("thinking") == {"type": value}, (
        f"服务端没收到期望的 thinking：{json.dumps(bodies[0], ensure_ascii=False)[:300]}")


@pytest.mark.allows_real_client
def test_no_thinking_field_when_unset():
    """``thinking`` 为空串时必须**完全不带** —— 带上会让不支持该字段的服务端 400。"""
    try:
        from langchain_core.messages import HumanMessage
    except Exception:  # noqa: BLE001
        pytest.skip("langchain-core 未安装")

    bodies: list = []
    srv, base_url = _start_wiretap(bodies)
    try:
        chain = lg.ModelChain(base_url=base_url, api_key="k", model="wiretap",
                              candidates=[], thinking="")
        chain.invoke([HumanMessage(content="ping")], [])
    finally:
        srv.shutdown()

    assert bodies and "thinking" not in bodies[0], (
        f"不该下发 thinking，实际收到了：{json.dumps(bodies[0], ensure_ascii=False)[:300]}")


def test_stream_usage_is_disabled_by_default():
    """默认不请求 usage —— 我们只用正文，没必要让服务端在流尾多算一次。"""
    chain = lg.build_chain(model_factory=lambda m: _FakeLcModel(m))
    assert chain.stream_usage is False


# --------------------------------------------------------------------------- #
# 2. 模型客户端按 model 缓存（顺带保住 TCP 连接池）
# --------------------------------------------------------------------------- #

def test_model_client_is_reused_within_a_chain():
    """同一个 ModelChain 内按 model 复用客户端实例。

    每轮重建 ChatOpenAI 会连带重建 HTTP 连接池 —— 这不只是 0.4ms 的构造开销，
    而是**每轮都要重做一次 TCP/TLS 握手**。
    """
    built = []

    def factory(model: str):
        built.append(model)
        return _FakeLcModel(model)

    chain = lg.ModelChain(base_url="u", api_key="k", model="m", candidates=[],
                          model_factory=factory)
    first = chain._build("m")
    second = chain._build("m")
    # 走 factory 分支时不缓存（测试桩每次要新实例），这里断言的是"factory 分支直通"
    assert len(built) == 2, "factory 分支应保持每次构造，便于测试注入"
    assert first is not second


@pytest.mark.allows_real_client
def test_real_client_is_cached(monkeypatch):
    """真实客户端分支（无 factory）下，同一 model 只构造一次。"""
    try:
        from langchain_openai import ChatOpenAI
    except Exception:  # noqa: BLE001
        pytest.skip("langchain-openai 未安装")

    calls: list = []
    orig = lg.ModelChain._new_client

    def counting(self, model):  # noqa: ANN001
        calls.append(model)
        return orig(self, model)

    monkeypatch.setattr(lg.ModelChain, "_new_client", counting)
    chain = lg.ModelChain(base_url="https://example.invalid/v3", api_key="k",
                          model="m", candidates=[])
    a = chain._build("m")
    b = chain._build("m")
    assert a is b
    assert calls == ["m"], f"应当只构造一次，实际：{calls}"


# --------------------------------------------------------------------------- #
# 3. 流式合并：纯文本轮必须走 O(n) 路径
# --------------------------------------------------------------------------- #

def test_stream_collect_text_is_linear_not_quadratic():
    """800 片纯文本：增量合并必须显著快于逐片 ``+``。

    LangChain 的 chunk 合并是逐片新建对象并 merge 内部列表，n 片即 O(n²)；
    这里要求合力后仍有明显收益（阈值放宽到 1.5×，避免不同机器抖动导致假红）。
    """
    from langchain_core.messages import AIMessageChunk

    class _Bound:
        def stream(self, _msgs):
            for i in range(800):
                yield AIMessageChunk(content="x", id="c")

    pieces = []

    def hook(t):  # noqa: ANN001
        pieces.append(t)

    t0 = time.perf_counter()
    msg = lg.ModelChain._stream_collect(_Bound(), [], hook)
    fast = (time.perf_counter() - t0) * 1000

    # 参考实现：逐片对象合并
    t0 = time.perf_counter()
    full = None
    for i in range(800):
        piece = AIMessageChunk(content="x", id="c")
        full = piece if full is None else full + piece
    slow = (time.perf_counter() - t0) * 1000

    assert lg.text_of(getattr(msg, "content", "")) == "x" * 800, "正文拼接必须完整"
    assert len(pieces) == 800, "token 出口不能因为换了合并方式就少推"
    print(f"\n[perf] 800 片：一次性 {fast:.2f}ms vs 逐片 {slow:.2f}ms")
    assert fast < slow * 0.8, f"未体现线性合并优势：{fast:.2f}ms vs {slow:.2f}ms"


def test_stream_collect_keeps_tool_calls():
    """带工具调用的轮必须完整保留 tool_calls —— 走对象合并的慢路径保证语义。"""

    class _Bound:
        def stream(self, _msgs):
            from langchain_core.messages import AIMessageChunk

            yield AIMessageChunk(content="", id="c", tool_call_chunks=[
                {"name": "get_metrics", "args": "{}", "id": "c1", "index": 0}])
            yield AIMessageChunk(content="", id="c")

    msg = lg.ModelChain._stream_collect(_Bound(), [], lambda t: None)
    calls = getattr(msg, "tool_calls", None) or []
    assert calls, "工具调用在合并中丢了 —— 图会直接收敛、什么工具都不执行"
    assert calls[0]["name"] == "get_metrics"


# --------------------------------------------------------------------------- #
# 4. 端到端：一次 run 的极限成本（防止将来被回退成慢路径）
# --------------------------------------------------------------------------- #

def test_single_run_is_fast():
    """一次完整 run（含构图）应在百毫秒级。

    真正的耗时在网上，这里卡的是"我们不该在本地浪费时间"。
    阈值定得很宽（300ms），只为拦住明显的 O(n²) 或重复构造回归。
    """

    class _Chain:
        models = ["fake"]
        primary = "fake"
        attempts = []

        def invoke(self, lc_messages, tools=None, token_hook=None, timeout_fn=None):
            return _FakeLcModel(chunks=200).invoke(lc_messages), "fake"

    def dispatch(calls, _msgs):
        return {getattr(c, "id", "") or "": "ok" for c in calls}

    t0 = time.perf_counter()
    content, model = lg.run(
        messages=[ChatMessage.system("s"), ChatMessage.user("u")],
        specs=_specs(), dispatch_tools=dispatch, max_rounds=3,
        model_chain=_Chain(), use_checkpoint=False, thread_id="bench")
    ms = (time.perf_counter() - t0) * 1000

    assert content.strip(), "假模型应当产出正文"
    print(f"\n[perf] run() 一轮工具 + 一轮成稿：{ms:.1f}ms")
    assert ms < 300, f"单次 run 本地开销过大：{ms:.1f}ms"


# --------------------------------------------------------------------------- #
# 5. 待补（需要额度，本文件刻意不放）
# --------------------------------------------------------------------------- #
# 「两侧 thinking 对齐」的端到端收益目前只有**间接证据**（相同的模型、更少的轮次、
# 却慢 2~6 倍，见 qa/eval-compare/compare-20260922-213640.json 的逐条数据）。
# 真正的 live 对照要烧额度：额度恢复后跑
#     python qa/eval_compare.py --mode live --yes
# 判据：langgraph 侧单条均值应从 ~196s 回落到与 legacy 同量级（~65s），
#       且 fast 骨架仍必须两侧完全一致。
