"""模型自检（``GET /api/ai/diagnose``）与 ``capabilities()`` 的契约回归。

为什么单独给这个端点写测试
--------------------------
它曾经是**唯一一个没移植的端点**：Python 侧直接抛 501，界面上点「模型自检」得到一段红字
（"尚未移植…当前请用 Java 版的 GET /api/ai/diagnose"），而 Java 侧那时代码已经被删掉了 ——
也就是说这个功能当时是"两边都没有"。

补它的时候连着挖出三个"接口在、行为不对"的静默失败，每个都用一条断言钉住：

1. ``capabilities()`` 原本返回的是 Python 自研结构（configured / baseUrl / …），
   而前端「服务商能力体检」卡片按 ``chat``/``jsonMode``/``tools``/``embedding`` 取值 ——
   键名对不上，四项会全部显示成"不可用"。
2. ``/models/adopt`` 只回 ``ok``/``current``，前端拿它整份覆盖模型下拉的数据源，
   点一下"自动选一个能用的"下拉就空了（Java 是 ``m.putAll(modelInfo(false))``）。
3. ``LlmClient.list_available_models()`` 构造 ``ModelDiscovery`` 时传了不存在的
   ``timeout`` / ``http`` 参数 → 每次必抛 ``TypeError``；唯一的调用方把异常吞在
   ``log.debug`` 里，对外表现是"自动降级从来没生效过，只报主模型不可用"。

全部离线、零 token：底座用假的 HTTP 客户端替身。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.ai import llm_client as lc
from app.ai.discovery import ModelDiscovery, StaticDiscovery
from app.ai.llm_client import LlmClient

#: Java ``AiController.diagnose()`` 的键集合（对账基准，别随手改）
JAVA_DIAGNOSE_KEYS = {
    "ok", "baseUrl", "model", "embeddingModel", "keyHint", "keySource", "configSource",
    "embedding", "capabilities", "ragIndex", "models",
    "processStartedAt", "processUptimeSeconds", "envHint", "webSearch",
}
#: Java ``AiController.modelInfo()`` 的键集合
JAVA_MODEL_INFO_KEYS = {
    "current", "candidates", "available", "availableCount", "recommended",
    "autoFallback", "discovery", "unavailable", "lastSwitch", "autoConsume", "hint",
}
#: 前端「服务商能力体检」卡片读的四个键
CAP_KEYS = {"chat", "jsonMode", "tools", "embedding"}


# ------------------------------------------------------------------
# 替身
# ------------------------------------------------------------------


def _settings(**over):
    """最小可用的 Settings 替身：只带 LlmClient 真正读到的字段。"""
    ai = dict(
        enabled=True,
        base_url="http://127.0.0.1:9/v1",
        api_key="test-key-abcdefghijklmn",
        model="mock-model",
        model_candidates="",
        model_auto_fallback=True,
        model_discovery=True,
        auto_consume=False,
        chat_timeout_seconds=600,
        thinking_type="disabled",
        embedding_model="",
    )
    rag = dict(
        embedding_enabled=True,
        embedding_base_url="",
        embedding_api_key="",
        embedding_model="",
        embedding_dimensions=0,
    )
    ai.update(over.pop("ai", {}))
    rag.update(over.pop("rag", {}))
    return SimpleNamespace(ai=SimpleNamespace(**ai), rag=SimpleNamespace(**rag))


class _Resp:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


def _default_reply(url, body):
    """一个"合规服务商"的应答：caps 的三个探测都走绿路径。"""
    if "ping_tool" in json.dumps(body, ensure_ascii=False):
        return _Resp(200, json.dumps({
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None,
                                     "tool_calls": [{"id": "c1", "type": "function",
                                                     "function": {"name": "ping_tool",
                                                                  "arguments": "{}"}}]}}]}))
    if (body.get("response_format") or {}).get("type") == "json_object":
        return _Resp(200, json.dumps({
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"ok":true}'}}]}))
    return _Resp(200, json.dumps({
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "pong"}}]}))


class FakeHttp:
    """按请求内容回放的假底座。``reply`` 是 ``(url, body) -> _Resp``。"""

    def __init__(self, reply=None) -> None:
        self.reply = reply or _default_reply
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None):  # noqa: A002 - 与 httpx 同名
        self.calls.append({"url": url, "body": json})
        return self.reply(url, json)

    def close(self):
        pass


def _client(http=None, **over) -> LlmClient:
    """``http`` 既可以是 :class:`FakeHttp`，也可以直接给一个 ``(url, body) -> _Resp``。"""
    h = http if isinstance(http, FakeHttp) else FakeHttp(http)
    return LlmClient(http=h, settings=_settings(**over))


# ------------------------------------------------------------------
# capabilities()
# ------------------------------------------------------------------


class TestCapabilities:
    def test_key_set_matches_frontend_contract(self):
        """键名即契约：前端按这 4 个键取值，多一个少一个都会让整张体检表失真。"""
        caps = _client().capabilities()
        assert set(caps) == CAP_KEYS

    def test_green_path(self):
        caps = _client().capabilities()
        assert caps["chat"] == "ok"
        assert caps["jsonMode"] == "ok"
        assert caps["tools"] == "ok"
        assert caps["embedding"].startswith("本地向量"), caps["embedding"]

    def test_json_probe_is_not_limited_by_max_tokens(self):
        """推理型模型思考 token 会吃满预算 → 必须给足，否则会被误判成"不支持 JSON 模式"。"""
        http = FakeHttp()
        _client(http=http).capabilities()
        json_call = [c for c in http.calls
                     if (c["body"].get("response_format") or {}).get("type") == "json_object"]
        assert json_call, "没有发出 JSON 模式探测请求"
        assert json_call[0]["body"]["max_tokens"] >= 1024

    def test_truncated_json_probe_reports_why(self):
        def reply(url, body):
            if (body.get("response_format") or {}).get("type") == "json_object":
                return _Resp(200, json.dumps({
                    "choices": [{"index": 0, "finish_reason": "length",
                                 "message": {"role": "assistant", "content": None}}]}))
            return _default_reply(url, body)

        caps = _client(http=FakeHttp(reply)).capabilities()
        assert "截断" in caps["jsonMode"], caps["jsonMode"]

    def test_missing_tool_calls_is_reported(self):
        def reply(url, body):
            if "ping_tool" in json.dumps(body, ensure_ascii=False):
                return _Resp(200, json.dumps({
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "好的"}}]}))
            return _default_reply(url, body)

        caps = _client(http=FakeHttp(reply)).capabilities()
        assert "tool_calls" in caps["tools"], caps["tools"]

    def test_embedding_disabled_is_not_a_failure(self):
        """配置状态不等于故障：「关掉了」要如实说关掉了，而不是报红。"""
        caps = _client(rag={"embedding_enabled": False}).capabilities()
        assert caps["embedding"].startswith("已关闭")

    def test_embedding_remote_without_key_says_how_to_fix(self):
        """配了独立向量化端点却没给 Key：直接说怎么配，别白打一次注定 401 的请求。"""
        http = FakeHttp()
        caps = _client(http=http, rag={"embedding_base_url": "https://emb.example.com/v1",
                                       "embedding_model": "bge-m3"}).capabilities()
        assert "APP_EMBEDDING_API_KEY" in caps["embedding"]
        assert all("emb.example.com" not in c["url"] for c in http.calls), \
            "未配置 Key 时不该真的去请求第三方 embedding"


# ------------------------------------------------------------------
# 模型列举：曾经必抛 TypeError 的那条路
# ------------------------------------------------------------------


class TestModelDiscoveryWiring:
    def test_ctor_kwargs_are_accepted_by_real_class(self, monkeypatch):
        """把传给 ModelDiscovery 的关键字**真的喂给真身** —— 参数名写错就该在这里炸。"""
        captured: dict = {}

        class Spy(ModelDiscovery):
            def __init__(self, **kw):
                captured.update(kw)
                super().__init__(**kw)

        monkeypatch.setattr(lc, "ModelDiscovery", Spy, raising=False)
        monkeypatch.setattr("app.ai.discovery.ModelDiscovery", Spy)

        client = _client()
        client.list_available_models()  # 不得抛 TypeError
        assert "timeout" not in captured and "http" not in captured
        assert set(captured) <= {"base_url", "api_key", "enabled", "transport"}

    def test_missing_key_does_not_hit_network(self):
        """没配 Key 时列举应当直接返回空并留下原因，而不是发一个注定 401 的请求。"""
        client = _client(ai={"api_key": ""})
        assert client.list_available_models() == []
        assert "Key" in (client.get_discovery_issue() or "")

    def test_discovered_models_feed_the_fallback_chain(self):
        """降级第二轮的招牌动作就是"拉一次账号可用模型再补试"——这里钉住这条缝合线。"""
        client = LlmClient(http=FakeHttp(), settings=_settings(),
                           discovery=StaticDiscovery(["alive-1", "alive-2"]))
        assert list(client._discover_models(True)) == ["alive-1", "alive-2"]

    def test_discovery_failure_leaves_a_reason(self):
        """列举失败要留下原因（自检页要显示"为什么这里是空的"），不能静默返回空列表。"""
        client = _client()  # base_url 指向没人监听的端口：连接被拒
        assert client._discover_models(True) == []
        assert client.get_discovery_issue()


class TestGetModelSemantics:
    """``get_model()`` 是**当前生效**模型，不是配置值 —— 对齐 Java ``getModel()``。

    Java 真身（``LlmClient.noteModelSuccess``）里是直接 ``model = m``，降级会把
    这个字段改写掉。Python 侧一度返回构造时读到的配置值，于是自动降级生效后，
    界面顶部横幅仍显示旧模型、右侧卡片显示新模型，**同一屏两处自相矛盾**。
    """

    def test_reflects_auto_downgrade(self):
        client = _client(ai={"model": "old-model"})
        assert client.get_model() == "old-model"
        client._cooldown.note_success("new-model", "连通性自检", "429 限流", True)
        assert client.get_model() == "new-model"

    def test_runtime_switch_compares_against_effective_model(self):
        """已降级到 new-model 时，用户再「应用」配置值 old-model 必须真的切回去。

        用 ``self.model``（配置值）做比较基准会判成"没变化"而直接跳过 ——
        界面提示"应用成功"，实际仍在用降级后的模型，属于静默失败。
        """
        client = _client(ai={"model": "old-model"})
        client._cooldown.note_success("new-model", "自检", "429", True)
        assert client.get_model() == "new-model"
        assert client.apply_runtime_config(None, "old-model", None) is None
        assert client.get_model() == "old-model", "提交配置值必须真的切回去"


# ------------------------------------------------------------------
# /diagnose 与 /models 的键集合
# ------------------------------------------------------------------


class TestDiagnoseContract:
    def _patch(self, monkeypatch, llm):
        from app.api import ai as ai_api

        monkeypatch.setattr(ai_api, "_llm_for_diagnose", lambda: llm)
        monkeypatch.setattr(ai_api, "_embedding_info", lambda _llm: {"mode": "local"})
        monkeypatch.setattr(ai_api, "_rag_index_stats", lambda: {"available": True, "enabled": True})
        monkeypatch.setattr(ai_api, "web_search_info", lambda: {"enabled": True})
        monkeypatch.setattr(ai_api, "model_info", lambda refresh=False: {"current": "m1"})
        return ai_api

    def test_key_set_is_java_identical(self, monkeypatch):
        llm = _client()
        api = self._patch(monkeypatch, llm)
        data = api.diagnose().data
        assert set(data) == JAVA_DIAGNOSE_KEYS
        assert data["ok"] is True
        assert data["capabilities"]["chat"] == "ok"
        assert data["processStartedAt"] and data["processUptimeSeconds"] >= 0
        assert data["envHint"]

    def test_disabled_switch_is_distinguishable_from_missing_key(self, monkeypatch):
        """``app.ai.enabled=false`` 才等价于 Java 的 ``llm == null``；
        "开关开着但没配 Key"要报可修的 Key 问题，而不是"未启用外部大模型"。"""
        api = self._patch(monkeypatch, None)
        data = api.diagnose().data
        assert set(data) == {"ok", "issue", "webSearch", "embedding", "ragIndex"}
        assert data["ok"] is False and "未启用外部大模型" in data["issue"]

        api = self._patch(monkeypatch, _client(ai={"api_key": ""}))
        data = api.diagnose().data
        assert data["ok"] is False
        assert "AI_API_KEY" in data["issue"], data["issue"]

    def test_model_info_key_set(self, monkeypatch):
        from app.api import ai as ai_api

        monkeypatch.setattr(ai_api, "_llm_for_diagnose", lambda: _client())
        data = ai_api.model_info(False)
        # Java 的 11 个键 + Python 侧补充的 discoveryIssue（"为什么这里是空的"）
        assert set(data) == JAVA_MODEL_INFO_KEYS | {"discoveryIssue"}
        assert data["current"] == "mock-model"
        assert data["hint"]


@pytest.mark.parametrize("key", sorted(CAP_KEYS))
def test_capability_values_are_human_readable(key):
    """"ok" 之外的每一项都必须是一句能照着动手的原因，不能是异常类名或空串。"""
    caps = _client(http=FakeHttp(lambda url, body: _Resp(500, "boom"))).capabilities()
    val = caps[key]
    assert isinstance(val, str) and val.strip()
    assert not val.startswith("Traceback")
