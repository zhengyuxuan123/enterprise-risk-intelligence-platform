"""阶段 4 回归：联网检索三通道的编排与解析。

全部**不联网**：解析层喂固定 HTML，通道层用假对象。
真正发请求的部分（httpx）由注入的假 client 顶替。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import List, Optional

import pytest

from app.web.direct_client import (
    DirectWebSearchClient,
    Hit,
    clean,
    decode_entities,
    domain_of,
    engine_url,
    normalize_url,
    parse_bing,
    parse_so360,
)
from app.web.mcp_client import McpWebSearchClient
from app.web import mcp_client, mcp_server
from app.web.search_service import (
    SearchOutcome,
    WebSearchService,
    _collect_annotations,
    _describe,
    _extract_from_text,
    _normalize_mode,
    _parse_sources,
    mask_key,
)


class _Resp:
    def __init__(self, text="", status=200, payload=None):
        self.text = text
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload if self._payload is not None else json.loads(self.text)


class _FakeHttp:
    """按预设队列应答；记录被请求的 URL 用于断言引擎选择。"""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.urls: List[str] = []
        self.bodies: List[dict] = []

    def get(self, url, **kw):
        self.urls.append(url)
        r = self.responses.pop(0) if self.responses else _Resp("")
        return r

    def post(self, url, **kw):
        self.urls.append(url)
        self.bodies.append(kw.get("json") or {})
        return self.responses.pop(0) if self.responses else _Resp("{}", payload={})

    def close(self):
        pass


# ------------------------------------------------------------------
# 解析层
# ------------------------------------------------------------------


def test_decode_entities_named_and_numeric():
    assert decode_entities("a&amp;b") == "a&b"
    assert decode_entities("&lt;x&gt;") == "<x>"
    assert decode_entities("&#20320;&#22909;") == "你好"
    assert decode_entities("&nbsp;a&nbsp;b") == " a b"
    assert decode_entities("无实体") == "无实体"


def test_clean_strips_tags_and_collapses_whitespace():
    assert clean("<b>标题</b><span>x</span>") == "标题x"
    assert clean("a \n\t b") == "a b"
    assert clean(None) == ""


def test_domain_of_strips_www():
    assert domain_of("https://www.example.com/a?b=1") == "example.com"
    assert domain_of("not-a-url") == "not-a-url"


def test_engine_url_zh_goes_domestic_en_goes_global():
    """中文走国内站、纯英文走国际站。

    同一串英文短语在 cn.bing.com 上会被当成「某个单词什么意思」来匹配，
    拿回来的全是词义辨析页。
    """
    assert "cn.bing.com" in engine_url("bing", "客户流失率")
    assert "www.bing.com" in engine_url("bing", "customer churn")
    assert "mkt=en-US" in engine_url("bing", "customer churn")
    assert "www.so.com" in engine_url("so360", "客户流失率")
    with pytest.raises(ValueError):
        engine_url("unknown", "x")


def test_normalize_url_decodes_bing_redirect():
    real = "https://real.example.com/path"
    token = base64.b64encode(real.encode("utf-8")).decode("ascii")
    wrapped = f"https://www.bing.com/ck/a?u=a1{token}&ntb=1"
    assert normalize_url(wrapped, False) == real


def test_normalize_url_drops_search_engine_pages():
    # 搜索引擎自己的站内页面不是「来源」
    assert normalize_url("https://cn.bing.com/search?q=x", False) is None
    assert normalize_url("https://www.so.com/s?q=x", False) is None
    # 备用引擎的跳转链在 allowRedirect 时保留：主引擎挂掉时「一条能点开的链接」好过「零条」
    assert normalize_url("https://www.so.com/link?m=a", False) is None
    assert normalize_url("https://www.so.com/link?m=a", True) == "https://www.so.com/link?m=a"
    assert normalize_url("relative/path", False) is None
    assert normalize_url(None, False) is None


def test_parse_bing_extracts_title_url_snippet():
    html = (
        '<li class="b_algo"><h2><a href="https://a.example.com/x">流失预警实践</a></h2>'
        '<div class="b_caption"><p>这是摘要</p></div></li>'
        '<li class="b_algo"><h2><a href="https://cn.bing.com/search?q=x">站内页应丢弃</a></h2></li>'
    )
    hits = parse_bing(html, 10)
    assert len(hits) == 1
    assert hits[0].title == "流失预警实践"
    assert hits[0].url == "https://a.example.com/x"
    assert hits[0].snippet == "这是摘要"
    assert hits[0].engine == "bing"


def test_parse_so360_uses_cite_when_title_missing():
    html = (
        '<li class="res-list"><h3 class="res-title"><a href="https://www.so.com/link?m=a"></a></h3>'
        '<p class="res-desc">描述</p><cite>https://www.site.cn/p</cite></li>'
    )
    hits = parse_so360(html, 5)
    assert len(hits) == 1
    # 标题空 → 用 cite 里的域名（Java 是 split("[\s/>]")[0]，所以路径部分会保留）
    assert hits[0].title == "site.cn/p"
    assert hits[0].snippet == "描述"


def test_explain_only_results_are_filtered():
    """释义 / 百科 / 词典这类结果常年占据首页，但几乎不可能构成业务依据。

    这是纯粹的本地规则判断（零 token），剔掉它们才能让真正有信息量的来源浮上来。
    """
    from app.web.direct_client import _is_explain_only

    assert _is_explain_only(Hit("SaaS是什么意思", "https://x.com/a", "", "bing"))
    assert _is_explain_only(Hit("词条", "https://baike.baidu.com/item/x", "", "bing"))
    assert _is_explain_only(Hit("词", "https://en.wikipedia.org/wiki/X", "", "bing"))
    assert not _is_explain_only(Hit("2026 SaaS 流失率基准报告", "https://x.com/a", "", "bing"))


def test_direct_client_dedups_by_url_and_caps():
    c = DirectWebSearchClient(engines=["bing"], timeout_seconds=1)
    html = "".join(
        f'<li class="b_algo"><h2><a href="https://a.example.com/{i}">标题{i}</a></h2>'
        f'<div class="b_caption"><p>摘要{i}</p></div></li>' for i in range(8)
    )
    http = _FakeHttp([_Resp(html)])
    # 候选池按 2 倍抓（n=3 → 抓 6），再截取前 3
    hits = c.search("客户流失率", 3, client=http)
    assert [h.url for h in hits] == [f"https://a.example.com/{i}" for i in range(3)]
    assert c.get_last_engine() == "bing"


def test_direct_client_retries_once_per_engine():
    """每个引擎给两次机会：抓结果页是公网请求，TLS 握手被重置这类抖动很常见。"""
    c = DirectWebSearchClient(engines=["bing"], timeout_seconds=1)
    html = '<li class="b_algo"><h2><a href="https://a.example.com/1">T</a></h2></li>'

    class _Flaky(_FakeHttp):
        def __init__(self):
            super().__init__()
            self.n = 0

        def get(self, url, **kw):
            self.urls.append(url)
            self.n += 1
            if self.n == 1:
                raise RuntimeError("Remote host terminated the handshake")
            return _Resp(html)

    http = _Flaky()
    hits = c.search("x", 1, client=http)
    assert len(hits) == 1
    assert len(http.urls) == 2  # 第一次失败、第二次成功


def test_direct_client_blank_query():
    c = DirectWebSearchClient(engines=["bing"])
    assert c.search("  ", 3, client=_FakeHttp()) == []
    assert c.get_last_error() == "缺少检索词"


# ------------------------------------------------------------------
# MCP 通道
# ------------------------------------------------------------------


class _McpCfg:
    enabled = True
    transport = "http"
    command = ""
    url = "http://mcp.local/rpc"
    headers = ""
    tool_name = ""
    args_template = '{"query":"{q}","count":{n}}'
    timeout_seconds = 5


def test_mcp_picks_search_tool_by_hint():
    c = McpWebSearchClient(cfg=_McpCfg())
    assert c._pick_tool(["read_file", "brave_web_search", "sql_query"]) == "brave_web_search"
    # 含 search 但属于"抓取/代码"类的要跳过
    assert c._pick_tool(["code_search", "fetch_url"]) is None
    assert c._pick_tool([]) is None


def test_mcp_build_args_from_template():
    c = McpWebSearchClient(cfg=_McpCfg())
    assert c._build_args("客户流失率", 5) == {"query": "客户流失率", "count": 5}
    c.args_template = '{"q":"{q}","numResults":{n}}'
    assert c._build_args("a", 2) == {"q": "a", "numResults": 2}
    # 模板坏了不能让检索崩掉
    c.args_template = "{not json"
    assert c._build_args("a", 2) == {"query": "a", "count": 2}


def test_mcp_parses_json_array_and_text():
    c = McpWebSearchClient(cfg=_McpCfg())
    res = {"content": [{"type": "text", "text": json.dumps([
        {"title": "报告A", "url": "https://a.example.com/1", "snippet": "摘要A"},
        {"title": "报告B", "url": "https://b.example.com/2", "description": "摘要B"},
    ], ensure_ascii=False)}]}
    hits = c._parse_hits(res, 5)
    assert [h.title for h in hits] == ["报告A", "报告B"]
    assert hits[1].snippet == "摘要B"


def test_mcp_parses_markdown_text():
    c = McpWebSearchClient(cfg=_McpCfg())
    res = {"content": [{"type": "text",
                        "text": "1. 2026 SaaS 基准报告\nhttps://r.example.com/report\n正文"}]}
    hits = c._parse_hits(res, 5)
    assert hits[0].url == "https://r.example.com/report"
    assert "基准报告" in hits[0].title


def test_mcp_extract_handles_sse_and_error():
    c = McpWebSearchClient(cfg=_McpCfg())
    assert c._extract('data: {"result":{"ok":1}}') == {"ok": 1}
    assert c._extract('noise {"result":{"ok":2}} tail') == {"ok": 2}
    with pytest.raises(RuntimeError):
        c._extract('{"error":{"code":-1,"message":"boom"}}')


def test_mcp_status_reason_when_unconfigured():
    class _Bare:
        enabled = True
        transport = ""
        command = ""
        url = ""
        headers = ""
        tool_name = ""
        args_template = ""
        timeout_seconds = 5

    c = McpWebSearchClient(cfg=_Bare())
    assert c.is_enabled() is False
    assert "未配置" in (c.status_reason() or "")


def test_mcp_headers_parsing():
    c = McpWebSearchClient(cfg=_McpCfg())
    assert c._parse_headers("A=1\nB=2") == {"A": "1", "B": "2"}
    assert c._parse_headers("") == {}
    assert c._parse_headers('X="v"') == {"X": "v"}


def test_stdio_session_read_line_obeys_timeout():
    """A silent MCP child must not block the analysis thread indefinitely."""
    import time
    from types import SimpleNamespace

    class _BlockingStdout:
        def readline(self):
            time.sleep(1)
            return ""

    session = mcp_client._StdioSession(SimpleNamespace(stdout=_BlockingStdout()))
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="MCP stdio"):
        session.read_line(0.03)
    assert time.monotonic() - started < 0.3


def test_builtin_mcp_server_contract(monkeypatch):
    init = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init["result"]["capabilities"]["tools"] == {}

    listed = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert listed["result"]["tools"][0]["name"] == "web_search"

    monkeypatch.setattr(mcp_server._client, "search", lambda q, n: [
        Hit("SaaS report", "https://example.com/report", "retention benchmark", "mcp")
    ])
    called = mcp_server.handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "web_search", "arguments": {"query": "retention", "count": 3}},
    })
    content = json.loads(called["result"]["content"][0]["text"])
    assert content[0]["url"] == "https://example.com/report"


# ------------------------------------------------------------------
# 服务编排
# ------------------------------------------------------------------


@dataclass
class _FakeDirect:
    hits: List[Hit]
    fail: bool = False

    def search(self, q, n):
        return [] if self.fail else self.hits

    def get_last_engine(self):
        return "bing"

    def get_last_error(self):
        return "引擎全部失败" if self.fail else None


def test_render_builds_indexed_sources():
    svc = WebSearchService(direct=_FakeDirect([]), mcp=None)
    o = svc._render([Hit("标题A", "https://a.example.com/1", "摘要A", "bing"),
                     Hit("标题B", "https://b.example.com/2", "", "bing")], "direct:bing")
    assert o.ok()
    assert [s.index for s in o.sources] == [1, 2]
    assert "[1] 标题A" in o.summary
    assert "来源：https://a.example.com/1" in o.summary
    # 摘要为空的那条只写标题与来源，不留空行
    assert o.sources[1].snippet == ""
    assert o.sources[0].site_name == "a.example.com"


def test_search_prefers_mcp_then_direct():
    #! 假数据的标题/摘要必须与查询**真的相关**：
    #! 2026-09-22 起通道编排多了一道相关度质量门（直连/MCP 给了无关网页不算成功），
    #! 沿用「MCP标题」这种占位文本会被正确地判掉 —— 这正说明那道门生效了。
    mcp_hits = [Hit("客户流失率上升的三大原因", "https://mcp.example.com/1",
                    "客户流失率环比上升 12%，主要来自沉默用户", "mcp")]

    class _Mcp:
        def is_enabled(self):
            return True

        def search(self, q, n):
            return mcp_hits

        def get_last_tool(self):
            return "brave_web_search"

        def get_last_error(self):
            return None

        def status_reason(self):
            return None

        def get_transport(self):
            return "http"

    svc = WebSearchService(direct=_FakeDirect([Hit("客户流失预警模型实践", "https://d.example.com",
                                                  "流失预警：提前识别高流失风险客户", "bing")]),
                           mcp=_Mcp())
    o = svc.search("客户流失率", 3)
    assert o.ok()
    assert svc.get_last_channel() == "mcp:brave_web_search"


def test_search_falls_back_to_direct_when_mcp_empty():
    class _Mcp:
        def is_enabled(self):
            return True

        def search(self, q, n):
            return []

        def get_last_tool(self):
            return None

        def get_last_error(self):
            return "未暴露检索工具"

        def status_reason(self):
            return None

        def get_transport(self):
            return "http"

    svc = WebSearchService(direct=_FakeDirect([Hit("直连", "https://d.example.com", "s", "bing")]),
                           mcp=_Mcp())
    o = svc.search("x", 2)
    assert o.ok()
    assert svc.get_last_channel() == "direct:bing"


def test_mode_direct_never_touches_api_even_if_key_present():
    """``mode=direct`` 绝不下发密钥：失败也不降级到 API。"""
    svc = WebSearchService(direct=_FakeDirect([], fail=True), mcp=None)
    svc.mode = "direct"
    svc.api_key = "sk-should-not-be-used"
    o = svc.search("x", 2)
    assert not o.ok()
    assert "不会使用任何 API Key" in o.error


def test_mode_mcp_does_not_fall_back():
    class _Mcp:
        def is_enabled(self):
            return True

        def search(self, q, n):
            return []

        def get_last_tool(self):
            return None

        def get_last_error(self):
            return "boom"

        def status_reason(self):
            return None

        def get_transport(self):
            return "stdio"

    svc = WebSearchService(direct=_FakeDirect([Hit("直连", "https://d.example.com", "s", "bing")]),
                           mcp=_Mcp())
    svc.mode = "mcp"
    o = svc.search("x", 2)
    assert not o.ok()
    assert "不会回退到直连" in o.error


def test_search_rejects_blank_query_and_disabled():
    svc = WebSearchService(direct=_FakeDirect([]), mcp=None)
    assert svc.search("  ", 3).error == "缺少检索词"
    svc.enabled = False
    assert "已关闭" in svc.search("x", 3).error


def test_status_reason_api_mode_requires_key_model_url():
    svc = WebSearchService(direct=_FakeDirect([]), mcp=None)
    svc.mode = "api"
    svc.api_key = ""
    svc._fallback_key = ""  # 本机有 AI_API_KEY 环境变量，不清掉就会走到"模型未配置"那条
    assert "未配置 API Key" in (svc.status_reason() or "")


def test_api_search_describes_ark_404_as_plugin_not_open():
    """未开通联网插件时方舟返回 **404**，极易被误读成「接口不存在」而白折腾。"""
    msg = _describe(404, '{"error":{"code":"ToolNotOpen"}}')
    assert "尚未开通" in msg and "免费开通" in msg
    assert "接口不存在" not in msg
    assert "无权调用" in _describe(401, "invalid_api_key")
    assert "限流" in _describe(429, "RateLimit")
    assert "/responses" in _describe(404, "NotFound")


def test_quota_limit_is_not_described_as_plain_rate_limit():
    """「安心体验模式」到额不是普通限流：模型服务被暂停，不人工调整永远不会恢复。

    把它说成"请稍后重试"等于让用户对着一个永久故障一直等。
    """
    body = ('{"error":{"code":"SetLimitExceeded","message":"has reached the set inference '
            'limit ... close the \\"Safe Experience Mode\\""}}')
    msg = _describe(429, body)
    assert "安心体验模式" in msg
    assert "请稍后重试" not in msg, "这不是等一等就会好的限流"
    assert "欠费" in _describe(429, '{"error":{"code":"Arrearage"}}')


class _FakeApi:
    """冒充方舟 Responses 的 HTTP client。"""

    def __init__(self, ok=True, sources=1):
        self.ok = ok
        self.sources = sources
        self.called = 0

    def post(self, url, json=None, headers=None, **kw):
        self.called += 1
        if not self.ok:
            return _Resp('{"error":{"code":"ToolNotOpen"}}', 404)
        content = [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": "出口退税率已调整。",
                "annotations": [{"url": f"https://gov.example.com/p{i}",
                                 "title": f"政策公告{i}"}
                                for i in range(self.sources)],
            }],
        }]
        return _Resp('', 200, payload={"output": content})


_JUNK_HITS = [Hit("2026年放假安排一览", "https://rili.example.com/2026", "日历表", "bing"),
              Hit("芒果TV-天生青春", "https://mgtv.example.com", "视频网站", "bing"),
              Hit("万年历_日历网", "https://rili.example.com/w", "农历查询", "bing")]


def test_auto_prefers_api_over_direct():
    """有契约的 API 必须排在免 Key 抓取之前（见 search_service 模块头部的实测结论）。"""
    fake = _FakeApi(ok=True, sources=2)
    svc = WebSearchService(direct=_FakeDirect([Hit("直连标题", "https://d.example.com", "", "bing")]),
                           mcp=None, http=fake)
    svc.api_key = "sk-x"
    svc._model = "m"
    svc.base_url = "https://api.example.com/v3"
    o = svc.search("出口退税政策调整", 3)
    assert o.ok()
    assert fake.called == 1, "API 可用时应当优先使用它"
    assert "gov.example.com" in o.sources[0].url


def test_direct_results_must_pass_relevance_gate():
    """★ 核心修复：免 Key 通道给了"看起来像结果"的垃圾，不能算成功。

    cn.bing 对任意查询都会老实返回十条网页（日历网 / 随机门户 / 芒果TV），
    若不经判定就返回，上层会以为"联网成功"，于是用户看到的不过是
    一份引用着无关网页的"已核实"结论——比干脆说"没联网"更危险。

    这里构造的场景是真实故障形态：**付费通道没开通**（方舟 ToolNotOpen），
    于是自动兜底到直连，而直连给回来的全是无关网页。
    """
    svc = WebSearchService(direct=_FakeDirect(_JUNK_HITS), mcp=None,
                           http=_FakeApi(ok=False))  # 404 ToolNotOpen
    svc.api_key = "sk-x"
    svc._model = "m"
    svc.base_url = "https://api.example.com/v3"
    o = svc.search("出口退税政策调整 税务总局公告", 3)
    assert not o.ok(), "直连的无关结果不能被当成联网成功"
    assert o.sources == [], "一条垃圾来源都不该往外传"
    assert any("不相关" in n for n in svc.last_notes), svc.last_notes
    # 失败原因要说全：付费通道为什么没用上 + 直连为什么被否
    assert "ToolNotOpen" in o.error or "尚未开通" in o.error
    assert any("不相关" in n for n in svc.last_notes)


def test_direct_only_mode_still_rejects_junk():
    """``mode=direct``（用户明确要零成本）时也要守同一把尺子。

    想要省钱是合理的诉求，但不能因此放松质量：宁可如实说「这次没联网」，
    也不能把日历网当政策依据递给模型。
    """
    svc = WebSearchService(direct=_FakeDirect(_JUNK_HITS), mcp=None)
    svc.mode = "direct"
    o = svc.search("出口退税政策调整 税务总局公告", 3)
    assert not o.ok()
    assert "不相关" in o.error
    assert "不会使用任何 API Key" in o.error


def test_direct_failure_surfaces_in_error_text():
    """直连拿不到东西时，失败原因要写进最后那句话——而不是含糊的"联网失败"。"""
    svc = WebSearchService(direct=_FakeDirect([], fail=True), mcp=None)
    svc.mode = "direct"
    o = svc.search("x", 2)
    assert not o.ok()
    assert "引擎全部失败" in o.error
    assert "不会使用任何 API Key" in o.error


def test_probe_is_zero_cost_by_default():
    """自检默认不许发任何请求：体检页常驻刷新时不该悄悄烧额度。"""
    fake = _FakeApi(ok=True, sources=1)
    svc = WebSearchService(direct=_FakeDirect([]), mcp=None, http=fake)
    svc.api_key = "sk-x"
    svc._model = "m"
    svc.base_url = "https://api.example.com/v3"
    cold = svc.probe("出口退税")
    assert cold["live"] is False
    assert fake.called == 0, "默认探活绝不能真发请求"
    assert cold["mode"] == "auto"
    assert cold["enabled"] is True or cold["reason"]
    live = svc.probe("出口退税", live=True)
    assert live["live"] is True
    assert fake.called == 1, "显式要求时才发一次"
    assert live["sources"]


def test_collect_annotations_dedups_by_url():
    out = []
    _collect_annotations([{"url": "https://a.com", "title": "A"},
                          {"url": "https://a.com", "title": "重复"},
                          {"url": "https://b.com"}], out, 5)
    assert [s.url for s in out] == ["https://a.com", "https://b.com"]
    assert out[1].title == "https://b.com"  # 没 title 就用 url


def test_extract_from_text_strips_trailing_punctuation():
    src = _extract_from_text("见 https://a.example.com/x. 另见 https://b.example.com/y，", 5)
    assert [s.url for s in src] == ["https://a.example.com/x", "https://b.example.com/y"]


@pytest.mark.parametrize("raw, want", [
    ("auto", "auto"), ("DIRECT", "direct"), ("crawl", "direct"), ("free", "direct"),
    ("api", "api"), ("ark", "api"), ("mcp", "mcp"), ("weird", "auto"), (None, "auto"),
])
def test_normalize_mode(raw, want):
    assert _normalize_mode(raw) == want


def test_parse_sources_and_mask():
    assert _parse_sources("a, b c，d") == ["a", "b", "c", "d"]
    assert _parse_sources('"x"') == ["x"]
    assert _parse_sources(None) == []
    assert mask_key(None) == "(未配置)"
    assert mask_key("1234567890abcdef") == "123456…cdef"
    assert mask_key("short") == "****"


def test_outcome_ok_requires_sources():
    assert SearchOutcome("x", [], None).ok() is False
    assert SearchOutcome(None, [], "err").ok() is False


class _FakeLlm:
    """冒充主模型客户端：提供当前模型与候选，供联网通道轮换使用。"""

    def __init__(self, model="m-main", candidates=("m-b", "m-c")):
        self._model = model
        self._candidates = list(candidates)

    def get_model(self):
        return self._model

    def get_model_candidates(self):
        return list(self._candidates)


class _FakeApiByModel:
    """按请求里的 model 分派应答：用于验证「换模型重试」而不是「通道直接报废」。"""

    def __init__(self, failing_models, fail_status=429, fail_body='{"error":{"code":"SetLimitExceeded"}}'):
        self.failing_models = set(failing_models)
        self.fail_status = fail_status
        self.fail_body = fail_body
        self.models: List[str] = []

    def post(self, url, json=None, headers=None, **kw):
        m = (json or {}).get("model") or ""
        self.models.append(m)
        if m in self.failing_models:
            return _Resp(self.fail_body, self.fail_status)
        content = [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": "出口退税率已调整。",
                "annotations": [{"url": "https://gov.example.com/p1", "title": "政策公告"}],
            }],
        }]
        return _Resp('', 200, payload={"output": content})


def _service_with_models(http, llm=None, **kw):
    svc = WebSearchService(direct=_FakeDirect(_JUNK_HITS), mcp=None, http=http, llm=llm)
    svc.api_key = "sk-x"
    svc._model = kw.get("model", "m-main")
    svc.base_url = "https://api.example.com/v3"
    return svc


def test_api_channel_rotates_model_on_per_model_quota():
    """★ 「安心体验模式」的上限是**按模型**计的：撞到就该换下一个模型。

    实测（2026-09-22）：同一把 Key 打 deepseek-v4-1-flash / deepseek-v4-flash-ga 都是
    429 SetLimitExceeded，换 doubao / glm 立刻 200 且 web_search 正常执行。
    若这里不轮换，api 通道会整体报废并悄悄回退到免 Key 直连——
    于是用户拿到的是引用着无关网页的"已核实"结论，比明说"没联网"更糟。
    """
    http = _FakeApiByModel(failing_models={"m-main", "m-b"})
    svc = _service_with_models(http, llm=_FakeLlm("m-main", ("m-b", "m-c")))
    o = svc.search("出口退税政策调整", 3)
    assert o.ok(), o.error
    assert http.models == ["m-main", "m-b", "m-c"], "应逐个模型重试直到可用"
    assert "gov.example.com" in o.sources[0].url
    assert any("自动切换" in n and "m-c" in n for n in svc.last_notes), svc.last_notes


def test_api_channel_does_not_rotate_on_key_error():
    """Key/账号级错误换模型只会拿同一把坏 Key 再撞一次，还会误导排障方向。"""
    http = _FakeApiByModel(failing_models={"m-main"}, fail_status=401,
                           fail_body='{"error":{"code":"invalid_api_key"}}')
    svc = _service_with_models(http, llm=_FakeLlm("m-main", ("m-b", "m-c")))
    o = svc.search("出口退税政策调整", 3)
    assert not o.ok()
    assert http.models == ["m-main"], "401 不该换模型重试"
    assert "Key" in (o.error or "")


def test_api_model_rotation_is_capped():
    """轮换必须有上限：一次检索不该把整个候选表刷一遍。"""
    from app.web.search_service import API_MODEL_TRIES

    http = _FakeApiByModel(failing_models={"m-main", "m-b", "m-c", "m-d"})
    svc = _service_with_models(http, llm=_FakeLlm("m-main", ("m-b", "m-c", "m-d")))
    svc.search("出口退税政策调整", 3)
    assert len(http.models) <= API_MODEL_TRIES


def test_web_search_and_llm_are_always_wired_at_construction():
    """★ 装配契约：这两处漏传都是**静默**的，只有真实分析时才暴露。

    - ``RiskAgentTools(web_search=...)`` 漏传 → 模型调 web_search 时走进
      ``if self.web_search is None`` 直接返回「联网通道未装配」，请求根本没发出去。
      现象是"联网开着却没结果"，**换模型、开插件都毫无变化**（曾据此误判为额度问题）。
    - ``WebSearchService(llm=...)`` 漏传 → ``_api_configured()`` 恒 False，方舟通道被整条跳过，
      "Key 配好了却永远只在抓网页"。

    两者都值得用测试钉住：它们不报错、不抛异常、日志也很安静，但功能整体不存在。
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    missing: List[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            if name == "RiskAgentTools":
                if not any(k.arg == "web_search" for k in node.keywords):
                    missing.append(f"{path.name}:{node.lineno} RiskAgentTools 缺 web_search=")
            elif name == "WebSearchService":
                if not any(k.arg == "llm" for k in node.keywords):
                    missing.append(f"{path.name}:{node.lineno} WebSearchService 缺 llm=")
    assert not missing, "装配缺失（会让功能静默失效）：\n" + "\n".join(missing)
