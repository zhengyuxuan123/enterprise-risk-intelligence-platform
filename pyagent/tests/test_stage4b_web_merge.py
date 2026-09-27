"""阶段 4b 回归：跨引擎候选池合并 + 跳转链还原。

对应线上故障（2026-09-22 修）：「联网搜索的外部结果还是没有」。

根因不是搜索引擎没结果，而是**候选池被第一个引擎占满就收工**——
中文业务词在 cn.bing.com 上命中 ``企业`` 这个 bigram，返回的全是门户站；
360 里真正对题的结果因为排在引擎列表第二位，压根没被请求过。

全部**不联网**：引擎结果页用固定 HTML，跳转页用假响应对象。
"""

from __future__ import annotations

import pytest

from app.web.direct_client import (
    DirectWebSearchClient,
    _final_url_of,
    _is_redirect_url,
)

QUERY = "企业级软件 SLA 服务标准 行业做法"

#: Bing 对这类中文业务词的真实表现：十条全是工商查询 / 协同办公门户站。
_BING_PORTAL = "".join(
    '<li class="b_algo"><h2><a href="https://p{0}.example.com/">{1}</a></h2>'
    '<div class="b_caption"><p>提供工商查询、企业信息查询、法人查询等服务，'
    '可查询企业工商信息、股东信息与经营异常名录。</p></div></li>'.format(i, t)
    for i, t in enumerate(["企业微信", "爱企查", "企查查", "天眼查", "风鸟企业查询",
                           "企名录", "国家企业信用信息公示系统", "企业查询网",
                           "企业黄页", "企业服务市场"])
)

#: 360 对同一个词的表现：正中靶心。
_SO_GOOD = (
    '<li class="res-list"><h3 class="res-title"><a href="https://www.so.com/link?m=AAA">'
    '服务等级协议(SLA)标准体系解析与应用实践</a></h3>'
    '<p class="res-desc">面向企业级软件的服务等级协议 SLA 标准体系，'
    '给出可用性、响应时间、故障恢复的行业通行做法与指标口径。</p>'
    '<cite>doc.example.com</cite></li>'
)


class _Resp:
    def __init__(self, text="", status=200, payload=None):
        self.text = text
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload if self._payload is not None else {}


class _FakeHttp:
    """按预设队列应答；记录被请求的 URL 用于断言「第二个引擎到底有没有被问到」。"""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.urls: list[str] = []

    def get(self, url, **kw):
        self.urls.append(url)
        return self.responses.pop(0) if self.responses else _Resp("")

    def close(self):
        pass


class TestCrossEngineCandidateMerge:
    """引擎顺序只决定同分时的先后，绝不决定谁能进结果。"""

    def test_second_engine_is_queried_even_when_first_fills_the_pool(self):
        """Bing 单独就抓满了候选池，360 **必须**照样被请求——这是本次修复的核心。"""
        c = DirectWebSearchClient(engines=["bing", "so360"], timeout_seconds=1,
                                  resolve_redirects=False)
        http = _FakeHttp([_Resp(_BING_PORTAL), _Resp(_SO_GOOD)])
        c.search(QUERY, 3, client=http)
        assert len(http.urls) == 2
        assert "cn.bing.com" in http.urls[0]
        assert "so.com" in http.urls[1]

    def test_relevant_result_from_second_engine_beats_portal_from_first(self):
        c = DirectWebSearchClient(engines=["bing", "so360"], timeout_seconds=1,
                                  resolve_redirects=False)
        http = _FakeHttp([_Resp(_BING_PORTAL), _Resp(_SO_GOOD)])
        hits = c.search(QUERY, 3, client=http)
        assert hits, "两个引擎都有结果，不该空手而归"
        assert "服务等级协议" in hits[0].title
        assert "SLA" in hits[0].title

    def test_engine_contributions_are_recorded_for_diagnostics(self):
        """排障时第一件事就是看「是不是只有 Bing 在说话」。"""
        c = DirectWebSearchClient(engines=["bing", "so360"], timeout_seconds=1,
                                  resolve_redirects=False)
        c.search(QUERY, 5, client=_FakeHttp([_Resp(_BING_PORTAL), _Resp(_SO_GOOD)]))
        assert c.last_engine_hits.get("bing") == 10
        assert c.last_engine_hits.get("so360") == 1
        assert c.get_last_engine() == "bing+so360"

    def test_same_article_from_two_engines_keeps_only_one(self):
        """同一篇文章常被两个引擎同时收录：URL 不同但标题几乎一样。"""
        bing = ('<li class="b_algo"><h2><a href="https://a.example.com/1">'
                '服务等级协议(SLA)标准体系解析与应用实践</a></h2>'
                '<div class="b_caption"><p>面向企业级软件的 SLA 标准体系。</p></div></li>')
        so = ('<li class="res-list"><h3 class="res-title"><a href="https://b.example.com/1">'
              '服务等级协议(SLA)标准体系解析与应用实践</a></h3>'
              '<p class="res-desc">面向企业级软件的 SLA 标准体系。</p></li>')
        c = DirectWebSearchClient(engines=["bing", "so360"], timeout_seconds=1,
                                  resolve_redirects=False)
        hits = c.search(QUERY, 5, client=_FakeHttp([_Resp(bing), _Resp(so)]))
        assert len(hits) == 1

    def test_ranking_applies_within_a_single_engine_too(self):
        """门户站与对题结果同在一个引擎里时，也按相关度排而不是照引擎原文顺序。

        对题的那条故意放在**最后**（第 11 位）：它之所以还能被选中，
        全靠候选池够大 + 按相关度重排，两者缺一它就会被门户站挤掉。
        """
        html = _BING_PORTAL + (
            '<li class="b_algo"><h2><a href="https://good.example.com/sla">'
            '服务等级协议(SLA)标准体系解析与应用实践</a></h2>'
            '<div class="b_caption"><p>面向企业级软件的 SLA 标准体系，'
            '可用性、响应时间、故障恢复的行业通行做法。</p></div></li>')
        c = DirectWebSearchClient(engines=["bing"], timeout_seconds=1,
                                  resolve_redirects=False)
        hits = c.search(QUERY, 6, client=_FakeHttp([_Resp(html)]))
        assert hits and "服务等级协议" in hits[0].title

    def test_one_engine_failing_does_not_kill_the_other(self):
        class _Flaky(_FakeHttp):
            def get(self, url, **kw):
                self.urls.append(url)
                if "bing" in url:
                    raise RuntimeError("Remote host terminated the handshake")
                return self.responses.pop(0) if self.responses else _Resp("")

        c = DirectWebSearchClient(engines=["bing", "so360"], timeout_seconds=1,
                                  resolve_redirects=False)
        hits = c.search(QUERY, 3, client=_Flaky([_Resp(_SO_GOOD)]))
        assert hits and "服务等级协议" in hits[0].title
        assert c.get_last_engine() == "so360"


# ------------------------------------------------------------------
# 跳转链还原（来源可核对 = 可追溯的最低要求）
# ------------------------------------------------------------------


class _FakeResp:
    def __init__(self, url="", status=200, text=""):
        self.url = url
        self.status_code = status
        self.text = text


JUMP = "https://www.so.com/link?m=e9fC"


class TestRedirectResolution:

    def test_jump_link_is_detected(self):
        assert _is_redirect_url(JUMP) is True
        assert _is_redirect_url("https://www.baidu.com/link?url=x") is True
        assert _is_redirect_url("https://www.docin.com/p-1.html") is False
        assert _is_redirect_url(None) is False

    def test_js_location_replace_is_extracted(self):
        """360 的跳转页不做 302：它返回一个 380 字节的壳页，靠 JS 把浏览器送走。"""
        body = ('<meta content="always" name="referrer">'
                '<script>window.location.replace("https://doc.example.com/p-1.html")</script>')
        assert _final_url_of(JUMP, _FakeResp(JUMP, 200, body)) == "https://doc.example.com/p-1.html"

    def test_meta_refresh_is_extracted_as_fallback(self):
        body = ('<noscript><meta http-equiv="refresh" '
                "content=\"0;URL='https://doc.example.com/p-2.html'\"></noscript>")
        assert _final_url_of(JUMP, _FakeResp(JUMP, 200, body)) == "https://doc.example.com/p-2.html"

    def test_dead_link_is_not_adopted(self):
        """跳到错误页的真实地址比原来的跳转链更糟——会让人以为这条来源点不开。"""
        body = '<script>window.location.replace("https://wenku.so.com/error/404?page=x")</script>'
        assert _final_url_of(JUMP, _FakeResp(JUMP, 200, body)) is None

    def test_plain_http_redirect_is_adopted(self):
        r = _FakeResp("https://doc.example.com/p-3.html", 200, "")
        assert _final_url_of(JUMP, r) == "https://doc.example.com/p-3.html"

    def test_unresolvable_body_keeps_original(self):
        assert _final_url_of(JUMP, _FakeResp(JUMP, 200, "<html></html>")) is None

    def test_switch_off_keeps_the_jump_link(self):
        """关掉开关就省掉这一跳 HTTP；代价是来源保持跳转链形态。

        用 360 的页面结构：Bing 的解析器本来就丢弃 ``so.com/link``（allow_redirect=False），
        只有 360 那条路径才会把跳转链留在结果里。
        """
        c = DirectWebSearchClient(engines=["so360"], timeout_seconds=1,
                                  resolve_redirects=False)
        html = ('<li class="res-list"><h3 class="res-title"><a href="' + JUMP + '">'
                '服务等级协议(SLA)标准体系解析与应用实践</a></h3>'
                '<p class="res-desc">面向企业级软件的 SLA 标准体系。</p></li>')
        hits = c.search(QUERY, 3, client=_FakeHttp([_Resp(html)]))
        assert hits and hits[0].url == JUMP


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
