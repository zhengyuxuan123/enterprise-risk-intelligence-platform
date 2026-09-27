"""免 Key 的联网检索：直接抓搜索引擎结果页并解析（对应 Java ``DirectWebSearchClient``）。

**不调用任何付费接口、不下发任何密钥。**

引擎与降级
----------
默认 ``bing → so360``。选 Bing 打头是因为它的结果页结构最规整
（``<li class="b_algo">`` 里标题/链接/摘要齐备）；360 作为国内备用；
DuckDuckGo 在国内不可达、搜狗有反爬拦截，都不在列表里。

.. WARNING::
   **绝不能「按引擎顺序凑够条数就收工」。** 实测中文业务词（「企业级软件 SLA 服务标准」）
   在 cn.bing.com 上命中 ``企业`` 这个 bigram，返回的全是「企业微信 / 爱企查 / 企查查 /
   天眼查 / 国家企业信用信息公示系统」这类门户站，一条有用的都没有；同一个词在 360 上
   返回的是「服务等级协议(SLA)标准体系解析与应用实践」这类正中靶心的结果。
   旧实现里 Bing 先抓满候选池就 ``break``，360 根本没机会跑，于是「外部来源」永远是垃圾。

现在的做法是**跨引擎合并候选池**：所有引擎并行抓，合并去重后用统一的相关度打分排序，
再取前 N。引擎顺序只影响「同分时的先后」，不再决定生死。

候选池按**三倍**抓：搜索引擎排序靠前的常常是「XX是什么意思」这类科普页，
抓满 n 条再过滤一遍就什么都不剩了。
"""

from __future__ import annotations

import base64
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

log = logging.getLogger(__name__)

#: 伪装成普通浏览器，否则部分引擎直接返回精简页或跳转页。
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

_BING_BLOCK = re.compile('<li class="b_algo"')
_BING_TITLE_A = re.compile(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>')
_BING_CAPTION = re.compile(r'<div class="b_caption"[^>]*>([\s\S]*?)</div>')

_SO_BLOCK = re.compile('<li class="res-list"')
_SO_TITLE_A = re.compile(
    r'<h3[^>]*class="[^"]*res-title[^"]*"[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>')
_SO_DESC = re.compile(r'<p class="res-desc"[^>]*>([\s\S]*?)</p>')

_ANY_P = re.compile(r"<p[^>]*>([\s\S]*?)</p>")
_CITE = re.compile(r"<cite[^>]*>([\s\S]*?)</cite>")
_TAG = re.compile(r"<[^>]+>")
_NUM_ENTITY = re.compile(r"&#(\d+);")
_BING_CK = re.compile(r"[?&]u=a1([^&]+)")

#: 「释义类」结果：只解释某个词是什么意思，本身不含任何业务信息。
#: 它们常年占据搜索引擎首页，递给模型只会拉低整体相关度。
_EXPLAIN_ONLY_TITLE = re.compile(
    "(是什么意思|什么意思|怎么读|如何读|怎么念|拼音|汉语词典|现代汉语词典"
    "|百度百科|百科词条|_百度|维基百科|翻译|近义词|反义词"
    "|周历|日历|黄历|万年历)"
)

_EXPLAIN_ONLY_HOSTS = (
    "baike.baidu.com", "baike.so.com", "baike.sogou", "dict.baidu", "dict.youdao",
    "dict.hjenglish", "iciba.com", "zdic.net", "hanyu.baidu", "wikipedia.org",
    "translate.google", "cidian.xpcha",
)


@dataclass(frozen=True)
class Hit:
    """一条直连检索结果：标题 / 可核对的 URL / 引擎给的摘要 / 来自哪个引擎。

    ``site`` 是结果页上"给用户看的那个域名"（搜索引擎自己的跳转链域名不算）。
    360 大量结果是 ``so.com/link?m=...``，光看 URL 根本不知道来源是谁，
    ``site`` 至少让「这条来自豆丁网 / CSDN」在界面上可读、可核对。
    """

    title: str
    url: str
    snippet: str
    engine: str
    site: str = ""


def _http():
    import httpx

    return httpx.Client(
        timeout=httpx.Timeout(10.0),
        follow_redirects=True,
        headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml",
                 "Accept-Language": "zh-CN,zh;q=0.9"},
    )


class DirectWebSearchClient:
    """跨引擎合并候选池，按相关度取前 topN。永不抛异常。"""

    def __init__(self, engines: Optional[List[str]] = None, timeout_seconds: Optional[int] = None,
                 resolve_redirects: Optional[bool] = None,
                 max_attempts: int = 2) -> None:
        from ..config import get_settings

        cfg = get_settings().web_search
        raw = engines if engines is not None else cfg.engine_list
        self.engines = [e.strip().lower() for e in raw if e and e.strip()] or ["bing", "so360"]
        self.timeout_seconds = int(
            cfg.direct_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        self.resolve_redirects = (bool(getattr(cfg, "resolve_redirects", True))
                                  if resolve_redirects is None else bool(resolve_redirects))
        self.max_attempts = max(1, min(int(max_attempts or 1), 3))
        self.last_engine: Optional[str] = None
        self.last_error: Optional[str] = None
        #: 每个引擎本次各贡献了多少条（排障时看「是不是只有 Bing 在说话」）
        self.last_engine_hits: Dict[str, int] = {}

    def get_engines(self) -> List[str]:
        return list(self.engines)

    def get_last_engine(self) -> Optional[str]:
        return self.last_engine

    def get_last_error(self) -> Optional[str]:
        return self.last_error

    # -- 检索 --------------------------------------------------------

    def search(self, query: Optional[str], top_n: int, client=None) -> List[Hit]:
        self.last_error = None
        self.last_engine_hits = {}
        if query is None or not query.strip():
            self.last_error = "缺少检索词"
            return []
        n = max(1, min(top_n if top_n > 0 else 5, 10))
        # 候选池刻意大于 n：引擎自己的排序不代表与本次问题相关，
        # 池子太小就没得挑，只能把排在最前面的门户站递给模型。
        pool_size = min(max(n * 3, n + 5), 18)

        per_engine, errors = self._fetch_all(query, pool_size, client)

        merged: List[Hit] = []
        seen_url: set = set()
        seen_title: set = set()
        for engine in self.engines:
            hits = per_engine.get(engine) or []
            self.last_engine_hits[engine] = len(hits)
            for h in hits:
                if _is_explain_only(h):
                    continue
                key = _norm_url_key(h.url)
                tkey = _norm_title_key(h.title)
                if key in seen_url or (tkey and tkey in seen_title):
                    continue
                seen_url.add(key)
                if tkey:
                    seen_title.add(tkey)
                merged.append(h)

        contributors = [e for e in self.engines if self.last_engine_hits.get(e)]
        self.last_engine = "+".join(contributors) if contributors else None

        ranked, scored = _rank(query, merged)
        out = ranked[:n]
        if out and self.resolve_redirects:
            out = _resolve_redirects(out, min(self.timeout_seconds, 6))
        if not out and errors:
            self.last_error = "；".join(str(e) for e in errors)
        raw_total = sum(len(v or []) for v in per_engine.values())
        log.info(
            "[WebSearch] 直连检索「%s」命中 %s 条（引擎 %s 各贡献 %s，"
            "原始 %s 条 → 合并去重并剔除释义/百科后 %s 条 → 按相关度取 %s 条）"
            "首条=%s 首条相关度=%s",
            _trim(query, 40), len(out), self.last_engine, self.last_engine_hits,
            raw_total, len(merged), n,
            "-" if not out else _trim(out[0].title, 40) + " " + out[0].url,
            "-" if not scored else round(scored[0], 3),
        )
        return out

    def _fetch_all(self, query: str, n: int, client) -> Tuple[Dict[str, Optional[List[Hit]]], List[str]]:
        """并行抓所有引擎。

        注入了 ``client`` 时（测试用假通道）退回串行——httpx.Client 不是线程安全的，
        而且假通道本来就是内存往返，并行没有意义。
        """
        errors: List[str] = []
        out: Dict[str, Optional[List[Hit]]] = {}
        if client is not None or len(self.engines) == 1:
            for engine in self.engines:
                hits, err = self._fetch_with_retry(engine, query, n, client)
                out[engine] = hits
                if err:
                    errors.append(f"{engine}: {err}")
            return out, errors

        # 每个线程自带一个 client：并行下共用连接会互相踩踏
        def one(engine: str) -> Tuple[str, Optional[List[Hit]], Optional[str]]:
            own = None
            try:
                own = _http()
                hits, err = self._fetch_with_retry(engine, query, n, own)
                return engine, hits, err
            finally:
                if own is not None:
                    try:
                        own.close()
                    except Exception:  # noqa: BLE001
                        pass

        with ThreadPoolExecutor(max_workers=min(len(self.engines), 4)) as ex:
            for engine, hits, err in ex.map(one, self.engines):
                out[engine] = hits
                if err:
                    errors.append(f"{engine}: {err}")
        return out, errors

    def _fetch_with_retry(self, engine: str, query: str, n: int,
                          client) -> Tuple[Optional[List[Hit]], Optional[str]]:
        """每个引擎给两次机会：抓结果页是公网请求，TLS 握手被重置这类抖动很常见。"""
        last: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                return self._fetch(engine, query, n, client or _http()), None
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt + 1 < self.max_attempts:
                    time.sleep(0.4)
                continue
        log.warning("直连检索引擎 %s 失败: %s", engine, last)
        return None, str(last)

    def _fetch(self, engine: str, query: str, n: int, client) -> List[Hit]:
        url = engine_url(engine, query)
        resp = client.get(url, timeout=self.timeout_seconds)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}")
        if engine == "bing":
            return parse_bing(resp.text, n)
        if engine == "so360":
            return parse_so360(resp.text, n)
        return []


def engine_url(engine: str, query: str) -> str:
    """拼搜索引擎地址。

    **中文检索词走国内站、纯英文走国际站**：同一串英文短语在 cn.bing.com 上会被当成
    「某个单词什么意思」来匹配，返回的全是词义辨析页——跟业务毫无关系。
    """
    zh = any(0x4E00 <= ord(c) <= 0x9FFF for c in query)
    q = quote(query, safe="")
    if engine == "bing":
        if zh:
            return f"https://cn.bing.com/search?q={q}&ensearch=0"
        return f"https://www.bing.com/search?q={q}&setlang=en&mkt=en-US"
    if engine == "so360":
        return f"https://www.so.com/s?q={q}"
    raise ValueError(f"未知引擎: {engine}")


def parse_bing(html: str, n: int) -> List[Hit]:
    out: List[Hit] = []
    blocks = _BING_BLOCK.split(html)
    for b in blocks[1:]:
        if len(out) >= n:
            break
        ma = _BING_TITLE_A.search(b)
        if not ma:
            continue
        url = normalize_url(ma.group(1), False)
        if url is None:
            continue
        title = clean(ma.group(2))
        mc = _BING_CAPTION.search(b)
        scope = mc.group(1) if mc else b
        mp = _ANY_P.search(scope)
        snip = clean(mp.group(1)) if mp else ""
        out.append(Hit(title or domain_of(url), url, snip, "bing", cite_of(b)))
    return out


def parse_so360(html: str, n: int) -> List[Hit]:
    out: List[Hit] = []
    blocks = _SO_BLOCK.split(html)
    for b in blocks[1:]:
        if len(out) >= n:
            break
        ma = _SO_TITLE_A.search(b)
        if not ma:
            continue
        # 360 的结果大量是 so.com/link?... 跳转链，真实 URL 取不出来。
        # 它是备用引擎，主引擎失败时「有一条能点开的链接」好过「一条都没有」。
        url = normalize_url(ma.group(1), True)
        if url is None:
            continue
        title = clean(ma.group(2))
        md = _SO_DESC.search(b)
        snip = clean(md.group(1)) if md else ""
        site = cite_of(b)
        if not title:
            title = site or domain_of(url)
        out.append(Hit(title, url, snip, "so360", site))
    return out


def normalize_url(raw: Optional[str], allow_redirect: bool) -> Optional[str]:
    """链接清洗：解码 HTML 实体 + 还原 Bing 的跳转包装。

    Bing 的部分结果是 ``/ck/a?...&u=a1<base64>``，直接给用户会跳一层看不出来源。
    解不开的站内跳转链接直接丢弃——「来源可核对」比「多一条结果」重要。
    """
    if raw is None or not raw.strip():
        return None
    u = decode_entities(raw).strip()
    if "bing.com/ck/a" in u:
        m = _BING_CK.search(u)
        if not m:
            return None
        try:
            u = base64.b64decode(m.group(1)).decode("utf-8")
        except Exception:  # noqa: BLE001
            return None
    if not u.startswith("http://") and not u.startswith("https://"):
        return None
    host = domain_of(u)
    search_site = (
        host in ("bing.com", "cn.bing.com", "www.bing.com", "so.com", "www.so.com")
        or host.endswith("360.cn")
        or (host.endswith("microsoft.com") and "translate" in u)
    )
    if search_site:
        if allow_redirect and "so.com/link" in u:
            return u
        return None
    return u


def cite_of(block: str) -> str:
    m = _CITE.search(block)
    if not m:
        return ""
    c = re.sub(r"^https?://", "", clean(m.group(1))).split()
    return re.sub(r"^www\.", "", c[0]) if c else ""


def clean(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", decode_entities(_TAG.sub("", s))).strip()


def decode_entities(s: Optional[str]) -> str:
    if s is None or "&" not in s:
        return s or ""
    t = (s.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
         .replace("&apos;", "'").replace("&lt;", "<").replace("&gt;", ">")
         .replace("&nbsp;", " ").replace("&ensp;", " ").replace("&emsp;", " ")
         .replace("&#0183;", "·").replace("&hellip;", "…"))
    return _NUM_ENTITY.sub(_num_repl, t)


def _num_repl(m) -> str:
    try:
        return chr(int(m.group(1)))
    except (ValueError, OverflowError):
        return ""


def domain_of(url: str) -> str:
    try:
        h = urlparse(url).hostname
        return re.sub(r"^www\.", "", h) if h else url
    except Exception:  # noqa: BLE001
        return url


def _is_explain_only(h: Hit) -> bool:
    t = h.title or ""
    u = (h.url or "").lower()
    if _EXPLAIN_ONLY_TITLE.search(t):
        return True
    return any(x in u for x in _EXPLAIN_ONLY_HOSTS)


#: 搜索引擎自己的跳转链域名——URL 停在这一层就等于"来源不可核对"。
_REDIRECT_HOSTS = ("so.com/link", "sogou.com/link", "baidu.com/link", "google.com/url")

_JS_REDIRECT = re.compile(r"""window\.location(?:\.replace|\.href)?\s*[=(]\s*["']([^"']+)["']""")
_META_REFRESH = re.compile(
    r"""http-equiv=["']refresh["'][^>]*?content=["'][^"']*?URL=["']?([^"'>]+)""", re.I)


def _is_redirect_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return any(x in url for x in _REDIRECT_HOSTS)


def _final_url_of(original: str, resp) -> Optional[str]:
    """从跳转页里抠出真实地址。

    360 的 ``so.com/link?m=...`` **不做 HTTP 302**——它返回一个 380 字节的壳页，
    靠 ``window.location.replace(...)`` 和 ``<meta refresh>`` 把浏览器送走。
    所以只跟重定向是跟不出来的，必须读壳页。
    """
    try:
        final = str(resp.url)
    except Exception:  # noqa: BLE001
        final = original
    if final != original and not _is_redirect_url(final):
        # 判断"是个死链"比"是不是跳转链"更重要：跳到 /error/404 的话宁可保留原链
        if resp.status_code < 400 and "/error/" not in final:
            return final
        return None
    body = ""
    try:
        body = resp.text or ""
    except Exception:  # noqa: BLE001
        return None
    for pat in (_JS_REDIRECT, _META_REFRESH):
        m = pat.search(body)
        if m:
            u = decode_entities(m.group(1)).strip().strip("'\"")
            if u.startswith("http://") or u.startswith("https://"):
                if "/error/" in u:
                    return None
                return u
    return None


def _resolve_redirects(hits: List[Hit], timeout_seconds: int) -> List[Hit]:
    """把跳转链还原成真实地址（并行、短超时）。

    **为什么要还原**：审计时「这条结论依据的是哪篇材料」必须一眼可核对。
    ``https://www.so.com/link?m=e9fC...`` 这种链接点进去是引擎的中转页，
    既看不出来源站，也可能随时失效——它不满足"可追溯"的最低要求。

    还原失败的保留原链（有一条能点开的链接好过一条都没有），
    只是界面上仍然看不出真实来源，这一点如实反映在 ``site`` 为空上。
    """
    todo = [h for h in hits if _is_redirect_url(h.url)]
    if not todo:
        return hits

    def one(h: Hit) -> Tuple[str, Optional[str]]:
        try:
            import httpx

            with httpx.Client(timeout=timeout_seconds, follow_redirects=True,
                              headers={"User-Agent": UA, "Accept": "text/html,*/*",
                                       "Accept-Language": "zh-CN,zh;q=0.9"}) as cl:
                r = cl.get(h.url)
            return h.url, _final_url_of(h.url, r)
        except Exception as e:  # noqa: BLE001
            log.debug("跳转链还原失败 %s: %s", _trim(h.url, 60), e)
            return h.url, None

    resolved: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(len(todo), 6)) as ex:
        for src, dst in ex.map(one, todo):
            if dst:
                resolved[src] = dst

    if not resolved:
        return hits
    log.info("[WebSearch] 跳转链还原 %s/%s 条", len(resolved), len(todo))

    out: List[Hit] = []
    seen: set = set()
    for h in hits:
        u = resolved.get(h.url) or h.url
        key = _norm_url_key(u)
        if key in seen:
            continue
        seen.add(key)
        out.append(Hit(h.title, u, h.snippet, h.engine, h.site or domain_of(u)))
    return out


def _norm_url_key(url: Optional[str]) -> str:
    """去重用的 URL 归一：去协议、去 www、去末尾斜杠与片段。"""
    if not url:
        return ""
    u = (url or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.split("#", 1)[0].rstrip("/")
    return u


def _norm_title_key(title: Optional[str]) -> str:
    """去重用的标题归一：同一篇文章常被两个引擎同时收录，URL 不同但标题几乎一样。"""
    if not title:
        return ""
    t = re.sub(r"[\s\-,_|｜·、：:。.！!？?（）()【】\[\]「」\"'’“”]+", "", title.lower())
    return t[:40]


_RELEVANCE = None


def _relevance():
    """候选池排序用的相关度打分器（懒加载，避免 web → rag 的导入时序问题）。

    与 ``app.rag.relevance.SourceRelevanceFilter`` 用的是**同一套判据**：
    词面覆盖 + 标题覆盖 + 本地余弦，零 token、确定性。
    两处共用一份实现，免得「直连层觉得相关、回检层觉得不相关」。
    """
    global _RELEVANCE
    if _RELEVANCE is None:
        from ..rag.embedding import LocalEmbedding
        from ..rag.relevance import SourceRelevanceFilter

        _RELEVANCE = SourceRelevanceFilter(local=LocalEmbedding())
    return _RELEVANCE


def _rank(query: str, hits: List[Hit]) -> Tuple[List[Hit], List[float]]:
    """按与检索词的相关度降序重排候选池。

    **为什么不能照引擎给的顺序取前 N**：引擎的排序服务于「最可能满足搜索者」，
    不服务于「与本次业务问题相关」。中文业务词在 Bing 上常被 ``企业`` 这类通用
    bigram 带偏到门户站，那些站排在最前面却一条有用信息都没有；360 里真正对题的
    结果因为排在引擎列表第二位而被整体丢弃——这正是「外部来源永远没用」的根因。
    """
    if not hits:
        return [], []
    try:
        from ..rag.relevance import WebSource

        flt = _relevance()
        srcs = [WebSource(i + 1, h.title, h.url, domain_of(h.url), h.snippet)
                for i, h in enumerate(hits)]
        res = flt.filter(None, query, srcs)
        pos = {s.url: i for i, s in enumerate(srcs)}
        scored = sorted(res.detail, key=lambda d: (-d.score, pos.get(d.source.url, 0)))
        ranked = [hits[pos[d.source.url]] for d in scored if d.source.url in pos]
        return ranked, [float(d.score) for d in scored]
    except Exception as e:  # noqa: BLE001
        # 打分器挂了不能让整条联网链路失效：退回引擎顺序，总好过一条都没有
        log.warning("候选池相关度排序失败，退回引擎原始顺序: %s", e)
        return list(hits), []


def _trim(s: Optional[str], max_len: int) -> str:
    if s is None:
        return ""
    return s if len(s) <= max_len else s[:max_len] + "…"
