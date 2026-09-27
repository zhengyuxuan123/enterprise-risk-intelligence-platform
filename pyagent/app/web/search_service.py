"""联网检索服务（对应 Java ``WebSearchService``）：多通道编排 + 来源结构化。

``mode`` 取值：``auto``（默认，见下方通道顺序）、
``mcp``（只用 MCP）、``direct``（只用直连，绝不下发密钥）、``api``（只用方舟 Responses）。

.. NOTE::
   **通道顺序在 2026-09-22 改过：方舟 Responses 提到了直连之前。**

   原因不是喜好问题，是本机实测的结论：所有"抓取搜索引擎结果页"的免 Key 通道都已
   不可用或不返回值——cn.bing / www.bing 对无 Cookie 的自动化请求返回**与查询无关的
   随机网页**（同一查询连问三次，分别得到日历网、芒果TV、美国房产），360 跳到验证码页，
   百度/搜狗 SSL 不可达，Brave 429，Ecosia 403。它们的危害不是"慢"，而是会带着一堆
   **看起来像结果**的垃圾走进最后的 "| 得过相关度回检" 那一层，于是内部循环永远不再尝试
   其它通道——用户看到的就只是「本次未联网核实」，而后端以为自己成功了一件事也没干。

   两道防线因此是必须的：
   1) 顺序上把**有契约的** API 放在前面（付费但可用 > 免费但随机）；
   2) 任何自称成功的免费通道都要过 :func:`_quality_pass` —— 同一把本地相关度的尺子，
      通不过就当作"这条通道没给到东西"，继续往下走。

**永不抛异常**：失败收敛成 :attr:`SearchOutcome.error`，让上层把「本次没联网」如实写进结论。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse

from ..ai.fallback import is_model_unavailable
from ..config import get_settings
from .direct_client import DirectWebSearchClient, Hit, domain_of
from .mcp_client import McpWebSearchClient

#: 联网检索最多试几个模型。**按模型**的上限是常态（见 _api_search 注释），
#: 但也不能为了一次检索把候选全刷一遍 —— 3 个足够绕开常见限流。
API_MODEL_TRIES = 3

log = logging.getLogger(__name__)

#: 只做检索与整理，不让它代替主分析师下结论（仅 API 通道使用）。
SYSTEM = (
    "你是联网检索助手。请基于检索到的真实网页内容回答，"
    "在陈述事实处用 [1][2] 形式标注来源序号。"
    "禁止编造未在检索结果中出现的链接或数据；查不到就明确说查不到。"
)

#: 方舟「联网内容插件」开通页（免费）。
ACTIVATE_URL = "https://console.volcengine.com/common-buy/CC_content_plugin"

#: 正文里抠 URL 的兜底正则（annotations 缺失时用）。
URL_PATTERN = re.compile(r"https?://[^\s()（）\"'<>【】\[\]]+")


@dataclass
class WebSource:
    """一条网页来源。``index`` 与正文里的 ``[n]`` 引用标记对应。

    ``snippet`` 是搜索引擎给的摘要——**相关性回检要靠它**，光有标题判断不了内容。
    """

    index: int
    title: str
    url: str
    site_name: str
    snippet: str = ""


@dataclass
class SearchOutcome:
    """检索结果：``summary`` 是带 [n] 标注的摘要，``sources`` 是结构化来源。"""

    summary: Optional[str] = None
    sources: List[WebSource] = field(default_factory=list)
    error: Optional[str] = None

    def ok(self) -> bool:
        return self.error is None and bool(self.sources)


class WebSearchService:
    """三通道编排。构造参数可注入，测试里能换成假通道。"""

    def __init__(self, direct: Optional[DirectWebSearchClient] = None,
                 mcp: Optional[McpWebSearchClient] = None, llm=None, http=None) -> None:
        s = get_settings()
        w = s.web_search
        self.enabled = bool(w.enabled)
        self.mode = _normalize_mode(w.mode)
        self.base_url = (w.base_url or "").strip().rstrip("/")
        self._model = (w.model or "").strip()
        self.api_key = (w.api_key or "").strip().strip('"').strip("'")
        self.max_keyword = max(1, min(w.max_keyword if w.max_keyword > 0 else 3, 50))
        self.sources_cfg = _parse_sources(w.sources)
        self.default_top_n = int(w.max_results or 6)
        self.timeout_seconds = int(w.timeout_seconds or 60)
        self.direct = direct if direct is not None else DirectWebSearchClient()
        self.mcp = mcp if mcp is not None else McpWebSearchClient()
        self.llm = llm
        self._http = http
        self.last_failure: Optional[str] = None
        self.last_channel: Optional[str] = None
        #: 本次检索中每条被否掉的通道及其理由。**成功时也要留**——
        #: "直连其实返回了一堆无关网页"这件事在排障时比"这次成功了"更有价值，
        #: 而 ``last_failure`` 只在彻底失败时才设，看不出这种"走过但没用上"。
        self.last_notes: List[str] = []
        #: 本次检索**逐通道**的尝试记录（含成功时被跳过的通道与其原因）。
        #: 与 last_notes 的区别：notes 只记"质量门否掉的结果"，tried 记"每个通道的结局"。
        self.last_tried: List[str] = []

        # 环境变量兜底用的 Key；显式配置了独立 Key 时不使用它
        import os

        self._fallback_key = _first_non_blank(os.environ.get("AI_API_KEY"),
                                              os.environ.get("OPENAI_API_KEY"))

    # -- 状态 --------------------------------------------------------

    def get_mode(self) -> str:
        return self.mode

    def get_last_channel(self) -> Optional[str]:
        return self.last_channel

    def get_last_failure(self) -> Optional[str]:
        return self.last_failure

    def get_model(self) -> str:
        return self._model or (str(self.llm.get_model()) if self.llm is not None else "")

    def get_endpoint(self) -> str:
        return self._url()

    def get_key_source(self) -> str:
        """Key 的实际来源，体检页用来说明「到底用的哪把」。"""
        if self.mode == "direct":
            return "无需 Key（直连抓取搜索引擎结果页）"
        if self.mode == "mcp":
            return "由 MCP Server 自带凭据（本平台不下发任何 Key）"
        if self.api_key:
            return "APP_WEB_SEARCH_API_KEY（独立，仅 API 通道使用）"
        if self._fallback_key:
            return "AI_API_KEY / OPENAI_API_KEY（环境变量，仅 API 通道使用）"
        if self.llm is not None and getattr(self.llm, "is_configured", lambda: False)():
            return "主推理客户端（仅 API 通道使用）"
        return "未配置（默认走免 Key 直连）"

    def mcp_info(self) -> Dict[str, object]:
        """MCP 通道的自检信息。

        给的是**连接态**而不是探测结果——探活会启动子进程、消耗一次工具往返，
        不该为了显示一个状态就去做。
        """
        if self.mcp is None:
            return {"assembled": False, "reason": "MCP 通道未装配"}
        m: Dict[str, object] = {
            "assembled": True,
            "enabled": self.mcp.is_enabled(),
            "transport": self.mcp.get_transport(),
            "status": self.mcp.status_reason() or "可用",
        }
        if self.mcp.get_last_tool():
            m["lastTool"] = self.mcp.get_last_tool()
        if self.mcp.get_last_error():
            m["lastError"] = self.mcp.get_last_error()
        return m

    def status_reason(self) -> Optional[str]:
        """不可用原因；``None`` 表示可用。

        只做**配置层面**的判断。「插件没开通」「Key 无权调用」这类只有真发一次请求才知道，
        放在 :attr:`last_failure` 里，避免体检接口为了探活而额外消耗一次调用。
        """
        if not self.enabled:
            return "已关闭（app.web-search.enabled=false）"
        if self.mode == "api":
            if not self._key():
                return "mode=api 但未配置 API Key（设置 APP_WEB_SEARCH_API_KEY，或改回 mode=auto）"
            if not self.get_model():
                return "未配置检索模型（app.web-search.model，或确保主底座模型可用）"
            if not self._url():
                return "无法确定检索端点（未配置 app.web-search.base-url，且主底座 base-url 为空）"
            return None
        if self.mode == "mcp":
            if self.mcp is None:
                return "mode=mcp 但 MCP 通道未装配"
            return self.mcp.status_reason()
        # auto：MCP 与直连至少有一条能跑，就不算不可用
        mcp_ok = self.mcp is not None and self.mcp.is_enabled()
        direct_ok = self.direct is not None
        if not mcp_ok and not direct_ok:
            return "MCP 与免 Key 直连两条通道都不可用"
        return None

    def is_enabled(self) -> bool:
        return self.status_reason() is None

    # -- 检索 --------------------------------------------------------

    def probe(self, query: str = "", live: bool = False, top_n: int = 3) -> Dict[str, object]:
        """体检。**默认零成本**——只汇报配置层判定与上一次失败原因，不发任何请求。

        ``live=True`` 才真的发一次检索。这条必须显式传入：曾经"顺手探活"的代价是
        每次打开体检页都在烧额度，而这个页面通常只是想知道「联网到底能不能用」。
        """
        out: Dict[str, object] = {
            "enabled": self.is_enabled(),
            "mode": self.get_mode(),
            "reason": self.status_reason(),
            "keySource": self.get_key_source(),
            "endpoint": self.get_endpoint() or "",
            "model": self.get_model(),
            "order": self._channels(),
            "lastChannel": self.get_last_channel(),
            "lastFailure": self.get_last_failure(),
            "lastNotes": list(self.last_notes),
            "mcp": self.mcp_info(),
            "live": False,
        }
        out["channels"] = [{
            "name": c,
            "note": (self._api_unconfigured_reason() if (c == "api" and not self._api_configured())
                     else ("可用" if c != "mcp" or self.mcp.is_enabled() else "未装配")),
        } for c in self._channels()]
        # 「可用」不等于「好用」：只有免 Key 抓取能跑时，界面上必须点明这一点，
        # 否则运维会把"联网已开启"当成"联网结果可信"。
        hints: List[str] = []
        if self.mode in ("auto", "") and "api" in self._channels() and not self._api_configured():
            hints.append("方舟 Responses 通道未装配（" + self._api_unconfigured_reason()
                         + "），当前实际只能用免 Key 直连；"
                           "直连在本机实测已不可靠（见模块头部），结果会被相关度质量门大量丢弃。"
                           "要让联网真正可用，请配置 APP_WEB_SEARCH_API_KEY 或确保主底座可用。")
        out["hints"] = hints
        if self.direct is not None:
            #! 通道对象可能是测试注入的简易替身，缺哪个属性都不该让自检挂掉——
            #! 体检接口自己崩了，比"某条通道不好用"更糟：用户连原因都看不到。
            out["direct"] = {
                "engines": _safe(lambda: self.direct.get_engines(), []),
                "lastError": _safe(lambda: self.direct.get_last_error(), None),
                "lastHits": _safe(lambda: getattr(self.direct, "last_engine_hits", {}), {}),
            }
        if not live:
            return out
        t0 = time.time()
        q = (query or "").strip() or "出口退税政策调整"
        o = self.search(q, top_n)
        out["live"] = True
        out["query"] = q
        out["elapsedMs"] = int((time.time() - t0) * 1000)
        out["channel"] = self.get_last_channel()
        out["ok"] = o.ok()
        out["error"] = o.error
        #! 最终用的通道只是结局，排查要看的是"被跳过的通道为什么没被采纳"
        #! —— 尤其 api 被 401/429/插件未开通挡掉、悄悄回退到直连这种情况。
        out["tried"] = list(self.last_tried)
        #! 上面那批字段是**检索前**的快照，必须回填成本次的：
        #! "自动切换到 glm 才联网成功"这种信息只在检索过程中产生，
        #! 不回填的话体检页永远显示上一次的残留，排障时看的是假象。
        out["lastNotes"] = list(self.last_notes)
        out["lastFailure"] = self.get_last_failure()
        out["lastChannel"] = self.get_last_channel()
        out["mcp"] = self.mcp_info()
        out["sources"] = [{"index": s.index, "title": s.title, "url": s.url,
                           "site": s.site_name} for s in o.sources]
        return out

    def search(self, query: Optional[str], top_n: int = 0) -> SearchOutcome:
        """按当前 mode 的通道顺序尝试，谁先给出**真正相关**的来源就用谁。

        每一步失败都留下原因，最后汇总成一句人能看懂的话；
        这样界面上的「本次未联网核实」不再是黑箱，而是「为什么没联网」。
        """
        if not self.is_enabled():
            return self._fail(f"联网检索不可用：{self.status_reason()}")
        if query is None or not query.strip():
            return self._fail("缺少检索词")
        n = max(1, min(top_n if top_n > 0 else self.default_top_n, 10))
        #! tried 与 last_tried 必须是同一个列表对象：成功路径在中途 return，
        #! 若只在失败时才回填，"这次明明走了 api 却被跳过"这件事就查不到了
        #! —— 排障时恰恰最需要知道"它试过什么、各自为什么没被采纳"。
        self.last_tried = []
        tried: List[str] = self.last_tried
        self.last_notes = []

        for ch in self._channels():
            if ch == "mcp":
                o = self._mcp_search(query, n)
                if o is not None and self._accept(ch, query, o, tried):
                    return o
                if o is None:
                    why = "MCP：" + (self.mcp.get_last_error() if self.mcp else "通道不可用")
                    tried.append(why)
                    self.last_notes.append(why)
                continue
            if ch == "api":
                if not self._api_configured():
                    tried.append(self._api_unconfigured_reason())
                    continue
                o = self._api_search(query, n)
                if o.ok() and self._accept(ch, query, o, tried):
                    return o
                why = f"方舟 API：{o.error or '未返回可引用来源'}"
                tried.append(why)
                self.last_notes.append(why)
                continue
            # direct
            o = self._direct_search(query, n)
            if o is not None and self._accept(ch, query, o, tried):
                return o
            err = self.direct.get_last_error() if self.direct else ""
            tried.append(f"直连：{err or '未取到结果'}")

        msg = "；".join(tried) or "没有任何可用通道"
        #! 单通道模式下必须写明「为什么不去试别的」：用户看到"联网失败"时最想知道的
        #! 恰恰是"那它试过别的通道了吗"——这句边界说明留着，别为了简洁删掉。
        if self.mode == "direct":
            msg += "（mode=direct，不会使用任何 API Key）"
        elif self.mode == "mcp":
            msg += "（mode=mcp，不会回退到直连或 API）"
        elif self.mode == "api":
            msg += "（mode=api，只走方舟 Responses API）"
        return self._fail(msg)

    def _channels(self) -> List[str]:
        """当前模式下要依次尝试的通道。

        auto 的顺序是 **mcp → api → direct**：先看有没有自带凭据的 MCP，
        否则走有契约的方舟 Responses，最后才碰免 Key 抓取。
        （旧顺序把直连排在 API 前面，代价见模块头部注释。）
        """
        if self.mode == "mcp":
            return ["mcp"]
        if self.mode == "direct":
            return ["direct"]
        if self.mode == "api":
            return ["api"]
        out: List[str] = []
        if self.mcp is not None and self.mcp.is_enabled():
            out.append("mcp")
        out.append("api")
        if self.direct is not None:
            out.append("direct")
        return out

    def _accept(self, channel: str, query: str, o: "SearchOutcome", tried: List[str]) -> bool:
        """免费/抓取型通道必须过本地相关度这一关才算"拿到了东西"。

        这是整套改动里最关键的一条：搜索引擎返回 200 + 十条结果 ≠ 检索成功。
        实测 cn.bing 对无关 query 也会老实返回十条（日历网、地图、随机门户），
        不经判定就返回的话，上层会以为"联网成功了"，然后引用来历不明的网页。
        """
        if channel == "api":
            return True  # Responses API 自己做过一轮检索与筛选，且带 annotations
        if not o.sources:
            return False
        if _quality_pass(query, o.sources):
            return True
        # 不是"失败"，而是"这条通道给的东西不能用" —— 记下来继续往下试
        for s in o.sources[:3]:
            log.warning("[WebSearch] 通道 %s 的结果未通过相关度质量门，已弃用：%s",
                        channel, _trim(s.title, 40))
        why = f"{channel}：取到 {len(o.sources)} 条但全部与本次问题不相关（已弃用）"
        tried.append(why)
        self.last_notes.append(why)
        return False

    def _api_unconfigured_reason(self) -> str:
        if not self._key():
            return "方舟 API 未配置 Key（设 APP_WEB_SEARCH_API_KEY，或由 AI_API_KEY 兜底）"
        if not self.get_model():
            return "方舟 API 未配置检索模型（app.web-search.model，或确保主底座模型可用）"
        return "方舟 API 未配置端点（app.web-search.base-url，且主底座 base-url 为空）"

    def _mcp_search(self, query: str, n: int) -> Optional[SearchOutcome]:
        if self.mcp is None or not self.mcp.is_enabled():
            return None
        try:
            hits = self.mcp.search(query, n)
        except Exception as e:  # noqa: BLE001
            log.warning("MCP 检索异常: %s", e)
            return None
        if not hits:
            return None
        self.last_failure = None
        self.last_channel = "mcp:" + (self.mcp.get_last_tool() or "-")
        return self._render(hits, self.last_channel)

    def _direct_search(self, query: str, n: int) -> Optional[SearchOutcome]:
        if self.direct is None:
            return None
        try:
            hits = self.direct.search(query, n)
        except Exception as e:  # noqa: BLE001
            log.warning("直连检索异常: %s", e)
            return None
        if not hits:
            return None
        self.last_failure = None
        self.last_channel = "direct:" + (self.direct.get_last_engine() or "-")
        return self._render(hits, self.last_channel)

    @staticmethod
    def _render(hits: List[Hit], channel: str) -> SearchOutcome:
        src: List[WebSource] = []
        parts: List[str] = []
        for h in hits:
            idx = len(src) + 1
            src.append(WebSource(idx, h.title, h.url, domain_of(h.url), h.snippet))
            parts.append(f"[{idx}] {h.title}\n")
            if h.snippet:
                parts.append(h.snippet + "\n")
            parts.append(f"来源：{h.url}\n\n")
        log.info("[WebSearch] 通道 %s 返回 %s 条可引用来源", channel, len(src))
        return SearchOutcome("".join(parts).strip(), src, None)

    # -- 方舟 Responses API ------------------------------------------

    def _api_models(self) -> List[str]:
        """联网检索依次可试的模型：专用检索模型 → 主模型 → 候选模型。

        实测（2026-09-22）：方舟的「安心体验模式」上限是 **按模型** 计的——
        同一把 Key 打 deepseek-v4-1-flash / deepseek-v4-flash-ga 都 429 SetLimitExceeded，
        换 doubao-seed-2-0-pro / glm-5-3-flash 立刻 200 且 web_search 正常执行。
        所以撞到模型级错误就该换下一个，而不是让整条 api 通道报废、
        悄悄回退到免 Key 直连去抓一堆随机网页。
        """
        out: List[str] = []

        def add(v: Optional[str]) -> None:
            t = str(v or "").strip()
            if t and t not in out:
                out.append(t)

        add(self.get_model())
        if self.llm is not None:
            #! 主模型链路切换过的模型通常就是「当前真能用的那个」，排在候选前面
            add(_safe(lambda: str(self.llm.get_model() or ""), ""))
            for c in _safe(lambda: list(self.llm.get_model_candidates()), []) or []:
                add(c)
        return out[:API_MODEL_TRIES]

    def _api_search(self, query: str, n: int) -> SearchOutcome:
        if not self._api_configured():
            if not self._key():
                return self._fail("未配置 API Key，无法使用方舟联网检索（或改用 mode=auto 走免 Key 直连）")
            if not self.get_model():
                return self._fail("未配置检索模型（app.web-search.model）")
            return self._fail("无法确定检索端点（app.web-search.base-url 为空且主底座 base-url 为空）")

        models = self._api_models()
        first = models[0] if models else ""
        last: Optional[SearchOutcome] = None
        attempted: List[str] = []
        # One search gets one wall-clock budget. Previously each candidate model
        # received the full timeout, turning a 15/60 second setting into 45/180s.
        deadline = time.monotonic() + max(0.1, float(self.timeout_seconds))
        for idx, model in enumerate(models):
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                break
            attempted.append(model)
            o, retryable = self._api_try(query, n, model, timeout_seconds=remaining)
            if o.ok():
                if idx > 0:
                    self.last_notes.append(
                        f"方舟 API：模型 {first} 当前不可用，已自动切换到 {model} 完成联网检索")
                self.last_failure = None
                self.last_channel = "api"
                return o
            last = o
            if not retryable:
                # Key/账号级错误：换模型只是拿同一把坏 Key 再撞一次
                break
        if len(attempted) > 1:
            self.last_notes.append("方舟 API：已依次尝试模型 " + "、".join(attempted) + " 均未取到来源")
        if len(attempted) < len(models):
            self.last_notes.append(
                f"方舟 API：已达到本次检索总预算 {self.timeout_seconds} 秒，停止继续切换模型"
            )
        return last or self._fail("方舟 API 未返回可引用来源")

    def _api_try(self, query: str, n: int, model: str,
                 timeout_seconds: Optional[float] = None):
        """单次 Responses 调用。返回 ``(outcome, 是否值得一试下一个模型)``。"""
        body = {
            "model": model,
            "stream": False,
            "input": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "input_text", "text": query}]},
            ],
        }
        tool: Dict[str, object] = {"type": "web_search", "max_keyword": self.max_keyword}
        if self.sources_cfg:
            tool["sources"] = self.sources_cfg
        body["tools"] = [tool]

        try:
            import httpx

            timeout = max(0.1, float(timeout_seconds or self.timeout_seconds))
            client = self._http or httpx.Client(timeout=timeout)
            r = client.post(self._url(), json=body, headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            })
            if r.status_code >= 400:
                log.warning("联网检索失败 HTTP %s（model=%s）: %s", r.status_code, model, r.text[:300])
                #! 是否换下一个模型，必须用同一把尺子：与主模型链路共用
                #! is_model_unavailable，否则两边对"这个错能不能靠换模型解决"判断不一致。
                return self._fail(_describe(r.status_code, r.text)), \
                    is_model_unavailable(r.status_code, r.text)
            root = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("联网检索异常: %s", e)
            # 网络/超时类异常与具体模型无关，换模型多半只是再等一次
            return self._fail(f"联网检索异常: {e}"), False

        found: List[WebSource] = []
        summary: Optional[str] = None
        for item in root.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for cc in item.get("content") or []:
                if not isinstance(cc, dict) or cc.get("type") != "output_text":
                    continue
                text = str(cc.get("text") or "").strip()
                if text and summary is None:
                    summary = text
                _collect_annotations(cc.get("annotations"), found, n)
        # annotations 为空时退一步：从摘要正文里抠 URL，至少让结论可溯源
        if not found and summary:
            found.extend(_extract_from_text(summary, n))

        found.sort(key=lambda s: s.index)
        found = found[:n]
        self.last_failure = None
        if not found:
            # 调用其实成功了，只是没拿到可引用链接 —— 换一个模型可能就有来源
            self.last_failure = "检索已执行但未返回任何可引用链接（可能是本次问题无需联网，或插件未返回来源）"
            return SearchOutcome(summary, [], self.last_failure), True
        self.last_channel = "api"
        return SearchOutcome(summary, found, None), True

    # -- 内部 --------------------------------------------------------

    def _url(self) -> str:
        b = self.base_url
        if not b and self.llm is not None:
            v = getattr(self.llm, "get_base_url", lambda: "")()
            b = (v or "").strip()
        if not b:
            return ""
        b = re.sub(r"/+$", "", b)
        return b if b.endswith("/responses") else b + "/responses"

    def _key(self) -> str:
        """独立优先，否则回退环境变量，最后才是主客户端的 Key。"""
        if self.api_key:
            return self.api_key
        if self._fallback_key:
            return self._fallback_key
        if self.llm is not None and getattr(self.llm, "is_configured", lambda: False)():
            return getattr(self.llm, "get_api_key", lambda: "")() or ""
        return ""

    def _api_configured(self) -> bool:
        return bool(self._key() and self.get_model() and self._url())

    def _fail(self, msg: str) -> SearchOutcome:
        self.last_failure = msg
        return SearchOutcome(None, [], msg)


def _normalize_mode(m: Optional[str]) -> str:
    v = (m or "").strip().lower()
    if v in ("direct", "crawl", "free"):
        return "direct"
    if v in ("api", "llm", "ark"):
        return "api"
    if v in ("mcp", "mcp-search"):
        return "mcp"
    return "auto"


def _parse_sources(raw: Optional[str]) -> List[str]:
    if not raw or not raw.strip():
        return []
    out = []
    for s in re.split(r"[,，\s]+", raw):
        t = s.strip().replace('"', "").replace("'", "")
        if t:
            out.append(t)
    return out


def _first_non_blank(*vals: Optional[str]) -> str:
    for v in vals:
        if v and v.strip():
            return v.strip()
    return ""


def _describe(status: int, body: Optional[str]) -> str:
    """方舟特有的报错可操作说明。

    未开通联网插件时返回的是 **404**，很容易被误读成「接口不存在」而白折腾。
    """
    b = body or ""
    if "ToolNotOpen" in b or "has not activated web search" in b:
        return ("方舟「联网内容插件」尚未开通，联网检索暂时拿不到来源。"
                f"请到 {ACTIVATE_URL} 免费开通后重试（无需改代码、无需重启）；"
                "或把 app.web-search.mode 设为 direct，走免 Key 的直连通道（质量不可控）。")
    if "SetLimitExceeded" in b or "Safe Experience Mode" in b:
        # 这不是"稍后重试就会好"的普通限流：安心体验模式是**账号级的额度上限**，
        # 到顶后模型服务直接暂停，不手动调整永远不会恢复。把它说成限流会让人干等。
        return ("联网检索调用被暂停：方舟账号已达到「安心体验模式」的推理额度上限，"
                "模型服务处于暂停状态。这是**账号级**限制，不是本次请求的限流，"
                "换模型、等一会儿都不会恢复——同一账号下的主模型推理会一并失败。"
                "请到方舟控制台「模型开通」页调整或关闭安心体验模式；"
                "期间可把 app.web-search.mode 设为 direct 走免 Key 直连（质量不可控）。")
    if "Arrearage" in b or "insufficient" in b.lower() or "overdue" in b.lower():
        return ("联网检索失败：方舟账号余额不足/已欠费，请充值后重试。"
                "期间可把 app.web-search.mode 设为 direct 走免 Key 直连（质量不可控）。")
    if status == 401 or "invalid_api_key" in b or "Authentication" in b:
        return (f"联网检索鉴权失败（HTTP {status}）：这把 Key 无权调用 Responses API，"
                "请为联网检索单独配置一把有权限的 Key（APP_WEB_SEARCH_API_KEY），或改用免 Key 直连（mode=direct）。")
    if status == 429 or "RateLimit" in b or "quota" in b:
        return f"联网检索触发限流（HTTP {status}），请稍后重试。"
    if status == 404 or "NotFound" in b:
        return (f"联网检索端点不存在（HTTP {status}）：请检查 app.web-search.base-url 是否指向支持 Responses API 的地址。"
                "（注意：方舟的 OpenAI 兼容 /chat/completions 不支持 web_search，必须是 /responses。）")
    return f"联网检索 HTTP {status}：{_trim(b, 160)}"


def _collect_annotations(ann, out: List[WebSource], limit: int) -> None:
    """按「任何带 url 字段的条目」来收，免得厂商小幅改字段名就让整条链路失效。"""
    if not isinstance(ann, list):
        return
    for a in ann:
        if not isinstance(a, dict):
            continue
        url = str(a.get("url") or "").strip()
        if not url or any(s.url == url for s in out):
            continue
        title = str(a.get("title") or "").strip() or url
        out.append(WebSource(len(out) + 1, title, url, str(a.get("site_name") or "").strip()))
        if len(out) >= limit:
            return


def _extract_from_text(text: str, limit: int) -> List[WebSource]:
    out: List[WebSource] = []
    for m in URL_PATTERN.finditer(text):
        if len(out) >= limit:
            break
        url = re.sub(r"[.,;。，；]+$", "", m.group(0))
        if any(s.url == url for s in out):
            continue
        out.append(WebSource(len(out) + 1, domain_of(url), url, domain_of(url)))
    return out


def _trim(s: Optional[str], max_len: int) -> str:
    if s is None:
        return ""
    return s if len(s) <= max_len else s[:max_len] + "…"


def _safe(fn, default):
    """取一个"可能不存在"的属性值：自检里宁可给默认值，也不让接口崩在半路。"""
    try:
        v = fn()
        return v if v is not None else default
    except Exception:  # noqa: BLE001
        return default


#: 相关度过滤器（懒加载单例：内部持有本地哈希向量，构造一次就够）
_RELEVANCE = None


def _relevance_impl():
    global _RELEVANCE
    if _RELEVANCE is None:
        from ..rag.embedding import LocalEmbedding
        from ..rag.relevance import SourceRelevanceFilter

        _RELEVANCE = SourceRelevanceFilter(local=LocalEmbedding())
    return _RELEVANCE


def _quality_pass(query: str, sources: List["WebSource"]) -> bool:
    """一条都没通过本地相关度 = 这条通道对本次问题是瞎给的。

    用的是**最后 differentiation tool** 的同款判据（词面 + 标题 + 本地余弦，零 token），
    目的不是"再筛一遍"，而是让「这条来源能不能递给用户」在通道编排这一层就有个是非判断。
    打分器本身出问题时按"通过"处理：宁可多给几条让下游回检去丢，也不要因为工具异常把
    整条联网链路判死。
    """
    if not sources:
        return False
    try:
        res = _relevance_impl().filter(None, query, [s for s in sources])
        #! 同样必须用严格口径。用 ``res.kept`` 会有一个后果：
        #! 它在全部不相关时仍会补一条最高分的（为了界面不留白），
        #! 于是"这条通道给的是垃圾"会被判成"这条通道给可用的东西"。
        strict = getattr(res, "strict_kept", None)
        if strict is None:  # 兜底：老实现没有该字段时按 Scored.keep 自己算
            strict = [d.source for d in getattr(res, "detail", []) if getattr(d, "keep", False)]
        return bool(strict)
    except Exception as e:  # noqa: BLE001
        log.warning("联网通道质量门判定失败（按通过处理）：%s", e)
        return True


def _host_of(url: str) -> str:
    try:
        h = urlparse(url).hostname
        return re.sub(r"^www\.", "", h) if h else url
    except Exception:  # noqa: BLE001
        return url


# 与 Java 的 ``WebSearchService.mask`` 对应（体检页展示用，不落库）
def mask_key(k: Optional[str]) -> str:
    if not k:
        return "(未配置)"
    return "****" if len(k) <= 8 else k[:6] + "…" + k[-4:]


_ = json  # 保留导入：响应体解析失败时用于定位，实际解析走 httpx 的 .json()
