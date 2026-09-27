"""大模型客户端（对应 Java ``LlmClient``）：补全 / 工具对话 / 流式 / JSON 模式 + 多模型降级。

两轮降级策略（与 Java 一致）
----------------------------
* **第一轮**只试已配置的模型（主模型 + ``AI_MODEL_CANDIDATES``），不给主链路增加额外网络往返；
* **第二轮**在第一轮全败后才拉一次 ``GET /v3/models``（只读列举，0 token），
  把「账号里确实可用」且还没试过的补上再试。

切/不切的分界线：**错误出在模型上**（404 ModelNotOpen / Model.NotExist / 429 / 5xx）就换；
**出在 Key/账号上**（401/403 invalid_api_key、Arrearage 欠费）不换。

关于成本
--------
``auto_consume`` 默认 False：后台自动任务不消耗额度，
只有用户主动点的分析/自检才调模型。这条红线不要动。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from ..config import get_settings
from .fallback import (
    ModelCooldown,
    ModelUnavailableException,
    describe as describe_status,
    is_model_unavailable,
    parse_candidates,
    upstream_error,
)
from .keys import mask_key, resolve_source, strip_trailing_slash
from .model_catalog import rank_chat_models

log = logging.getLogger(__name__)

#: Java ``\s``（默认不开启 UNICODE_CHARACTER_CLASS）等价物：只含这 6 个 ASCII 空白。
_JAVA_WS = re.compile(r"[ \t\n\x0B\f\r]+")

#: **单次调用的超时覆盖**（线程 / 任务本地）。由 :func:`llm_call_timeout` 设置。
#:
#: 为什么用 ContextVar 而不是给 ``chat_with_tools`` 加一个 ``timeout=`` 参数：
#: ① 客户端是**进程内共享单例**，直接改 ``self.chat_timeout_seconds`` 会让并发的
#:    另一次分析跟着一起变（A 的预算掐死 B 的调用）；② 加参数则要动所有调用点
#:    与测试里的假客户端签名。ContextVar 天然按执行上下文隔离，且对不读它的
#:    调用方完全无感 —— 不设就是原来的行为。
_CALL_TIMEOUT: ContextVar[Optional[float]] = ContextVar("llm_call_timeout", default=None)


@contextmanager
def llm_call_timeout(seconds: Optional[float]) -> Iterator[None]:
    """在一段代码内收紧模型调用的单次超时（秒）。``None`` / ``<=0`` 表示不收紧。

    与 :class:`~app.agent.deadline.Deadline` 配套：
    剩下多少时间，就只给这一次调用多少时间，
    避免"一次卡住的请求吃掉整份预算"——那是「到点得不到结果」最常见的成因。
    """
    if not seconds or seconds <= 0:
        yield
        return
    token = _CALL_TIMEOUT.set(float(seconds))
    try:
        yield
    finally:
        _CALL_TIMEOUT.reset(token)


def current_call_timeout() -> Optional[float]:
    """当前生效的单次调用超时覆盖（``None`` 表示沿用客户端默认）。排障与测试用。"""
    return _CALL_TIMEOUT.get()


#: 消息类型现在**统一由 :mod:`app.ai.messages` 提供**（LangChain 消息是唯一内部表示）。
#:
#: 这里 re-export 是为了既有导入点（``from ..ai.llm_client import ChatMessage, ToolCall``）
#: 不用改；新代码请直接从 :mod:`app.ai.messages` 导入。
#: 注意 ``ChatMessage.system(...)`` 现在返回的是**真的 LangChain 消息对象**。
from .messages import (  # noqa: E402  - 放在这里是为了让上面的注释紧邻它
    ChatMessage,
    ToolCall,
    args_of,
    as_message,
    id_of,
    name_of,
    text_of,
    to_payload,
    to_payloads,
)


@dataclass
class ChatResult:
    """一次 chat 调用的结果：要么有最终文本 ``content``，要么有 ``tool_calls``。

    .. NOTE::
        它是**结果 DTO**，不是消息对象。进入编排链路（LangGraph / 消息列表）之前
        请先 :meth:`to_ai_message` —— 生态组件只认 :class:`AIMessage`。
    """

    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    #: 思维链（reasoning_content）的字符数。
    #:
    #: 推理型模型"想了很久却没输出"时，正文为空而思维链很长——这是必须能看见的诊断信号，
    #: 否则只会看到一句"没有结果"，无法判断是模型的问题还是取数的问题。
    #: 思维链本身**不会**被当作正文使用。
    reasoning_chars: int = 0

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def is_blank_content(self) -> bool:
        """正文为空：调用方需要据此触发重试或本地兜底，绝不能把空正文当成结论。"""
        return self.content is None or not self.content.strip()

    def to_ai_message(self) -> Any:
        """→ :class:`~langchain_core.messages.AIMessage`。

        ``finish_reason`` / ``reasoning_chars`` 挂进 ``response_metadata`` 带走，
        这样从消息对象上也能看到"为什么正文是空的"。
        """
        from langchain_core.messages import AIMessage

        #! 走辅助函数而不是属性访问：``ToolCall`` 是 dict 子类，
        #! ``getattr(tc, "id")`` 拿不到 dict 里的 key（dict 没有这个属性），会静默变空串。
        calls: List[Dict[str, Any]] = [{
            "id": id_of(tc),
            "name": name_of(tc),
            "args": args_of(tc) or _safe_args(getattr(tc, "arguments_json", None)),
            "type": "tool_call",
        } for tc in self.tool_calls or []]
        meta: Dict[str, Any] = {}
        if self.finish_reason:
            meta["finish_reason"] = self.finish_reason
        if self.reasoning_chars:
            meta["reasoning_chars"] = self.reasoning_chars
        return AIMessage(content=self.content or "", tool_calls=calls or [],
                         response_metadata=meta or {})

    @staticmethod
    def from_ai_message(msg: Any) -> "ChatResult":
        """:class:`AIMessage` → ``ChatResult``（反向：给自研链路与测试桩用）。"""
        from .messages import tool_calls_of

        meta = getattr(msg, "response_metadata", None) or {}
        calls = [ToolCall.of(c.get("id") or "", c.get("name") or "", c.get("args") or {})
                 for c in tool_calls_of(msg)]
        return ChatResult(
            content=text_of(getattr(msg, "content", "")),
            tool_calls=calls or None,
            finish_reason=meta.get("finish_reason"),
            reasoning_chars=int(meta.get("reasoning_chars") or 0),
        )


def _safe_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        v = json.loads(raw)
    except Exception:  # noqa: BLE001
        return {}
    return v if isinstance(v, dict) else {}


class LlmClient:
    """OpenAI 兼容端点的客户端。模型降级状态由 :class:`ModelCooldown` 承载。"""

    def __init__(self, http=None, discovery=None, settings=None) -> None:
        s = settings or get_settings()
        ai = s.ai
        self._ai_settings = ai
        self.base = strip_trailing_slash(ai.base_url)
        self.model = (ai.model or "").strip()
        self.chat_timeout_seconds = int(ai.chat_timeout_seconds)
        self.thinking_type = (ai.thinking_type or "disabled").strip()
        self.thinking_active = self.thinking_type == "enabled"

        src = resolve_source(ai.api_key, None)
        self.key_source = src
        self._key = (src.value if src else "") or ""
        self.config_source = "启动配置（环境变量 / application.yml）"

        self.auto_consume = bool(ai.auto_consume)
        self._http = http

        # ---- 向量化：可完全独立，未配置时整个跟随推理底座（与 Java 同名同默认值）----
        # 为什么单独回显：它能独立于推理底座配到第三方，所以"打到哪家"必须看得见，
        # 否则换了服务商也无从确认有没有真的走对。
        rag = s.rag
        self.rag_embedding_enabled = bool(rag.embedding_enabled)
        self.embedding_base_url = (rag.embedding_base_url or "").rstrip("/")
        self.embedding_model = (rag.embedding_model or "").strip() or (ai.embedding_model or "").strip()
        self.embedding_dimensions = int(rag.embedding_dimensions or 0)
        emb_key_raw = (rag.embedding_api_key or "").strip()
        self.embedding_independent = bool(self.embedding_base_url or emb_key_raw)
        if emb_key_raw:
            self.embedding_key = emb_key_raw
        elif self.embedding_base_url:
            # 配了独立的向量化端点却没给 Key：宁可直接判定「未配置」，
            # 也不要拿推理底座的 Key 去打第三方——那只会换来一串看不懂的 401，
            # 然后被误读成"服务有问题"。
            self.embedding_key = ""
        else:
            self.embedding_key = self._key

        self._cooldown = ModelCooldown(
            primary=self.model,
            candidates=parse_candidates(ai.model_candidates),
            auto_fallback=bool(ai.model_auto_fallback),
            discovery=bool(ai.model_discovery),
            discover=self._discover_models,
        )
        self._discovery = discovery
        self._lock = threading.RLock()

    # -- 状态 --------------------------------------------------------

    def is_configured(self) -> bool:
        """``app.ai.enabled=false`` 是**总开关**，优先级高于"有没有 Key"。

        它存在的意义就是让「一个开关就能彻底不烧额度」这句话成立：
        否则排查时想临时关掉模型，只能去改 Key 或改 base-url，
        而那两个一旦改错就成了"配置的锅"，很难一眼看出是人为关掉的。
        """
        if not bool(getattr(self._ai_settings, "enabled", True)):
            return False
        return bool(self._key and self.base)

    def get_model(self) -> str:
        """**当前生效**模型 —— 会被自动降级改写（与 Java ``getModel()`` 一致）。

        Java 的 ``noteModelSuccess`` 里是直接 ``model = m``，所以降级一旦发生，
        ``getModel()`` 返回的就是**切换后**的模型。Python 侧曾经返回构造时读到的
        配置值（``self.model`` 从不被降级改写），后果是：自动降级**确实生效了**，
        但界面顶部横幅仍显示那个已经失效的旧模型名，而同一屏右侧「模型服务」卡片
        显示的是切换后的模型 —— **同一屏两处自相矛盾**，比直接报错更难查。

        配置值与生效值的分工：
        * ``self.model``        —— 配置里写的（环境变量 / 运行时入口设的）；
        * ``self._cooldown.primary`` —— 真正在用的，降级时被 promote 改写。
        """
        return self._cooldown.primary or self.model

    def get_base_url(self) -> str:
        return self.base

    def get_api_key(self) -> str:
        """原始 Key：只在同一进程内共享给其它服务（联网检索/向量化），不会写入任何日志。"""
        return self._key

    def get_key_hint(self) -> str:
        return mask_key(self._key)

    def get_key_source_name(self) -> str:
        return self.key_source.name if self.key_source else "未配置"

    def get_config_source(self) -> str:
        return self.config_source

    # -- 向量化（诊断回显用；实际检索走 rag.EmbeddingClient）----------

    def effective_embedding_base(self) -> str:
        """实际打请求的向量化地址：没配独立的就跟推理底座。"""
        return self.embedding_base_url or self.base

    def effective_embedding_key(self) -> str:
        """实际使用的向量化 Key。

        一旦配置了**独立**的向量化端点，就必须同时给对应的 Key，否则判为「未配置」——
        绝不回落到推理底座那把 Key。
        """
        if self.embedding_base_url and not self.embedding_key:
            return ""
        return self.embedding_key or self._key

    def get_embedding_base_url(self) -> str:
        return self.effective_embedding_base()

    def get_embedding_model(self) -> str:
        return self.embedding_model

    def get_embedding_key_hint(self) -> str:
        return mask_key(self.effective_embedding_key())

    def get_embedding_dimensions(self) -> int:
        return self.embedding_dimensions

    def is_embedding_independent(self) -> bool:
        """向量化这一路是否已独立配置（用来在诊断里说清楚"跟谁的"）。"""
        return self.embedding_independent

    def is_rag_embedding_enabled(self) -> bool:
        return self.rag_embedding_enabled

    def is_embedding_configured(self) -> bool:
        """向量化是否可用（只看配置，不发请求）。"""
        return (self.rag_embedding_enabled and bool(self.embedding_model)
                and bool(self.effective_embedding_key()))

    def get_last_model_switch(self) -> Optional[str]:
        return self._cooldown.last_switch

    def get_model_candidates(self) -> List[str]:
        return list(self._cooldown._candidates)

    def get_unavailable_models(self) -> Dict[str, str]:
        return self._cooldown.unavailable()

    def get_discovery_issue(self) -> Optional[str]:
        """最近一次拉取账号模型列表的失败原因（没失败过就是 ``None``）。

        供自检页回答「为什么"账号可用模型"是空的」：配置层面看不出是 Key 无效、
        账号欠费，还是这个服务商压根没有 ``GET /models``。
        """
        return self._discovery.last_issue if self._discovery is not None else None

    def is_auto_consume_allowed(self) -> bool:
        return self.auto_consume

    def set_auto_consume(self, v: bool) -> None:
        if self.auto_consume != v:
            self.auto_consume = bool(v)
            log.info("[AI] 后台自动消耗模型额度：%s", "已允许" if v else "已禁止")

    def set_model_auto_fallback(self, v: bool) -> None:
        self._cooldown.set_auto_fallback(v)

    def add_model_candidate(self, model_id: Optional[str]) -> None:
        if model_id and model_id.strip():
            self._cooldown.add_candidate(model_id.strip())

    # -- 运行期热更新 ------------------------------------------------

    def apply_runtime_config(self, new_base: Optional[str] = None,
                             new_model: Optional[str] = None,
                             new_key: Optional[str] = None) -> Optional[str]:
        """运行期切换整套模型服务配置（**Windows 上改系统环境变量对已启动进程无效**）。

        换厂商往往 base-url、模型名、Key 必须同时换，只换其中一两个必然 401，
        使用者很难自己定位。这里给出一条「不重启也能换整套底座」的通道。

        返回 ``None`` 表示可用；否则返回不可读原因（同 :meth:`probe`）。
        """
        with self._lock:
            parts: List[str] = []
            if new_base and new_base.strip():
                b = strip_trailing_slash(new_base.strip())
                if b != self.base:
                    self.base = b
                    parts.append(f"base-url={b}")
                    # 换了底座，之前发现到的「可用模型」和「失效模型」全部作废——
                    # 它们描述的是上一家的账号权限，留着只会误导降级链
                    self._cooldown = ModelCooldown(
                        primary=self.model,
                        candidates=self.get_model_candidates(),
                        discover=self._discover_models,
                    )
                    self._discovery = None
            if new_model and new_model.strip():
                m = new_model.strip()
                # 比较基准必须用**生效模型**而不是配置值：降级已经把 primary 改写成 X
                # 之后，若用户提交恰好等于配置值 Y 的模型名，用 self.model 比会得到
                # "没变化" → 直接跳过 → primary 仍是 X。界面上点了"应用"、提示成功，
                # 实际用的还是降级后的模型，属于静默失败。
                if m != self.get_model():
                    self.model = m
                    self._cooldown.set_primary(m)
                    self.add_model_candidate(m)
                    parts.append(f"model={m}")
            if new_key and new_key.strip():
                self._key = new_key.strip()
                self.key_source = None
                parts.append("key=" + mask_key(self._key))
            self.config_source = "界面热更新（重启后失效，请同步写入环境变量）"
            log.info("[AI] 运行期配置更新：%s", "；".join(parts) or "无变化")
        return self.probe()

    def apply_runtime_key(self, new_key: str) -> Optional[str]:
        return self.apply_runtime_config(None, None, new_key)

    def apply_embedding_config(self, new_base: Optional[str] = None,
                               new_model: Optional[str] = None,
                               new_key: Optional[str] = None) -> Optional[str]:
        """运行期切换**向量化**这一路（推理底座不动）。

        用途：推理留在火山方舟，语义检索挂到第三方 embedding（OpenAI 兼容即可）。
        换完立刻发一条真实向量化请求自证，不用重启。

        返回 ``None`` 表示可用；否则返回可读原因（同 :meth:`probe`）。
        """
        with self._lock:
            parts: List[str] = []
            if new_base and new_base.strip():
                b = strip_trailing_slash(new_base.strip())
                if b != self.embedding_base_url:
                    self.embedding_base_url = b
                    parts.append(f"base-url={b}")
            if new_model and new_model.strip():
                m = new_model.strip()
                if m != self.embedding_model:
                    self.embedding_model = m
                    parts.append(f"model={m}")
            if new_key and new_key.strip():
                k = new_key.strip()
                if k != self.embedding_key:
                    self.embedding_key = k
                    parts.append("key=" + mask_key(k))
            if parts:
                self.embedding_independent = True
            self.config_source = "界面热更新（重启后失效，请同步写入环境变量）"
            log.info("[AI] 向量化配置更新：%s", "；".join(parts) or "无变化")

        try:
            v = self.embed(["capability probe"],
                           base_url=self.effective_embedding_base(),
                           model=self.embedding_model,
                           key=self.effective_embedding_key(),
                           dimensions=self.embedding_dimensions)
            if not v or not v[0]:
                return (f"向量化返回了空向量：模型 {self.embedding_model} 在 "
                        f"{self.effective_embedding_base()} 上未产出结果。")
            log.info("[AI] 向量化自检通过：%s 维", len(v[0]))
            return None
        except Exception as e:  # noqa: BLE001 - 自检失败要把原因带回去给人看
            return str(e)

    # -- 调用 --------------------------------------------------------

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        return self._cooldown.with_fallback("补全", lambda m: self._complete_once(m, system, user, temperature))

    def complete_json(self, system: str, user: str, temperature: float = 0.2) -> str:
        return self._cooldown.with_fallback(
            "JSON补全", lambda m: self._complete_json_once(m, system, user, temperature))

    def chat_with_tools(self, tools: Sequence[Any], messages: Sequence[ChatMessage],
                        temperature: float = 0.2, max_tokens: int = 0) -> ChatResult:
        """携带工具声明的多轮对话。``max_tokens <= 0`` 表示不显式设置，沿用服务商默认。

        「空正文自愈」会在重试时显式给足输出预算，绕开
        「推理型模型思考 token 占满默认预算、正文为空」这个坑。
        """
        return self._cooldown.with_fallback(
            "工具对话",
            lambda m: self._chat_once(m, list(tools), list(messages), temperature, max_tokens),
            promote=False,
        )

    def stream(self, system: str, user: str, temperature: float,
               on_token: Callable[[str], None]) -> str:
        """流式补全：逐 token 回调，最后返回完整文本。"""
        return self._cooldown.with_fallback(
            "流式补全", lambda m: self._stream_once(m, system, user, temperature, on_token))

    def embed(self, inputs: Sequence[str], base_url: str = "", model: str = "",
              key: str = "", dimensions: int = 0, timeout: float = 20.0) -> List[List[float]]:
        """向量化。参数由 :class:`~app.rag.embedding_client.EmbeddingClient` 传入。"""
        if not inputs:
            return []
        body: Dict[str, Any] = {"model": model, "input": list(inputs)}
        if dimensions > 0:
            body["dimensions"] = dimensions
        raw = self._post(f"{strip_trailing_slash(base_url)}/embeddings", body,
                         timeout=timeout, key=key)
        data = json.loads(raw).get("data") or []
        return [list(d.get("embedding") or []) for d in data]

    # -- 自检 --------------------------------------------------------

    def probe(self) -> Optional[str]:
        """连通性自检：发一个 1 token 的极小请求。

        把「Key 无效 / 账户欠费 / 模型无权限」这类问题直接说清楚，
        而不是把一段原始 JSON 抛给使用者。返回 ``None`` 表示正常。
        """
        if not self.is_configured():
            return "未配置 API Key：请设置环境变量 AI_API_KEY（或 OPENAI_API_KEY）后重启后端"
        try:
            self._cooldown.with_fallback("连通性自检", lambda m: self._post(
                f"{self.base}/chat/completions",
                {"model": m, "max_tokens": 1, "messages": to_payloads([ChatMessage.user("ping")])},
                timeout=20))
            return None
        except RuntimeError as e:
            return str(e)

    def adopt_working_model(self) -> Optional[str]:
        """自动探测一个「现在真的能用」的模型并把它变成当前模型。"""
        if not self.is_configured():
            return "未配置 API Key"
        self.list_available_models(True)
        return self.probe()

    def list_available_models(self, force: bool = False) -> List[str]:
        """``GET /v3/models`` —— **只读列举，0 token**。"""
        if self._discovery is None:
            from .discovery import ModelDiscovery

            # 参数名必须与 ModelDiscovery.__init__ 对齐（base_url / api_key / enabled / transport）。
            # 历史上这里照着 Java 的 `new ModelDiscovery(baseUrl, key, 20)` 传了 timeout/http，
            # 于是**每次调用都抛 TypeError**；而唯一的调用方（第一轮全败后的"补试一轮"）
            # 把异常吞在 log.debug 里，表现成"自动降级从来不生效、只报主模型不可用"。
            # 教训：`except Exception: log.debug(...)` 会把"代码写错"伪装成"上游不可用"。
            self._discovery = ModelDiscovery(
                base_url=self.base, api_key=self._key,
                enabled=self._cooldown.discovery,
            )
        ids = self._discovery.list_available_models(force=force)
        self._cooldown.set_candidates(list(self._cooldown._candidates))
        return ids

    def recommended_models(self, refresh: bool = False, limit: int = 8) -> List[str]:
        ids = self.list_available_models(force=refresh)
        return rank_chat_models(ids)[:limit]

    def capabilities(self) -> Dict[str, Any]:
        """能力体检：换 LLM 厂商后先看这张表，缺哪项就少了哪项功能。

        键集合必须与 Java ``LlmClient.capabilities()`` **完全一致**
        （``chat`` / ``jsonMode`` / ``tools`` / ``embedding``）：前端「服务商能力体检」
        卡片是拿这 4 个键去自己的标签表里取值的，键名一变整张表就全变成"不可用"——
        属于"接口在、行为不对"的静默失败。

        每项都发一个极小请求**实测**（失败原因用 :func:`describe` 翻译成人话），
        所以它只该由使用者主动点击触发（「模型自检」/「应用并自检」），不做后台调用。
        """
        caps: Dict[str, Any] = {}

        chat = self.probe()
        caps["chat"] = "ok" if chat is None else chat

        # ① JSON 模式
        # 注意预算：推理型模型（如 DeepSeek-R1/V4 系列、QwQ）会先花掉一部分 token 做思考，
        # 预算给小了会出现 content 为空 + finish_reason=length，从而被误判成「不支持 JSON 模式」。
        try:
            raw = self._cooldown.with_fallback("体检·JSON模式", lambda m: self._post(
                f"{self.base}/chat/completions",
                {
                    "model": m,
                    "max_tokens": 1024,
                    "messages": [
                        to_payload(ChatMessage.system("只输出 JSON，不要输出其它内容。")),
                        to_payload(ChatMessage.user('返回 {"ok":true}')),
                    ],
                    "response_format": {"type": "json_object"},
                },
                timeout=150))
            content = _extract_content(raw)
            if content is not None and "ok" in content:
                caps["jsonMode"] = "ok"
            elif content is None or not content.strip():
                caps["jsonMode"] = (
                    "响应被 max_tokens 截断（推理型模型思考 token 占满预算）。"
                    "本平台正式调用不限制 max_tokens，一般不受影响。"
                    if _is_truncated(raw)
                    else "模型返回空内容，疑似不支持 response_format=json_object")
            else:
                caps["jsonMode"] = "返回内容不是预期 JSON：" + _trim(content, 120)
        except Exception as e:  # noqa: BLE001 - 体检不该让整个接口炸掉
            caps["jsonMode"] = _trim(str(e), 200)

        # ② 工具调用：给一个假工具，若模型返回 tool_calls 即视为支持
        try:
            raw = self._cooldown.with_fallback("体检·工具调用", lambda m: self._post(
                f"{self.base}/chat/completions",
                {
                    "model": m,
                    "max_tokens": 1024,
                    "messages": to_payloads([ChatMessage.user(
                        "Call ping_tool now. 请立即调用 ping_tool，不要只解释。")]),
                    "tools": [{"type": "function", "function": {
                        "name": "ping_tool",
                        "description": "Return a fixed string, used to probe tool-calling support.",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    }}],
                    "tool_choice": "auto",
                },
                timeout=150))
            if raw is not None and '"tool_calls"' in raw:
                caps["tools"] = "ok"
            elif _is_truncated(raw):
                caps["tools"] = "响应被 max_tokens 截断，未能判定（推理型模型思考占满预算）"
            else:
                caps["tools"] = "模型未返回 tool_calls（该服务商可能不支持函数调用）"
        except Exception as e:  # noqa: BLE001
            caps["tools"] = _trim(str(e), 200)

        # ③ 向量化：先分清「没开」和「不支持」，别把配置状态报成故障；
        #    可用时把「打到哪家、什么模型、几维」一并回显，换服务商后一眼能确认没走错路径。
        if not self.rag_embedding_enabled:
            caps["embedding"] = "已关闭（APP_RAG_EMBEDDING_ENABLED=false，检索走关键词匹配）"
        elif not self.embedding_model:
            # 没配向量化模型 = 默认使用本地零成本向量，这不是故障，要说清楚
            caps["embedding"] = ("本地向量（未配置第三方 embedding，"
                                 "使用零成本本地哈希向量：不联网、无需 Key）")
        elif not self.effective_embedding_key():
            # 端点已指向第三方却没给对应的 Key：直接说明该怎么配，
            # 不要为了"探活"白打一次注定失败的请求。
            caps["embedding"] = (
                f"已配置独立的向量化服务 {self.effective_embedding_base()} · {self.embedding_model}"
                "，但未配置对应的 Key。请设置环境变量 APP_EMBEDDING_API_KEY"
                "（或在「模型自检」卡片里热更新），配置前检索走关键词匹配。")
        else:
            try:
                v = self.embed(["capability probe"],
                               base_url=self.effective_embedding_base(),
                               model=self.embedding_model,
                               key=self.effective_embedding_key(),
                               dimensions=self.embedding_dimensions)
                dims = len(v[0]) if v and v[0] else 0
                caps["embedding"] = (
                    f"ok（{self.effective_embedding_base()} · {self.embedding_model} · {dims} 维"
                    f"{' · 独立配置' if self.embedding_independent else ' · 跟随推理底座'}）")
            except Exception as e:  # noqa: BLE001
                caps["embedding"] = _trim(str(e), 260)
        return caps

    # -- 内部：单次调用 ----------------------------------------------

    def _complete_once(self, model: str, system: str, user: str, temperature: float) -> str:
        body = self._build_body(model, system, user, None, temperature, 0)
        raw = self._post_chat(body)
        return _extract_content(raw)

    def _complete_json_once(self, model: str, system: str, user: str, temperature: float) -> str:
        body = self._build_body(model, system, user, None, temperature, 0)
        body["response_format"] = {"type": "json_object"}
        return _extract_content(self._post_chat(body))

    def _chat_once(self, model: str, tools: Sequence[Any], messages: Sequence[ChatMessage],
                   temperature: float, max_tokens: int) -> ChatResult:
        body = self._build_body_from_messages(model, tools, messages, temperature, max_tokens)
        raw = self._post_chat(body)
        return _parse_chat_result(raw)

    def _stream_once(self, model: str, system: str, user: str, temperature: float,
                     on_token: Callable[[str], None]) -> str:
        body = self._build_body(model, system, user, None, temperature, 0)
        body["stream"] = True
        url = f"{self.base}/chat/completions"
        buf: List[str] = []
        for chunk in self._post_stream(url, body):
            if not chunk:
                continue
            buf.append(chunk)
            on_token(chunk)
        return "".join(buf)

    # -- 内部：HTTP --------------------------------------------------

    def _post_chat(self, body: Dict[str, Any]) -> str:
        return self._post(f"{self.base}/chat/completions", body, timeout=self._timeout())

    def _timeout(self) -> float:
        """本次调用的超时：``llm_call_timeout`` 覆盖优先，否则用配置的 ``chat_timeout_seconds``。"""
        override = current_call_timeout()
        if override and override > 0:
            return float(override)
        return float(self.chat_timeout_seconds)

    def _post(self, url: str, body: Dict[str, Any], timeout: float, key: Optional[str] = None) -> str:
        import httpx

        client = self._http or httpx.Client(timeout=timeout)
        try:
            r = client.post(url, json=body, headers={
                "Authorization": f"Bearer {key or self._key}",
                "Content-Type": "application/json",
            })
            if r.status_code >= 400:
                # 只有「模型层面」的错误才换模型；Key/账号层面的错误直接抛，换也没用
                if is_model_unavailable(r.status_code, r.text):
                    raise ModelUnavailableException(r.status_code, r.text[:200])
                raise upstream_error(r.status_code, r.text[:400],
                                     auto_fallback=self._cooldown.auto_fallback)
            return r.text
        finally:
            if self._http is None:
                client.close()

    def _post_stream(self, url: str, body: Dict[str, Any]):
        """SSE 流式：逐块产出增量文本。"""
        import httpx

        client = self._http or httpx.Client(timeout=self._timeout())
        try:
            with client.stream("POST", url, json=body, headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            }) as r:
                if r.status_code >= 400:
                    r.read()
                    if is_model_unavailable(r.status_code, r.text):
                        raise ModelUnavailableException(r.status_code, r.text[:200])
                    raise upstream_error(r.status_code, r.text[:400],
                                         auto_fallback=self._cooldown.auto_fallback)
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            yield piece
        finally:
            if self._http is None:
                client.close()

    # -- 内部：请求体 ------------------------------------------------

    def _build_body(self, model: str, system: Optional[str], user: Optional[str],
                    tools: Optional[Sequence[Any]], temperature: float, max_tokens: int) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append(to_payload(ChatMessage.system(system)))
        if user:
            messages.append(to_payload(ChatMessage.user(user)))
        return self._finish_body(model, messages, tools, temperature, max_tokens)

    def _build_body_from_messages(self, model: str, tools: Sequence[Any],
                                  messages: Sequence[Any], temperature: float,
                                  max_tokens: int) -> Dict[str, Any]:
        #! ``messages`` 现在是 **LangChain 消息对象**（不再是自研 DTO），
        #! 线格式只在这一行产生 —— 这是自研 HTTP client 与生态之间的唯一边界。
        return self._finish_body(model, to_payloads(messages),
                                 tools, temperature, max_tokens)

    def _finish_body(self, model: str, messages: List[Dict[str, Any]],
                     tools: Optional[Sequence[Any]], temperature: float,
                     max_tokens: int) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages,
                                "temperature": temperature, "stream": False}
        if tools:
            body["tools"] = [{"type": "function",
                              "function": {"name": t.name, "description": t.description,
                                           "parameters": t.parameters}} for t in tools]
        if max_tokens and max_tokens > 0:
            body["max_tokens"] = max_tokens
        # disabled（默认）/ enabled / 空串=完全不下发该字段
        if self.thinking_type == "enabled":
            body["thinking"] = {"type": "enabled"}
        elif self.thinking_type == "disabled":
            body["thinking"] = {"type": "disabled"}
        return body

    def _discover_models(self, force: bool) -> Sequence[str]:
        try:
            return self.list_available_models(force=force)
        except Exception as e:  # noqa: BLE001 - 补试一轮拿不到列表不该阻断主链路
            # 这里用 warning 而不是 debug：能走到 except 说明是**列举过程本身出错**
            # （代码缺陷 / 极端网络），而不是"上游说没有模型"。用 debug 会把它藏起来——
            # 曾经就有一个构造参数名写错在这里躺了很久，对外表现只是"自动降级从来没生效"。
            log.warning("[AI] 模型发现失败（不阻断本次调用，但降级链少了补试一轮）：%s: %s",
                        type(e).__name__, e)
            return []


# ------------------------------------------------------------------
# 响应解析
# ------------------------------------------------------------------


def _is_truncated(raw: Optional[str]) -> bool:
    """判断是否被 max_tokens 截断——推理型模型的典型症状是"想了很久但没输出"。

    Java 侧是 ``raw.contains("\\"finish_reason\\":\\"length\\"")``：只在底座返回**紧凑 JSON**
    时成立（方舟就是这样）。这里改成先解析再判定，对带空格的 JSON 同样有效，
    结论不会与 Java 相反 —— 只是少了一个"格式一变动就静默判错"的隐患。
    """
    if raw is None:
        return False
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return '"finish_reason":"length"' in raw
    return any(str(c.get("finish_reason") or "") == "length"
               for c in (obj.get("choices") or []))


def _trim(s: Optional[str], max_len: int) -> str:
    """等价 Java ``LlmClient.trim``：``replaceAll("\\s+"," ")`` + ``String.trim()`` + 超长截断。

    刻意复用 :mod:`app.rag.textnorm` 的 ``trim``（Java 的 trim 只吃 <= U+0020，
    **不删全角空格 U+3000**），避免同一个语义在项目里出现第二个版本。
    """
    if s is None:
        return ""
    from ..rag.textnorm import trim

    t = trim(_JAVA_WS.sub(" ", str(s)))
    return t if len(t) <= max_len else t[:max_len] + "…"


def _extract_content(raw: str) -> str:
    """抽取首条 choices 的正文；拿不到就返回空串（不再抛，交给调用方判定）。"""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return ""
    for choice in obj.get("choices") or []:
        msg = choice.get("message") or {}
        c = msg.get("content")
        if c:
            return str(c)
    return ""


def _parse_chat_result(raw: str) -> ChatResult:
    obj = json.loads(raw)
    choices = obj.get("choices") or []
    if not choices:
        return ChatResult(content=None, finish_reason=None)
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    reasoning = msg.get("reasoning_content") or ""
    calls: List[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        #! 归一化成 **LangChain 形状**（``{id,name,args,type}``）：
        #! 线格式的 ``function.arguments`` 是字符串，留在里面后面每一步都要记得 parse。
        calls.append(ToolCall.of(tc.get("id"), fn.get("name"), fn.get("arguments")))
    return ChatResult(
        content=None if content is None else str(content),
        tool_calls=calls or None,
        finish_reason=choices[0].get("finish_reason"),
        reasoning_chars=len(reasoning),
    )


def describe(status: int, body: Optional[str]) -> str:
    """把上游错误码翻译成人话（复用阶段 2b 已对账过的实现）。"""
    return describe_status(status, body)
