# -*- coding: utf-8 -*-
"""
模型降级层：冷却状态机 + 上游错误翻译。

对应 Java ``LlmClient`` 里 ``MODEL_*`` 相关的部分（约 431~560 行、884~915 行、
1292~1343 行）。这一片**不烧 token 就能对账**，所以放在阶段 2b 的第一刀：

* ``describe`` / ``is_model_unavailable`` / ``trim`` / ``parse_candidates``
  是纯函数，基准由 ``qa/_java_ground.jsh`` 用 jshell 跑 Java 真身取回。
* 冷却状态机是纯状态逻辑，用注入的时钟做单元测试。

.. NOTE::
   字符串处理**一律走 Python 原生语义**（与 :mod:`app.rag.textnorm` 同源）：
   空白按 Unicode、长度与切片按码点。

   这里原来为了对齐 Java 保留了三条非原生规则（``re.ASCII`` 的空白折叠、
   ``<= U+0020`` 的 trim、按 UTF-16 码元截断），它们的共同特点是
   **不报错、只在遇到全角空格或 emoji 时悄悄偏一点**。Java 侧已下线，
   再留着就是纯负债 —— 详见 ``app/rag/textnorm.py`` 的模块说明。
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, TypeVar

from ..core.logging import get_logger

_log = get_logger("AI")

# ---- 常量 ----
MODEL_DEAD_COOLDOWN_MS = 10 * 60 * 1000  # 10 分钟
MODEL_MAX_ATTEMPTS = 4

T = TypeVar("T")

# 逗号、CR、LF 任一都是分隔符
_SPLIT_CANDIDATES = re.compile(r"[,\r\n]")
# 空白折叠：Python 原生（Unicode 感知），全角空格也会被折成一个空格
_WS_RUN = re.compile(r"\s+")


def _trim(s: str) -> str:
    """去首尾空白（Unicode 感知）。"""
    return s.strip() if s else s


def _len(s: str) -> int:
    """长度按**码点**。"""
    return len(s)


def _head(s: str, n: int) -> str:
    """取前 ``n`` 个**码点**。判断与切片必须同量纲，所以两个函数成对出现。"""
    return s[:n] if n > 0 else ""


def trim(s: Optional[str], max_len: int) -> str:
    """折叠空白后截断。

    * ``None`` → 空串（用在「原因可能没给」的地方）
    * 截断后追加 ``…``；``max_len`` 按码点计
    """
    if s is None:
        return ""
    t = _trim(_WS_RUN.sub(" ", s))
    return (_head(t, max_len) + "…") if _len(t) > max_len else t


def is_model_unavailable(status: int, body: Optional[str]) -> bool:
    """这个报错「换一个模型就可能好」吗？对应 ``LlmClient.isModelUnavailable``。

    分界线是**问题出在模型上，还是出在 Key/账号上**：401/403 和欠费属于后者，
    换模型只会拿同一把坏 Key 再撞一次，还会把「Key 错了」误导成「模型不对」。
    """
    b = body or ""
    if status == 401 or status == 403:
        return False
    if "invalid_api_key" in b or "Arrearage" in b or "overdue-payment" in b:
        return False
    if status == 404 or status == 429 or status >= 500:
        return True
    return (
        "ModelNotOpen" in b
        or "Model.NotExist" in b
        or "model_not_found" in b
        or "InvalidEndpointOrModel" in b
        or "do not have access" in b
        or "not activated the model" in b
    )


def describe(status: int, body: Optional[str]) -> str:
    """把上游错误码翻译成人话。对应 ``LlmClient.describe``（public static）。

    判断顺序**不能调整**：``Arrearage`` 在 ``invalid_api_key`` 之前、
    ``ModelNotOpen`` 在 404 兜底之前 —— 顺序错了会把「欠费」报成「Key 无效」，
    把「没开通」报成「接口不存在」，都是指向错误排障方向的事故级误导。
    """
    b = body or ""
    if "Arrearage" in b or "overdue-payment" in b:
        return (
            "大模型服务账户欠费或额度已用尽（HTTP " + str(status) + "，Arrearage / overdue-payment）。"
            "免费额度用尽或过期后同样报这个错：请到服务商控制台确认余额，或充值后重试。"
        )
    if "invalid_api_key" in b or status == 401:
        return (
            "API Key 无效（HTTP " + str(status) + "，invalid_api_key），与额度无关——服务根本没认这把 Key。"
            "常见原因：① 控制台里这把 Key 已被删除/禁用或换过；② 复制时少了字符；"
            "③ Key 地域与调用域名不匹配（火山方舟是 https://ark.cn-beijing.volces.com/api/v3）；"
            "④ 环境变量里放的是别的平台的 Key。"
            "请到当前模型服务商的控制台复制一把有效 Key，在本页直接粘贴热更新即可，无需重启。"
        )
    if "ModelNotOpen" in b or "not activated the model" in b:
        return (
            "该模型在你的账号下尚未开通（HTTP " + str(status) + "，ModelNotOpen）。"
            "注意方舟把「未开通」报成 404，很容易被误读成「接口不存在」——它其实只是没开这个模型。"
            "本平台已开启自动降级：会按候选链自动切到账号里确实可用的模型（无需改配置）。"
            "若想固定用某一个，请到方舟控制台开通它，或在 AI 分析页的【模型服务】卡片里从可用列表下拉选择。"
        )
    if "Model.NotExist" in b or "model_not_found" in b or status == 404:
        return (
            "模型或接口不存在（HTTP " + str(status) + "）。三种情况最常见："
            "① 模型名不对——请核对该服务商的模型 ID（火山方舟要填控制台里的模型 ID 或接入点 ep-xxx）；"
            "② 该模型在控制台<b>没开通</b>——注意它的表现就是 404，很容易被误读成「接口不存在」"
            "（实测：方舟 /v3/models 能列出 doubao-embedding-*，但这把 Key 直接调用就是 404）；"
            "③ 该服务商确实没这个接口——此时请关掉对应能力（如 RAG 向量化 APP_RAG_EMBEDDING_ENABLED=false），"
            "检索会自动退化为关键词匹配。"
        )
    if status == 429:
        return "模型服务限流（HTTP 429）。请稍后重试或降低并发。"
    return "模型服务返回 HTTP " + str(status) + "：" + (_head(b, 200) if _len(b) > 200 else b)


def parse_candidates(raw: Optional[str]) -> List[str]:
    """逗号 / 换行分隔的候选模型列表。对应 ``LlmClient.parseCandidates``。

    保留首次出现顺序、去空、去重。
    """
    if raw is None or not _trim(raw):
        return []
    out: List[str] = []
    for s in _SPLIT_CANDIDATES.split(raw):
        t = _trim(s)
        if t and t not in out:
            out.append(t)
    return out


class ModelUnavailableException(RuntimeError):
    """能靠换模型解决的失败。抛它才会触发降级链；其余异常一律不做无谓重试。"""

    def __init__(self, status: int, body: Optional[str]):
        super().__init__(describe(status, body))
        self.status = status
        self.body = body


def upstream_error(status: int, body: Optional[str], auto_fallback: bool = True) -> RuntimeError:
    """统一处理上游错误状态码。对应 ``LlmClient.upstreamError``。

    :return: 能换模型解决 → :class:`ModelUnavailableException`；否则普通异常
    """
    if auto_fallback and is_model_unavailable(status, body):
        return ModelUnavailableException(status, body)
    return RuntimeError(describe(status, body))


class ModelCooldown:
    """模型降级的状态机：候选链排序、冷却记账、自动切换记录。

    对应 Java ``LlmClient`` 里 ``modelChain`` / ``orderAliveFirst`` / ``isCooling``
    / ``markModelDead`` / ``noteModelSuccess`` / ``withModelFallback`` /
    ``getUnavailableModels`` / ``getLastModelSwitch`` 这一组。

    :param clock: 返回**毫秒**时间戳的可调用对象，便于测试里注入假时钟。
    :param discover: ``(force: bool) -> list[str]``，返回**已按「新且强」排好序**的
        账号可用模型。第二轮降级时用它补候选。

    .. NOTE::
       :meth:`unavailable` 的返回顺序与 Java **不保证一致**：Java 遍历的是
       ``ConcurrentHashMap``，迭代顺序本身未定义；Python 用 dict 的插入序。
       这是「Java 自己也不保证顺序」而非移植偏差，比对时按集合比（见对账脚本）。
    """

    def __init__(
        self,
        primary: str = "",
        candidates: Sequence[str] = (),
        *,
        cooldown_ms: int = MODEL_DEAD_COOLDOWN_MS,
        max_attempts: int = MODEL_MAX_ATTEMPTS,
        auto_fallback: bool = True,
        discovery: bool = True,
        discover: Optional[Callable[[bool], Sequence[str]]] = None,
        clock: Callable[[], int] = lambda: int(time.time() * 1000),
    ):
        self._primary = primary or ""
        self._candidates = list(candidates)
        self._cooldown_ms = cooldown_ms
        self._max_attempts = max_attempts
        self.auto_fallback = auto_fallback
        self.discovery = discovery
        self._discover = discover
        self._clock = clock
        self._lock = threading.RLock()
        self._dead_until: Dict[str, int] = {}
        self._dead_reason: Dict[str, int] = {}  # type: ignore[assignment]  # 见下
        self._dead_reason = {}
        self.last_switch: Optional[str] = None

    # ---------------- 配置 ----------------

    @property
    def primary(self) -> str:
        with self._lock:
            return self._primary

    def set_primary(self, v: str) -> None:
        with self._lock:
            self._primary = v or ""

    def set_candidates(self, v: Sequence[str]) -> None:
        with self._lock:
            self._candidates = list(v)

    def add_candidate(self, model_id: str) -> bool:
        """运行期追加候选；已存在则不重复加（对齐 ``addModelCandidate``）。"""
        if not model_id:
            return False
        with self._lock:
            if model_id in self._candidates:
                return False
            self._candidates.append(model_id)
            return True

    def set_auto_fallback(self, v: bool) -> None:
        with self._lock:
            self.auto_fallback = bool(v)

    # ---------------- 冷却记账 ----------------

    def is_cooling(self, model: Optional[str]) -> bool:
        if not model:
            return False
        with self._lock:
            until = self._dead_until.get(model)
            return until is not None and until > self._clock()

    def mark_dead(self, model: Optional[str], reason: Optional[str]) -> None:
        """记录某个模型不可用，并进入冷却。"""
        if not model:
            return
        with self._lock:
            self._dead_until[model] = self._clock() + self._cooldown_ms
            self._dead_reason[model] = trim(reason, 200)

    def note_success(
        self,
        model: Optional[str],
        op: str,
        reason: Optional[str] = None,
        promote: bool = True,
    ) -> None:
        """调用成功：解除冷却；``promote`` 时把用通的模型提升为当前模型。

        :param promote: ``False`` = 只解除冷却，**不动当前模型**。调用方按请求
            临时指定了另一个模型时必须传 ``False``，否则主模型会被静默换成那个
            临时模型 —— 后续分析跟着降档且没有任何报错，表现出来是
            「最近分析质量突然变差」，极难归因。
        """
        if not model:
            return
        with self._lock:
            self._dead_until.pop(model, None)
            self._dead_reason.pop(model, None)
            if not promote or model == self._primary:
                return
            from_ = self._primary
            self._primary = model
            self.last_switch = (
                _now_hhmmss() + "　" + op + " 时自动切换模型："
                + (from_ or "(未指定)") + " → " + model
                + "（原因：" + (reason or "原模型不可用") + "）"
            )
            _log.info("[AI] 模型自动降级：%s → %s（触发：%s）", from_, model, op)

    def unavailable(self) -> Dict[str, str]:
        """当前处于冷却期的模型 → 失效原因。对应 ``getUnavailableModels``。"""
        now = self._clock()
        with self._lock:
            return {k: v for k, v in self._dead_reason.items() if self._dead_until.get(k, 0) > now}

    def cooling_names(self) -> List[str]:
        now = self._clock()
        with self._lock:
            return [k for k, u in self._dead_until.items() if u > now]

    # ---------------- 候选链 ----------------

    def order_alive_first(self, chain: Iterable[str]) -> List[str]:
        """冷却期内的模型排到最后：不指望它，但也不至于完全不试。

        注意截断：最多保留 :data:`MODEL_MAX_ATTEMPTS` 个，撞太多只会拖长响应时间。
        """
        now = self._clock()
        alive: List[str] = []
        cooling: List[str] = []
        with self._lock:
            for m in chain:
                until = self._dead_until.get(m)
                if until is not None and until > now:
                    cooling.append(m)
                else:
                    alive.append(m)
        alive.extend(cooling)
        return alive[: self._max_attempts]

    def model_chain(self) -> List[str]:
        """主模型 + 候选（去重、保序）；都没配且开了发现，就取账号可用列表。"""
        with self._lock:
            primary = self._primary
            cands = list(self._candidates)
            discovery = self.discovery
            discover = self._discover
        seen = []
        chain: List[str] = []
        if primary:
            chain.append(primary)
            seen.append(primary)
        for m in cands:
            if m not in seen:
                chain.append(m)
                seen.append(m)
        if not chain and discovery and discover is not None:
            for m in discover(False):
                if m not in seen:
                    chain.append(m)
                    seen.append(m)
        return self.order_alive_first(chain)

    # ---------------- 降级执行 ----------------

    def with_fallback(
        self,
        op: str,
        call: Callable[[str], T],
        promote: bool = True,
    ) -> T:
        """带模型降级的调用外壳。对应 ``withModelFallback``。

        两轮策略：**第一轮**只试已配置的模型（主模型 + 候选），不给主链路增加
        额外网络往返；**第二轮**在第一轮全败后才拉一次 ``GET /v3/models``，
        把「账号里确实可用」且还没试过的补上再试。

        :param call: 给定模型名执行调用；抛 :class:`ModelUnavailableException` 才换下一个
        """
        chain = self.model_chain()
        if not chain:
            raise RuntimeError("没有可用模型：" + _model_config_hint())

        last: Optional[ModelUnavailableException] = None
        for m in chain:
            try:
                r = call(m)
                self.note_success(m, op, last.args[0] if last else None, promote)
                return r
            except ModelUnavailableException as e:
                self.mark_dead(m, str(e))
                last = e
                _log.warning("[AI] 模型 %s 不可用：%s", m, str(e))

        extra: List[str] = []
        if self.discovery and self._discover is not None:
            owned = list(self._discover(True))
            # 第二轮是"救火"，顺序决定成败：**必须按「新且强」挑，不能照上游返回的
            # 原始顺序取前几个**。实测（2026-09-22，134 个模型）：方舟的
            # ``GET /v3/models`` 按上架时间从旧到新返回，且列的是**平台全量**而不是
            # 本账号开通的 —— 91 个压根不存在、16 个未开通。照原序取前 4 个必然挑到
            # 2024 年的老模型，全部 404，于是"自动降级"对外看起来永远不生效。
            # ``rank_chat_models`` 同时做两件事：剔掉 embedding/图像等非对话模型，
            # 再按代次 + 版本日期倒序。
            try:
                from .model_catalog import rank_chat_models

                owned = rank_chat_models(owned)
            except Exception as e:  # noqa: BLE001 - 排序只是优化，失败仍按原序兜底
                _log.debug("[AI] 候选排序失败，按上游原序补试：%s", e)
            for mid in owned:
                if mid not in chain and not self.is_cooling(mid):
                    extra.append(mid)
        extra = extra[: self._max_attempts]
        for m in extra:
            try:
                r = call(m)
                # 第二轮一律 promote：走到这一步说明原模型确实不可用
                self.note_success(m, op, str(last) if last else None, True)
                return r
            except ModelUnavailableException as e:
                self.mark_dead(m, str(e))
                last = e
                _log.warning("[AI] 模型 %s 不可用：%s", m, str(e))

        tried = list(chain) + extra
        raise RuntimeError(
            "已尝试 " + str(len(tried)) + " 个模型（" + "、".join(tried) + "）均不可用。"
            "最后一个失败原因：" + (str(last) if last else "未知") + _model_config_hint()
        )


def _now_hhmmss() -> str:
    """``LocalTime.now().withNano(0)`` 的等价物：``HH:MM:SS``。"""
    return time.strftime("%H:%M:%S", time.localtime())


def _model_config_hint() -> str:
    return (
        " 处理办法：在「AI Agent 智能分析」页的【模型服务】卡片里，从「账号可用模型」下拉直接选一个"
        "（列表由 GET /v3/models 只读拉取，不消耗额度），或设环境变量 AI_MODEL（主模型）/"
        " AI_MODEL_CANDIDATES（逗号分隔的降级候选）后重启后端。"
    )
