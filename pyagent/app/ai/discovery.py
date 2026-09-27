"""模型发现：``GET {base}/models`` 只读列举（不消耗 token）。

移植来源：``LlmClient.listAvailableModels(boolean force)``。

三件事值得单独说明，因为它们都是"没写会很难查"的那种：

1. **负缓存**。失败也要记时间戳（``discoveredAttemptAt``），否则每次调用都白打一次
   注定失败的列举请求，把响应时间拖长 —— 而调用它的地方（``/api/ai/models``、
   诊断页）往往是排障时反复刷新的。
2. **只读列举 ≠ 能调用**。方舟的 embedding 模型就是典型：列表里有、直接调 404。
   所以这里只把它当**候选来源**，能不能用仍以真实调用结果为准。
3. **没有 Key 就不发请求**。``isConfigured()`` 为假时直接返回缓存（通常是空列表），
   而不是发一个必然 401 的请求去污染日志。
"""

from __future__ import annotations

import time
from typing import List, Optional

import httpx

from ..core.logging import get_logger
from .keys import strip_trailing_slash

_log = get_logger("ai.discovery")

#: 成功结果的缓存时长。Java: ``MODEL_DISCOVERY_TTL_MS = 10 * 60 * 1000L``
MODEL_DISCOVERY_TTL_MS = 10 * 60 * 1000
#: 失败后的重试间隔（负缓存）。Java: ``MODEL_DISCOVERY_RETRY_MS = 60 * 1000L``
MODEL_DISCOVERY_RETRY_MS = 60 * 1000
#: 列举请求自身的超时。Java 里是 ``Duration.ofSeconds(8)`` —— 比一次推理短得多，
#: 因为它是"锦上添花"的元数据请求，不能把主链路堵住。
DISCOVERY_TIMEOUT_SECONDS = 8


def _trim(s: Optional[str], max_len: int) -> str:
    """Java ``trim(String s, int max)``：null 安全 + 超长截断。"""
    if s is None:
        return ""
    return s[:max_len] if len(s) > max_len else s


class ModelDiscovery:
    """账号可用模型的只读列举 + 缓存。

    线程安全性：Java 侧靠 ``synchronized`` / 字段可见性；这里用纯 Python 单进程语义 ——
    FastAPI 的请求处理在同一事件循环里，赋值是原子的，最坏情况是并发时多打一次列举请求，
    后果只是多一次只读调用，**不会**产生费用。故不额外加锁。
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        enabled: bool = True,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = strip_trailing_slash(base_url)
        self.api_key = api_key or ""
        self.enabled = enabled
        self._transport = transport

        self._models: List[str] = []
        self._expire_at: float = 0.0
        self._attempt_at: float = 0.0
        #: 最近一次列举的结果说明，给自检接口用（不是异常，是"为什么是空的"）
        self.last_issue: Optional[str] = None

    # ------------------------------------------------------------ 对外只读视图

    @property
    def cached_models(self) -> List[str]:
        return list(self._models)

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None and self.api_key.strip() != ""

    # ------------------------------------------------------------ 主入口

    def list_available_models(self, force: bool = False) -> List[str]:
        """拉取「本账号实际可用的模型」。

        ``GET /v3/models`` 是只读的权限列举接口：**不消耗 token、不产生推理费用**，
        因此可以被自动调用。模型 ID 会随版本变，写死一个名字迟早过期（额度用尽、
        模型下架、没开通都会命中），这个列表才是「能用方舟里所有模型」的真正来源。

        :param force: ``True`` 时跳过节流，强制重新拉一次（页面上"刷新"按钮用）。
        """
        now = time.time() * 1000.0

        if not force:
            if self._models and now < self._expire_at:
                return self._models
            # 失败也要做负缓存：否则每次调用都白打一次注定失败的列举请求
            if now - self._attempt_at < MODEL_DISCOVERY_RETRY_MS:
                return self._models

        if not self.is_configured:
            self.last_issue = "未配置 API Key，跳过模型列举（不会发请求）"
            return self._models

        self._attempt_at = now
        url = f"{self.base_url}/models"
        try:
            with httpx.Client(
                timeout=DISCOVERY_TIMEOUT_SECONDS,
                transport=self._transport,
            ) as c:
                resp = c.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
            if resp.status_code >= 400:
                self.last_issue = f"HTTP {resp.status_code}：{_trim(resp.text, 160)}"
                _log.warning(
                    "[AI] 拉取可用模型列表失败：HTTP %s %s",
                    resp.status_code,
                    _trim(resp.text, 160),
                )
                return self._models

            # 延迟导入避免循环依赖（model_catalog 不认识 discovery）
            from .model_catalog import parse_model_ids

            ids = parse_model_ids(resp.text)
            if ids:
                self._models = ids
                self._expire_at = now + MODEL_DISCOVERY_TTL_MS
                self.last_issue = None
                _log.info("[AI] 发现账号可用模型 %s 个：%s", len(ids), ids)
            else:
                self.last_issue = "响应里没有解析出任何模型 ID（结构可能变了）"
        except Exception as e:  # noqa: BLE001 - 任何异常都不该让主链路崩
            self.last_issue = f"{type(e).__name__}：{e}"
            _log.warning("[AI] 拉取可用模型列表失败：%s", e)
        return self._models


class StaticDiscovery(ModelDiscovery):
    """离线实现：直接用给定列表，不发任何网络请求。

    用途是**对账** —— 拿 Java 版 ``/api/ai/models`` 返回的 ``available`` 列表
    喂给 Python 的排序逻辑，比较 ``recommended`` 是否逐项相等。
    这样可以在完全确定性的条件下验证打分排序，不受网络波动影响。
    """

    def __init__(self, model_ids: List[str]) -> None:
        super().__init__(base_url="", api_key="", enabled=False)
        self._models = list(model_ids)
        self._expire_at = float("inf")

    def list_available_models(self, force: bool = False) -> List[str]:  # noqa: ARG002
        return list(self._models)

    @property
    def is_configured(self) -> bool:
        return True
