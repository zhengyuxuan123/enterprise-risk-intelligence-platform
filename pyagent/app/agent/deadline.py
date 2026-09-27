"""时间预算与超时降级（对应「指定时间得不到结果就降级」）。

为什么还要再来一层
------------------
``AgentBudget`` 里**已经**有一份挂钟预算（``Plan.time_budget_ms``：快答 75s /
标准 240s / 深度 600s），但旧链路里它只做了**一件事**：到点后不再给工具
（``tool_restriction_reason``）。缺的三件事才是"等不到结果"的真正来源：

1. **理由没送到模型耳朵里**。旧代码拿到那段「预算已用尽，请立即成稿」的文案后，
   只用它决定 ``round_specs = None``，**文案本身被丢掉了** —— 模型并不知道该收敛，
   于是继续空转，剩下的轮次照样一轮一轮烧下去。
2. **没有"到点就不再发起调用"**。到点之后最贵的行为是**又发了一次注定来不及的模型调用**：
   它既烧额度，又把用户按在等待里，而等到它回来时早已超时。
3. **没有降级出口**。到点后应当立刻落到**零 token** 的本地确定性报告
   （:meth:`AgentService._local_report`），而不是等所有轮次跑完再说。

三态
----
::

    RUNNING ──(用掉 soft_ratio)──▶ CONVERGING ──(预算用尽)──▶ EXPIRED
                                       │                        │
                          不再给工具                    不再发起任何模型调用
                          + 把理由原样交给模型          + 收紧单次调用超时
                          + 收紧单次调用超时            + 直接落本地确定性报告

**超时不是错误，是一次降级交付**。所以这里不抛异常、不返回 500：
用户拿到的是一份短而完整、且明确写清降级原因的结论。
宁可交付一份短而完整的结论，也不要交付一份没人等得起的鸿篇。

成本视角
--------
这一层是**省** token 的，不是多花的：它砍掉的正是那些"注定来不及、回来也用不上"的调用。
真正零成本的兜底（本地确定性报告）反而是它的终点。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

#: 预算充足，正常跑。
RUNNING = "running"
#: 软截止：进入收敛 —— 不再给工具、把理由交给模型、只再跑这一轮。
CONVERGING = "converging"
#: 硬截止：不再发起任何模型调用，直接走零 token 兜底。
EXPIRED = "expired"

#: 用掉预算的多少比例就进入收敛。0.7 意味着"留最后 30% 专心写结论"。
DEFAULT_SOFT_RATIO = 0.7
#: 单次模型调用的时间下限（秒）。再赶也要给它这么多 ——
#: 给 3 秒只会换来一次必定失败的请求，既浪费一次往返又拿不到东西。
DEFAULT_MIN_CALL_SECONDS = 15


class Deadline:
    """一次分析的挂钟预算。三态机，只出不进（时间不会倒流）。

    :param budget_ms: 预算毫秒。``<=0`` 表示**不设限**（该对象恒为 RUNNING，
        所有方法都退化成"不干预"），这样可以放心地把它传进任何链路。
    :param started_at_ms: 起点（wall clock 毫秒）。传 :class:`AgentBudget` 的
        起点可以保证两边看到的是同一个钟 —— 否则会出现
        "预算说没到点、deadline 说到点了"这种互相打脸的降级理由。
    """

    def __init__(self, budget_ms: int, started_at_ms: Optional[int] = None,
                 soft_ratio: float = DEFAULT_SOFT_RATIO,
                 min_call_seconds: int = DEFAULT_MIN_CALL_SECONDS,
                 converge_max_tokens: int = 0) -> None:
        self.budget_ms = max(0, int(budget_ms or 0))
        self._t0_ms = int(started_at_ms if started_at_ms is not None
                          else time.time() * 1000)
        # 边界全部夹一遍：配置写错（负数 / >1 / 0）不该让整条链路失去收敛能力。
        self.soft_ratio = min(0.95, max(0.05, float(soft_ratio or DEFAULT_SOFT_RATIO)))
        self.min_call_seconds = max(1, int(min_call_seconds or DEFAULT_MIN_CALL_SECONDS))
        #: 收敛轮的输出预算（token）。``0`` = 不显式设置（沿用服务商默认），
        #: 默认不动它：收紧会让深度档的长正文被截短，**是否收紧交给配置决定**。
        self.converge_max_tokens = max(0, int(converge_max_tokens or 0))

    # -- 工厂 --------------------------------------------------------

    @classmethod
    def for_plan(cls, plan_time_budget_ms: int, started_at_ms: Optional[int] = None,
                 settings: Any = None) -> "Deadline":
        """按深度档位的挂钟预算起一个 deadline。

        优先级：``APP_AGENT_DEADLINE_ENABLED=false``（整层关掉）
        > ``APP_AGENT_DEADLINE_SECONDS``（显式指定秒数）
        > 档位自带的 ``time_budget_ms``。
        """
        if settings is None:
            from ..config import get_settings  # 延迟导入：避免与配置层的循环依赖

            settings = get_settings()
        agent = getattr(settings, "agent", None)
        if agent is not None and not bool(getattr(agent, "deadline_enabled", True)):
            return cls(0)
        override = int(getattr(agent, "deadline_seconds", 0) or 0) if agent else 0
        ms = override * 1000 if override > 0 else int(plan_time_budget_ms or 0)
        return cls(
            ms,
            started_at_ms=started_at_ms,
            soft_ratio=float(getattr(agent, "deadline_soft_ratio", DEFAULT_SOFT_RATIO))
            if agent else DEFAULT_SOFT_RATIO,
            min_call_seconds=int(getattr(agent, "deadline_min_call_seconds",
                                         DEFAULT_MIN_CALL_SECONDS)) if agent
            else DEFAULT_MIN_CALL_SECONDS,
            converge_max_tokens=int(getattr(agent, "deadline_converge_tokens", 0) or 0)
            if agent else 0,
        )

    # -- 计量 --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.budget_ms > 0

    @property
    def budget_seconds(self) -> int:
        return self.budget_ms // 1000

    @property
    def soft_ms(self) -> int:
        return int(self.budget_ms * self.soft_ratio)

    def elapsed_ms(self) -> int:
        return int(time.time() * 1000) - self._t0_ms

    def elapsed_seconds(self) -> float:
        return self.elapsed_ms() / 1000.0

    def remaining_ms(self) -> Optional[int]:
        """剩余毫秒。不设限时返回 ``None``（不是"无限大"，避免调用方拿它做算术）。"""
        if not self.enabled:
            return None
        return self.budget_ms - self.elapsed_ms()

    def remaining_seconds(self) -> Optional[float]:
        r = self.remaining_ms()
        return None if r is None else r / 1000.0

    # -- 三态 --------------------------------------------------------

    @property
    def expired(self) -> bool:
        r = self.remaining_ms()
        return r is not None and r <= 0

    @property
    def converging(self) -> bool:
        """软截止已过但还没硬截止。"""
        if not self.enabled or self.expired:
            return False
        return self.elapsed_ms() >= self.soft_ms

    def state(self) -> str:
        if not self.enabled:
            return RUNNING
        if self.expired:
            return EXPIRED
        if self.converging:
            return CONVERGING
        return RUNNING

    # -- 给调用方用的两个"怎么收" ------------------------------------

    def call_timeout(self, default_seconds: float) -> Optional[float]:
        """这一轮单次模型调用的超时上限（秒）。

        ``min(服务商默认超时, 剩余时间)``，但不低于 ``min_call_seconds`` ——
        剩下的时间不够跑完一次调用时，宁可让它略超一点预算，
        也不要发一个 3 秒就断、注定拿不到东西的请求。
        不设限时返回 ``None``（沿用客户端默认）。
        """
        if not self.enabled:
            return None
        left = self.remaining_ms()
        if left is None:
            return None
        want = max(self.min_call_seconds, left / 1000.0)
        return float(min(float(default_seconds or 0) or want, want))

    def converge_max_tokens_for(self, default_tokens: int = 0) -> int:
        """收敛轮的输出预算。没配就沿用调用方给的默认值。"""
        return self.converge_max_tokens or int(default_tokens or 0)

    # -- 文案（必须真的送到模型面前，否则收敛只是"我们以为收敛了"）----

    def converge_hint(self) -> str:
        """软截止时追加给模型的收敛指令。"""
        left = self.remaining_seconds()
        left_txt = f"{max(1, int(left))} 秒" if left is not None else "不多"
        return (
            f"【时间预算提醒】本次分析只剩 {left_txt}。"
            "请立即停止调用任何工具，直接基于上面已经取到的证据给出最终结论，"
            "按「一、结论（内部数据）二、结论（外部公开资料）三、建议动作四、不确定性」四节输出；"
            "未联网核实的，第二节原样写「本次未联网核实」。宁可短，不要没写完。"
        )

    def expire_reason(self) -> str:
        """硬截止时的降级理由（写进 ``degrade_reason`` 与留痕）。"""
        return (
            f"分析用时超过预算（{self.budget_seconds} 秒），"
            "已停止后续模型调用并生成本地确定性报告"
        )

    def to_dict(self) -> Dict[str, Any]:
        left_ms = self.remaining_ms()
        return {
            "state": self.state(),
            "enabled": self.enabled,
            "budgetSeconds": self.budget_seconds,
            "elapsedSeconds": round(self.elapsed_seconds(), 2),
            "remainingSeconds": None if left_ms is None else round(left_ms / 1000.0, 2),
            "softRatio": self.soft_ratio,
        }

    def __repr__(self) -> str:  # pragma: no cover - 排障用
        return f"<Deadline {self.state()} {self.elapsed_ms()}ms/{self.budget_ms}ms>"
