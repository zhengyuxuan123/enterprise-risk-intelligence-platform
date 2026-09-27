"""稳定性三件套（对应 Java ``Resilience``）：分桶熔断 + 半开探测 + 请求单飞。

分桶为什么按「企业 + 用户」
--------------------------
一次模型故障通常是**账号级或底座级**的（欠费、限流、模型下线），
按请求 URL 分桶毫无意义；按「企业 + 用户」分桶才能做到
「这一个租户的调用被熔断了，别人的分析照样能跑」。

单飞（single-flight）
--------------------
同一个用户、同一家企业、同一个问题在短时间内重复提交，
应当复用同一次执行而不是并发跑两遍——既省额度，也避免用户看到两份不同的答案。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Optional, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _Bucket:
    failures: int = 0
    opened_at: Optional[float] = None
    half_open_until: Optional[float] = None


class CircuitBreaker:
    """分桶熔断 + 半开探测。

    :param threshold: 连续失败几次后跳闸。
    :param cooldown_seconds: 跳闸后多久进入半开（放一个探测请求）。
    """

    def __init__(self, threshold: int = 2, cooldown_seconds: int = 120) -> None:
        self.threshold = max(1, int(threshold))
        self.cooldown = max(1, int(cooldown_seconds))
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.RLock()
        #: 与 Java ``Resilience.snapshot()`` 同源的三个累计量：
        #  opened=跳闸次数 / closed=半开成功后关闭次数 / merged=单飞合并掉的重复请求数。
        self.opened_total = 0
        self.closed_total = 0

    def _key(self, company_id: Optional[int], user_id: Optional[int]) -> str:
        return f"{company_id}:{user_id}"

    def allows(self, company_id: Optional[int], user_id: Optional[int]) -> bool:
        b = self._buckets.get(self._key(company_id, user_id))
        if b is None or b.opened_at is None:
            return True
        now = time.time()
        if now - b.opened_at >= self.cooldown:
            # 半开：放行一个探测请求
            return True
        return False

    def note_failure(self, company_id: Optional[int], user_id: Optional[int]) -> None:
        with self._lock:
            b = self._buckets.setdefault(self._key(company_id, user_id), _Bucket())
            b.failures += 1
            if b.failures >= self.threshold and b.opened_at is None:
                b.opened_at = time.time()
                self.opened_total += 1
                log.warning("[Resilience] 熔断开启 company=%s user=%s（连续失败 %s 次）",
                            company_id, user_id, b.failures)

    def note_success(self, company_id: Optional[int], user_id: Optional[int]) -> None:
        with self._lock:
            k = self._key(company_id, user_id)
            b = self._buckets.get(k)
            if b is None:
                return
            if b.opened_at is not None:
                log.info("[Resilience] 半开探测成功，熔断关闭 company=%s user=%s", company_id, user_id)
            self._buckets.pop(k, None)
            self.closed_total += 1

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        """逐桶明细。``Resilience.snapshot()`` 要的是**汇总**口径，别拿这个直接对外。"""
        with self._lock:
            return {k: {"failures": v.failures, "opened": v.opened_at is not None}
                    for k, v in self._buckets.items()}

    def open_count(self) -> int:
        with self._lock:
            return len(self._buckets)


@dataclass
class _Flight(Generic[T]):
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[T] = None
    error: Optional[BaseException] = None


class SingleFlight:
    """同 key 的并发请求只执行一次，其余等待同一份结果。"""

    def __init__(self, wait_seconds: int = 180) -> None:
        self.wait_seconds = max(1, int(wait_seconds))
        self._flights: Dict[str, _Flight] = {}
        self._lock = threading.RLock()
        #: 被"搭便车"合并掉的重复请求数（Java: merged）
        self.merged_total = 0

    def run(self, key: str, fn: Callable[[], T]) -> T:
        with self._lock:
            f = self._flights.get(key)
            if f is None:
                f = _Flight()
                self._flights[key] = f
                owner = True
            else:
                owner = False
                self.merged_total += 1

        if not owner:
            f.done.wait(self.wait_seconds)
            if f.error is not None:
                raise f.error
            if f.done.is_set():
                return f.result  # type: ignore[return-value]
            raise TimeoutError(f"等待同 key 请求超时（{self.wait_seconds}s）")

        try:
            r = fn()
            f.result = r
            return r
        except BaseException as e:  # noqa: BLE001 - 要把异常传给等待方
            f.error = e
            raise
        finally:
            f.done.set()
            with self._lock:
                self._flights.pop(key, None)

    def in_flight(self) -> int:
        with self._lock:
            return len(self._flights)


class Resilience:
    """熔断 + 单飞的组合入口。"""

    def __init__(self, threshold: int = 2, cooldown_seconds: int = 120,
                 single_flight: bool = True, wait_seconds: int = 180) -> None:
        self.breaker = CircuitBreaker(threshold, cooldown_seconds)
        self.flight = SingleFlight(wait_seconds)
        self.single_flight_enabled = bool(single_flight)

    def allows(self, company_id: Optional[int], user_id: Optional[int]) -> bool:
        return self.breaker.allows(company_id, user_id)

    def guards(self, key: str, company_id: Optional[int], user_id: Optional[int],
               fn: Callable[[], T]) -> T:
        if not self.allows(company_id, user_id):
            raise RuntimeError("该账号的分析调用已被熔断（连续失败过多），请稍后重试或更换底座配置")
        if not self.single_flight_enabled:
            return self._guarded(company_id, user_id, fn)
        return self.flight.run(key, lambda: self._guarded(company_id, user_id, fn))

    def _guarded(self, company_id: Optional[int], user_id: Optional[int], fn: Callable[[], T]) -> T:
        try:
            r = fn()
        except Exception:
            self.breaker.note_failure(company_id, user_id)
            raise
        self.breaker.note_success(company_id, user_id)
        return r

    def snapshot(self) -> Dict[str, object]:
        """与 Java ``Resilience.snapshot()`` **逐字段同名**（前端体检页按这些键渲染）。

        早期这里返回的是 ``{"buckets": {...}, "inFlight": n}`` —— 结构自洽、读起来也合理，
        但转发过去之后体检页的"熔断/半开"那几行会**整块消失**（键名对不上，前端取到 undefined）。
        这正是迁移里最危险的失败模式：接口在、行为不对。
        """
        return {
            "threshold": self.breaker.threshold,
            "cooldownSeconds": self.breaker.cooldown,
            "openBuckets": self.breaker.open_count(),
            "openedTotal": self.breaker.opened_total,
            "closedTotal": self.breaker.closed_total,
            "singleFlight": self.single_flight_enabled,
            "mergedTotal": self.flight.merged_total,
            "inflight": self.flight.in_flight(),
        }
