"""工具执行上下文（对应 Java ``ToolContext``）。

承载三样东西：

1. **本次分析的全局联网来源编号**。一次分析里模型可能多次调用 web_search，
   而底层检索每次返回的序号都从 1 开始，直接用会让正文里的 ``[n]`` 指向多条不同来源。
   这里按 URL 分配全局唯一且稳定的序号（同一 URL 复用首次分配的序号）。
2. **联网检索预算闸门**。没有闸门时会发生什么：实测一轮提问里模型发了 **11 次**联网检索
   （16 秒内 7 次），每拿到垃圾结果就换个词重搜——既拖慢响应，也把来源池越搅越浑。
3. **规模指标**（高风险事件数、超 SLA 投诉数）。Agent 的最终正文要等模型写完才知道风险等级，
   但用户在流式等待时最想知道的恰恰是"这次到底严不严重"。工具执行完就能拿到这些确定的统计量，
   于是可以把「风险等级 / 来源构成」先推给前端，不等正文。
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional


class ToolContext:
    """线程安全：工具可能在线程池里并发执行（Java 版用 synchronized 同理）。"""

    def __init__(
        self,
        company_id: Optional[int] = None,
        top_k: int = 0,
        user_id: Optional[int] = None,
        scope: Optional[Callable[[int], bool]] = None,
        question: Optional[str] = None,
        max_web_calls: int = 4,
    ) -> None:
        self.company_id = company_id
        self.top_k = top_k
        self.user_id = user_id
        #: 数据权限判定回调；``None`` 表示不做越权检查（批处理 / 评估脚本）
        self.scope = scope
        self.question = question
        self.max_web_calls = max_web_calls
        self.rag_diagnostics: Optional[Dict[str, object]] = None

        self._lock = threading.RLock()
        self._web_index_by_url: Dict[str, int] = {}
        self._web_seq = 0
        self._web_call_count = 0
        self._web_trails: List[Dict[str, object]] = []
        self._stats: Dict[str, int] = {}

    # -- 联网预算 ----------------------------------------------------

    def try_consume_web_call(self) -> bool:
        with self._lock:
            if self._web_call_count >= self.max_web_calls:
                return False
            self._web_call_count += 1
            return True

    def refund_web_call(self) -> None:
        """回滚一次计数（仅在检索被前置规则驳回时用，避免"没真的搜"也算一次）。"""
        with self._lock:
            if self._web_call_count > 0:
                self._web_call_count -= 1

    @property
    def web_call_count(self) -> int:
        with self._lock:
            return self._web_call_count

    # -- 联网来源编号 ------------------------------------------------

    def web_index_for(self, url: Optional[str]) -> tuple[int, int]:
        """取该 URL 的全局来源序号；首次出现则分配新序号。

        返回 ``(序号, 是否本次新分配)``，1 = 新分配、0 = 复用。
        """
        key = (url or "").strip()
        with self._lock:
            exist = self._web_index_by_url.get(key)
            if exist is not None:
                return exist, 0
            self._web_seq += 1
            self._web_index_by_url[key] = self._web_seq
            return self._web_seq, 1

    def web_source_count(self) -> int:
        with self._lock:
            return self._web_seq

    def web_urls_seen(self) -> List[str]:
        with self._lock:
            return list(self._web_index_by_url.keys())

    # -- 检索痕迹 ----------------------------------------------------

    def add_web_trail(self, trail: Dict[str, object]) -> None:
        with self._lock:
            self._web_trails.append(trail)

    def web_trails(self) -> List[Dict[str, object]]:
        with self._lock:
            return list(self._web_trails)

    # -- 规模指标 ----------------------------------------------------

    def stat_max(self, key: Optional[str], value: int) -> None:
        """记录一个统计量；同一 key 多次调用取最大值。

        取最大而不是累加，是为了防止模型反复调用同一工具把计数累加虚高。
        """
        if key is None:
            return
        with self._lock:
            old = self._stats.get(key)
            if old is None or value > old:
                self._stats[key] = int(value)

    def stats_snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def rule_risk_level(self) -> Optional[str]:
        """由已取到的统计证据直接判定风险等级。

        证据不足时返回 ``None``，让上层继续等模型结论，而不是凭空给个等级。
        """
        with self._lock:
            s = self._stats
            high = s.get("riskHigh", 0)
            mid = s.get("riskMedium", 0)
            sla = s.get("complaintSla", 0)
            churn = s.get("complaintChurnHigh", 0)
            has_any = bool(high or mid or sla or churn
                           or "riskTotal" in s or "complaintTotal" in s)
            if high:
                return "HIGH"
            if mid or sla or churn:
                return "MEDIUM"
            return "LOW" if has_any else None
