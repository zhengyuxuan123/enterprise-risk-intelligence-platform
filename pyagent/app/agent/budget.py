"""分析预算 —— 逐行移植自 ``AgentBudget.java``。

把「一次分析最多允许多贵」变成可声明、可复现的硬约束。约束三件事：
生成多少字（最大的杠杆）、调几次模型、单个工具被调用几次；再加一道挂钟兜底。

**三笔账分开算，这是两次线上事故换来的**（详见 2026-09-21 的 QA 文档 §8.2）：

1. **模型调用**：受 ``per_tool_limit`` + ``total_tool_limit`` 约束；
2. **预取证**：受 ``prefetch_limit`` 约束，**不占用**模型的总配额。
   合在一起算时，快答档总额 5 次会被预取证一次用满，之后模型一次都调不动，
   而预取的证据又被「配额到点即停」整批丢掉 —— 两头落空；
3. **写动作**（``propose_*``）：受 ``write_tool_limit`` 约束，
   也不占用只读取数的总配额（否则"该不该建工单"这个最有价值的判断会被前面的取数挤掉）。

这里的每一次"制止"都应写进 ``degrade_reason``，界面上表现为 PARTIAL 降级。
**宁可交付一份短而完整的结论，也不要交付一份没人等得起的鸿篇。**
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Depth(Enum):
    """分析深度档位。"""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"

    @property
    def key(self) -> str:
        return self.value


@dataclass(frozen=True)
class Plan:
    """每档预算。字段全部不可变 → 同一个 question + depth 永远得到同一种执行。"""

    depth: Depth
    #: 允许的工具轮数（round trip 上限）。
    max_iterations: int
    #: 正文预算（字）。写进 system 提示约束模型，不是事后截断。
    answer_max_chars: int
    #: 单个工具最多被调用几次。
    per_tool_limit: int
    #: 本轮分析的工具调用总数上限。
    total_tool_limit: int
    #: 挂钟预算（毫秒）：到点后下一轮不再给工具，直接成稿。
    time_budget_ms: int
    #: 联网检索次数上限（成本最高，单独控）。
    web_search_limit: int
    #: 预取证能取几个工具（走独立配额，见模块文档）。
    prefetch_limit: int
    #: 写动作（``propose_*``）的次数上限，与只读取数分账。
    write_tool_limit: int
    #: 是否走「预取证 + 单轮成稿」。
    single_round: bool
    #: 是否跳过「结构化抽取」那一次独立模型调用。
    skip_structure_extract: bool


#                       轮数 正文字数 单工具 总工具 挂钟(ms)   联网 预取证 写动作 单轮  跳抽取
QUICK = Plan(Depth.QUICK, 2, 700, 1, 5, 75_000, 1, 4, 2, True, True)
STANDARD = Plan(Depth.STANDARD, 3, 1500, 2, 9, 240_000, 3, 5, 3, False, False)
DEEP = Plan(Depth.DEEP, 6, 3600, 3, 16, 600_000, 5, 6, 4, False, False)


def plan_of(depth: Optional[str]) -> Plan:
    """解析深度档位；认不出来（含 None / 空串）时按 STANDARD。

    **绝不退化成"无限深"** —— 传错参数只会得到默认档，不会失去上界。
    """
    if depth is not None and depth.strip():
        d = depth.strip().lower()
        #! 逐字对齐 Java 的匹配方式，**不要"顺手统一成前缀匹配"**：
        #! quick 是前缀（"quicker"/"quickish?" 都算），但 **fast 是全等** ——
        #! Java 原文是 `d.startsWith("quick") || d.equals("fast")`。
        #! 若把 fast 也改成前缀，"fastest" 就会落到 QUICK 而 Java 落到 STANDARD，
        #! 两边同一份配置跑出不同档位，双跑对照直接失真。
        if d.startswith("quick") or d == "fast":
            return QUICK
        if d.startswith("deep") or d.startswith("full"):
            return DEEP
        if d.startswith("standard") or d.startswith("normal"):
            return STANDARD
    return STANDARD


class AgentBudget:
    """运行时记账。一次分析一个实例，**不是线程安全的**。"""

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self._started_at_ms = int(time.time() * 1000)
        #: 模型发起的工具调用计数（预取证不计入 —— 它和模型抢的是两笔不同的账）。
        self._tool_calls = 0
        #: 预取证调用计数（平台自己发起）。
        self._prefetch_calls = 0
        #: 写动作调用计数，与只读取数分账。
        self._write_calls = 0

    @staticmethod
    def start(depth_key: Optional[str]) -> "AgentBudget":
        """按深度档位起一本预算。"""
        return AgentBudget(plan_of(depth_key))

    @property
    def plan(self) -> Plan:
        return self._plan

    @property
    def started_at_ms(self) -> int:
        """起点（wall clock 毫秒）。

        :class:`~app.agent.deadline.Deadline` 用它对齐同一个钟 ——
        两边各自计时会出现"预算说没到点、deadline 说到点了"，
        而降级理由一旦自相矛盾，排障时第一反应就是不信它。
        """
        return self._started_at_ms

    def elapsed_ms(self) -> int:
        return int(time.time() * 1000) - self._started_at_ms

    # ------------------------------------------------------------------ 闸门

    def tool_restriction_reason(self) -> Optional[str]:
        """这一轮还允不允许再调工具。

        :return: ``None`` 表示放行；否则返回**必须当面告诉模型的理由**
            （本方法不替模型做决定，只把现实约束讲清楚，让它基于已有证据作答）。
        """
        left = self._plan.time_budget_ms - self.elapsed_ms()
        if left <= 0:
            return (
                f"本次分析的用时预算（{self._plan.time_budget_ms // 1000} 秒）已用尽，"
                "请立即基于已获取的证据给出最终结论，不要再调用任何工具。"
            )
        if self._tool_calls >= self._plan.total_tool_limit:
            return (
                f"本次分析的工具调用总配额（{self._plan.total_tool_limit} 次）已用尽，"
                "请基于已获取的证据给出最终结论。"
            )
        return None

    def try_consume(
        self, tool_name: str, calls_so_far_for_this_tool: int, prefetch: bool = False
    ) -> Optional[str]:
        """尝试记一次工具调用。

        .. note::
           **``calls_so_far_for_this_tool`` 由调用方维护，本类不自增它。**
           Java 侧 ``AgentService.consumeBudget()`` 的写法是::

               int used = toolUsage.getOrDefault(toolName, 0);
               String denied = budget.tryConsume(toolName, used, prefetch);
               toolUsage.put(toolName, used + 1);   // 被拒也照样 +1

           两个**需要原样保留**、且在阶段 5 移植 ``AgentService`` 时必须一起搬的行为：

           1. 被拒绝的那一次**也**会推高计数器；
           2. 预取证与模型调用**共用同一张 ``toolUsage`` 表** —— 也就是说在快答档
              （``per_tool_limit=1``）下，某个工具被预取证用过一次之后，
              模型再请求同一个工具就会撞上「已调用 1 次，达到上限」。

           第 2 条是不是有意的、会不会又变成一次"取证取够了反而失败"，
           要看 ``runTools`` 里 ``callCache`` 与配额检查的先后顺序 —— **阶段 5 必须实测确认**，
           不能靠读代码下结论。这里保留原行为，不擅自"顺手修好"。

        :param prefetch: ``True`` 表示这次调用是平台预取证发起的
        :return: ``None`` 表示放行；否则返回拒绝理由（该工具本轮已用够 / 总配额已满），
            调用方应把理由作为工具结果回灌给模型，而不是让工具真的再跑一遍。
        """
        web = tool_name == "web_search"
        write = bool(tool_name) and tool_name.startswith("propose_")

        if web:
            per_limit = self._plan.web_search_limit
        elif write:
            per_limit = self._plan.write_tool_limit
        elif prefetch:
            per_limit = self._plan.prefetch_limit
        else:
            per_limit = self._plan.per_tool_limit
        #! 【已知的提示语与实现不符，原样保留待你拍板】
        #! 上面这四个上限**全都是"按工具"**的，不是"按类别总量"：
        #! ``used = calls_so_far_for_this_tool``，判定是 ``used >= per_limit``。
        #! 对 web_search 无害（全网只有一个联网工具，按工具 == 总量）。
        #! 但 ``propose_*`` 有 **3 个**工具，于是 ``writeToolLimit=2`` 实际含义是
        #! 「每个提案工具最多 2 次」→ 最多可提 6 个动作，
        #! 而拒绝时的文案写的是「本次分析允许的动作提案（2 个）已提满」——
        #! 字段注释也写着"次数上限"。**三处口径不一致**。
        #! 另外 ``prefetch_limit`` 同时被当作「预取清单长度」（AgentService.buildPrefetchCalls
        #! 用它截断）和「单个工具最多预取几次」，一个字段担了两个语义。
        #! 迁移期**不擅自改**（改了就和 Java 的行为对不上，双跑对照会失真）；
        #! 是否要收敛成真正的"总量"语义，等阶段 5 把 AgentService 搬完之后再定。

        used = calls_so_far_for_this_tool
        if used >= per_limit:
            if web:
                return (
                    f"本次分析允许的联网检索次数（{per_limit} 次）已用尽，请基于已有证据作答；"
                    "若确需外部公开信息佐证，请在「四、不确定性 / 需人工确认」"
                    "写明「本次未联网核实」。"
                )
            if write:
                return (
                    f"本次分析允许的动作提案（{per_limit} 个）已提满，"
                    "请基于已有证据给出最终结论，不要再新增提案。"
                )
            if prefetch:
                return f"预取证条数已达上限（{per_limit} 条）"
            return (
                f"工具 {tool_name} 本次分析已调用 {used} 次，达到上限（{per_limit} 次）。"
                "重复检索同一证据源不会再有新信息，请基于已返回的结果作答。"
            )

        # 写动作与只读取数分账：取数用得多，不能把「该不该建工单」这个判断挤掉
        if not write and not prefetch and self._tool_calls >= self._plan.total_tool_limit:
            return (
                f"本次分析的工具调用总配额（{self._plan.total_tool_limit} 次）已用尽，"
                "请立即基于已获取的证据给出最终结论。"
            )

        if prefetch:
            self._prefetch_calls += 1
        elif write:
            self._write_calls += 1
        else:
            self._tool_calls += 1
        return None

    # ------------------------------------------------------------------ 观测

    def usage_snapshot(self) -> str:
        """供留痕与排障：三笔账各自用了多少。"""
        return (
            f"模型工具 {self._tool_calls}/{self._plan.total_tool_limit}"
            f"，预取证 {self._prefetch_calls}/{self._plan.prefetch_limit}"
            f"，写动作 {self._write_calls}/{self._plan.write_tool_limit}"
        )

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    @property
    def prefetch_calls(self) -> int:
        return self._prefetch_calls

    @property
    def write_calls(self) -> int:
        return self._write_calls
