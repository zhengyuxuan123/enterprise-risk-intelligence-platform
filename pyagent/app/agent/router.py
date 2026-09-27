"""专员路由 —— 逐行移植自 ``AgentRouter.java``。

**零 token** 地决定这次提问要派哪几个 Agent 上场。为什么不让模型自己挑：
让模型选 Agent 等于为一件高度确定的事额外付一次推理，而且不可复现
（同一个问题今天派张三明天派李四，出了问题没法复盘）。

三层保险，保证不会因为路由漏判而"无工具可用"：

1. 每个专员按关键词信号打分，取前 ``MAX_SPECIALISTS`` 个；
2. 一个信号都没命中时落到默认专员组；
3. 无论如何都附加 ``AgentRegistry.BASE_TOOLS`` 这几个低成本内部只读工具。

**注意：Agent 少 ≠ 工具全开** —— 工具裁剪下沉到 Agent 内部的「工具分组」，
只有命中的分组才会把它的工具放进本轮白名单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.logging import get_logger
from .registry import _AGENTS, AgentDef, AgentRegistry

log = get_logger("pyagent.agent.router")


@dataclass
class Route:
    """路由结果。

    :param agents: 参与本次分析的 Agent（核心角色 + 命中的专员）
    :param tools: 本轮工具白名单（兜底工具 ∪ 命中分组的工具）
    :param scores: 各专员得分
    :param reasons: 每条判定依据（页面与日志可查）
    :param opened_tools: 每个专员本轮实际被开放的工具（可能少于其名下全部工具）
    :param force_web: 用户**显式要求**联网（"请联网核实""必须查公开资料"）。
        与「候补开放」的区别在于：候补只是把工具放进白名单、由模型自行决定用不用；
        force_web 还会让提示词硬性要求模型先调用 ``web_search``，
        并在评估里作为一条可判定的契约。
    """

    agents: List[AgentDef] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    reasons: Dict[str, str] = field(default_factory=dict)
    opened_tools: Dict[str, List[str]] = field(default_factory=dict)
    force_web: bool = False
    #: 本次分析是否涉及联网（force_web 或白名单里含 web_search），由 ``__post_init__`` 填充。
    needs_web: bool = False

    def __post_init__(self) -> None:
        # needs_web 曾经只被 getattr(route, "needs_web", False) 读到 —— 而 Route 上
        # 根本没有这个属性，于是留痕里的 needsWeb 永远是 false，"这次该不该联网"看不出来。
        # 现在收敛成一个真实属性：要么用户点名要联网，要么白名单里确实有联网工具。
        self.needs_web = bool(self.force_web or "web_search" in (self.tools or []))

    def needs(self, agent_id: str) -> bool:
        return any(a.id == agent_id for a in self.agents)


class AgentRouter:
    """规则路由。无状态（策略从配置读入构造参数），可当单例。"""

    #: 单次最多派几个非核心专员（太多等于没有分工，还会把工具清单重新撑大）。
    MAX_SPECIALISTS = 2

    #: 外部对照意图信号：比 web-researcher 那组「政策/监管」宽，
    #: 但不至于宽到「提到指标就想上网」。命中任一条即把联网作为候补开放。
    EXTERNAL_HINT_SIGNALS: List[str] = [
        # 明确要求联网 / 外部
        "联网", "上网", "网上", "搜索一下", "查一下公开", "外部", "公开", "公开数据", "公开信息",
        # 横向对照
        "行业", "同业", "同行", "同行做法", "业界", "标杆", "最佳实践", "惯例", "通用做法",
        "对照", "对标", "横向", "benchmark", "其它公司", "其他公司", "市面上",
        # 时效 / 动态
        "最新", "近期", "当下", "目前市场", "趋势如何", "市场情况", "行情", "新闻", "动态",
        # 与内部数据做印证
        "是否普遍", "普遍吗", "行业平均", "平均水平", "行业水平", "正常吗", "算高吗",
    ]

    #: **强指令**信号：用户不只是"提了个带有外部意味的问题"，而是明确下达了
    #: 「去联网」这个动作。命中即 ``force_web``，与上面那组"候补开放"有本质区别 ——
    #: 候补只是把 ``web_search`` 放进白名单交给模型自行取舍（模型完全可以不调，
    #: 然后写一句"本次未联网核实"交差），强指令则要求提示词把「必须调用」讲死。
    #:
    #: 词表刻意只用**意图明确的多字短语**，不收录"联网""外部""公开"这类单词：
    #: 它们已经在上面的候补表里，再收进来会让"是否联网？"这种元问题也触发强制联网。
    FORCE_WEB_SIGNALS: List[str] = [
        "必须联网", "一定要联网", "务必联网", "强制联网", "需要联网", "要去联网",
        "请联网", "帮我联网", "联网查", "联网搜", "联网核实", "联网查询", "联网检索",
        "上网查", "上网搜", "网上查一下", "网上搜一下",
        "查一下公开", "查公开资料", "查外部资料", "找公开资料",
        "只信外部", "只看外部", "必须查外部", "必须用外部",
    ]

    #: Natural-language opt-out. A negative phrase contains words such as
    #: "联网" too, so checking positive keywords alone reverses user intent.
    NO_WEB_SIGNALS: List[str] = [
        "不要联网", "不用联网", "无需联网", "不需要联网", "禁止联网",
        "不要上网", "不用上网", "无需上网", "仅使用内部", "只用内部",
        "不要查询外部", "不查外部", "不要外部资料",
    ]

    def __init__(self, registry: Optional[AgentRegistry] = None, offer_policy: str = "signal"):
        self.registry = registry or AgentRegistry()
        #: 候补开放策略的**原始字符串**，与 Java 侧 ``@Value("${app.web-search.always-offer:signal}")``
        #: 对齐：认不出来的写法一律按 signal，避免写成错别字就退回"永不联网"。
        self._always_offer_raw = offer_policy

    # ------------------------------------------------------------------ 主入口

    def route(self, question: Optional[str], allow_web: bool,
              force_web_requested: bool = False) -> Route:
        """按问题路由。

        :param question: 用户原始提问
        :param allow_web: 全局联网开关（关掉时即便命中外部专员也不给 web_search）
        :param force_web_requested: 调用方**显式**要求联网（前端「必须联网」开关）。
            与词表识别互为补充：开关不依赖用户怎么说，词表不依赖前端是否传参。
            两者任一为真即视为强指令。
        """
        q = (question or "").lower()
        no_web = bool(self._hit_no_web(q)) and not force_web_requested
        web_allowed = bool(allow_web and not no_web)

        scores: Dict[str, float] = {}
        reasons: Dict[str, str] = {}
        opened: Dict[str, List[str]] = {}

        for a in self.registry.all():
            if a.core or not a.bundles:
                continue
            sc = AgentRegistry.match_score(a, q)
            if sc <= 0:
                continue

            bundles = AgentRegistry.match_bundles(a, q)
            tools_of_agent: List[str] = []
            seen = set()
            for ts in bundles.values():
                for t in ts:
                    if t not in seen:
                        seen.add(t)
                        tools_of_agent.append(t)

            # 只命中"报告导出"这类无工具分组时，不占专员名额 —— 它没有工具可贡献。
            if not tools_of_agent and a.id != "orchestrator":
                log.debug("[AgentRouter] 专员 %s 命中的分组无工具，跳过", a.id)
                continue

            scores[a.id] = sc
            reasons[a.id] = f"命中分组：{'、'.join(bundles.keys())}（信号：{self._hit_signals(a, q)}）"
            opened[a.id] = list(tools_of_agent)

        # 按「得分降序，同分按 id 升序」排序。同分兜底用 id 是必须的：
        # 否则同一个问题在两次运行里可能派出不同的专员组合，路由就不再可复现了。
        order = sorted(scores.keys(), key=lambda k: (-scores[k], k))

        specialists: List[AgentDef] = []
        for agent_id in order:
            if len(specialists) >= self.MAX_SPECIALISTS:
                break
            if not web_allowed and agent_id == "web-researcher":
                continue
            d = self.registry.get(agent_id)
            if d is not None:
                specialists.append(d)

        weak = len(specialists) == 0
        if weak:
            # 一个信号都没命中：给内部数据专员 + 知识库专员（联网开着再补外部情报员），
            # 免得模型只能在"完全没工具"和"全部工具"之间二选一。
            fallback = ["internal-analyst", "knowledge-agent"]
            if web_allowed:
                fallback.append("web-researcher")
            for agent_id in fallback:
                d = self.registry.get(agent_id)
                if d is not None:
                    specialists.append(d)
                    reasons[agent_id] = "未识别出专项意图，按通用风控分析兜底配置"
                    opened[agent_id] = list(d.all_tools())

        # ---- 用户点名要联网：这是硬要求，不是"候补" ----
        force_signals = self._hit_force_web(q)
        if force_web_requested and not force_signals:
            force_signals = ["前端开关：必须联网"]
        force_web = bool(web_allowed and force_signals)

        # ---- 候补补位：让「说了要对照外部、但没说政策/行情」的提问也能联网 ----
        # 上面两条路径都只在「命中强外部信号」或「一个信号都没命中」时才可能带出 web_search。
        # 真实提问大多两者都不是：问「客户流失为什么升高，联网对照行业做法」命中了内部专员，
        # 按旧逻辑就和联网彻底绝缘。这里补一刀：命中「外部对照意图」信号时把外部情报员作为
        # 候补加进来（不占专员名额、不影响其他专员的工具）。
        already_offered = any(t == "web_search" for ts in opened.values() for t in ts)
        if web_allowed and not already_offered:
            offer = self._offer_policy()
            hints = self._hit_external_hints(q)
            # force_web 直接压过策略配置：用户都点名了，不能因为 always_offer=never 就不给。
            offer_it = force_web or offer == "always" or (offer == "signal" and len(hints) > 0)
            if offer_it:
                wr = self.registry.get("web-researcher")
                if wr is not None:
                    specialists.append(wr)
                    opened[wr.id] = ["web_search"]
                    if force_web:
                        reasons[wr.id] = (
                            f"用户明确要求联网（命中：{'、'.join(force_signals)}），"
                            "强制开放联网检索，并要求本轮必须真实调用"
                        )
                    else:
                        reasons[wr.id] = (
                            "候补开放：未命中政策/监管等强联网信号，"
                            f"但问题带有外部对照意图（{'、'.join(hints)}），"
                            "保留联网检索能力；仅在确实需要外部公开资料时使用"
                        )

        # 参与本次分析的完整名单 = 核心角色 + 命中的专员
        participants: List[AgentDef] = [a for a in self.registry.all() if a.core]
        participants.extend(specialists)

        # 工具白名单 = 兜底工具 ∪ 各专员命中分组开放的工具
        tools: List[str] = []
        seen_tools = set()
        for t in _AGENTS_BASE_TOOLS:
            if t not in seen_tools:
                seen_tools.add(t)
                tools.append(t)
        for a in specialists:
            for t in opened.get(a.id, []):
                if t not in seen_tools:
                    seen_tools.add(t)
                    tools.append(t)
        if not web_allowed and "web_search" in tools:
            tools.remove("web_search")
        # 强指令兜底：专员名额已满、或 web-researcher 恰好没命中任何分组时，
        # 上面的候补路径会整条跳过 —— 用户点名要联网，工具就必须交出去。
        if force_web and "web_search" not in tools:
            tools.append("web_search")

        log.info(
            "[AgentRouter] 问题「%s」→ 专员：%s；开放工具：%s%s",
            (q[:30] + "…") if len(q) > 30 else q,
            [a.name for a in specialists],
            tools,
            "；【用户要求联网】" if force_web else "",
        )

        return Route(
            agents=participants,
            tools=tools,
            scores=scores,
            reasons=reasons,
            opened_tools=opened,
            force_web=force_web,
        )

    # ------------------------------------------------------------------ 内部

    def _offer_policy(self) -> str:
        """候补开放策略：认不出来的写法一律按 signal。"""
        v = (self._always_offer_raw or "").strip().lower()
        if v in ("always", "true", "on", "1"):
            return "always"
        if v in ("never", "false", "off", "0"):
            return "never"
        return "signal"

    @staticmethod
    def _hit_external_hints(lower_question: Optional[str]) -> List[str]:
        """命中的外部对照意图信号（最多列 4 条，进路由理由）。"""
        hits: List[str] = []
        if not lower_question:
            return hits
        for s in AgentRouter.EXTERNAL_HINT_SIGNALS:
            if s.lower() in lower_question:
                hits.append(s)
        return hits[:4]

    @staticmethod
    def _hit_force_web(lower_question: Optional[str]) -> List[str]:
        """命中的「明确要求联网」短语（最多列 3 条，进路由理由与留痕）。"""
        hits: List[str] = []
        if not lower_question:
            return hits
        for s in AgentRouter.FORCE_WEB_SIGNALS:
            if s.lower() in lower_question:
                hits.append(s)
        return hits[:3]

    @staticmethod
    def _hit_no_web(lower_question: Optional[str]) -> List[str]:
        """Return explicit opt-out phrases; the frontend force switch overrides them."""
        if not lower_question:
            return []
        return [s for s in AgentRouter.NO_WEB_SIGNALS
                if s.lower() in lower_question][:3]

    @staticmethod
    def _hit_signals(agent: AgentDef, lower_question: str) -> str:
        """返回命中的信号词（进 diagnostics，方便日后调规则）。"""
        hits: List[str] = []
        for b in agent.bundles:
            for s in b.signals:
                if s and s.lower() in lower_question:
                    hits.append(s)
        return "、".join(hits[:4])


#: 从 registry 引入兜底工具（放在这里是为了让本模块的 import 结构更直白）。
_AGENTS_BASE_TOOLS: List[str] = AgentRegistry.BASE_TOOLS


# ---------------------------------------------------------------------- 前端视图


def describe_participants(
    agents: List[AgentDef], route: Route
) -> List[Dict[str, object]]:
    """组装"本次参与的智能体"清单（按路由结果顺序）。

    Java 侧这是 ``AgentRegistry.describe(agents, route)`` 的重载。搬到本模块是为了
    避开 ``registry`` 与 ``router`` 的循环导入 —— ``registry`` 不该反过来知道 ``Route``。
    """
    out: List[Dict[str, object]] = []
    for a in agents:
        m = AgentRegistry.describe(a)
        sc = route.scores.get(a.id)
        # Java: Math.round(sc * 100.0) / 100.0 —— 保留两位小数（四舍五入）
        m["score"] = 0.0 if sc is None else round(sc * 100.0) / 100.0
        why = route.reasons.get(a.id)
        if why is not None:
            m["reason"] = why
        opened = route.opened_tools.get(a.id)
        if opened:
            m["openedTools"] = opened
        out.append(m)
    return out
