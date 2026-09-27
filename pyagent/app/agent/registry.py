"""多智能体分工表 —— 逐行移植自 ``AgentRegistry.java``。

为什么收敛到 5 个 Agent、为什么用「工具分组」而不是「多拆 Agent」、
为什么路由不交给模型 —— 理由见 Java 侧注释，这里不重复，只保留代码与关键约束。

**移植时的两个保真点**（很容易做错、做错了不会报错但行为会漂）：

1. **大小写折叠**：Java 用 ``toLowerCase(Locale.ROOT)``，它在 ASCII 之外**不做**特殊折叠
   （土耳其语 I 之类的坑它没有）。Python 的 ``str.lower()`` 语义更宽但不会影响
   这里的中文/ASCII 信号词，所以直接用 ``.lower()`` 即可；两边对同一批信号词的结果一致。
2. **信号词命中的是子串**，不是分词。所以"流失"既会命中"客户流失"也会命中"流失率" ——
   这是**有意的**（中文没有词边界，用分词反而会漏）。移植时保持 ``in`` 判断，不要"顺手优化"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..core.logging import get_logger

log = get_logger("pyagent.agent.registry")


@dataclass(frozen=True)
class ToolBundle:
    """工具分组：一个 Agent 内部的"子专长"。

    :param name: 分组名（写进路由理由，方便排查为什么给了这个工具）
    :param tools: 该分组开放的具名工具（全集来自 ``RiskAgentTools.specs()``）
    :param signals: 命中即开放本组工具的关键词信号（子串匹配）
    """

    name: str
    tools: List[str]
    signals: List[str]


@dataclass(frozen=True)
class AgentDef:
    """一个 Agent 的定义。"""

    id: str
    name: str
    duty: str
    bundles: List[ToolBundle] = field(default_factory=list)
    weight: float = 0.0
    core: bool = False

    def all_tools(self) -> List[str]:
        """该 Agent 名下全部工具（用于页面展示与自检，不代表本轮全开）。

        Java 用 ``LinkedHashSet`` 去重且**保持插入顺序**，故这里同样按顺序去重。
        """
        out: List[str] = []
        seen = set()
        for b in self.bundles:
            for t in b.tools:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return out


#: 兜底工具：无论路由结果如何都保留 —— 避免"专员没被选中导致模型无工具可用"的饿死情况。
BASE_TOOLS: List[str] = [
    "get_company_profile",
    "get_metrics",
    "get_risk_events",
    "search_knowledge",
]

_AGENTS: List[AgentDef] = [
    AgentDef(
        id="orchestrator",
        name="风控总监",
        duty=(
            "拆解问题、指派专员、决定要不要联网与是否提议动作，并把各路证据汇总成一份风控结论；"
            "同时负责成稿复核（有没有编造证据里不存在的数据、引用是否悬空）、"
            "合规护栏（敏感凭证、个人信息、越权表述命中即降级）、报告导出，"
            "以及在事件驱动场景下的主动预警哨兵职责（受配额约束，默认不自动消耗模型额度）。"
        ),
        bundles=[],
        weight=0,
        core=True,
    ),
    AgentDef(
        id="internal-analyst",
        name="内部数据专员",
        duty=(
            "负责本公司内部结构化数据的取数与解读：企业档案、经营指标、风险事件、客户投诉、竞品态势。"
            "只讲数据口径范围内的结论，查不到就说查不到。"
        ),
        bundles=[
            ToolBundle(
                "企业档案",
                ["get_company_profile"],
                [
                    "基本情况", "公司简介", "企业简介", "客户等级", "所属行业", "所在的地区",
                    "区域", "规模", "背景", "是什么公司",
                ],
            ),
            ToolBundle(
                "经营指标",
                ["get_metrics"],
                [
                    "营收", "收入", "利润", "毛利", "净利", "指标", "续费", "留存", "流失",
                    "客户数", "现金流", "成本", "费用", "客单价", "同比增长", "同比", "环比",
                    "增长率", "毛利率", "增长", "下滑", "下降", "经营状况", "业绩", "趋势",
                ],
            ),
            ToolBundle(
                "风险事件",
                ["get_risk_events"],
                [
                    "风险事件", "预警", "告警", "预警事件", "风险等级", "阈值", "触发",
                    "处置状态", "高风险", "中风险", "未处置", "待处置", "风险清单", "风险点",
                    "爆雷", "逾期",
                ],
            ),
            ToolBundle(
                "客户投诉",
                ["get_complaints"],
                [
                    "投诉", "客诉", "抱怨", "不满", "售后", "SLA", "满意度", "退费", "退款",
                    "重复投诉", "客户流失", "流失率", "流失原因", "客户体验", "服务态度",
                ],
            ),
            ToolBundle(
                "竞品态势",
                ["get_competitors"],
                [
                    "竞品", "竞争对手", "同行", "友商", "竞争", "市场份额", "价格战",
                    "对标", "同业", "竞争格局",
                ],
            ),
        ],
        # 内部数据是主力证据源，但单条信号的权重略低于外部/处置：
        # 避免"只要提到一个指标词就把外部检索、动作提案全挤掉"。
        weight=0.8,
    ),
    AgentDef(
        id="knowledge-agent",
        name="资料库检索专员",
        duty=(
            "负责本地资料库的语义 + 关键字混合检索：制度、SOP、处置预案、合规要求等知识库文档，"
            "以及经营指标、风险事件、客户投诉、竞品、企业档案、风控规则与聚合统计，"
            "产出可被引用核对的「来源ID」。适合需要跨模块找线索的问题。"
        ),
        bundles=[
            ToolBundle(
                "本地资料库",
                ["search_knowledge"],
                [
                    "制度", "规定", "办法", "手册", "指引", "SOP", "预案", "流程", "规范",
                    "知识库", "内部文档", "历史案例", "既往", "合规", "标准要求", "官方文档",
                    "资料", "有没有", "类似", "之前", "哪些",
                ],
            )
        ],
        weight=1.0,
    ),
    AgentDef(
        id="web-researcher",
        name="外部情报员",
        duty=(
            "负责联网检索公开资料：政策监管、行业行情、公开舆情。检索词会被清洗与相关性回检后才能使用；"
            "单次分析最多 4 次检索，且默认不自动消耗模型额度。"
        ),
        bundles=[
            ToolBundle(
                "联网检索",
                ["web_search"],
                [
                    "政策", "监管", "法规", "新规", "法条", "行情", "市场情况", "公开数据",
                    "新闻", "舆情", "外部", "行业报告", "最新动态", "红头", "海关",
                    "税务政策", "出口退税", "standard", "据说", "网传",
                ],
            )
        ],
        # 联网成本最高，需要更强的信号才启用。
        weight=1.1,
    ),
    AgentDef(
        id="action-officer",
        name="处置审批专员",
        duty=(
            "负责把结论转成可执行的产出：建工单、通知责任人、变更事件状态，或按要求导出 PDF/Word 报告；"
            "所有动作只提出、不执行，一律送人工审批。"
        ),
        bundles=[
            ToolBundle(
                "动作提案",
                [
                    "propose_create_ticket", "propose_notify_owner",
                    "propose_update_event_status",
                ],
                [
                    "怎么办", "建议怎么", "处置", "整改措施", "跟进", "派单", "建单", "工单",
                    "通知责任人", "责任人", "关闭事件", "变更状态", "催办", "整改",
                    "下一步动作", "给出建议", "措施建议",
                ],
            ),
            # 报告导出没有工具（导出走的是独立链路），所以命中它**不占专员名额**。
            ToolBundle("报告导出", [], ["导出", "下载", "生成报告", "word", "pdf", "报告文件"]),
        ],
        weight=1.0,
    ),
]


class AgentRegistry:
    """分工表查询。无状态，可以安全地当单例用。"""

    BASE_TOOLS = BASE_TOOLS

    def all(self) -> List[AgentDef]:
        """全部 Agent（供页面展示与自检）。故意返回**同一份列表对象**，
        与 Java 的 ``return AGENTS`` 一致 —— 调用方不该改动它。"""
        return _AGENTS

    def get(self, agent_id: str) -> Optional[AgentDef]:
        for a in _AGENTS:
            if a.id == agent_id:
                return a
        return None

    @staticmethod
    def match_score(agent: AgentDef, lower_question: str) -> float:
        """命中判定：逐分组统计命中数，换算成加权得分。

        单个分组内命中数上限取 3，避免"重复堆砌同义词"把分数刷高、挤掉其他专员。

        :return: 0 表示未命中
        """
        if not agent.bundles or not lower_question or not lower_question.strip():
            return 0.0
        total = 0.0
        for b in agent.bundles:
            hit = AgentRegistry.count_hit(b.signals, lower_question)
            if hit > 0:
                total += agent.weight * min(hit, 3)
        return total

    @staticmethod
    def match_bundles(agent: AgentDef, lower_question: str) -> Dict[str, List[str]]:
        """本次应当开放的工具：只给命中的分组。

        :return: 命中的分组名 → 该分组的工具（保持插入顺序）
        """
        out: Dict[str, List[str]] = {}
        if not agent.bundles or not lower_question or not lower_question.strip():
            return out
        for b in agent.bundles:
            if AgentRegistry.count_hit(b.signals, lower_question) > 0:
                out[b.name] = b.tools
        return out

    @staticmethod
    def count_hit(signals: Sequence[str], lower_question: str) -> int:
        """统计 signals 中有多少个出现在问题里（大小写不敏感）。"""
        if not signals or lower_question is None:
            return 0
        hit = 0
        for s in signals:
            if s and s.lower() in lower_question:
                hit += 1
        return hit

    @staticmethod
    def describe(agent: AgentDef) -> Dict[str, object]:
        """导出给前端的精简信息。字段名与 Java 侧 ``LinkedHashMap`` 完全一致。"""
        return {
            "id": agent.id,
            "name": agent.name,
            "duty": agent.duty,
            "tools": agent.all_tools(),
            "bundles": [b.name for b in agent.bundles],
            "core": agent.core,
        }
