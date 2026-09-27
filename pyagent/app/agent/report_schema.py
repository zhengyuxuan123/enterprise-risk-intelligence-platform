"""成稿的**结构化**契约：让模型直接交出四节字段，而不是我们从正文里正则切。

背景：一个已经兑现过两次的失败
------------------------------
界面上的「结论 / 建议动作 / 不确定性」四格，一直是从模型正文里**切**出来的：

.. code-block:: python

    conclusion_internal = _section(draft, "一")   # 找行首「一、结论」
    conclusion_web     = _section(draft, "二")
    uncertainties      = _section(draft, "四")

这是移植期 Java ``AgentService`` 的实现（Java 里对模型输出做字符串切分很常见）。
它的脆弱性已经在 2026-09-23 兑现过两次：

1. 模型把标题写成 ``**一、结论（…）**``（Markdown 加粗）→ 裸前缀匹配不上 →
   **正文 1801 字、四格全空**。接口 200、正文也在，只有界面那一格是白的。
2. 「四」的判据写死成"必须含、结论"，而模板标题是「四、不确定性」→
   不确定性那一格**从来没被填过**，没人发现。

补一个正则分支只能解决"加粗"这一种写法；模型换 ``1. 结论``、``### 一、总结``、
``第一节``，它就再崩一次 —— 而且每次都是**静默**的：正文在、接口 200，
只有前端那一格空着。

通用做法：让模型**自己交出结构**
--------------------------------
主流 Agent 框架的做法是让模型直接产出结构化数据（OpenAI 官方也建议用
function calling / JSON 模式做结构化输出），而不是事后从自由文本里抽。
本模块提供这条通道：

* :data:`STRUCTURED_HINT` —— 追加到 system prompt 的回执要求；
* :func:`extract` —— 从正文里取出回执块并解析（**解析失败返回 None，绝不抛**）；
* :func:`render_markdown` —— 结构化 → 四节正文（模型只给了结构、没给正文时用）。

**回退是设计的一部分**：解析不到结构就退回切分。所以最坏情况等于现状，
启用这条通道不会让事情变得更糟 —— 这一点由
``tests/test_stage14_agent_ergonomics.py`` 的回退用例钉住。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

#: 回执块的围栏。用 ``json`` 语言标签，模型认得它。
_FENCE_RE = re.compile(r"```(?:json|JSON)\s*(\{.*?\})\s*```", re.DOTALL)


class Recommendation(BaseModel):
    """一条建议动作。**形状与旧的 ``_recs()`` 逐字段一致**（前端契约）。"""

    action: str = ""
    basis: str = ""
    refs: List[str] = Field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "basis": self.basis, "refs": list(self.refs)}


class RiskAnalysisReport(BaseModel):
    """一次分析的四节结构。字段全部可空 —— **宽容优先于严格**。

    !!! 不要用必填字段。模型漏填某个字段是常态，把它变成校验错误会导致
    "整份结构都被丢掉、退回切分"，那等于这条通道白开了。宁可某个字段空着。
    """

    #: 一句话综合结论（前端「结论」卡的标题行）
    headline: str = ""
    #: HIGH / MEDIUM / LOW / UNKNOWN
    risk_level: str = ""
    #: 一、结论（基于内部知识库与经营数据）
    conclusion_internal: str = ""
    #: 二、结论（基于外部公开资料）
    conclusion_web: str = ""
    #: 三、建议动作（一）
    recommendations_internal: List[Recommendation] = Field(default_factory=list)
    #: 三、建议动作（二）
    recommendations_web: List[Recommendation] = Field(default_factory=list)
    #: 四、不确定性 / 需人工确认
    uncertainties: List[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}

    @property
    def has_any(self) -> bool:
        """是否交出了**任何**有效结构。

        全空的结构（模型敷衍地给了个 ``{}``）不能算成功 —— 用它覆盖切分结果
        会让四格变成空的，比不开这条通道更糟。
        """
        return bool(
            self.headline.strip() or self.risk_level.strip()
            or self.conclusion_internal.strip() or self.conclusion_web.strip()
            or self.recommendations_internal or self.recommendations_web
            or self.uncertainties)

    def recs_as_dicts(self, internal: bool) -> List[Dict[str, Any]]:
        src = self.recommendations_internal if internal else self.recommendations_web
        return [r.as_dict() for r in (src or []) if (r.action or "").strip()]

    def uncertainties_text(self) -> str:
        return "\n".join(f"{i + 1}. {u}" for i, u in enumerate(
            x for x in (self.uncertainties or []) if str(x).strip()))


#: 追加到 system prompt 的回执要求。
#:
#: 三处措辞是刻意的，改之前想清楚：
#: * 「**可精简**」—— 不要求把正文复制一遍，否则输出 token 翻倍；
#: * 「以正文为准」—— 正文仍是给用户看的东西，结构只是给界面四格用的回执；
#: * 「不要额外说明」—— 模型爱在 JSON 外面再写一句"以下是 JSON"，那会让剥离失败。
STRUCTURED_HINT = (
    "\n【结构化回执】在正文之后另起一段，输出且仅输出一个 ```json 代码块作为回执，形如：\n"
    "```json\n"
    '{"risk_level":"HIGH|MEDIUM|LOW|UNKNOWN",\n'
    ' "headline":"一句话综合结论",\n'
    ' "conclusion_internal":"一、结论（内部）的核心判断，可精简",\n'
    ' "conclusion_web":"二、结论（外部）的核心判断；未联网就写「本次未联网核实」",\n'
    ' "recommendations_internal":[{"action":"动作","basis":"依据","refs":["来源ID=x"]}],\n'
    ' "recommendations_web":[{"action":"动作","basis":"依据","refs":["[n]"]}],\n'
    ' "uncertainties":["证据缺口或需人工确认的事项"]}\n'
    "```\n"
    "要求：只输出这一个代码块，前后不要附加说明；字段以正文为准；"
    "某一节没有内容就给空字符串或空数组，不要省略字段。"
)


def extract(draft: str) -> Tuple[Optional[RiskAnalysisReport], str]:
    """从正文里取出结构化回执。返回 ``(报告, 剥离回执后的正文)``。

    **永不抛异常**：解析不了就返回 ``(None, 原文)``，由调用方退回切分。
    这里一旦抛异常，整次分析会 500 —— 为了"多拿点结构"把结论弄没，不值得。

    只认**最后一个**围栏块：模型偶尔会先写个示例 JSON 再写正文，
    取最后一个才是对的那份（也顺带避开了正文里本来就有的示例代码）。
    """
    text = draft or ""
    matches = list(_FENCE_RE.finditer(text))
    if not matches:
        return None, text
    last = matches[-1]
    try:
        data = json.loads(last.group(1))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, text
    if not isinstance(data, dict):
        return None, text
    try:
        report = RiskAnalysisReport.model_validate(data)
    except Exception as e:  # noqa: BLE001 - 结构不合预期就当没给，退回切分
        log.debug("[report-schema] 结构化回执校验失败，退回切分：%s", e)
        return None, text
    if not report.has_any:
        return None, text
    # 剥离：把回执块从正文里去掉，用户不该看到一堆 JSON。
    # 只切掉最后那一个块（前面的示例/引用留着，那是正文的一部分）。
    cleaned = text[: last.start()].rstrip() + text[last.end():].lstrip()
    return report, cleaned.strip()


def render_markdown(report: "RiskAnalysisReport") -> str:
    """结构化 → 四节正文。

    只在"模型只交了结构、没写正文"时兜底成稿用。标题与
    ``guardrail.build_system_prompt`` 的模板**逐字一致** —— 不一致的话
    引用核对与切分兜底都会跟着错位。
    """
    b: List[str] = []
    if report.risk_level.strip():
        b.append(f"风险等级：{report.risk_level.strip()}")
        if report.headline.strip():
            b.append(f"综合结论：{report.headline.strip()}")
        b.append("")
    elif report.headline.strip():
        b.append(f"综合结论：{report.headline.strip()}")
        b.append("")

    b.append("一、结论（基于内部知识库与经营数据）")
    b.append(report.conclusion_internal.strip() or "- 内部证据未覆盖。")
    b.append("")
    b.append("二、结论（基于外部公开资料）")
    b.append(report.conclusion_web.strip() or "- 本次未联网核实")
    b.append("")
    b.append("三、建议动作")
    b.append("（一）基于内部证据的动作")
    items = report.recs_as_dicts(True)
    b.extend(f"- {x['action']}" + (f" —— 依据：{x['basis']}" if x["basis"] else "")
             for x in items) if items else b.append("- 无")
    b.append("（二）基于外部资料的动作")
    items_w = report.recs_as_dicts(False)
    b.extend(f"- {x['action']}" + (f" —— 依据：{x['basis']}" if x["basis"] else "")
             for x in items_w) if items_w else b.append("- 无")
    b.append("")
    b.append("四、不确定性 / 需人工确认")
    b.append(report.uncertainties_text() or "- 无")
    return "\n".join(b).strip()


__all__ = [
    "Recommendation",
    "RiskAnalysisReport",
    "STRUCTURED_HINT",
    "extract",
    "render_markdown",
]
