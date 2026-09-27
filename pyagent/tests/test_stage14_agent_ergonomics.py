"""阶段 14：把「从 Java 移植来的写法」换成 Python Agent 生态的通用写法。

这一组用例钉的是**契约不变**，不是"代码更好看了"。理由写在
``app/ai/tool_schemas.py`` 的模块文档里，这里只重复最关键的一条：

    schema 是**模型看到的契约**。任何字段名 / 类型 / 描述的变化都会改变模型
    行为，而离线测试全是脚本化的假模型，**压根不读 schema** —— 也就是说这类
    变化在本仓库里是测不出来的。所以必须拿移植期那份手写 JSON 当黄金副本，
    逐字段对照。

三组对照
--------
1. **schema 等价**：Pydantic 生成的 == 移植期手写的（逐字段）。
2. **解析等价**：``coerce_args`` 相对旧 ``_parse_args`` + ``_get_int`` 只放宽不收紧。
3. **生态可达**：``as_langchain_tools()`` 产出的对象能被 ``bind_tools`` 消费，
   且**没有绕过权限闸门**。
"""

from __future__ import annotations

import inspect
import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.ai.tool_schemas import TOOL_ARG_MODELS, coerce_args, schema_for  # noqa: E402
from app.ai.tools import RiskAgentTools, ToolSpec  # noqa: E402


# --------------------------------------------------------------------------- #
# 移植期手写 schema 的黄金副本
#
# 这是 2026-09-23 之前 ``RiskAgentTools.specs()`` 里逐字写死的内容。
# **不要"顺手优化"这里** —— 它是对照基准，改它等于把尺子改短。
# --------------------------------------------------------------------------- #

LEGACY_SCHEMAS: dict = {
    "get_company_profile": {"type": "object", "properties": {}},
    "get_metrics": {
        "type": "object",
        "properties": {
            "days": {"type": "integer", "description": "最近N天，默认全部"},
            "limit": {"type": "integer", "description": "返回条数，默认15，最大30"},
        },
    },
    "get_risk_events": {
        "type": "object",
        "properties": {
            "level": {"type": "string", "description": "风险等级过滤"},
            "limit": {"type": "integer", "description": "返回条数，默认15"},
        },
    },
    "get_complaints": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "投诉类别过滤"},
            "limit": {"type": "integer", "description": "返回条数，默认15"},
        },
    },
    "get_competitors": {
        "type": "object",
        "properties": {"limit": {"type": "integer", "description": "返回条数，默认15"}},
    },
    "search_knowledge": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索词，需为具体业务问题或关键词"},
            "topK": {"type": "integer", "description": "返回条数，默认5"},
        },
    },
    "web_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "一个具体检索短语，3-5 个关键词，空格分隔，不要写整句"},
            "topN": {"type": "integer", "description": "返回来源条数，默认5，最大6"},
        },
    },
    "propose_create_ticket": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "工单标题"},
            "detail": {"type": "string", "description": "处置内容与建议动作"},
            "riskLevel": {"type": "string", "description": "HIGH/MEDIUM/LOW"},
            "reason": {"type": "string",
                       "description": "必填。为什么建议该动作，需附上证据来源（来源ID= 或 [n]）"},
        },
    },
    "propose_notify_owner": {
        "type": "object",
        "properties": {
            "eventId": {"type": "integer", "description": "风险事件ID"},
            "assigneeUserId": {"type": "integer", "description": "责任人用户ID，可省略"},
            "message": {"type": "string", "description": "通知内容"},
            "reason": {"type": "string", "description": "必填。依据说明，需附来源"},
        },
    },
    "propose_update_event_status": {
        "type": "object",
        "properties": {
            "eventId": {"type": "integer", "description": "风险事件ID"},
            "status": {"type": "string", "description": "目标状态，如 HANDLING / CLOSED / OPEN"},
            "comment": {"type": "string", "description": "处置说明"},
            "reason": {"type": "string", "description": "必填。依据说明，需附来源"},
        },
    },
}


# --------------------------------------------------------------------------- #
# 1. schema 等价
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(LEGACY_SCHEMAS))
def test_generated_schema_matches_legacy_handwritten(name):
    """Pydantic 生成的 schema 必须与移植期手写的**逐字段一致**。

    这条是本次改造的安全带。它红了只有两种可能：
    (a) 有人改了 ``tool_schemas.py`` 的字段描述/类型 —— 那会改变模型行为，必须想清楚；
    (b) ``_compact()`` 没折叠干净（多了 title / default / anyOf）。
    """
    assert schema_for(name) == LEGACY_SCHEMAS[name], (
        f"{name} 的 schema 与移植期手写的不一致。\n"
        f"  生成：{json.dumps(schema_for(name), ensure_ascii=False)}\n"
        f"  手写：{json.dumps(LEGACY_SCHEMAS[name], ensure_ascii=False)}")


def test_every_declared_tool_has_a_registered_args_model():
    """``specs()`` 声明的每个工具都必须在 ``TOOL_ARG_MODELS`` 登记。

    漏登记的后果是 ``from_args_model`` 直接抛 ``KeyError``（服务起不来），
    但更隐蔽的是"顺手改成手写 schema 绕过报错" —— 那条路会把我们拉回
    移植期的老问题。所以这里把声明集合与登记集合钉成相等。
    """
    tools = RiskAgentTools().specs()
    declared = {s.name for s in tools}
    assert declared == set(TOOL_ARG_MODELS), (
        f"specs() 声明的工具与 TOOL_ARG_MODELS 登记的不一致："
        f"仅声明={sorted(declared - set(TOOL_ARG_MODELS))}，"
        f"仅登记={sorted(set(TOOL_ARG_MODELS) - declared)}")


def test_specs_source_has_no_handwritten_json_schema_anymore():
    """``specs()`` 源码里不得再出现手写 JSON schema 字符串。

    这是一条**结构性**断言（扫源码而不是扫产物）：只要有人图省事把
    ``'{"type":"object",...}'`` 写回 ``specs()``，schema 就又与参数模型分居两地了。
    """
    src = inspect.getsource(RiskAgentTools.specs)
    assert '"type":"object"' not in src, (
        "specs() 里又出现了手写 JSON schema —— schema 必须来自 "
        "app.ai.tool_schemas 的 Pydantic 模型（描述与类型同源）。")


def test_from_args_model_rejects_unregistered_tool():
    """未登记的工具必须**报错**，而不是静默生成空 schema。

    静默生成空 schema 的表现是"模型怎么都不给这个工具传参数"，
    排查时会一路怀疑到模型，很难想到是登记漏了。
    """
    with pytest.raises(KeyError):
        ToolSpec.from_args_model("some_new_tool", "一个还没登记的工具")


def test_parameters_are_parsed_dicts_not_strings():
    """``ToolSpec.parameters`` 必须是 dict（下游 ``to_openai_tools`` 的约定）。"""
    for s in RiskAgentTools().specs():
        assert isinstance(s.parameters, dict), f"{s.name} 的 parameters 不是 dict"
        assert s.parameters.get("type") == "object"


# --------------------------------------------------------------------------- #
# 2. 解析等价（coerce_args 相对旧手工兜底只放宽，不收紧）
# --------------------------------------------------------------------------- #

def test_numeric_string_is_coerced_to_int():
    """模型很爱给整数参数传字符串。旧的 ``_get_int`` 能转，这里也必须能转。"""
    assert coerce_args("get_metrics", {"days": "7", "limit": "20"}) == {"days": 7, "limit": 20}


def test_bad_field_is_dropped_without_killing_the_whole_call():
    """一个字段坏掉只丢该字段，其余照留。

    旧实现里 ``_get_int`` 遇到坏值会落回默认值，尚可；但一旦有人以后写成
    "整体 try/except 返回 {}"，模型就会以为工具坏了、转去乱试别的工具。
    """
    got = coerce_args("get_metrics", {"days": "abc", "limit": 5})
    assert got == {"limit": 5}


def test_broken_json_yields_empty_args_not_exception():
    """入参是模型给的**不可信输入**，坏 JSON 不能变成 500。"""
    assert coerce_args("get_metrics", "{not json") == {}
    assert coerce_args("get_metrics", None) == {}
    assert coerce_args("get_metrics", "[]") == {}


def test_unregistered_tool_is_passed_through_untouched():
    """没登记参数模型的工具按原样透传（不做无谓的类型猜测）。"""
    assert coerce_args("unknown_tool", {"query": "x"}) == {"query": "x"}


def test_none_values_are_dropped():
    """显式 null 视为"没给"，与移植期 ``_str(None) == ""`` 的语义一致。"""
    assert coerce_args("search_knowledge", {"query": "流失率", "topK": None}) == {"query": "流失率"}


def test_raw_json_string_input_still_works():
    """模型传 JSON 字符串（真实链路就是这么传的）必须能解析。"""
    assert coerce_args("get_complaints", '{"category": "服务", "limit": 3}') == {
        "category": "服务", "limit": 3}


# --------------------------------------------------------------------------- #
# 3. 生态可达：as_langchain_tools
# --------------------------------------------------------------------------- #

def _ctx(company_id=None, scope=None):
    from app.ai.tool_context import ToolContext

    return ToolContext(question="客户流失率为何上升", company_id=company_id, scope=scope)


def test_as_langchain_tools_produces_bindable_tools():
    """产出的必须是 LangChain 原生工具：可被 ``bind_tools`` 消费。

    这是"工具层真正接入生态"的判据 —— 以前只有自研分发认得它们。
    """
    lc_tools = RiskAgentTools().as_langchain_tools(_ctx(company_id=1))
    assert lc_tools, "as_langchain_tools() 返回了空列表"
    names = {t.name for t in lc_tools}
    assert {"get_metrics", "web_search"} <= names

    # 能被 langchain 的标准转换函数吃下 = 生态可达
    # （langchain-core 1.x 里它不在 ``langchain_core.tools`` 下，
    #  直接 import 会 ImportError —— 位置随版本变过，别写死在错的那个包上）
    from langchain_core.utils.function_calling import convert_to_openai_tool

    payload = convert_to_openai_tool(lc_tools[0])
    assert payload["type"] == "function"
    assert payload["function"]["name"] == lc_tools[0].name
    assert payload["function"]["parameters"] == schema_for(lc_tools[0].name)


def test_langchain_tool_still_enforces_the_permission_gate():
    """包成 LangChain 工具后**权限闸门不能消失**。

    这是本次改造最危险的一处：一旦包出来的对象绕过 ``execute_with_meta``，
    等于给模型开了一条不做 companyId 校验、不过数据权限的越权通道。
    """
    tools = RiskAgentTools()

    # company_id 缺失 → 闸门拦截（返回错误文本，不执行）
    assert "缺少企业ID" in tools.as_langchain_tools(_ctx(company_id=None))[0].invoke({})

    # scope 判定不通过 → 抛权限错误
    strict = _ctx(company_id=1, scope=lambda cid: False)
    with pytest.raises(PermissionError):
        tools.as_langchain_tools(strict)[0].invoke({})


def test_langchain_tool_returns_the_same_text_as_native_dispatch():
    """两条入口必须返回同一份内容：外观换了，行为不能换。"""
    tools = RiskAgentTools()  # 未装配 rag/web → 走"不可用"分支，零 I/O、零 token
    ctx = _ctx(company_id=1)
    lc = {t.name: t for t in tools.as_langchain_tools(ctx)}
    assert lc["search_knowledge"].invoke({"query": "流失率"}) == \
        tools.execute_with_meta("search_knowledge", '{"query":"流失率"}', ctx).text


# --------------------------------------------------------------------------- #
# 4. 成稿结构化：让模型交出四节，而不是我们从正文里切
# --------------------------------------------------------------------------- #

from app.agent.report_schema import (  # noqa: E402
    RiskAnalysisReport,
    extract,
    render_markdown,
)

RECEIPT = """```json
{"risk_level":"HIGH",
 "headline":"客户流失率处于 15.8% 高位",
 "conclusion_internal":"客户流失率处于高位（指标：客户流失率=15.8%）。",
 "conclusion_web":"本次未联网核实",
 "recommendations_internal":[{"action":"启动续费挽留专项","basis":"流失率 15.8%","refs":["来源ID=7"]}],
 "recommendations_web":[],
 "uncertainties":["缺少近三个月的流失明细"]}
```"""


def test_extract_parses_and_strips_the_receipt():
    """能解析出结构，并且**把回执从正文里剥掉**（用户不该看到 JSON）。"""
    draft = "风险等级：HIGH\n综合结论：流失率偏高。\n\n一、结论…\n正文内容。\n\n" + RECEIPT
    report, body = extract(draft)
    assert report is not None
    assert report.conclusion_internal.startswith("客户流失率处于高位")
    assert report.recs_as_dicts(True)[0]["action"] == "启动续费挽留专项"
    assert report.uncertainties_text().startswith("1. 缺少近三个月")
    assert "```json" not in body, "回执块没被剥掉，用户会看到一堆 JSON"
    assert "正文内容。" in body, "剥离时把正文也切掉了"


@pytest.mark.parametrize("draft", [
    "一、结论（内部）\n- 无内部证据。",          # 根本没有回执
    "正文。\n```json\n{不是合法 JSON\n```",      # 回执是坏 JSON
    "正文。\n```json\n{\"unknown\": 1}\n```",    # 合法 JSON 但结构全空
    "",
])
def test_extract_never_raises_and_falls_back(draft):
    """解析不了就返回 ``(None, 原文)`` —— **绝不抛异常**。

    这条是开关能不能默认打开的前提：万一模型不遵守格式，最坏情况必须等于
    "退回旧行为"，而不是把整次分析搞成 500。
    """
    report, body = extract(draft)
    assert report is None
    assert body == draft, "回退时必须原样返回正文，不能顺手改动用户看到的内容"


def test_empty_structure_is_not_treated_as_success():
    """模型敷衍地给个 ``{}`` 不能算成功。

    用它覆盖切分结果会让界面四格变空 —— 那比不开这条通道更糟。
    """
    report, _ = extract("正文。\n```json\n{}\n```")
    assert report is None
    assert not RiskAnalysisReport().has_any


def test_render_markdown_reproduces_the_four_sections():
    """只拿到结构时，渲染回的四节必须与提示词模板**标题一致**。

    不一致的话，引用核对与切分兜底都会跟着错位（那一格又空了）。
    """
    report = RiskAnalysisReport(
        risk_level="HIGH", headline="流失率偏高",
        conclusion_internal="内部判断。", conclusion_web="本次未联网核实",
        recommendations_internal=[{"action": "启动挽留", "basis": "流失率 15.8%", "refs": []}],
        uncertainties=["缺少明细"])
    md = render_markdown(report)
    for title in ("一、结论（基于内部知识库与经营数据）",
                  "二、结论（基于外部公开资料）",
                  "三、建议动作", "四、不确定性 / 需人工确认"):
        assert title in md, f"渲染结果缺少小节标题：{title}"
    assert "启动挽留" in md and "内部判断。" in md


# ---- 端到端：跑一次真实的主链路（仍零 token，全部走假模型）----

from app.ai.llm_client import ChatResult  # noqa: E402
from app.agent.agent_service import AgentService  # noqa: E402
from fake_langchain import adapt_factory  # noqa: E402


class _FakeLlm:
    """脚本化的假模型：队列应答，用完返回 ``default``。"""

    def __init__(self, content: str = "") -> None:
        self.default = ChatResult(content=content or "一、结论\n- 无。", finish_reason="stop")
        self.calls: list = []

    def chat_with_tools(self, tools, messages, temperature=0.2, max_tokens=0) -> ChatResult:
        self.calls.append({"tools": [t.name for t in (tools or [])]})
        return self.default

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        return self.default.content or ""

    @property
    def model(self) -> str:
        return "fake-model"

    def is_configured(self) -> bool:
        return True

    def get_model(self) -> str:
        return "fake-model"


def _svc(content: str, structured: bool = True):
    llm = _FakeLlm(content)
    svc = AgentService(llm=llm, lg_model_factory=adapt_factory(llm))
    svc.cache_enabled = False
    svc.cache.clear()
    svc.structured_output = structured
    svc.long_term_write = False       # 别把测试的假结论沉淀成长期记忆
    return svc


DRAFT_WITH_RECEIPT = (
    "一、结论（基于内部知识库与经营数据）\n- 流失率偏高。\n\n"
    "二、结论（基于外部公开资料）\n- 本次未联网核实\n\n"
    "三、建议动作\n（一）基于内部证据的动作\n- 无\n（二）基于外部资料的动作\n- 无\n\n"
    "四、不确定性 / 需人工确认\n- 无\n\n" + RECEIPT)


def test_end_to_end_receipt_fills_the_four_slots():
    """模型交了回执 → 四格用它，**且正文里不再有 JSON**。"""
    svc = _svc(DRAFT_WITH_RECEIPT)
    out = svc.analyze(1, "客户流失率为何上升", use_multi_agent=False)
    aj = out.answer_json or {}
    assert aj.get("structure_source") == "structured-receipt"
    assert "客户流失率处于高位" in (aj.get("conclusion_internal") or "")
    assert aj.get("recommendations_internal")[0]["action"] == "启动续费挽留专项"
    assert "缺少近三个月" in (aj.get("uncertainties") or "")
    assert aj.get("risk_level") == "HIGH"
    assert "```json" not in (out.content or ""), "回执块泄漏到了用户看到的正文里"


def test_end_to_end_without_receipt_falls_back_to_splitting():
    """模型没交回执 → 退回切分，**行为与改造前完全一致**（这条是回退保证）。"""
    svc = _svc(DRAFT_WITH_RECEIPT, structured=False)
    out = svc.analyze(1, "客户流失率为何上升", use_multi_agent=False)
    aj = out.answer_json or {}
    assert aj.get("structure_source") == "markdown-split"
    assert "流失率偏高" in (aj.get("conclusion_internal") or ""), "切分兜底必须仍然有效"


def test_end_to_end_structure_only_answer_still_yields_a_report():
    """模型只交结构、没写正文 → 渲染成完整四节，用户不会看到空白。"""
    svc = _svc(RECEIPT)
    out = svc.analyze(1, "客户流失率为何上升", use_multi_agent=False)
    assert "一、结论（基于内部知识库与经营数据）" in (out.content or "")
    assert (out.answer_json or {}).get("structure_source") == "structured-receipt"
