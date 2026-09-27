"""Agent 工具的参数模型（Pydantic）—— 工具 schema 的**单一数据源**。

为什么要这一层
--------------
从 Java 移植过来时，工具参数是以**手写 JSON Schema 字符串**的形式散在
``RiskAgentTools.specs()`` 里的：

.. code-block:: python

    ToolSpec("get_metrics", "查询企业经营指标…",
             '{"type":"object","properties":{'
             '"days":{"type":"integer","description":"最近N天，默认全部"},'
             '"limit":{"type":"integer","description":"返回条数，默认15，最大30"}}}')

那是 Java Spring AI ``@Tool`` 的写法（注解上挂一段 JSON 字符串）。在 Python 侧
它有三个硬伤，每一个都已经在项目里兑现过：

1. **schema 与执行代码分居两地**。``_metrics()`` 里读 ``args.get("limit")``，
   而 schema 躺在几百行外的字符串常量里。改一边忘另一边**不会报错**，
   只会让模型拿到"描述里写了、参数里没有"的东西 —— 属于典型的静默失败。
2. **没有类型**。模型把 ``"abc"`` 传进来只能靠 ``_get_int`` 这类手搓函数
   一个个兜；每加一个字段就要再手写一个兜底分支，而这类分支写漏了
   同样是静默的（直接落默认值，没人知道模型其实传了东西）。
3. **进不了 LangChain 生态**。``StructuredTool`` / ``ToolNode`` / ``bind_tools``
   全都要求 ``args_schema``（Pydantic 模型）或可调用对象，手写字符串它们吃不下。
   于是工具只能永远留在自研分发里，图里那一步也得自己造轮子。

改法：参数定义成 Pydantic 模型，schema 由 ``model_json_schema()`` 生成，
执行时用**同一份**模型校验入参。

!!! **生成结果必须逐字段等价于移植期那份手写 schema。**

这不是洁癖：schema 是**模型看到的契约**，字段名 / 类型 / 描述任何一处变化都会
改变模型行为，而这类变化在离线测试里**根本看不出来** —— 离线用例全是脚本化的
假模型，压根不读 schema。所以 ``tests/test_stage14_agent_ergonomics.py`` 里钉了
一条新旧逐字段对照的用例，改这里会当场变红。

``_compact()`` 为什么必须存在
-----------------------------
Pydantic 生成的 schema **不等于**手写那份，差异有三处，而每一处都会改变
模型看到的契约：

=========================================  ===========================
Pydantic 生成                               移植期手写
=========================================  ===========================
根对象 / 每个属性都带 ``title``              没有
``Optional[int]`` → ``anyOf:[integer,null]``  ``"type":"integer"``
每个属性带 ``default: null``                没有
=========================================  ===========================

``anyOf`` 对 OpenAI function calling 是合法的，但它把"这个参数可以不填"
表达成了联合类型，而弱模型在联合类型上的参数准确率明显更差。所以这里统一
折叠回单一 ``type``：**可空信息不进 schema**（模型不填即没有），只在校验时体现。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


class ToolArgs(BaseModel):
    """所有工具参数模型的基类。本身不带字段。"""


class EmptyArgs(ToolArgs):
    """无参数工具（``get_company_profile``）。"""


class GetMetricsArgs(ToolArgs):
    days: Optional[int] = Field(None, description="最近N天，默认全部")
    limit: Optional[int] = Field(None, description="返回条数，默认15，最大30")


class GetRiskEventsArgs(ToolArgs):
    level: Optional[str] = Field(None, description="风险等级过滤")
    limit: Optional[int] = Field(None, description="返回条数，默认15")


class GetComplaintsArgs(ToolArgs):
    category: Optional[str] = Field(None, description="投诉类别过滤")
    limit: Optional[int] = Field(None, description="返回条数，默认15")


class GetCompetitorsArgs(ToolArgs):
    limit: Optional[int] = Field(None, description="返回条数，默认15")


class SearchKnowledgeArgs(ToolArgs):
    query: Optional[str] = Field(None, description="检索词，需为具体业务问题或关键词")
    topK: Optional[int] = Field(None, description="返回条数，默认5")


class WebSearchArgs(ToolArgs):
    query: Optional[str] = Field(
        None,
        description="一个具体检索短语，3-5 个关键词，空格分隔，不要写整句")
    topN: Optional[int] = Field(None, description="返回来源条数，默认5，最大6")


class ProposeCreateTicketArgs(ToolArgs):
    title: Optional[str] = Field(None, description="工单标题")
    detail: Optional[str] = Field(None, description="处置内容与建议动作")
    riskLevel: Optional[str] = Field(None, description="HIGH/MEDIUM/LOW")
    reason: Optional[str] = Field(
        None, description="必填。为什么建议该动作，需附上证据来源（来源ID= 或 [n]）")


class ProposeNotifyOwnerArgs(ToolArgs):
    eventId: Optional[int] = Field(None, description="风险事件ID")
    assigneeUserId: Optional[int] = Field(None, description="责任人用户ID，可省略")
    message: Optional[str] = Field(None, description="通知内容")
    reason: Optional[str] = Field(None, description="必填。依据说明，需附来源")


class ProposeUpdateEventStatusArgs(ToolArgs):
    eventId: Optional[int] = Field(None, description="风险事件ID")
    status: Optional[str] = Field(None, description="目标状态，如 HANDLING / CLOSED / OPEN")
    comment: Optional[str] = Field(None, description="处置说明")
    reason: Optional[str] = Field(None, description="必填。依据说明，需附来源")


#: 工具名 → 参数模型。**新增工具必须同时登记到这里**，
#: 否则 :func:`coerce_args` 会走"不校验"分支，等于又退回手搓时代。
TOOL_ARG_MODELS: Dict[str, Type[ToolArgs]] = {
    "get_company_profile": EmptyArgs,
    "get_metrics": GetMetricsArgs,
    "get_risk_events": GetRiskEventsArgs,
    "get_complaints": GetComplaintsArgs,
    "get_competitors": GetCompetitorsArgs,
    "search_knowledge": SearchKnowledgeArgs,
    "web_search": WebSearchArgs,
    "propose_create_ticket": ProposeCreateTicketArgs,
    "propose_notify_owner": ProposeNotifyOwnerArgs,
    "propose_update_event_status": ProposeUpdateEventStatusArgs,
}


def args_model(name: str) -> Optional[Type[ToolArgs]]:
    """取工具的参数模型；未登记返回 ``None``（调用方按"不校验"处理）。"""
    return TOOL_ARG_MODELS.get(str(name or "").strip())


def _compact(schema: Dict[str, Any]) -> Dict[str, Any]:
    """把 Pydantic 生成的 schema 折叠成"移植期手写"的同一形态。

    三件事：去掉 ``title``、``default``；把 ``anyOf:[T,null]`` 折叠回 ``T``。
    详见模块文档 —— **不要因为"反正合法"就省掉这一步**。
    """
    out = dict(schema or {})
    out.pop("title", None)
    # Pydantic 会把类的 docstring 当成根 ``description``。这里统一去掉：
    # 工具说明由 ``ToolSpec.description`` 承载（那才是模型看到的
    # ``function.description``），根上再挂一份既重复又会改变请求体。
    out.pop("description", None)
    props = out.get("properties")
    if not isinstance(props, dict):
        out.setdefault("type", "object")
        out.setdefault("properties", {})
        return out
    cleaned: Dict[str, Any] = {}
    for key, val in props.items():
        if not isinstance(val, dict):
            cleaned[key] = val
            continue
        v = dict(val)
        v.pop("title", None)
        v.pop("default", None)
        any_of = v.get("anyOf")
        if isinstance(any_of, list):
            non_null = [x for x in any_of if isinstance(x, dict) and x.get("type") != "null"]
            # 只有"单一类型 + null"这种形态能安全折叠；真正的多类型联合原样保留，
            # 折叠它会把信息弄丢（宁可 schema 复杂也不能骗模型）。
            if len(non_null) == 1 and len(any_of) == len(non_null) + 1:
                v.pop("anyOf")
                v.update(non_null[0])
        cleaned[key] = v
    out["properties"] = cleaned
    out.setdefault("type", "object")
    return out


def schema_for(name: str) -> Dict[str, Any]:
    """生成某个工具的参数 schema（OpenAI function 格式）。"""
    model = args_model(name)
    if model is None:
        return {"type": "object", "properties": {}}
    try:
        return _compact(model.model_json_schema())
    except Exception as e:  # noqa: BLE001 - schema 生成失败绝不能让工具整体不可用
        log.warning("[tool-schemas] %s 的 schema 生成失败，退回空 schema：%s", name, e)
        return {"type": "object", "properties": {}}


def schema_json_for(name: str) -> str:
    """:func:`schema_for` 的字符串形式（``ToolSpec`` 的构造入口要的是字符串）。"""
    return json.dumps(schema_for(name), ensure_ascii=False)


def _as_dict(raw: Any) -> Dict[str, Any]:
    """模型给的是 JSON 字符串，内部调用传 dict —— 两种都收。"""
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    if raw is None:
        return {}
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            loaded = json.loads(s)
        except json.JSONDecodeError:
            return {}
        return {str(k): v for k, v in loaded.items()} if isinstance(loaded, dict) else {}
    return {}


def coerce_args(name: str, raw: Any) -> Dict[str, Any]:
    """按参数模型校验并归一化工参。

    三条原则，顺序不能变：

    1. **能转就转**。Pydantic 的 lax 模式会把 ``"15"`` 转成 ``15`` —— 模型很爱
       给整数参数传字符串，这在以前是 ``_get_int`` 手写兜底的活。
    2. **坏字段只丢自己**。整体校验失败时逐字段重试，一个字段坏掉不该让整次
       工具调用退化成"什么参数都没给"（那会让模型以为工具坏了，于是它换一个
       工具乱试，白烧一轮）。
    3. **永不抛异常**。工具入参来自模型，是**不可信输入**；把它变成 500
       等于让模型的一句话决定服务可用性。
    """
    data = _as_dict(raw)
    model = args_model(name)
    if model is None:
        return data
    try:
        return model.model_validate(data).model_dump(exclude_none=True)
    except ValidationError:
        pass
    out: Dict[str, Any] = {}
    for k, v in data.items():
        try:
            model.model_validate({k: v})
        except ValidationError:
            log.debug("[tool-schemas] %s 的入参 %s 校验失败，已丢弃该字段", name, k)
            continue
        out[str(k)] = v
    return out


__all__ = [
    "TOOL_ARG_MODELS",
    "EmptyArgs",
    "GetComplaintsArgs",
    "GetCompetitorsArgs",
    "GetMetricsArgs",
    "GetRiskEventsArgs",
    "ProposeCreateTicketArgs",
    "ProposeNotifyOwnerArgs",
    "ProposeUpdateEventStatusArgs",
    "SearchKnowledgeArgs",
    "ToolArgs",
    "WebSearchArgs",
    "args_model",
    "coerce_args",
    "schema_for",
    "schema_json_for",
]
