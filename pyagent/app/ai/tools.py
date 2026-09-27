"""Agent 可调用的取数工具（对应 Java ``RiskAgentTools``）。

每个工具只返回紧凑、可读的结果，由模型在 ReAct 循环中按需调用，
而不是把全量业务数据一次性塞进 prompt。

联网检索的三道闸门
------------------
1. **次数预算**：模型拿到不相关结果会惯性重搜，必须外力刹住。
2. **检索词确定性清洗**：长串关键词会被搜索引擎抓偏（2024→日历、如何→词典）。
3. **来源相关性回检**（零 token）：搜得到 ≠ 与本次问题有关。

第 3 道闸门通过后才会分配 ``[n]`` 序号——**未通过回检的来源不分配序号**，
否则会出现「来源卡里有 62 条、正文只引用了 3 条」的错位。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

from sqlalchemy import desc, select

from ..config import get_settings
from ..db import tables as T
from ..db.engine import Database, get_db
from ..rag.embedding import LocalEmbedding
from ..rag.rag_service import RagService
from ..rag.relevance import SourceRelevanceFilter, WebSource
from ..rag.web_cleaner import clean as clean_query
from ..web.search_service import WebSearchService
from .tool_context import ToolContext
from .tool_schemas import args_model, coerce_args

log = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """工具声明（OpenAI function 格式）。``parameters`` 直接持有已解析的 JSON Schema。

    **不要再用字符串字面量写 schema** —— 那是移植期 Java Spring AI ``@Tool`` 的写法，
    与执行代码分居两地、改一边忘另一边不报错。请走 :meth:`from_args_model`，
    让 :mod:`app.ai.tool_schemas` 里的 Pydantic 模型生成它（描述与类型同源）。
    """

    name: str
    description: str
    parameters: Dict = field(default_factory=dict)

    def __init__(self, name: str, description: str, parameters_json: str = "{}") -> None:
        self.name = name
        self.description = description
        try:
            self.parameters = json.loads(parameters_json)
        except json.JSONDecodeError:
            self.parameters = {"type": "object", "properties": {}}

    @classmethod
    def from_args_model(cls, name: str, description: str) -> "ToolSpec":
        """按工具名从 :mod:`app.ai.tool_schemas` 生成 schema。

        **工具名必须已在 ``TOOL_ARG_MODELS`` 登记**；否则这里会静默生成空 schema，
        模型就看不到任何参数。所以未登记时直接报错 —— 这种错必须在启动时炸，
        不能等到"模型怎么都不传参数"才被发现。
        """
        from .tool_schemas import args_model, schema_json_for

        if args_model(name) is None:
            raise KeyError(
                f"工具 {name!r} 未在 app.ai.tool_schemas.TOOL_ARG_MODELS 登记。"
                f"新增工具必须同时补上参数模型（否则模型看不到它的入参）。")
        return cls(name, description, schema_json_for(name))


@dataclass
class ToolResult:
    """文本 + 结构化来源 + 来源类型（``web`` / ``knowledge`` / ``internal``）。"""

    text: str
    sources: List[Dict] = field(default_factory=list)
    source_type: str = "internal"


class RiskAgentTools:
    """按名称分发执行。``approvals`` 是写动作的落点（只入审批队列，不直接改数据）。"""

    def __init__(
        self,
        rag: Optional[RagService] = None,
        web_search: Optional[WebSearchService] = None,
        db: Optional[Database] = None,
        web_filter: Optional[SourceRelevanceFilter] = None,
        approvals: Optional[Callable[..., int]] = None,
        max_web_calls: Optional[int] = None,
        max_sources: Optional[int] = None,
    ) -> None:
        s = get_settings()
        self.db = db or get_db()
        self.rag = rag
        self.web_search = web_search
        self.web_filter = web_filter or SourceRelevanceFilter(local=LocalEmbedding())
        self.approvals = approvals
        self.max_web_calls = int(max_web_calls if max_web_calls is not None else s.web.max_calls)
        self.max_sources = int(max_sources if max_sources is not None else s.web.max_sources)

    # -- 声明 --------------------------------------------------------

    def specs(self) -> List[ToolSpec]:
        """工具声明。

        **schema 一律由 :mod:`app.ai.tool_schemas` 的 Pydantic 模型生成**，
        不再手写 JSON 字符串 —— 描述与类型同源，改字段不会漏改另一处。
        """
        S = ToolSpec.from_args_model
        return [
            S("get_company_profile",
              "获取企业基础档案：名称、行业、区域、客户等级、规模等。"),
            S("get_metrics",
              "查询企业经营指标（营收、利润、客户数、续费率等），按日期倒序。可指定最近天数与条数。"),
            S("get_risk_events",
              "查询企业风险事件，可按风险等级(HIGH/MEDIUM/LOW)过滤。返回等级、标题、状态、触发值与阈值。"),
            S("get_complaints",
              "查询客户投诉，可按类别过滤。返回重复投诉、超SLA、高流失统计与典型投诉摘要。"),
            S("get_competitors",
              "查询竞争对手/竞品：竞品名、产品、价格、卖点、竞争风险等级等。"),
            S("search_knowledge",
                     "在本地资料库中做语义 + 关键字混合检索，覆盖：知识库文档（行业报告、SOP、合规与风控文档）、"
                     "经营指标、风险事件、客户投诉、竞品、企业档案、风控规则，以及指标走势等聚合统计。"
                     "返回标题、摘要与来源ID；引用其内容时必须写成「来源ID=x」的形式，便于核对内部依据。"
                     "【用法】问「哪些指标在恶化」「有没有类似的投诉」「竞品怎么打」这类需要跨数据找线索的问题，"
                     "优先用它；要精确取某一类数据的最新明细时再用 get_metrics 等工具。"),
            S("web_search",
                     "联网检索外部公开资料：行业政策与监管要求、最新市场行情、同业做法、公开舆情等。"
                     "返回带全局序号的网页来源（标题 + URL + 站点）。"
                     "当问题需要企业外部信息、最新政策/行情，或内部知识库检索不到时调用。"
                     "引用其内容时必须用 [n] 标注（n 为该来源序号），并在末尾「参考来源（网页）」逐条列出标题与完整 URL，"
                     "只列正文真正引用过的来源。"
                     "【检索词写法】query 必须是「一个具体的检索短语」，用空格分隔 3-5 个核心关键词，"
                     "例如「SaaS 客户流失率 预警指标」「跨境电商 出口退税 新政 2026」；"
                     "禁止堆砌一长串同义词，禁止写疑问句或整句话；单次分析最多调用 4 次 web_search。"),
            # ---- 写动作：一律只「提议」，必须经人工审批才生效 ----
            S("propose_create_ticket",
              "【需人工审批】提议创建一个风险处置工单。调用后动作进入审批队列，由人确认后才会真正创建；"
              "你的回答里必须说明这是「待审批的提议」，不能说成已创建。",
              ),
            S("propose_notify_owner",
              "【需人工审批】提议通知风险责任人跟进某个风险事件。调用后进入审批队列，确认后才会写入责任人。"),
            S("propose_update_event_status",
              "【需人工审批】提议变更某个风险事件的处置状态（如 OPEN→HANDLING/CLOSED）。调用后进入审批队列，确认后才生效。"),
        ]

    # -- 分发 --------------------------------------------------------

    def execute_with_meta(self, name: Optional[str], arguments_json: Optional[str],
                          ctx: ToolContext) -> ToolResult:
        if ctx.company_id is None:
            return ToolResult('{"error":"缺少企业ID"}', [], "internal")
        if ctx.scope is not None and not ctx.scope(ctx.company_id):
            raise PermissionError("AI工具不能绕过当前用户数据权限")

        #! 入参走 Pydantic 校验（``tool_schemas.coerce_args``），不再用手搓的
        #! ``_parse_args`` + ``_get_int``。它做的是同一件事但有三点不同：
        #!   · 类型不符的字符串（"15"）会被转成 15，而不是落回默认值；
        #!   · 坏字段只丢自己，不会让整次调用退化成"没给任何参数"；
        #!   · 参数定义与 schema **同源**，不存在"改了 schema 忘了改解析"。
        args = coerce_args(name or "", arguments_json)
        try:
            if name == "web_search":
                return self._web_search(args, ctx)
            if name == "search_knowledge":
                return self._knowledge(args, ctx)
            if name and name.startswith("propose_"):
                return self._propose(name, args, ctx)
            text = {
                "get_company_profile": lambda: self._profile(ctx),
                "get_metrics": lambda: self._metrics(args, ctx),
                "get_risk_events": lambda: self._risks(args, ctx),
                "get_complaints": lambda: self._complaints(args, ctx),
                "get_competitors": lambda: self._competitors(args, ctx),
            }.get(name or "", lambda: '{"error":"未知工具 ' + str(name) + '"}')()
            return ToolResult(text, [], "internal")
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001
            return ToolResult('{"error":"工具执行失败: ' + str(e) + '"}', [], "internal")

    def execute(self, name: Optional[str], arguments_json: Optional[str], ctx: ToolContext) -> str:
        return self.execute_with_meta(name, arguments_json, ctx).text

    # -- LangChain 生态出口 -------------------------------------------

    def as_langchain_tools(self, ctx: ToolContext,
                           on_result: Optional[Callable[[str, Any], None]] = None) -> List[Any]:
        """把工具包成 LangChain ``StructuredTool``。

        有了它，这些工具才能被 ``bind_tools`` / LangGraph 的 ``ToolNode`` /
        各类 Agent 框架直接消费 —— 之前只有自研分发认得它们，等于把工具层
        关在了生态门外。

        :param on_result: 每次执行**成功后**的回调 ``(工具名, ToolResult)``。
            工具节点用它做留痕与证据池：``ToolResult`` 里才有 ``sources``，
            而 LangChain 工具只能返回文本——**不挂这个回调就会丢掉来源**，
            前端「证据溯源」会只剩预取那一批。

        !!! **包出来的工具仍然走 ``execute_with_meta``**，也就是说 companyId 校验、
        数据权限闸门、联网三道闸门、审批落点一个都不少。这里只是换了一层外观，
        **不是**给模型开第二条执行通道 —— 另开一条通道会绕过权限，那是事故级改动。
        """
        from langchain_core.tools import StructuredTool

        out: List[Any] = []
        for spec in self.specs():
            model = args_model(spec.name)
            if model is None:  # 防御：specs() 已保证登记过
                continue
            out.append(StructuredTool.from_function(
                func=self._lc_invoke(spec.name, ctx, on_result),
                name=spec.name,
                description=spec.description or "",
                args_schema=model,
            ))
        return out

    def _lc_invoke(self, name: str, ctx: ToolContext,
                   on_result: Optional[Callable[[str, Any], None]] = None) -> Callable[..., str]:
        """LangChain 调用入口：``**kwargs`` → 自研分发。"""

        def _invoke(**kwargs: Any) -> str:
            #! **必须滤掉 None**：``StructuredTool`` 会按 args_schema 把**所有字段**都传进来
            #! （模型只给了 ``limit``，这里会多一个 ``days=None``）。带着它进执行层，
            #! 表现是"模型明明没说要过滤最近 N 天，取数却按 presence 判断走了另一条分支"。
            args = {k: v for k, v in kwargs.items() if v is not None}
            res = self.execute_with_meta(
                name, json.dumps(args, ensure_ascii=False), ctx)
            if on_result is not None:
                try:
                    on_result(name, res)
                except Exception as e:  # noqa: BLE001 - 留痕失败不能让工具"执行失败"
                    log.debug("[Tools] 结果回灌失败（已忽略）：%s", e)
            return res.text

        return _invoke

    # -- 联网检索 ----------------------------------------------------

    def _web_search(self, args: Dict, ctx: ToolContext) -> ToolResult:
        raw_query = _str(args.get("query"))
        if not raw_query:
            return ToolResult('{"error":"未提供检索词 query"}', [], "web")

        # 闸门一：次数预算
        ctx.max_web_calls = self.max_web_calls
        if not ctx.try_consume_web_call():
            return ToolResult(
                f"本轮联网检索已达上限（最多 {self.max_web_calls} 次），"
                "请不要再调用 web_search，直接基于已获得的证据作答；"
                "若确实缺少外部依据，就在「二、结论（基于外部公开资料）」写明「本次未联网核实」。",
                [], "web")

        # 闸门二：检索词确定性清洗
        cq = clean_query(raw_query)
        if not cq.accepted:
            ctx.refund_web_call()
            ctx.add_web_trail(_trail(raw_query, None, 0, 0, "驳回：" + (cq.reason or "")))
            return ToolResult("联网检索被驳回：" + (cq.reason or ""), [], "web")

        top_n = min(_get_int(args, "topN", 5, self.max_sources), self.max_sources)
        if self.web_search is None:
            ctx.add_web_trail(_trail(raw_query, cq.query, 0, 0, "检索失败：联网通道未装配"))
            return ToolResult("联网检索未取得结果：联网通道未装配"
                              "（请用更具体的业务短语重试，或基于内部数据作答并说明未联网核实）", [], "web")

        r = self.web_search.search(cq.query, top_n)
        if not r.ok():
            ctx.add_web_trail(_trail(raw_query, cq.query, 0, 0,
                                     "检索失败：" + (r.error or "无结果")))
            return ToolResult(
                "联网检索未取得结果：" + (r.error or "搜索引擎无结果")
                + "（请用更具体的业务短语重试，或基于内部数据作答并说明未联网核实）", [], "web")

        # 闸门三：来源相关性回检（零 token）
        #! 判定一条都没过时必须用 ``strict_kept``：``kept`` 在全部不相关时也会补一条
        #! 用于展示的最高分来源，照它判断的话下面那个"全部不相关"分支永远不会执行，
        #! 于是搜索引擎抓回来的无关网页会被当成来源编上 [n] 递给模型。
        srcs = [WebSource(s.index, s.title, s.url, s.site_name, s.snippet) for s in r.sources]
        fr = self.web_filter.filter(ctx.question, cq.query, srcs)
        kept = fr.strict_kept
        ctx.add_web_trail(_trail(raw_query, cq.query, len(kept), fr.dropped_count(),
                                 "清洗说明：" + (cq.reason or "") + "；回检阈值 " + str(fr.threshold)))

        if not kept:
            return ToolResult(
                f"联网检索「{cq.query}」返回 {len(srcs)} 条结果，但全部被判定为与本次问题不相关"
                f"（相关度低于阈值 {fr.threshold}），已按「不把无关来源递给模型」的原则丢弃。"
                "请基于内部数据作答，或换用更贴近业务实体的检索词重试一次。", [], "web")

        numbered = _number_web_sources(kept, ctx)
        return ToolResult(self._render_web(cq.query, raw_query, numbered, fr), numbered, "web")

    def _render_web(self, query: str, raw_query: str, src: List[Dict], fr) -> str:
        """只用回检通过后的来源重建摘要。

        上游摘要是按搜索引擎自己的顺序编号的，一旦有来源被回检丢弃，编号就会错位——
        模型引用的 ``[n]`` 会指到一条它根本没见过的链接上。
        """
        b = [f"联网检索「{query}」返回 {len(src)} 条来源（序号为本次分析的全局序号，正文引用时用该序号）："]
        if raw_query and raw_query != query:
            b.append(f"（你提供的原始检索词是「{_trim(raw_query, 60)}」，已被清洗为上述短语）")
        for s in src:
            b.append(f"  [{s['index']}] {s['title']}")
            b.append(f"      URL: {s['url']}")
            if str(s.get("siteName") or "").strip():
                b.append(f"      站点: {s['siteName']}")
            if str(s.get("snippet") or "").strip():
                b.append(f"      摘要: {_trim(s['snippet'], 180)}")
            if s.get("reused") == "1":
                b.append("      （该来源在本次分析中已出现过，沿用原序号）")
        if fr is not None and fr.dropped_count() > 0:
            b.append(f"\n（另有 {fr.dropped_count()} 条因与本问题相关度低于阈值 {fr.threshold} "
                     "被过滤，未列出——不要引用任何未列出的来源。）")
        b.append("\n【引用要求】正文引用上述内容时必须写成 [1][2] 形式，"
                 "并在末尾「参考来源（网页）」逐条列出标题与完整 URL；"
                 "只列正文中真正引用过的来源，禁止编造未列出的链接。")
        return "\n".join(b).strip()

    # -- 知识库检索 --------------------------------------------------

    def _knowledge(self, args: Dict, ctx: ToolContext) -> ToolResult:
        """一次检索同时供「模型可读文本」与「结构化来源」两处使用。"""
        q = _str(args.get("query"))
        if not q:
            return ToolResult('{"error":"未提供检索词"}', [], "knowledge")
        top_k = _get_int(args, "topK", ctx.top_k if ctx.top_k > 0 else 5, 10)
        if self.rag is None:
            return ToolResult("本地资料库不可用。", [], "knowledge")

        res = self.rag.search_detailed(ctx.company_id, q, top_k)
        docs = res.hits
        ctx.rag_diagnostics = res.diagnostics
        if not docs:
            return ToolResult(
                f"本地资料库未检索到与「{q}」相关的内容。"
                "（资料库覆盖：知识库文档、经营指标、风险事件、客户投诉、竞品、企业档案、风控规则。"
                "可换更贴近业务实体的检索词重试一次，或改用 get_metrics 等取数工具。）",
                [], "knowledge")
        ctx.stat_max("kbHits", len(docs))

        b = [f"本地资料库检索(共{len(docs)}条)："]
        for i, d in enumerate(docs):
            # 用 ref() 而不是 document_id：结构化切片（metric:12）没有文档 ID，
            # 但同样要能被引用、被核对，否则模型引用了它却判成悬空。
            b.append(f"  [{i + 1}] 来源ID={d.ref()} 类型={_type_label(d.ref())} "
                     f"标题={d.title} 相关度={d.score:.1f}")
            b.append(f"     摘要：{_trim(d.snippet, 200)}")

        src: List[Dict] = []
        for i, d in enumerate(docs):
            m: Dict[str, object] = {
                "index": i + 1,
                "documentId": d.document_id,
                "title": d.title,
                "score": d.score,
            }
            if d.ref() is not None:
                m["sourceRef"] = d.ref()
            if d.rerank_score is not None:
                m["rerankScore"] = d.rerank_score
            if d.recall_paths:
                m["recallPaths"] = d.recall_paths
            src.append(m)
        return ToolResult("\n".join(b).strip(), src, "knowledge")

    # -- 写动作（只提议） --------------------------------------------

    def _propose(self, name: str, args: Dict, ctx: ToolContext) -> ToolResult:
        """**只入审批队列，绝不直接落业务数据**。

        返回给模型的措辞必须是「已提议、待审批」——否则模型会顺着「已创建工单」的说法
        向用户谎称事情已经办了，而实际上什么都没发生。这是这类工具最容易出的错。
        """
        action_type = {
            "propose_create_ticket": "CREATE_TICKET",
            "propose_notify_owner": "NOTIFY_OWNER",
            "propose_update_event_status": "UPDATE_EVENT_STATUS",
        }.get(name)
        if action_type is None:
            return ToolResult('{"error":"未知动作工具 ' + name + '"}', [], "internal")
        reason = _str(args.get("reason"))
        if not reason:
            return ToolResult('{"error":"必须提供 reason：说明建议该动作的依据，并附上证据来源'
                              '（内部写「来源ID=x」，外部写 [n]）。没有依据的动作不予受理。"}',
                              [], "internal")
        if self.approvals is None:
            return ToolResult('{"error":"审批通道未装配，动作不予受理"}', [], "internal")
        try:
            rid = self.approvals(ctx.company_id, action_type, args, reason,
                                 _str(args.get("riskLevel")) or None, None, "agent")
        except Exception as e:  # noqa: BLE001
            return ToolResult('{"error":"提议动作失败: ' + str(e) + '"}', [], "internal")
        return ToolResult(
            f"已生成待审批动作 #{rid}（{action_type}）。\n"
            "重要：该动作尚未执行，需由有权限的人员在「待审批动作」中确认后才会生效；"
            "你在回答中必须写成「建议…（待审批）」，不能说成已经完成。", [], "internal")

    # -- 取数工具 ----------------------------------------------------

    def _profile(self, ctx: ToolContext) -> str:
        row = self.db.fetch_one(select(T.company).where(T.company.c.id == ctx.company_id))
        if row is None:
            return '{"error":"企业不存在"}'
        return ("企业档案：名称={}，编码={}，行业={}，区域={}，客户等级={}，规模={}，状态={}".format(
            _nz(row.get("company_name")), _nz(row.get("company_code")), _nz(row.get("industry")),
            _nz(row.get("region")), _nz(row.get("customer_level")), _nz(row.get("company_scale")),
            _nz(row.get("status"))))

    def _metrics(self, args: Dict, ctx: ToolContext) -> str:
        limit = _get_int(args, "limit", 15, 30)
        stmt = (select(T.business_metric)
                .where(T.business_metric.c.company_id == ctx.company_id)
                .order_by(desc(T.business_metric.c.metric_date)))
        days = args.get("days")
        if isinstance(days, (int, float)) and int(days) > 0:
            stmt = stmt.where(T.business_metric.c.metric_date >= date.today() - timedelta(days=int(days)))
        rows = self.db.fetch_all(stmt.limit(limit))
        if not rows:
            return self._fallback_of(ctx, "经营指标")
        ctx.stat_max("metricTotal", len(rows))
        b = [f"经营指标(共{len(rows)}条)："]
        for m in rows:
            b.append(f"  · {_nz(m.get('metric_date'))} {_nz(m.get('metric_name'))}"
                     f"({_nz(m.get('metric_code'))})={_nz(m.get('metric_value'))}{_nz(m.get('unit'))}")
        return "\n".join(b).strip()

    def _risks(self, args: Dict, ctx: ToolContext) -> str:
        limit = _get_int(args, "limit", 15, 30)
        stmt = (select(T.risk_event).where(T.risk_event.c.company_id == ctx.company_id)
                .order_by(desc(T.risk_event.c.created_at)))
        level = args.get("level")
        if isinstance(level, str) and level.strip():
            stmt = stmt.where(T.risk_event.c.risk_level == level.strip().upper())
        rows = self.db.fetch_all(stmt.limit(limit))
        if not rows:
            return self._fallback_of(ctx, "风险事件")
        by_level: Dict[str, int] = {}
        for r in rows:
            k = _nz(r.get("risk_level"))
            by_level[k] = by_level.get(k, 0) + 1
        # 供 SSE meta 事件在正文之前推出风险等级
        ctx.stat_max("riskTotal", len(rows))
        ctx.stat_max("riskHigh", by_level.get("HIGH", 0))
        ctx.stat_max("riskMedium", by_level.get("MEDIUM", 0))
        b = [f"风险事件(共{len(rows)}条，等级分布{by_level})："]
        for r in rows:
            b.append(f"  · [{_nz(r.get('risk_level'))}] {_nz(r.get('risk_title'))}"
                     f" 状态={_nz(r.get('status'))} 触发值={_nz(r.get('trigger_value'))}"
                     f"/阈值={_nz(r.get('threshold_value'))} 日期={_nz(r.get('metric_date'))}")
        return "\n".join(b).strip()

    def _complaints(self, args: Dict, ctx: ToolContext) -> str:
        limit = _get_int(args, "limit", 15, 30)
        stmt = (select(T.complaint).where(T.complaint.c.company_id == ctx.company_id)
                .order_by(desc(T.complaint.c.complaint_date)))
        cat = args.get("category")
        if isinstance(cat, str) and cat.strip():
            stmt = stmt.where(T.complaint.c.category == cat.strip())
        rows = self.db.fetch_all(stmt.limit(limit))
        if not rows:
            return self._fallback_of(ctx, "客户投诉")
        repeat = sum(1 for x in rows if (x.get("repeat_flag") or 0) == 1)
        sla = sum(1 for x in rows if (x.get("sla_exceeded") or 0) == 1)
        churn = sum(1 for x in rows if "高" == _nz(x.get("churn_risk")))
        ctx.stat_max("complaintTotal", len(rows))
        ctx.stat_max("complaintRepeat", repeat)
        ctx.stat_max("complaintSla", sla)
        ctx.stat_max("complaintChurnHigh", churn)
        b = [f"投诉(共{len(rows)}条，重复={repeat}，超SLA={sla}，高流失={churn})："]
        for x in rows:
            b.append(f"  · {_nz(x.get('complaint_date'))} {_nz(x.get('category'))}"
                     f"｜严重度={_nz(x.get('severity'))}｜流失风险={_nz(x.get('churn_risk'))}"
                     f"｜{_trim(_nz(x.get('description')), 50)}")
        return "\n".join(b).strip()

    def _competitors(self, args: Dict, ctx: ToolContext) -> str:
        limit = _get_int(args, "limit", 15, 30)
        rows = self.db.fetch_all(
            select(T.competitor_product)
            .where(T.competitor_product.c.company_id == ctx.company_id)
            .order_by(desc(T.competitor_product.c.updated_date)).limit(limit))
        if not rows:
            return self._fallback_of(ctx, "竞品")
        ctx.stat_max("competitorTotal", len(rows))
        b = [f"竞品(共{len(rows)}条)："]
        for x in rows:
            b.append(f"  · {_nz(x.get('competitor_name'))}/{_nz(x.get('product_name'))}"
                     f"｜标准价={_nz(x.get('price'))}{_nz(x.get('price_unit'))}"
                     f"｜竞争风险={_nz(x.get('risk_level'))}")
        return "\n".join(b).strip()

    def _fallback_of(self, ctx: ToolContext, label: str) -> str:
        """取数工具查不到东西时转去本地资料库做一次语义检索。

        以前这类工具返回空就是一句「共 0 条」，模型只能据此说"没有数据"——
        但真实情况往往是"这张表确实没数据，线索藏在别处"（投诉里写了、事件描述里提了）。
        """
        b = [f"{label}：本企业暂无该类数据。"]
        if self.rag is None:
            return "\n".join(b).strip()
        try:
            q = ctx.question if (ctx.question and ctx.question.strip()) else label
            hits = self.rag.search(ctx.company_id, q, 3)
            if hits:
                b.append("\n本地资料库中与本次问题相关的片段（引用时写「来源ID=x」）：")
                for d in hits:
                    b.append(f"  · 来源ID={d.ref()} 类型={_type_label(d.ref())} "
                             f"标题={d.title}｜{_trim(d.snippet, 120)}")
        except Exception:  # noqa: BLE001
            # 兜底失败就当没有，绝不能因为"想多给点线索"把工具调用搞挂
            pass
        return "\n".join(b).strip()


# ------------------------------------------------------------------
# 小工具
# ------------------------------------------------------------------


def _number_web_sources(sources, ctx: ToolContext) -> List[Dict]:
    """为通过回检的来源分配全局序号（按 URL 去重、复用已有序号）。"""
    out: List[Dict] = []
    for s in sources or []:
        url = getattr(s, "url", None)
        if not url or not str(url).strip():
            continue
        idx, is_new = ctx.web_index_for(url)
        m = {"index": idx, "title": getattr(s, "title", ""), "url": url,
             "siteName": getattr(s, "site_name", ""), "snippet": getattr(s, "snippet", "")}
        if is_new == 0:
            m["reused"] = "1"
        out.append(m)
    return out


def _trail(raw: str, used: Optional[str], kept: int, dropped: int, note: str) -> Dict:
    """一次联网检索的完整经过（前端「联网检索经过」卡的原始数据）。"""
    return {"rawQuery": raw, "usedQuery": used, "kept": kept, "dropped": dropped, "note": note}


def _get_int(args: Dict, key: str, default: int, maximum: int) -> int:
    """取整数参数并**截断到上限**。

    Pydantic（``coerce_args``）负责"能不能转成整数"，它**不做上限截断** ——
    而这里的上限是硬性的：``limit=9999`` 会把整张表捞出来塞进上下文。
    所以这一层留着，别把它当成"手搓解析的残留"一并删掉。
    """
    v = args.get(key)
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return min(maximum, max(1, int(v)))
    if isinstance(v, str) and v.strip():
        try:
            return min(maximum, max(1, int(v.strip())))
        except ValueError:
            pass
    return default


def _str(v) -> str:
    return "" if v is None else str(v).strip()


def _nz(v) -> str:
    return "" if v is None else str(v)


def _trim(s, max_len: int) -> str:
    if s is None:
        return ""
    x = re.sub(r"\s+", " ", str(s)).strip()
    return x if len(x) <= max_len else x[:max_len] + "…"


def _type_label(ref: Optional[str]) -> str:
    """把引用标识翻译成人话，模型据此判断这条证据的分量。"""
    if not ref or not ref.strip():
        return "内部资料"
    if ":" not in ref:
        return "知识库文档"
    prefix = ref[: ref.index(":")]
    return {
        "metric": "经营指标",
        "event": "风险事件",
        "complaint": "客户投诉",
        "competitor": "竞品",
        "company": "企业档案",
        "rule": "风控规则",
    }.get(prefix, "内部资料")
