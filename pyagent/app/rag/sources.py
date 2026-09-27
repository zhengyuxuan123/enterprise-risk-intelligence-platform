"""8 个语料源：把一类业务数据渲染成可入库的切片（对应 ``service/rag/source/*``）。

「写入即索引」要求每一类数据都能回答三个问题：

* 某条记录变了 → :meth:`CorpusSource.of`：给我这条记录现在该有的切片。
* 要全量回填 → :meth:`CorpusSource.all_of`：给我这家企业（或全局）的全部切片。
* 巡检对账 → :meth:`CorpusSource.ids_of`：库里实际存在的主键有哪些
  （用来发现「state 里有、库里没了」的僵尸切片）。

与 Java 的一处结构性差异
------------------------
Java 用 MyBatis-Plus 的 ``LambdaQueryWrapper`` 表达查询，子类实现 :meth:`scope`
返回一个条件对象。Python 没有等价物，这里把「条件」拆成
**(where 列表, order_by 列表)** 二元组的显式构造——语义相同，
但查询是**可见的 SQL 片段**，排查"为什么少召回一批"时不用去猜 ORM 生成了什么。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from sqlalchemy import asc, desc, select

from ..db import tables as T
from ..db.engine import Database, get_db
from .corpus import CorpusChunk, T as CT, blank, date, num, nz, trim, yes
from .text import split as split_text

log = logging.getLogger(__name__)


class CorpusSource(ABC):
    """语料源接口（对应 Java ``CorpusSource``）。"""

    @abstractmethod
    def type(self) -> str:  # noqa: A003 - 与 Java 同名，不改名以免对照时错位
        ...

    @abstractmethod
    def of(self, source_id: Optional[int]) -> List[CorpusChunk]:
        ...

    @abstractmethod
    def all_of(self, company_id: Optional[int]) -> List[CorpusChunk]:
        ...

    @abstractmethod
    def ids_of(self, company_id: Optional[int]) -> List[int]:
        ...

    def per_company(self) -> bool:
        """False = 全局语料（如风控规则），入库时 companyId 记为 None。"""
        return True


class RowCorpusSource(CorpusSource):
    """「一行数据 → 一片语料」的骨架（对应 Java ``RowCorpusSource``）。

    子类只需说清楚四件事：来源类型、一行怎么渲染、按企业怎么查、主键怎么取。
    剩下的 ``of / all_of / ids_of`` 都是样板代码。

    ``all_of`` 与 ``ids_of`` 都带上限保护：导入可能产生上万行数据，
    全量回填时不能把整表捞进内存。
    """

    table = None  # 子类覆盖

    def __init__(self, db: Optional[Database] = None) -> None:
        self.db = db or get_db()

    def scan_limit(self) -> int:
        return 5000

    @abstractmethod
    def render(self, row: dict) -> List[CorpusChunk]:
        ...

    @abstractmethod
    def scope(self, company_id: Optional[int]) -> Tuple[list, list]:
        """返回 ``(where 条件列表, order_by 列表)``。"""
        ...

    def id_of(self, row: dict) -> Optional[int]:
        return row.get("id")

    # -- 样板实现 ----------------------------------------------------

    def of(self, source_id: Optional[int]) -> List[CorpusChunk]:
        if source_id is None:
            return []
        try:
            row = self.db.fetch_one(select(self.table).where(self.table.c.id == source_id))
        except Exception as e:  # noqa: BLE001
            log.warning("语料源 %s 单条查询失败 %s: %s", self.type(), source_id, e)
            return []
        return [] if row is None else self.render(row)

    def all_of(self, company_id: Optional[int]) -> List[CorpusChunk]:
        try:
            where, order = self.scope(company_id)
            stmt = select(self.table).where(*where).order_by(*order).limit(self.scan_limit())
            rows = self.db.fetch_all(stmt)
        except Exception as e:  # noqa: BLE001
            log.warning("语料源 %s 全量查询失败 company=%s: %s", self.type(), company_id, e)
            return []
        out: List[CorpusChunk] = []
        for r in rows:
            try:
                out.extend(self.render(r))
            except Exception as e:  # noqa: BLE001 - 一行渲染失败不能拖垮整批
                log.warning("语料源 %s 渲染失败 id=%s: %s", self.type(), r.get("id"), e)
        return out

    def ids_of(self, company_id: Optional[int]) -> List[int]:
        try:
            where, order = self.scope(company_id)
            stmt = select(self.table.c.id).where(*where).order_by(*order).limit(self.scan_limit())
            rows = self.db.fetch_all(stmt)
        except Exception as e:  # noqa: BLE001
            log.warning("语料源 %s 主键查询失败 company=%s: %s", self.type(), company_id, e)
            return []
        out: List[int] = []
        for r in rows:
            i = self.id_of(r)
            if i is not None:
                out.append(int(i))
        return out


def _company_scope(col, company_id: Optional[int]) -> list:
    """``companyId == null`` → ``IS NULL``；否则 ``= 值``（与 Java 各 Source 一致）。"""
    return [col.is_(None)] if company_id is None else [col == company_id]


# ------------------------------------------------------------------
# 企业档案
# ------------------------------------------------------------------


class CompanySource(RowCorpusSource):
    """企业档案 → 语料。

    档案本身很短，进语料的价值不在"被检索到"，而在**给别的切片提供背景**：
    「这家企业是华东的制造业大客户」一旦进入上下文，模型对指标异常的判断口径会稳很多。
    """

    table = T.company

    def type(self) -> str:
        return CT.COMPANY

    def scope(self, company_id: Optional[int]) -> Tuple[list, list]:
        # 档案的主键就是企业 ID，回填逐企业调用
        if company_id is None:
            return [], [desc(T.company.c.id)]
        return [T.company.c.id == company_id], []

    def render(self, c: dict) -> List[CorpusChunk]:
        if not c or c.get("id") is None:
            return []
        name = nz(c.get("company_name"))
        cid = c.get("id")
        body = (
            f"企业档案【{name}】：编码 {nz(c.get('company_code'))}"
            f"，行业 {nz(c.get('industry'))}"
            f"，区域 {nz(c.get('region'))}"
            f"，客户等级 {nz(c.get('customer_level'))}"
            f"，规模 {nz(c.get('company_scale'))}。"
        )
        return [CorpusChunk(f"co:{cid}", self.type(), str(cid), cid,
                            f"企业档案 · {name}", body, 1, c.get("dept_id"))]


# ------------------------------------------------------------------
# 经营指标
# ------------------------------------------------------------------


class MetricSource(RowCorpusSource):
    """经营指标 → 语料（**行级切片**）。

    一条指标一片。问「毛利率最近是多少」这类问题时，语义检索能直接命中具体的那一条，
    而不必等模型自己想到去调工具。

    「近 N 期走势」「环比」这类**跨行**聚合信息不在这里——它们随任何一行变化而失效，
    由 :class:`AggregateSource` 按企业维度单独生成。
    """

    table = T.business_metric

    def type(self) -> str:
        return CT.METRIC

    def scan_limit(self) -> int:
        # 指标行数可能很大，但历史越久价值越低，回填只取最近的
        return 3000

    def scope(self, company_id: Optional[int]) -> Tuple[list, list]:
        return _company_scope(T.business_metric.c.company_id, company_id), [
            desc(T.business_metric.c.metric_date)
        ]

    def render(self, m: dict) -> List[CorpusChunk]:
        if not m or m.get("id") is None:
            return []
        name = nz(m.get("metric_name"))
        if name == "未填写":
            name = nz(m.get("metric_code"))
        body = (
            f"经营指标【{name}】{date(m.get('metric_date'))} 为 "
            f"{num(m.get('metric_value'))}{blank(m.get('unit'))}。"
        )
        if blank(m.get("metric_code")):
            body += f"指标代码 {blank(m.get('metric_code'))}。"
        if blank(m.get("source_type")):
            body += f"数据来源 {blank(m.get('source_type'))}。"
        return [CorpusChunk(f"m:{m['id']}", self.type(), str(m["id"]), m.get("company_id"),
                            f"经营指标 · {name}", body, 1, None)]


# ------------------------------------------------------------------
# 风险事件
# ------------------------------------------------------------------


class EventSource(RowCorpusSource):
    """风险事件 → 语料。

    事件会被反复修改（指派 / 处置 / 复核 / 关闭），每次变更都要重新切片入库——
    否则索引里会长期停留在 OPEN 状态，问「这个事件处置得怎么样」答的是几个月前的内容。
    """

    table = T.risk_event

    def type(self) -> str:
        return CT.EVENT

    def scope(self, company_id: Optional[int]) -> Tuple[list, list]:
        return _company_scope(T.risk_event.c.company_id, company_id), [
            desc(T.risk_event.c.created_at)
        ]

    def render(self, r: dict) -> List[CorpusChunk]:
        if not r or r.get("id") is None:
            return []
        title = nz(r.get("risk_title"))
        body = (
            f"风险事件【{title}】：类型 {nz(r.get('risk_type'))}"
            f"，等级 {nz(r.get('risk_level'))}"
            f"，状态 {nz(r.get('status'))}"
        )
        if r.get("trigger_value") is not None or r.get("threshold_value") is not None:
            body += f"，触发值 {num(r.get('trigger_value'))}（阈值 {num(r.get('threshold_value'))}）"
        if r.get("metric_date") is not None:
            body += f"，数据日期 {date(r.get('metric_date'))}"
        body += "。"
        if blank(r.get("handle_result")):
            body += f"处置情况：{trim(r.get('handle_result'), 300)}。"
        if blank(r.get("review_comment")):
            body += f"复核意见：{trim(r.get('review_comment'), 300)}。"
        return [CorpusChunk(f"e:{r['id']}", self.type(), str(r["id"]), r.get("company_id"),
                            f"风险事件 · {title}", body, 1, None)]


# ------------------------------------------------------------------
# 客户投诉
# ------------------------------------------------------------------


class ComplaintSource(RowCorpusSource):
    """客户投诉 → 语料。

    **刻意不写客户姓名**：切片会进 prompt，也会被检索结果带进回答里。
    一旦模型把客户姓名复述出来，输出护栏的 PII 检查是拦不住"从内部语料里读到的真名"的
    （它只对"看起来像手机号/身份证"的串报警）。用类别/产品/根因定位问题已经足够。
    """

    table = T.complaint

    def type(self) -> str:
        return CT.COMPLAINT

    def scope(self, company_id: Optional[int]) -> Tuple[list, list]:
        return _company_scope(T.complaint.c.company_id, company_id), [
            desc(T.complaint.c.complaint_date)
        ]

    def render(self, c: dict) -> List[CorpusChunk]:
        if not c or c.get("id") is None:
            return []
        cat = nz(c.get("category"))
        body = (
            f"客户投诉【类别 {cat}】{date(c.get('complaint_date'))}"
            f"，产品 {nz(c.get('product_name'))}"
            f"，严重度 {nz(c.get('severity'))}"
            f"，流失风险 {nz(c.get('churn_risk'))}"
            f"，状态 {nz(c.get('status'))}。"
        )
        if yes(c.get("repeat_flag")):
            body += "该投诉为重复投诉。"
        if yes(c.get("sla_exceeded")):
            body += "处理已超出 SLA。"
        if c.get("solve_hours") is not None:
            body += f"解决耗时 {num(c.get('solve_hours'))} 小时。"
        if blank(c.get("description")):
            body += f"客户反馈：{trim(c.get('description'), 400)}。"
        if blank(c.get("root_cause")):
            body += f"根因：{trim(c.get('root_cause'), 300)}。"
        return [CorpusChunk(f"c:{c['id']}", self.type(), str(c["id"]), c.get("company_id"),
                            f"客户投诉 · {cat}", body, 1, None)]


# ------------------------------------------------------------------
# 竞品
# ------------------------------------------------------------------


class CompetitorSource(RowCorpusSource):
    """竞品 → 语料。

    竞品是唯一一类"既没有进 RAG、又不会被结构化切片覆盖"的数据——
    改造前只能在模型恰好调用 ``get_competitors`` 工具时才出现。
    """

    table = T.competitor_product

    def type(self) -> str:
        return CT.COMPETITOR

    def scope(self, company_id: Optional[int]) -> Tuple[list, list]:
        return _company_scope(T.competitor_product.c.company_id, company_id), [
            desc(T.competitor_product.c.updated_date)
        ]

    def render(self, x: dict) -> List[CorpusChunk]:
        if not x or x.get("id") is None:
            return []
        name = nz(x.get("competitor_name"))
        prod = nz(x.get("product_name"))
        body = (
            f"竞品【{name}/{prod}】："
            f"标准价 {num(x.get('price'))}{blank(x.get('price_unit'))}"
            f"，竞争风险 {nz(x.get('risk_level'))}。"
        )
        if blank(x.get("target_customer")):
            body += f"目标客户：{trim(x.get('target_customer'), 120)}。"
        if blank(x.get("selling_point")):
            body += f"其卖点：{trim(x.get('selling_point'), 200)}。"
        if blank(x.get("weakness")):
            body += f"其短板：{trim(x.get('weakness'), 200)}。"
        if blank(x.get("promotion")):
            body += f"促销动作：{trim(x.get('promotion'), 150)}。"
        if blank(x.get("delivery_cycle")):
            body += f"交付周期 {trim(x.get('delivery_cycle'), 60)}。"
        if blank(x.get("service_commitment")):
            body += f"服务承诺：{trim(x.get('service_commitment'), 150)}。"
        return [CorpusChunk(f"p:{x['id']}", self.type(), str(x["id"]), x.get("company_id"),
                            f"竞品 · {name}/{prod}", body, 1, None)]


# ------------------------------------------------------------------
# 知识库文档
# ------------------------------------------------------------------


class KnowledgeSource(RowCorpusSource):
    """知识库文档 → 语料：**唯一需要分片**的语料源。

    文档正文最长 6 万字，整篇入库会让索引体积与写入耗时失控，
    而且命中段落可能在八千里之外。按 700 字切、留 100 字重叠。

    密级与部门取自文档自身字段（``security_level`` / ``dept_id``），
    不是"谁上传的就按谁的权限入"——写入即索引之后，
    文档可能在没人查询时就已经入库了。
    """

    table = T.knowledge_document
    TARGET = 700
    OVERLAP = 100

    def type(self) -> str:
        return CT.KNOWLEDGE

    def scope(self, company_id: Optional[int]) -> Tuple[list, list]:
        filters = _company_scope(T.knowledge_document.c.company_id, company_id)
        filters.append(T.knowledge_document.c.deleted == 0)
        filters.append(T.knowledge_document.c.ingest_status.in_(["INDEX_PENDING", "INDEXED", "INDEX_RETRY"]))
        return filters, [
            desc(T.knowledge_document.c.id)
        ]

    def render(self, d: dict) -> List[CorpusChunk]:
        if not d or d.get("id") is None or bool(d.get("deleted")):
            return []
        raw = d.get("content") or ""
        from .textnorm import trim  # 局部导入：避免与顶层循环

        body = trim(raw)
        if not body:
            return []
        title_raw = d.get("title")
        base = (
            trim(title_raw)
            if title_raw and not title_raw.isspace()
            else (d.get("original_filename") or "未命名文档")
        )
        try:
            level = max(1, int(d.get("security_level") or 1))
        except (TypeError, ValueError):
            level = 1

        parts = split_text(body, self.TARGET, self.OVERLAP)
        out: List[CorpusChunk] = []
        for i, part in enumerate(parts):
            t = f"{base}（第 {i + 1}/{len(parts)} 节）" if len(parts) > 1 else base
            out.append(CorpusChunk(f"k:{d['id']}#{i}", self.type(), str(d["id"]),
                                   d.get("company_id"), t, self._prefix(d, part),
                                   level, d.get("dept_id")))
        return out

    @staticmethod
    def _prefix(d: dict, body: str) -> str:
        """把文档类型放在正文开头：元数据本身就是很好的检索命中点。"""
        dt = d.get("doc_type")
        return f"【{dt}】{body}" if dt and not dt.isspace() else body


# ------------------------------------------------------------------
# 风控规则（全局语料）
# ------------------------------------------------------------------


class RiskRuleSource(RowCorpusSource):
    """风控规则 → 语料（**全局语料**，不属于任何企业）。

    规则表没有 companyId，它是平台级的判定标准。入库时 companyId 记为 None，
    查询条件因此变成 ``company:<本企业> OR company 为空``，
    于是"毛利率低于多少算高风险"这种问题在任何企业下都能召回。

    只收录 ``enabled=1`` 的规则：停用规则出现在证据里只会误导判断。
    """

    table = T.risk_rule

    def type(self) -> str:
        return CT.RULE

    def per_company(self) -> bool:
        return False

    def scope(self, company_id: Optional[int]) -> Tuple[list, list]:
        return [T.risk_rule.c.enabled == 1], [desc(T.risk_rule.c.id)]

    def render(self, r: dict) -> List[CorpusChunk]:
        if not r or r.get("id") is None:
            return []
        enabled = r.get("enabled")
        if enabled is not None and int(enabled) != 1:
            return []
        name = nz(r.get("rule_name"))
        body = (
            f"风控规则【{name}】：当指标 {nz(r.get('metric_code'))} "
            f"{_op(r.get('operator_code'))} {num(r.get('threshold_value'))} 时，"
            f"判定为 {nz(r.get('risk_level'))} 级{nz(r.get('risk_type'))}风险。"
        )
        if blank(r.get("description")):
            body += f"规则说明：{trim(r.get('description'), 300)}。"
        return [CorpusChunk(f"r:{r['id']}", self.type(), str(r["id"]), None,
                            f"风控规则 · {name}", body, 1, None)]


def _op(code: Optional[str]) -> str:
    """操作符代码 → 中文。与 Java ``RiskRuleSource.op`` 逐值对齐。"""
    if code is None:
        return "满足"
    c = code.strip().upper()
    return {
        "GT": "高于",
        ">": "高于",
        "LT": "低于",
        "<": "低于",
        "GE": "不低于",
        ">=": "不低于",
        "LE": "不高于",
        "<=": "不高于",
        "EQ": "等于",
        "=": "等于",
        "NE": "不等于",
        "!=": "不等于",
    }.get(c, code.strip())


# ------------------------------------------------------------------
# 聚合切片
# ------------------------------------------------------------------


class AggregateSource(CorpusSource):
    """跨行聚合切片：指标走势、风险事件分布、投诉类别分布（对应 Java ``AggregateSource``）。

    **为什么要和行级切片分开**：「近 6 期营收走势」是跨行算出来的，
    任何一条新指标写入，它的内容就变了。若挂在行级写入上，
    导入 1000 条指标会触发 1000 次聚合重算——每次都要回扫几十行数据。

    所以聚合切片的粒度定成**企业**：任何指标/事件/投诉变更都发一条 ``aggregate:<企业ID>``，
    由索引服务的合并窗口把短时间内的重复事件并成一次，
    最多延迟几秒重建，换来写放大从 O(行数) 降到 O(1)。
    """

    def __init__(self, corpus=None, db: Optional[Database] = None) -> None:
        from .structured import StructuredCorpus  # 局部导入：避免包级循环

        self.corpus = corpus or StructuredCorpus(db=db)
        self.db = db or get_db()

    def type(self) -> str:
        return CT.AGGREGATE

    def of(self, company_id: Optional[int]) -> List[CorpusChunk]:
        if company_id is None or self.corpus is None:
            return []
        try:
            out: List[CorpusChunk] = []
            for c in self.corpus.chunks(company_id):
                out.append(CorpusChunk(f"agg:{company_id}:{c.source_id}", self.type(),
                                       str(company_id), company_id, c.title, c.text, 1, None))
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("聚合切片生成失败 company=%s: %s", company_id, e)
            return []

    def all_of(self, company_id: Optional[int]) -> List[CorpusChunk]:
        return self.of(company_id)

    def ids_of(self, company_id: Optional[int]) -> List[int]:
        if company_id is None:
            # 全局巡检：所有企业的聚合切片
            rows = self.db.fetch_all(select(T.company.c.id).order_by(asc(T.company.c.id)))
            return [int(r["id"]) for r in rows if r.get("id") is not None]
        row = self.db.fetch_one(select(T.company.c.id).where(T.company.c.id == company_id))
        return [] if row is None else [int(company_id)]


# ------------------------------------------------------------------
# 注册表
# ------------------------------------------------------------------


def build_sources(db: Optional[Database] = None) -> List[CorpusSource]:
    """全部语料源，顺序与 Java 的 Bean 注入顺序一致（影响回填时的写入次序）。"""
    d = db or get_db()
    return [
        KnowledgeSource(d),
        MetricSource(d),
        EventSource(d),
        ComplaintSource(d),
        CompetitorSource(d),
        CompanySource(d),
        RiskRuleSource(d),
        AggregateSource(db=d),
    ]
