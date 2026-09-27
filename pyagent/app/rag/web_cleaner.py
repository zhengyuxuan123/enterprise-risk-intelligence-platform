"""联网检索词的确定性出口清洗，对应 Java ``WebQueryCleaner``。

**为什么必须有这一层**：检索词是模型自己生成的。实测模型会把意图写成一长串关键词
``2024 行业报告 SaaS 客户流失 定义 投资 产品 行业趋势 客户成功``，
这串词被原样拼进 ``cn.bing.com/search?q=...`` 之后，搜索引擎会对它做宽松分词，
专挑**最容易命中的那个通用词**：``2024`` → 日历/年鉴/年终报道，
``如何`` → 汉语词典词条，``ChurnZero`` 被切成 ``cwnu`` → 某师范大学整站。
于是"参考来源"里全是垃圾，而模型拿到垃圾并不会自知，只会换词重搜、越搜越偏。

这一层不看模型脸色、也不问模型：按确定性规则处理，**零 token**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .textnorm import is_blank, trim, lower, length

LEAD_NOISE = [
    "请", "帮我", "麻烦", "我想知道", "我想了解", "了解一下", "查询", "查一下", "检索", "搜索",
    "请分析", "请说明", "请介绍", "请问", "解释一下", "分析一下", "介绍下", "关于", "针对", "有关",
]

LEAD_QUESTION = [
    "如何", "怎么", "怎样", "为啥", "为何", "为什么", "什么原因", "是什么", "什么是",
    "有哪些", "哪些", "哪个", "哪种", "是否", "能否", "可否", "会不会", "有没有",
]

INNER_NOISE = [
    "如何", "怎么", "怎样", "为什么", "为何", "是否", "有没有", "会不会", "能否", "可否",
    "的原因", "原因是什么", "的定义", "是什么", "什么意思", "什么意思?",
    "请", "帮我", "麻烦", "一下", "的", "地", "得", "了", "吗", "呢", "啊",
]

TAIL_NOISE = [
    "定义", "含义", "意思", "概念", "简介", "介绍", "说明", "概述", "描述", "解释",
    "情况", "状况", "分析", "研究", "报告", "资料", "信息", "数据", "背景",
    "建议", "措施", "对策", "方法", "办法", "方案", "策略", "手段", "思路", "做法",
    "影响", "趋势", "趋势分析", "变化", "原因", "成因", "风险点", "注意", "要点",
    "是什么", "有哪些", "以及相关", "等内容", "等方面",
]

PURE_NOISE = frozenset(
    {
        "定义", "含义", "意思", "概念", "简介", "介绍", "说明", "概述", "解释", "回答",
        "如何", "怎么", "怎样", "为什么", "为何", "是否", "请", "帮我", "麻烦", "一下",
        "客户", "公司", "企业", "风险", "问题", "情况", "建议", "措施", "方法", "对策",
        "最新", "相关", "有关", "具体", "详细", "以及其他", "等等", "一些", "某些",
    }
)

# 拆词分隔符：空白、中英文标点、全角符号（注意含全角空格 U+3000）
_PAT_SPLIT = re.compile(r"[\s　,，、;；/|+&()（）【】\[\]「」《》<>\"'：:!！?？.。~～*]+")
_PAT_PURE_ASCII = re.compile(r"[a-z0-9\-_\.]+")
_PAT_EDGE_PUNCT = re.compile(r"^[\-\._]+|[\-\._]+$")
_PAT_DIGITS = re.compile(r"\d+", re.ASCII)

ASCII_NOISE = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this",
        "what", "why", "how", "are", "was", "howto", "please", "about", "latest", "best", "top",
        "new", "you", "your", "can", "get", "use",
    }
)

MAX_TERMS = 5


@dataclass
class CleanedQuery:
    """清洗结果。``accepted=false`` 时调用方应**直接拒绝**这次联网检索。"""

    query: str | None
    accepted: bool
    reason: str
    terms: list[str] = field(default_factory=list)


def clean(raw: str | None) -> CleanedQuery:
    """清洗检索词。"""
    src = trim(raw) if raw else ""
    if is_blank(src):
        return CleanedQuery(None, False, "检索词为空", [])

    pieces: list[str] = []
    for p in _PAT_SPLIT.split(src):
        if p and not is_blank(p):
            pieces.append(trim(p))
    if not pieces:
        pieces.append(src)

    kept: dict[str, None] = {}
    reject_parts: list[str] = []

    for piece in pieces:
        t = trim(lower(piece))
        if is_blank(t):
            continue

        # 纯英文/数字串：整体处理，不参与中文剥词
        if _PAT_PURE_ASCII.fullmatch(t):
            a = _ascii_term(t)
            if a is not None:
                kept[a] = None
            else:
                reject_parts.append(t)
            continue

        c = _strip_lead_question(t)
        c = _strip_lead_noise(c)
        c = _strip_inner_noise(c)
        c = _strip_tail_noise(c)
        c = trim(c)

        if is_blank(c) or c in PURE_NOISE or length(c) < 2:
            reject_parts.append(t)
            continue
        kept[c] = None

    terms = list(kept)
    if len(terms) > MAX_TERMS:
        terms = terms[:MAX_TERMS]

    if not terms:
        return CleanedQuery(
            None,
            False,
            "检索词「" + src + "」清洗后无任何有效词：全是疑问词/通用词/年份，已驳回本次联网检索。"
            "请改写成具体的检索短语（3-5 个词），例如「SaaS 客户流失率 预警指标」。",
            [],
        )

    query = " ".join(terms)
    if len(query) < 2:
        return CleanedQuery(None, False, "清洗后检索词过短：「" + src + "」", terms)

    sb = "原始「" + src + "」→ 清洗「" + query + "」"
    if reject_parts:
        sb += "，剔除噪音词：" + " ".join(reject_parts).strip()
    return CleanedQuery(query, True, sb, terms)


def _ascii_term(t: str) -> str | None:
    """英文/数字串：去停用词、去纯年份、太短不要。"""
    s = _PAT_EDGE_PUNCT.sub("", t)
    if is_blank(s):
        return None
    if _PAT_DIGITS.fullmatch(s):
        v = int(s)
        if 1900 <= v <= 2099:
            return None  # 年份：没有检索价值，只会招来日历和年鉴
        return s if len(s) >= 2 else None
    if len(s) < 3:
        return None
    if s in ASCII_NOISE:
        return None
    return s


def _strip_lead_question(s: str) -> str:
    cur = s
    changed = True
    while changed:
        changed = False
        for p in LEAD_QUESTION:
            if len(cur) > len(p) and cur.startswith(p):
                cur = cur[len(p):]
                changed = True
                break
    return cur


def _strip_lead_noise(s: str) -> str:
    cur = s
    changed = True
    while changed:
        changed = False
        for p in LEAD_NOISE:
            if len(cur) > len(p) and cur.startswith(p):
                cur = cur[len(p):]
                changed = True
                break
    return cur


def _strip_inner_noise(s: str) -> str:
    cur = s
    for p in INNER_NOISE:
        if len(cur) > len(p) + 1:
            cur = cur.replace(p, "")
    return cur


def _strip_tail_noise(s: str) -> str:
    cur = s
    # 倒序匹配：优先剥最长的尾巴（"趋势分析" 先于 "分析"）
    tails = sorted(TAIL_NOISE, key=len, reverse=True)  # sorted 稳定，reverse 不破坏稳定性
    changed = True
    while changed:
        changed = False
        for p in tails:
            if len(cur) > len(p) and cur.endswith(p):
                cur = cur[: len(cur) - len(p)]
                changed = True
                break
    return cur
