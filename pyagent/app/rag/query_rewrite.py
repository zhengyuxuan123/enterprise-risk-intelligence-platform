"""检索查询改写（零 token），对应 Java ``QueryRewrite``。

用户提问与知识库文档的用语常常不一致：提问是口语化的疑问句（"云帆科技为什么留不住客户？"），
文档却是书面化的陈述（"客户流失预警与客户挽留 SOP"）。直接拿原句检索会漏召回。
这里用规则把问句改写成「检索式」，并做领域同义/上下位扩展，产出多个查询变体供多路召回。

为什么默认不用大模型改写：一次分析里 RAG 可能被多次调用，每次都多花一轮对话 token，
收益却不稳。因此在 Retrieval 阶段只用确定性规则。

.. WARNING::
   Java 侧 ``LEXICON`` 用 ``Map.ofEntries`` 构建，而 **``Map.of`` / ``Map.ofEntries``
   的迭代顺序在同一份代码里是不保证的**（JVM 用随机化 SALT 做探测表）。
   于是 ``expanded()`` 拼出来的扩展词顺序**每次 JVM 启动都可能不同**，
   同一句提问会产出不同顺序的扩展式查询 —— 这是 Java 自身的非确定性，不是移植偏差。
   对账时按"词集合"比，不按字符串比；Python 侧按词表声明顺序输出，反而是稳定的。
"""

from __future__ import annotations

import re

from .textnorm import is_blank, trim, lower

# .. DANGER::
#    **这个字典的顺序是有语义的，不要按"看起来顺"重排。**
#
#    Java 侧是 ``Map.ofEntries(...)`` —— 它的迭代顺序是**未规定的**（探测表顺序），
#    实测**不等于声明顺序**：下面这份顺序是用 jshell 反射 ``QueryRewrite.LEXICON``
#    的 ``keySet()`` 真实 dump 出来的（见 ``qa/_java_ground_rag.jsh`` 的 LEXICON_ORDER）。
#
#    顺序之所以重要：``expanded()`` 按词表迭代顺序拼扩展词，多个词条同时命中时
#    （如"现金流与债务风险"同时命中 现金流 与 风险），谁先谁后直接决定产出的查询串，
#    进而决定多路召回的第二条路径。排错一个位置，多命中场景就会与 Java 不一致。
#
#    对账用例 ``LEXICON_ORDER`` 会专门校验这份顺序；换 JDK 若导致探测顺序变化，测试会红。
LEXICON: dict[str, list[str]] = {
    "sla": ["响应时限", "服务等级", "响应时长", "赔付标准"],
    "故障": ["宕机", "可用性", "事故", "容灾", "故障响应"],
    "现金流": ["资金", "回款", "应收", "账期", "资金链"],
    "投诉": ["客诉", "满意度", "投诉处理", "服务质量"],
    "库存": ["周转率", "积压", "备货", "缺货"],
    "数据": ["口径", "数据质量", "缺失值", "统计口径"],
    "大促": ["促销", "峰值", "活动期"],
    "利润": ["毛利", "净利", "毛利率", "成本控制"],
    "团队": ["人力", "离职率", "组织结构", "编制"],
    "债务": ["负债率", "偿债", "有息负债", "授信"],
    "gmv": ["成交额", "交易额", "营收"],
    "供应链": ["供应商", "交付延迟", "断供", "采购"],
    "安全": ["数据安全", "权限管控", "脱敏", "泄露"],
    "合规": ["监管", "处罚", "资质", "license", "合规审计"],
    "竞品": ["竞争对手", "价格策略", "替代方案"],
    "客户满意": ["满意度", "NPS", "回访"],
    "监控": ["预警指标", "观测", "看板", "巡检"],
    "流失": ["客户流失", "退订", "挽留", "churn"],
    "成本": ["费用管控", "降本", "成本结构"],
    "风险": ["预警", "阈值", "风控", "风险评估"],
    "竞争": ["竞品", "价格战", "市场份额", "竞争对手"],
    "留存": ["续费", "客户流失", "retention"],
    "续费": ["续约", "留存", "复购", "renewal"],
}

# 口语化前缀：删掉后不影响检索语义
PREFIX = [
    "我想问一下", "我想知道", "请帮我分析", "帮我分析", "请分析", "帮我看看", "请告诉我",
    "能否", "是否可以", "请问", "麻烦帮我", "麻烦", "我想要", "我想", "请", "帮我",
]

# 疑问句尾巴。Java 的 `\s` 只认 ASCII 空白，这里必须 re.ASCII。
_PAT_LEAD_PUNCT = re.compile(r"^[,，、：:]+", re.ASCII)
_TAIL: list[re.Pattern] = [
    re.compile(
        r"[，,]?\s*(主要)?(有)?(哪些|哪些原因|哪几种|有什么|有哪些|是如何|怎么|怎样|如何)"
        r"(处理|应对|解决|规避|预防|监控|预警|判断|识别)?\s*[?？]*\s*$",
        re.ASCII,
    ),
    re.compile(
        r"[，,]?\s*(的)?(原因|因素|信号|措施|方法|做法|办法|策略|建议)"
        r"(有哪些|有哪些呢|是什么|为何|是啥)?\s*[?？]*\s*$",
        re.ASCII,
    ),
    re.compile(r"\s*(是什么|为何|为啥|怎么会|怎么办|该如何|应该怎么)\s*[?？]*\s*$", re.ASCII),
    re.compile(r"\s*[?？。；;，,]+\s*$", re.ASCII),
]


def variants(query: str | None) -> list[str]:
    """产出查询变体：**第一位始终是原文**（保证不会改写坏），随后是核心式与扩展式。"""
    q = trim(query) if query else ""
    if is_blank(q):
        return []

    out: dict[str, None] = {q: None}
    core_q = core(q)
    if core_q and core_q != q:
        out[core_q] = None
    exp = expanded(core_q if core_q else q)
    if exp and exp != q and exp != core_q:
        out[exp] = None
    return list(out)


def core(query: str | None) -> str:
    """去掉口语前缀与疑问尾缀，得到"检索式"核心短语。"""
    q = trim(query) if query else ""
    if is_blank(q):
        return ""
    for p in PREFIX:
        if q.startswith(p) and len(q) > len(p):
            q = q[len(p):]
            break
    q = trim(_PAT_LEAD_PUNCT.sub("", q))
    # 尾缀可能被多个 pattern 连续命中，循环到稳定
    for _ in range(3):
        before = q
        for pat in _TAIL:
            m = pat.search(q)
            if m:
                cand = trim(q[: m.start()])
                # 别把整句都吃掉：至少留 2 个字
                if len(cand) >= 2:
                    q = cand
        if q == before:
            break
    return trim(q)


def expanded(core_query: str | None) -> str:
    """命中词表的词条做同义扩展，拼出第二条召回路径的查询。"""
    if core_query is None or is_blank(core_query):
        return ""
    low = lower(core_query)
    add: dict[str, None] = {}
    for key, words in LEXICON.items():
        if key in low:
            for w in words:
                if len(add) >= 6:
                    break
                add[w] = None
    if not add:
        return ""
    return core_query + " " + " ".join(add)
