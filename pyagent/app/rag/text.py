"""RAG 文本处理 + 长文本分片。

全部为确定性计算，**不消耗任何 token**，因此可以用固定语料做回归对账
（对账基准在 ``tests/data/java_ground_rag.txt``，注意那里登记了"有意偏离"的条目）。

字符串语义一律走 Python 原生（见 :mod:`app.rag.textnorm`）：长度与下标按**码点**，
空白按 Unicode。分片规则变了，**已入库的 chunk 需要用新规则重建索引**。
"""

from __future__ import annotations

import regex as re

from .textnorm import is_blank, trim

# ---------------------------------------------------------------- 停用词

STOPWORDS = frozenset(
    {
        "如何", "应该", "怎样", "怎么", "什么", "哪些", "哪个", "请问", "一下",
        "我们", "你们", "他们", "可以", "需要", "是否", "以及", "对于", "关于",
        "这个", "那个", "这些", "那些", "进行", "相关", "方面", "问题", "如果",
        "那么", "但是", "而且", "所以", "因为", "由于", "通过", "根据", "可能",
        "一个", "时候", "为什么", "还有", "就是", "不是", "分析", "介绍", "说明",
        "情况", "多少", "主要", "目前", "现在", "最近",
    }
)

# Java 源码里是 `[^a-z0-9]+` 与 `[^\p{IsHan}]+`：
# Python 内置 re 不认识 \p{...}，必须换成 regex 模块的 \p{Han}（脚本类，与 Java IsHan 对齐）。
_PAT_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PAT_NON_HAN = re.compile(r"[^\p{Han}]+")
_PAT_NON_HAN_ALNUM = re.compile(r"[^\p{Han}a-z0-9]")


def lower(v: str | None) -> str:
    """小写化。``None`` → 空串。

    .. NOTE::
        这里不再走 ``textnorm.lower`` 的导入：本模块自己就导出 ``lower``，
        同名导入会把它变成递归调用（sed 批量改名时真的踩过一次，表现是
        ``RecursionError`` 而不是"函数不存在"，很容易误判）。
    """
    return "" if v is None else v.lower()


def tokenize(query: str | None) -> list[str]:
    """中英文混合分词：英文按单词，中文保留短词与滑动二元组。

    返回**有序去重**列表（Java 用 ``LinkedHashSet``，迭代顺序即插入顺序）。
    """
    x = lower(query)
    seen: dict[str, None] = {}
    for w in _PAT_NON_ALNUM.split(x):
        if len(w) >= 2:
            seen[w] = None
    for run in _PAT_NON_HAN.split(x):
        if not run:
            continue
        if len(run) <= 4:
            seen[run] = None
        for i in range(len(run) - 1):
            seen[run[i : i + 2]] = None
    for k in list(seen):
        if k in STOPWORDS:
            del seen[k]
    return list(seen)


def effective_terms(terms: list[str], corpus: list[str]) -> list[str]:
    """只保留语料中真实出现过的词（DF>0）。"""
    out: list[str] = []
    for t in terms:
        for body in corpus:
            if t in body:
                out.append(t)
                break
    return out


def bigrams(text: str | None) -> list[str]:
    """字级二元组（有序去重），用于邻接 proximity 与 MMR 冗余度。"""
    x = _PAT_NON_HAN_ALNUM.sub("", lower(text))
    seen: dict[str, None] = {}
    if len(x) < 2:
        return []
    for i in range(len(x) - 1):
        seen[x[i : i + 2]] = None
    return list(seen)


def jaccard(a, b) -> float:
    """两个集合的 Jaccard 相似度（0~1）。"""
    if not a or not b:
        return 0.0
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    sa = set(small)
    sb = set(big)
    inter = len(sa & sb)
    return inter / (len(a) + len(b) - inter)


def occurrences(text: str, term: str) -> int:
    """非重叠出现次数。

    Python 的 ``str.count`` 与 Java 的 ``indexOf`` 循环都是"左到右、非重叠"，语义一致。
    """
    if not term:
        return 0
    return text.count(term)


# ---------------------------------------------------------------- 分片

# 句读边界：。！？；\n .
_BREAK_CHARS = frozenset("。！？；\n.")
_PAT_WS_RUN = re.compile(r"[ \t\r]+")


def split(text: str | None, target: int, overlap: int) -> list[str]:
    """长文本分片：按句读切，累积到目标长度成片，片间保留重叠。

    为什么必须分片：知识库文档最长 6 万字。整篇当一片会让中文 bigram 分词产生
    十几万个 token，索引体积与写入耗时失控；而且检索只返回一篇的摘要，
    命中段落可能在八千里外。

    为什么留重叠：硬切会把跨边界的一句话劈成两半，两半都不像人话、各自召回率都很低。

    .. NOTE::
       **全程按码点操作**（Python 原生）。
       以前这里按 UTF-16 码元（为了对齐 Java ``String.substring``），
       emoji 这类 astral 字符一个算两个，分片长度会在它们出现的地方整体偏长 ——
       不报错，只是悄悄偏，正是最难查的那一类。
    """
    out: list[str] = []
    if text is None or is_blank(text):
        return out
    s = trim(_PAT_WS_RUN.sub(" ", text))
    n = len(s)
    if n <= target:
        return [s] if s else []

    start = 0
    while start < n:
        end = min(n, start + target)
        if end < n:
            end = _retreat_to_break(s, end, start + (target // 2))
        piece = trim(s[start:end])
        if piece:
            out.append(piece)
        if end >= n:
            break
        nxt = max(start + 1, end - overlap)
        if nxt <= start:
            nxt = end
        start = nxt
    return out


def _retreat_to_break(s: str, end: int, floor: int) -> int:
    """从 end 往前找最近的句读边界，找不到就原地返回（宁可硬切也不要死循环）。"""
    for i in range(end, floor, -1):
        if s[i - 1] in _BREAK_CHARS:
            return i
    return end
