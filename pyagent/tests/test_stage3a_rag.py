"""阶段 3a：检索确定性层的离线回归。

* 对账部分读冻结的 Java 基准，**不联网、不消耗 token、不需要 Java 在跑**。
* 后面另有一组纯语义单测，覆盖基准里不好表达、但恰恰最容易移植错的边界
  （全角空格、NBSP、emoji 码元、HALF_UP 舍入）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from app.rag.embedding import LocalEmbedding, cosine, tokenize as emb_tokenize
from app.rag.textnorm import (
    fmt2,
    head,
    is_blank,
    length,
    lower,
    round3,
    trim,
)
from app.rag.relevance import SourceRelevanceFilter, WebSource
from app.rag.text import bigrams, jaccard, split, tokenize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_parity import run as run_parity  # noqa: E402


# ---------------------------------------------------------------- 对账

def test_parity_against_java_ground_truth():
    """逐字节对齐 jshell 跑出来的 Java 真身输出。"""
    ok, bad, missing = run_parity()
    assert missing == 0, f"有 {missing} 条用例缺基准（基准过期？重跑 jshell 取真）"
    assert not bad, "与 Java 不一致：\n" + "\n".join(bad)
    assert ok > 0


# ---------------------------------------------------------------- 文本规范化（Python 原生）

def test_trim_removes_unicode_whitespace():
    """trim 删的是 **Unicode 空白**，不是 Java 那套「<= U+0020」。

    旧实现为了对齐 Java ``String.trim()`` 会留下全角空格 U+3000 与 NBSP U+00A0，
    于是「以全角空格开头的语料」分片起点比看起来靠后一个字符 —— 不报错，只是悄悄偏。
    """
    assert trim("　全角") == "全角"
    assert trim(" nbsp ") == "nbsp"
    assert trim("  a b  ") == "a b"
    assert trim(None) == ""


def test_is_blank_treats_unicode_whitespace_as_blank():
    """与 :func:`trim` 同一套空白定义（不会各判一套）。"""
    assert is_blank(" ")
    assert is_blank("　")
    assert is_blank("   ")
    assert is_blank("")
    assert is_blank(None)
    assert not is_blank("x")


def test_length_and_head_are_code_point_based():
    """长度与切片按**码点**：一个 emoji 算一个字符，不是两个。"""
    assert length("风险") == 2
    assert length("😀") == 1
    s = "😀" * 120
    assert length(s) == 120
    assert length(head(s, 100)) == 100
    assert head(s, 0) == ""
    assert head(None, 5) == ""
    assert head("风险可控", 2) == "风险"


def test_round3_and_fmt2_follow_python_rounding():
    """走 Python 默认（银行家舍入），不再为对齐 Java 走 HALF_UP。

    .. NOTE::
       不写死 ``fmt2(0.125) == "0.13"`` 这类字面量：那正是旧实现（HALF_UP）的期望值。
       改成与 Python 原生 `:func:`format`` 对照，语义「和 Python 一致」才被钉住。
    """
    assert round3(0.1234) == 0.123
    assert round3(1.0) == 1.0
    assert fmt2(0.125) == f"{0.125:.2f}"
    assert fmt2(0.0) == "0.00"
    assert fmt2(1.005) == f"{1.005:.2f}"


def test_lower_handles_none_as_empty_string():
    assert lower(None) == ""
    assert lower("AbC") == "abc"


# ---------------------------------------------------------------- 文本层

def test_split_never_grows_past_target_and_keeps_overlap():
    text = "。" .join(["客户流失预警管理条款第%d条" % i for i in range(1, 40)])
    pieces = split(text, 100, 30)
    assert len(pieces) > 1
    for p in pieces:
        assert length(p) <= 100 + 40  # 回退到句读可能略超，但不能失控


def test_split_is_deterministic():
    text = "现金流紧张。" * 30
    assert split(text, 50, 10) == split(text, 50, 10)


def test_tokenize_drops_stopwords_but_keeps_their_bigrams():
    """"为什么"是停用词，但它参与组成的 bigram（为什/么留）不是。"""
    toks = tokenize("为什么留不住")
    assert "为什么" not in toks
    assert "为什" in toks


def test_bigrams_ignores_punctuation():
    assert bigrams("风险，预警！") == ["风险", "险预", "预警"]
    assert bigrams("单") == []


def test_jaccard_symmetric_and_bounded():
    a = bigrams("客户流失预警")
    b = bigrams("客户流失")
    assert 0.0 <= jaccard(a, b) <= 1.0
    assert math.isclose(jaccard(a, b), jaccard(b, a))
    assert jaccard([], a) == 0.0


# ---------------------------------------------------------------- 向量

def test_local_embedding_is_deterministic_and_unit_length():
    le = LocalEmbedding(64)
    v1 = le.embed("客户流失风险")
    v2 = le.embed("客户流失风险")
    assert np.array_equal(v1, v2)
    assert math.isclose(float(np.linalg.norm(v1)), 1.0, abs_tol=1e-6)


def test_local_embedding_returns_none_for_blank():
    le = LocalEmbedding(64)
    assert le.embed("") is None
    assert le.embed("   ") is None
    assert le.embed(None) is None


def test_local_embedding_dim_is_clamped():
    assert LocalEmbedding(0).dim == 512
    assert LocalEmbedding(-5).dim == 512
    assert LocalEmbedding(1).dim == 64
    assert LocalEmbedding(99999).dim == 4096


def test_cosine_of_identical_text_is_one():
    le = LocalEmbedding(64)
    assert math.isclose(cosine(le.embed("现金流"), le.embed("现金流")), 1.0, abs_tol=1e-5)


def test_embedding_tokenize_single_cjk_char_is_kept():
    """单字（如"税"）也要能召回，否则短查询分词结果全空。"""
    assert "税" in emb_tokenize("税")


# ---------------------------------------------------------------- 来源回检

def _src(i: int, title: str, url: str, snip: str = "") -> WebSource:
    return WebSource(i, title, url, "site", snip)


def test_filter_drops_irrelevant_sources():
    flt = SourceRelevanceFilter(LocalEmbedding(512), 0.16, 0.18, 5)
    srcs = [
        _src(0, "客户流失预警指标", "https://good", "流失率阈值与挽留流程"),
        _src(1, "2024年日历", "https://bad", "全年日历查询"),
    ]
    res = flt.filter("客户流失怎么办", "客户流失 预警", srcs)
    urls = [s.url for s in res.kept]
    assert "https://good" in urls
    assert "https://bad" not in urls


def test_filter_keeps_one_when_all_below_threshold():
    """全部判无关时保留最优 1 条 —— "来源区一片空白"是更糟的体验。"""
    flt = SourceRelevanceFilter(LocalEmbedding(512), 0.16, 0.18, 5)
    srcs = [_src(0, "完全无关的内容", "https://x", "天气与星座运势")]
    res = flt.filter("客户流失率预警阈值", "客户流失 预警", srcs)
    assert len(res.kept) == 1
    assert not res.detail[0].keep  # 但标记仍是"未通过"


def test_strict_kept_is_the_only_truth_for_has_relevant():
    """★ ``kept`` 是给界面看的，``strict_kept`` 才是"有没有相关结果"的依据。

    2026-09-22 之前的坑：``kept`` 在全部不相关时也会补一条最高分的（为了不留白），
    上层若用 ``if not kept`` 判断"来源全都无关"，这条分支**永远不成立**——
    于是 Bing 抓回来的无关网页会被编上 ``[1]`` 当成证据递给模型。
    这个断言就是守那条防线不被改回去的。
    """
    flt = SourceRelevanceFilter(LocalEmbedding(512), 0.16, 0.18, 5)
    srcs = [_src(0, "2026年放假安排一览", "https://rili", "日历表"),
            _src(1, "芒果TV-天生青春", "https://mgtv", "视频网站")]
    res = flt.filter("出口退税政策调整", "出口退税 税务总局公告", srcs)
    assert res.kept, "展示层仍留一条，避免来源区空白"
    assert res.strict_kept == [], "★ 严格口径必须为空：一条都不相关"
    assert res.has_relevant() is False
    assert res.dropped_count() == len(srcs), "兜底留下的那条也算没通过"

    good = [_src(0, "关于调整出口退税政策的公告", "https://gov.cn", "税务总局公告出口退税率调整")]
    res2 = flt.filter("出口退税政策调整", "出口退税 调整", good)
    assert res2.strict_kept and res2.has_relevant()
    assert res2.dropped_count() == 0


def test_filter_empty_sources_is_noop():
    flt = SourceRelevanceFilter(LocalEmbedding(512))
    res = flt.filter("问题", "查询", [])
    assert res.kept == []
    assert res.total == 0


def test_long_question_does_not_dilute_relevance():
    """★ 问题写得长，不该把相关来源的分数压下去。

    修之前：``tools._web_search`` 调的是 ``filter(ctx.question, cq.query, srcs)``，
    而 ``filter`` 把整句问题与检索词**混成一个词集**去算覆盖率。覆盖率是个**比例**，
    分母随问题长度膨胀 —— 实测同一条明显相关的网页分数从 0.40 掉到 0.15，
    被阈值全部拦下。真实分析里因此出现「联网调了 4 次、来源 0 条、结论写未联网核实」。

    这条断言守的就是"检索词是判据主体、问题只是低权背景"这个分工不被改回去。
    """
    flt = SourceRelevanceFilter(LocalEmbedding(512), 0.16, 0.18, 5)
    short = "客户流失率 上升 原因 行业"
    long_q = ("为什么最近客户流失风险升高？请结合经营指标、投诉、竞品和知识库给出证据，"
              "并联网对照行业公开资料")
    srcs = [
        _src(0, "客户流失预警的关键指标与阈值设定方法", "https://good1", "流失率阈值与挽留流程"),
        _src(1, "客户管理中的流失危机:深度解析原因与应对策略", "https://good2", "流失原因分析"),
        _src(2, "2026年中国顾客满意度指数研究成果发布", "https://bad1", "消费者体验指数回升"),
    ]
    only_query = flt.filter("", short, srcs)
    with_question = flt.filter(long_q, short, srcs)

    kept_only = {s.url for s in only_query.strict_kept}
    kept_with = {s.url for s in with_question.strict_kept}
    assert len(kept_only) >= 2, "只用检索词时，相关来源应被保留"
    assert kept_with == kept_only, "★ 长问题的存在不该改变取舍结果"
    assert "https://bad1" not in kept_with, "无关来源仍须被丢弃"
