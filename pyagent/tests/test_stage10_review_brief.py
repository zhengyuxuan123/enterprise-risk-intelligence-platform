# -*- coding: utf-8 -*-
"""「检索员简报」回归。

这里钉住的是一个**被静默吞掉的**缺陷，不是新功能：

``RagHit.document_id`` 是 dataclass 字段（``Optional[int]``），却被写成 ``h.document_id()``
调用 —— 抛 ``'int' object is not callable``（document_id 为 None 时则是
``'NoneType' object is not callable``）。调用点在 ``AgentService`` 的 try/except 里，
异常只落一条 ``log.warning`` 就被吞掉，主链路照常往下走。

后果是**「检索员简报」这段从来没生效过**：多智能体召回的证据一条都没进过分析师上下文，
而界面、接口、门禁全都显示正常。它是被 ``--mode full`` 的日志刷出来才发现的——
零 token 的确定性测试覆盖不到"真有命中"这条路径。

所以这组用例的价值不在于断言文案，而在于：**只要有人再把字段当方法调用，测试立刻红。**
（顺带的教训：``except Exception + log.warning`` 会把功能失效伪装成"轻微告警"，
对于"静默失效即功能消失"的环节，要么让它显式失败，要么给它一条真断言。）
"""

from __future__ import annotations

from typing import List

import pytest

from app.agent.review_loop import ReviewLoopService
from app.rag.rag_service import RagHit


class _FakeRag:
    def __init__(self, hits: List[RagHit]) -> None:
        self._hits = hits
        self.seen: List[tuple] = []

    def search(self, company_id, question, top_k):
        self.seen.append((company_id, question, top_k))
        return self._hits


def _hit(title: str, score: float, snippet: str,
         document_id=None, source_ref=None) -> RagHit:
    return RagHit(document_id=document_id, title=title, snippet=snippet,
                  score=score, source_ref=source_ref)


def _brief(hits: List[RagHit]) -> str:
    return ReviewLoopService().retrieve(1, "毛利率为什么下滑", 5, _FakeRag(hits))


def test_structured_hit_is_annotated_with_source_id():
    """结构化切片（metric:12 这类）必须标出来源ID —— 分析师靠它写「来源ID=metric:12」引用。"""
    brief = _brief([_hit("毛利率指标", 88.0, "毛利率 32%", source_ref="metric:12")])
    assert "来源ID=metric:12" in brief, brief


def test_document_hit_has_no_source_id_annotation():
    """知识库文档命中走纯数字 documentId，不该再挂一个 `来源ID=` 前缀。"""
    brief = _brief([_hit("年度经营报告", 90.0, "毛利率同比下滑", document_id=7)])
    assert "来源ID=" not in brief, brief
    assert "年度经营报告" in brief


def test_mixed_hits_do_not_raise():
    """文档命中 + 结构化命中混在一起时不能抛异常 —— 这是原缺陷的直接回归。

    只要有人把 ``h.document_id`` 写回成 ``h.document_id()``，
    这条会立刻以 ``TypeError: 'int' object is not callable`` 失败。
    """
    hits = [
        _hit("年度经营报告", 91.0, "毛利率下滑 2pt", document_id=7),
        _hit("毛利率指标", 87.0, "毛利率 32%", source_ref="metric:12"),
        _hit("客户投诉记录", 80.0, "交付延迟投诉", source_ref="complaint:5"),
    ]
    brief = _brief(hits)
    assert "检索员简报" in brief
    assert "来源ID=metric:12" in brief
    assert "来源ID=complaint:5" in brief
    # 文档命中仍出现在简报里，只是不带来源ID标注
    assert "年度经营报告" in brief


def test_empty_hits_fall_back_to_explicit_sentence():
    """没命中时也要给一句明确的话，不能返回空串让分析师以为"检索过了没问题"。"""
    brief = _brief([])
    assert "未检索到" in brief


def test_top_k_and_company_are_passed_through():
    """参数顺序（company_id, question, top_k, rag）错了会以很隐蔽的方式错数据，钉一下。"""
    rag = _FakeRag([])
    ReviewLoopService().retrieve(3, "现金流风险", 7, rag)
    assert rag.seen == [(3, "现金流风险", 7)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
