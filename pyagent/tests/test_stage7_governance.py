"""阶段 7：追溯查询 / 复核回环 / 主动预警 / 通知通道。

这批模块共同的特点是**"不出错但悄悄不做"**的风险特别高：

* 追溯查不到 → 空对象，看起来跟"没问题"一样；
* 复核确定性信号干净 → 免模型复核，看起来跟"复核通过"一样；
* 预警被闸门挡下 → 返回 skipped，看起来跟"研判过了"一样；
* 通知没配 webhook → 回执 REGISTERED，**不能当成已送达**。

所以测试的重点不是"能跑通"，而是**把每种静默结果都钉死成可断言的状态**。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from app.agent.grounding import Check
from app.agent.notification import NotificationService, Receipt
from app.agent.proactive import (_build_question, _parse_headline, _parse_level,
                                 ProactiveRiskService)
from app.agent.review_loop import ReviewLoopService, ReviewResult
from app.agent.trace_query import CHUNK_PREFIX, TraceQueryService
from app.rag.rag_service import RagHit


def _params(stmt) -> dict:
    """SQLAlchemy 把值放在语句里而不是 ``params`` 实参里，所以要编译后取。"""
    try:
        return dict(stmt.compile().params)
    except Exception:  # noqa: BLE001
        return {}


def _sql(stmt) -> str:
    return str(stmt)


class FakeDb:
    """最小 DB 替身。

    值的传递方式是个坑：SQLAlchemy 的 ``insert().values(**row)`` 把值编译进语句，
    **不在** ``execute(stmt, params)`` 的第二个实参里。所以这里统一从
    ``stmt.compile().params`` 取，否则"写入了什么"永远断言不到。
    """

    def __init__(self, chunks: Optional[Dict[str, dict]] = None,
                 analysis_row: Optional[dict] = None,
                 trace_rows: Optional[List[dict]] = None, scalar: Any = 0) -> None:
        self.chunks = chunks or {}
        self.analysis_row = analysis_row
        self.trace_rows = trace_rows or []
        self.scalar_value = scalar
        self.inserted: List[dict] = []
        self.updated: List[dict] = []

    # -- 读 ----------------------------------------------------------

    def fetch_one(self, stmt, params=None):
        text = _sql(stmt)
        if "ai_analysis_trace" in text:
            return self.trace_rows[0] if self.trace_rows else None
        if "rag_chunk_state" in text:
            return _match_chunk(self.chunks, stmt)
        return self.analysis_row

    def fetch_all(self, stmt, params=None):
        text = _sql(stmt)
        if "ai_analysis_trace" in text:
            return list(self.trace_rows)
        return [self.analysis_row] if self.analysis_row else []

    def scalar(self, stmt, params=None, default=0):
        return self.scalar_value

    # -- 写 ----------------------------------------------------------

    def insert_id(self, stmt, params=None):
        self.inserted.append(_params(stmt))
        return 11

    def execute(self, stmt, params=None):
        self.updated.append(_params(stmt))
        return 1


def _match_chunk(chunks: Dict[str, dict], stmt) -> Optional[dict]:
    """按编译后的绑定值模拟三条还原规则：精确 / ``k:{n}#%`` / ``%{ref}``。"""
    p = _params(stmt)
    for v in p.values():
        if not isinstance(v, str):
            continue
        if v in chunks:
            return chunks[v]
        if v.endswith("#%"):                      # 知识库分片前缀
            base = v[:-1]
            for k, val in chunks.items():
                if k.startswith(base):
                    return val
        if v.startswith("%"):                     # 聚合切片后缀
            tail = v[1:]
            for k, val in chunks.items():
                if k.endswith(tail):
                    return val
    return None


# ==================================================================
# 追溯查询
# ==================================================================


class TestTraceQuery:
    def test_trace_not_found_is_explicit(self):
        """查不到要显式标 ``found=false``，不能返回空 dict 让人误读成"没留痕"。"""
        assert TraceQueryService(FakeDb()).trace(999999) == {"found": False}

    def test_lineage_not_found_is_explicit(self):
        assert TraceQueryService(FakeDb()).lineage(999999) == {"found": False}

    def test_trace_without_detail_rows_says_why(self):
        """留痕表上线前的历史分析：只有汇总字段，必须说明原因。"""
        db = FakeDb(analysis_row={"id": 5, "trace_id": "abc", "question": "q",
                                  "degrade_level": "NONE", "duration_ms": 12,
                                  "llm_calls": 1, "tool_calls": 0,
                                  "answer": "", "evidence_json": None})
        out = TraceQueryService(db).trace(5)
        assert out["found"] is True
        assert out["analysisId"] == 5
        assert out["detail"] is None
        assert "留痕" in out["detailNote"]

    def test_trace_maps_camel_case_keys(self):
        db = FakeDb(
            analysis_row={"id": 7, "trace_id": "t7", "degrade_level": "PARTIAL",
                          "duration_ms": 33, "llm_calls": 2, "tool_calls": 4,
                          "answer": "", "evidence_json": None},
            trace_rows=[{"trace_id": "t7", "iterations": 2,
                         "route_json": json.dumps({"agents": ["a"]}),
                         "grounding_json": json.dumps({"ok": True}),
                         "tool_detail": "not-json"}])
        out = TraceQueryService(db).trace(7)
        assert out["degradeLevel"] == "PARTIAL"
        assert out["durationMs"] == 33
        assert out["llmCalls"] == 2
        assert out["detail"]["route"] == {"agents": ["a"]}
        assert out["detail"]["grounding"] == {"ok": True}
        # 解析不了就原样返回字符串，不丢信息
        assert out["detail"]["toolDetail"] == "not-json"


class TestLineageRefResolution:
    """正文的 ref 与索引 chunk_id 不是同一套编码，还原规则必须逐条钉死。"""

    def _svc(self, chunks: Dict[str, dict]):
        return TraceQueryService(FakeDb(chunks=chunks))

    def test_chunk_prefix_table_covers_row_sources(self):
        assert CHUNK_PREFIX["knowledge"] == "k"
        assert CHUNK_PREFIX["metric"] == "m"
        assert CHUNK_PREFIX["event"] == "e"
        assert CHUNK_PREFIX["complaint"] == "c"
        assert CHUNK_PREFIX["competitor"] == "p"
        assert CHUNK_PREFIX["company"] == "co"

    def test_web_ref_is_never_resolved(self):
        """网页来源本就不在语料库里，解析不到不算异常。"""
        svc = self._svc({})
        assert svc._resolve_chunk("https://a.com/x", "web") is None

    def test_direct_chunk_id_hit(self):
        svc = self._svc({"m:12": {"chunk_id": "m:12", "source_type": "metric",
                                  "source_id": "12", "status": "INDEXED"}})
        hit = svc._resolve_chunk("m:12", "internal")
        assert hit is not None and hit["source_id"] == "12"

    def test_structured_ref_is_mapped_to_prefix(self):
        """``metric:12`` 在索引里是 ``m:12``。"""
        svc = self._svc({"m:12": {"chunk_id": "m:12", "source_type": "metric",
                                  "source_id": "12", "status": "INDEXED"}})
        assert svc._resolve_chunk("metric:12", "internal") is not None

    def test_knowledge_doc_no_falls_back_to_shard_prefix(self):
        """知识库 ref 是文档号（7），索引里带分片序号（k:7#0）。"""
        svc = self._svc({"k:7#0": {"chunk_id": "k:7#0", "source_type": "knowledge",
                                   "source_id": "7", "status": "INDEXED"}})
        hit = svc._resolve_chunk("7", "knowledge")
        assert hit is not None and hit["chunk_id"] == "k:7#0"


class TestCollectRefs:
    def test_prefers_ref_then_document_id_then_url(self):
        svc = TraceQueryService(FakeDb())
        ev = [
            {"tool": "search_knowledge", "sourceType": "knowledge",
             "sources": [{"documentId": "7", "title": "制度"}]},
            {"tool": "web_search", "sourceType": "web",
             "sources": [{"url": "https://x", "title": "报道"}]},
            {"tool": "get_metrics", "sourceType": "internal",
             "sources": [{"ref": "metric:12", "title": "流失率"}]},
            {"tool": "x", "sourceType": "internal", "sources": [{"title": "无标识"}]},
        ]
        items, refs = svc._collect_refs(json.dumps(ev, ensure_ascii=False))
        assert [i["ref"] for i in items] == ["7", "https://x", "metric:12"]
        assert [i["type"] for i in items] == ["knowledge", "web", "internal"]
        assert refs == ["7", "metric:12"]  # 网页 URL 不进 refs

    def test_garbage_json_yields_nothing(self):
        svc = TraceQueryService(FakeDb())
        assert svc._collect_refs("not json") == ([], [])
        assert svc._collect_refs(None) == ([], [])


# ==================================================================
# 复核回环
# ==================================================================


class FakeRag:
    #! **必须用真实的 ``RagHit``，不要自己捏一个"长得像"的假 hit**。
    #! 这里原来手写了 ``_Hit``，还把 ``document_id`` 定义成**方法**；而真实 ``RagHit``
    #! 里它是 dataclass **字段**。于是生产代码写成 ``h.document_id()`` 会在真机上抛
    #! ``'int' object is not callable``、被外层 except 吞掉（检索员简报整段失效），
    #! 而这组测试却一直显示绿灯 —— 假对象把类型缺陷盖住了。
    #! 教训：替身只要不是真实类型，就会在"字段 vs 方法"这类地方骗过断言。
    def _Hit(title: str, score: float, snippet: str,  # noqa: N802 - 保持原调用点不变
             ref: Optional[str], doc_id: Optional[int]) -> RagHit:
        return RagHit(title=title, snippet=snippet, score=score,
                      source_ref=ref, document_id=doc_id)

    def __init__(self, hits: Optional[List[Any]] = None) -> None:
        self.hits = hits or []

    def search(self, cid, q, k):
        return self.hits


class TestReviewLoop:
    def test_retrieve_empty_says_so(self):
        brief = ReviewLoopService().retrieve(1, "q", 5, FakeRag([]))
        assert "未检索到" in brief

    def test_retrieve_marks_structured_ref_for_citation(self):
        """结构化切片要在简报里标出「来源ID=」，否则模型不知道可以这样引用。"""
        rag = FakeRag([FakeRag._Hit("客户流失率", 88.0, "25%", "metric:16", None)])
        brief = ReviewLoopService().retrieve(1, "q", 5, rag)
        assert "来源ID=metric:16" in brief
        assert "88.0" in brief

    def test_retrieve_skips_ref_for_knowledge_docs(self):
        """知识库文档用 documentId 引用，简报里不该再塞一个「来源ID=」。"""
        rag = FakeRag([FakeRag._Hit("制度", 70.0, "…", "k:7#0", 7)])
        brief = ReviewLoopService().retrieve(1, "q", 5, rag)
        assert "来源ID=" not in brief

    def test_no_llm_review_is_honest(self):
        out = ReviewLoopService(llm=None).review("draft", "evidence")
        assert "跳过自动复核" in out

    def test_clean_deterministic_check_skips_model_by_default(self):
        """最关键的省钱逻辑：确定性信号干净 → 不烧模型，且 trigger 要写明是"免复核"。"""
        rr = ReviewLoopService(llm=None).review_structured("d", "e", Check())
        assert rr.severe is False
        assert rr.trigger == "确定性核对通过，已免模型复核"
        assert rr.text == ""

    def test_dangling_and_low_coverage_are_severe_without_model(self):
        ck = Check(web_dangling=[3], kb_dangling=["metric:9"], citation_coverage=0.2)
        rr = ReviewLoopService(llm=None).review_structured("d", "e", ck)
        assert rr.severe is True
        assert rr.trigger == "确定性：悬空引用"
        assert any("悬空引用" in i for i in rr.issues)
        assert any("来源ID" in i for i in rr.issues)
        assert "20%" in " ".join(rr.issues)

    def test_always_llm_forces_model_review(self, monkeypatch):
        calls: List[str] = []

        class L:
            def is_configured(self):
                return True

            def complete(self, system, user, temperature=0.2):
                calls.append(user)
                return "复核通过"

        rr = ReviewLoopService(llm=L(), always_llm=True).review_structured("d", "e", Check())
        assert calls, "开了 always_llm 就必须真的调一次模型"
        assert rr.severe is False  # 「复核通过」不是 severe

    def test_severe_wording_from_model_is_detected(self):
        class L:
            def is_configured(self):
                return True

            def complete(self, system, user, temperature=0.2):
                return "成稿存在编造数据的问题。"

        rr = ReviewLoopService(llm=L(), always_llm=True).review_structured("d", "e", Check())
        assert rr.severe is True
        assert rr.trigger == "复核员措辞判定"

    def test_revise_returns_none_without_model(self):
        """无法修订时返回 None，调用方保留原文 —— 绝不能返回残缺版本。"""
        assert ReviewLoopService(llm=None).revise("draft", "opinion", "q") is None

    def test_revise_rejects_suspiciously_short_output(self):
        class L:
            def is_configured(self):
                return True

            def complete(self, system, user, temperature=0.2):
                return "好的"

        draft = "正文" * 100  # 200 字
        assert ReviewLoopService(llm=L()).revise(draft, "意见", "q") is None

    def test_revise_accepts_full_rewrite(self):
        class L:
            def is_configured(self):
                return True

            def complete(self, system, user, temperature=0.2):
                return "修订后的完整成稿。" * 20

        out = ReviewLoopService(llm=L()).revise("正文" * 100, "意见", "q")
        assert out is not None and "修订后" in out


# ==================================================================
# 主动预警
# ==================================================================


class FakeLlm:
    def __init__(self, allowed: bool) -> None:
        self._allowed = allowed

    def is_auto_consume_allowed(self) -> bool:
        return self._allowed


class TestProactiveGates:
    def test_disabled_short_circuits(self):
        svc = ProactiveRiskService(enabled=False)
        assert "未启用" in svc.evaluate(1, "METRIC_THRESHOLD", "metric:1")["skipped"]

    def test_missing_company_short_circuits(self):
        svc = ProactiveRiskService()
        assert svc.evaluate(None, "X")["skipped"] == "缺少企业ID"

    def test_unauthorized_auto_consume_is_blocked(self):
        """成本红线：未经使用者主动触发，一律不烧 token。"""
        svc = ProactiveRiskService(llm=FakeLlm(False))
        out = svc.evaluate(1, "METRIC_THRESHOLD", "metric:1")
        assert "未授权" in out["skipped"]
        assert "APP_AI_AUTO_CONSUME" in out["skipped"]

    def test_daily_quota_blocks(self):
        db = FakeDb(scalar=5)
        svc = ProactiveRiskService(llm=None, db=db, daily_limit=5)
        out = svc.evaluate(1, "X", "r")
        assert "已达上限" in out["skipped"]
        assert db.inserted == [], "被配额挡下就不该再落占位记录"

    def test_placeholder_is_written_before_agent_runs(self):
        """配额占位必须先落库：否则这次消耗"没发生过"，同一事件能反复烧额度。"""
        db = FakeDb(scalar=0)
        svc = ProactiveRiskService(llm=None, db=db, agent_service=None)
        try:
            svc.evaluate(1, "X", "r")
        except Exception:
            pass
        assert db.inserted and db.inserted[0]["dispatched"] == "RUNNING"
        assert db.inserted[0]["risk_level"] == "PENDING"

    def test_fail_closed_when_placeholder_unwritable(self):
        """占位写不进去 → 宁可不做，也不能无上限地跑。"""
        db = FakeDb(scalar=0)
        db.insert_id = lambda stmt, params=None: None
        svc = ProactiveRiskService(llm=None, db=db, agent_service=None)
        assert "配额记录不可用" in svc.evaluate(1, "X", "r")["skipped"]


class TestProactiveDispatch:
    def test_high_dispatches_alerted_with_approval(self):
        class Agent:
            def analyze(self, cid, q, top_k, sid, multi, uid):
                class O:
                    answer_json = {"risk_level": "HIGH", "headline": "资金链紧张"}
                return O()

        db = FakeDb(scalar=0)
        approved: List[dict] = []
        svc = ProactiveRiskService(agent_service=Agent(), approvals=lambda *a: approved.append(a) or 5,
                                   llm=None, db=db, system_user_id=1)
        out = svc.evaluate(1, "METRIC_THRESHOLD", "metric:1")
        assert out["riskLevel"] == "HIGH"
        assert out["dispatched"] == "ALERTED"
        assert out["approvalId"] == 5
        assert approved, "HIGH 必须生成待审批动作"
        # 回填：无论研判成功与否，这一条都已计入当日配额
        assert db.updated and db.updated[0]["dispatched"] == "ALERTED"

    def test_low_only_archived(self):
        class Agent:
            def analyze(self, cid, q, top_k, sid, multi, uid):
                class O:
                    answer_json = {"risk_level": "LOW"}
                return O()

        db = FakeDb(scalar=0)
        approved: List[dict] = []
        svc = ProactiveRiskService(agent_service=Agent(), approvals=lambda *a: approved.append(a) or 1,
                                   llm=None, db=db, system_user_id=1)
        out = svc.evaluate(1, "X", "r")
        assert out["dispatched"] == "ARCHIVED"
        assert approved == [], "低危只归档，不去打扰人"

    def test_missing_system_user_fails_loudly(self):
        db = FakeDb(scalar=0)
        svc = ProactiveRiskService(agent_service=None, llm=None, db=db,
                                   system_username="不存在的账号")
        out = svc.evaluate(1, "X", "r")
        assert "error" in out


class TestProactiveParsing:
    def test_question_forbids_web_and_limits_tools(self):
        q = _build_question("METRIC_THRESHOLD", "metric:1")
        assert "不要调用联网检索工具" in q
        assert "最多调用 2 次取数工具" in q
        assert "METRIC_THRESHOLD" in q and "metric:1" in q

    def test_level_is_uppercased(self):
        assert _parse_level('{"risk_level":"high"}') == "HIGH"
        assert _parse_level('{"risk_level": "Medium"}') == "MEDIUM"
        assert _parse_level(None) is None
        assert _parse_level("no level here") is None

    def test_headline_is_truncated_at_300(self):
        long = "x" * 500
        assert len(_parse_headline('{"headline":"%s"}' % long)) == 300
        assert _parse_headline('{"headline":"短"}') == "短"
        assert _parse_headline(None) is None


# ==================================================================
# 通知通道
# ==================================================================


class TestNotification:
    def test_no_webhook_reports_registered_not_delivered(self):
        """没配 webhook 必须回执"待人工投递"，**不能假装已送达**。"""
        svc = NotificationService(webhook_url="")
        r = svc.notify_owner(1, None, "请关注")
        assert isinstance(r, Receipt)
        assert r.delivered is False
        assert r.channel == "REGISTERED"
        assert "待人工投递" in r.detail

    def test_empty_message_gets_placeholder(self):
        r = NotificationService(webhook_url="").notify_owner(1, 2, "   ")
        assert "(无正文)" in r.detail

    def test_timeout_is_clamped(self):
        assert NotificationService(webhook_url="", timeout=1).timeout() == 2
        assert NotificationService(webhook_url="", timeout=999).timeout() == 60
        assert NotificationService(webhook_url="", timeout=8).timeout() == 8

    def test_webhook_failure_is_reported_not_swallowed(self, monkeypatch):
        import urllib.error

        import app.agent.notification as mod

        def boom(*a, **k):
            raise urllib.error.URLError("connect refused")

        monkeypatch.setattr(mod.urlrequest, "urlopen", boom)
        r = NotificationService(webhook_url="https://hook").notify_owner(1, 2, "hi")
        assert r.delivered is False
        assert r.channel == "WEBHOOK_FAILED"


# ==================================================================
# 报告导出
# ==================================================================


class TestReportExport:
    """导出最容易"静默省掉"的就是来源清单 —— 那恰恰是最不能省的部分。"""

    def _analysis(self, answer_json: dict, answer: str = "正文") -> dict:
        return {"id": 103, "company_id": 1, "question": "客户流失率为何上升",
                "confidence": "中", "grounded": 1, "created_at": None,
                "answer": answer,
                "answer_json": json.dumps(answer_json, ensure_ascii=False)}

    def _svc(self):
        from app.agent.report_export import ReportExportService

        return ReportExportService(db=FakeDb())

    def test_sources_are_always_listed(self):
        a = self._analysis({
            "headline": "流失率上升",
            "web_sources": [{"title": "行业报告", "url": "https://example.com/a"}],
            "kb_sources": [{"sourceRef": "metric:16", "title": "客户流失率"}],
            "recommendations_internal": [{"action": "跟进重点客户"}, "补充访谈"],
        })
        lines = self._svc().build_lines(a, "示例企业")
        text = "\n".join(l.text for l in lines)
        assert "https://example.com/a" in text, "网页必须带完整 URL，否则无从核对"
        assert "[1] 行业报告" in text
        assert "来源ID=metric:16" in text
        assert "（内部）跟进重点客户" in text
        assert "（内部）补充访谈" in text
        assert "一、核心结论" in text and "四、参考来源" in text

    def test_no_sources_says_so_explicitly(self):
        lines = self._svc().build_lines(self._analysis({"headline": "x"}), "示例企业")
        text = "\n".join(l.text for l in lines)
        assert "本次分析未引用可列出的来源" in text
        assert "本次未给出具体动作建议" in text

    def test_degraded_and_revised_are_marked(self):
        a = self._analysis({"degraded": True, "review_revised": True,
                            "review_trigger": "确定性：悬空引用"})
        lines = self._svc().build_lines(a, "示例企业")
        styles = {l.style for l in lines}
        text = "\n".join(l.text for l in lines)
        assert "warn" in styles and "降级报告" in text
        assert "已根据复核意见修订" in text and "悬空引用" in text

    def test_grounded_flag_drives_traceability_line(self):
        a = self._analysis({})
        a["grounded"] = 0
        text = "\n".join(l.text for l in self._svc().build_lines(a, "示例企业"))
        assert "待人工核对" in text

    def test_docx_renders(self):
        from app.agent.report_export import Line

        out = self._svc().to_docx([Line("h1", "标题"), Line("p", "中文正文一段。")])
        assert out[:2] == b"PK" and len(out) > 1000

    def test_pdf_renders_with_chinese_font(self):
        from app.agent.report_export import Line

        out = self._svc().to_pdf([Line("h1", "企业经营风险分析报告"),
                                  Line("p", "客户流失率上升至 25%。")])
        assert out[:5] == b"%PDF-" and len(out) > 1000

    def test_pdf_wraps_long_lines(self):
        from app.agent.report_export import ReportExportService

        segs = ReportExportService._wrap("中" * 500, "simhei.ttf", 11.0, 495.0)
        assert len(segs) > 1 and all(s for s in segs)

    def test_missing_analysis_raises(self):
        from app.core.errors import BusinessError

        with pytest.raises(BusinessError, match="不存在"):
            self._svc().export(999999, "pdf")

    def test_scope_denies_other_company(self):
        from app.agent.report_export import ReportExportService
        from app.core.errors import BusinessError

        db = FakeDb(analysis_row={"id": 1, "company_id": 9, "answer_json": None,
                                  "answer": "", "grounded": 0})
        svc = ReportExportService(db=db, scope=lambda cid: cid == 1)
        with pytest.raises(BusinessError, match="其他企业"):
            svc.export(1, "pdf")
