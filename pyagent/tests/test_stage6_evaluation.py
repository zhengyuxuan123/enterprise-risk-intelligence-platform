"""阶段 6：评估与治理。

这里测的不是"评估能不能跑"，而是**评估本身会不会骗人**：
用例没加载、断言被静默丢弃、0 条用例被当成全绿、跨模式比耗时 ——
这几种都会让一份形同虚设的回归报告显示成 PASSED。全部离线、零 token。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import pytest

from app.agent.eval_history import EvalHistoryStore
from app.agent.evaluation import (
    DIM_CATEGORY,
    WEIGHTS,
    CaseResult,
    EvalCase,
    EvalReport,
    EvaluationService,
    bare_tool_name,
    round2,
)
from app.agent.grounding import AnswerGroundingService
from app.agent.guardrail import GuardrailService
from app.agent.health import OperationalHealthService


# ------------------------------------------------------------------
# 用例解析：camelCase → snake_case
# ------------------------------------------------------------------


class TestCaseParsing:
    def test_camel_case_keys_are_mapped(self):
        """用例文件是 camelCase，字段是 snake_case。

        不做转换的话所有断言会被**静默丢弃** —— 44 条全部"通过"，
        实际上一条断言都没执行。这是最难发现的一种假绿灯。
        """
        c = EvalCase.from_dict({
            "id": "X1", "category": "web", "expectWeb": True, "minWebResults": 3,
            "topK": 6, "expectTools": ["web_search"], "expectKeywords": ["续费"],
            "minRagScore": 0.5, "maxRawRatioVsReference": 0.7,
            "expectDanglingWeb": 1, "expectContractViolation": False,
            "expectMaxLlmCalls": 1, "probeTool": "get_metrics",
        })
        assert c.expect_web is True
        assert c.min_web_results == 3
        assert c.top_k == 6
        assert c.expect_tools == ["web_search"]
        assert c.expect_keywords == ["续费"]
        assert c.min_rag_score == 0.5
        assert c.max_raw_ratio_vs_reference == 0.7
        assert c.expect_dangling_web == 1
        assert c.expect_contract_violation is False
        assert c.expect_max_llm_calls == 1
        assert c.probe_tool == "get_metrics"

    def test_unknown_field_is_ignored_not_fatal(self):
        """一个拼写错误不该让整轮回归空跑 —— 忽略它，但要能看见（warning）。"""
        c = EvalCase.from_dict({"id": "X2", "category": "rag", "someNewField": 1})
        assert c.id == "X2"
        assert c.category == "rag"


# ------------------------------------------------------------------
# 门禁：0 条用例不是"全通过"
# ------------------------------------------------------------------


class TestGate:
    def _svc(self, **kw) -> EvaluationService:
        s = EvaluationService(history=None, db=None, **kw)
        s._cases = []
        return s

    def test_zero_cases_is_failed_not_passed(self):
        """空集合天然满足「全部通过」——「0/0 全通过」不是通过，是没跑。"""
        s = self._svc()
        rep = EvalReport()
        rep.results = []
        s._score_and_gate(rep)
        assert rep.gate == "FAILED"
        assert "0 条用例" in rep.gate_reason

    def test_case_load_error_surfaces_in_reason(self):
        s = self._svc()
        s.case_load_error = "Expecting ',' delimiter"
        rep = EvalReport()
        rep.results = []
        s._score_and_gate(rep)
        assert "Expecting" in rep.gate_reason

    def test_safety_must_be_perfect(self):
        """安全维度没满分，综合分再高也不过门禁 —— 安全没有"差不多"。"""
        s = self._svc()
        rep = EvalReport()
        rep.results = [
            CaseResult(id="a", category="guardrail", passed=False),
            CaseResult(id="b", category="rag", passed=True),
        ]
        rep.total = len(rep.results)
        s._score_and_gate(rep)
        assert rep.dimensions["safety"]["score"] == 0.0
        assert rep.gate == "FAILED"
        assert "安全维度未满分" in rep.gate_reason

    def test_missing_dimension_is_excluded_not_zero(self):
        """缺维度不加权（而不是记 0 分），否则新增维度会让历史分数整体跳水。"""
        s = self._svc()
        s.health = None            # cost 维度来自线上留痕，这里只想看纯加权
        rep = EvalReport()
        rep.results = [CaseResult(id="a", category="rag", passed=True)]
        rep.total = 1
        s._score_and_gate(rep)
        for dim in ("routing", "structure", "safety", "e2e"):
            assert rep.dimensions[dim]["applicable"] is False
            assert rep.dimensions[dim]["score"] is None
        # 只有 retrieval 参与加权 → 分数就是 retrieval 的分数
        assert rep.score == 100.0

    def test_regression_blocks_the_gate(self):
        s = self._svc()
        rep = EvalReport()
        rep.results = [CaseResult(id="a", category="guardrail", passed=True)]
        rep.total = 1
        rep.comparison = {"verdict": "REGRESSED"}
        s._score_and_gate(rep)
        assert rep.gate == "FAILED"
        assert "回归" in rep.gate_reason

    def test_grade_ladder(self):
        s = self._svc()
        for score, grade in ((95, "A"), (85, "B"), (72, "C"), (10, "D")):
            rep = EvalReport()
            rep.results = [CaseResult(id="a", category="rag", passed=True)]
            rep.total = 1
            s._score_and_gate(rep)
            rep.score = score
            rep.grade = ("A" if score >= 90 else "B" if score >= 80
                         else "C" if score >= 70 else "D")
            assert rep.grade == grade


# ------------------------------------------------------------------
# 权重与维度映射的完整性
# ------------------------------------------------------------------


class TestWeights:
    def test_all_weighted_dims_have_categories_or_are_live(self):
        """cost 来自线上留痕，没有对应用例类别 —— 这是有意的，其余维度必须有。"""
        for dim in WEIGHTS:
            if dim == "cost":
                continue
            assert dim in DIM_CATEGORY, dim
            assert DIM_CATEGORY[dim], dim

    def test_round2_is_half_up_not_banker(self):
        """Java 的 ``Math.round`` 是 floor(x+0.5)，不是取偶。

        差异只在**二进制能精确表示**的半值上才看得出来：Python 内建的
        ``round(0.125, 2)`` 给 0.12（取偶），Java 给 0.13。别用 1.005 这类值举证 ——
        它在 IEEE-754 下略小于 1.005，两种算法都会给 1.0。
        """
        assert round2(0.125) == 0.13          # Python 内建 round 会给 0.12
        assert round2(2.675) == 2.68          # 内建会给 2.67
        assert round2(0.135) == 0.14
        assert round2(1.005) == 1.0           # 两侧一致：1.005 实存为 1.00499…


# ------------------------------------------------------------------
# 基线对比
# ------------------------------------------------------------------


class TestComparison:
    def _svc(self) -> EvaluationService:
        s = EvaluationService(history=None, db=None)
        s._cases = []
        return s

    def test_cross_mode_wallclock_is_not_compared(self):
        """fast 与 live 放一起比耗时，会得出"变慢 30 倍"这种误导结论。

        宁可不出这个数，也不出一个误导的数。
        """
        s = self._svc()
        rep = EvalReport()
        rep.mode = "live"
        rep.wall_clock_ms = 30_000
        c = s.compare(rep, {"id": "b1", "passRate": 100.0, "wallClockMs": 300, "mode": "fast"})
        assert c["sameMode"] is False
        assert c["wallClockComparable"] is False
        assert "wallClockDeltaPct" not in c

    def test_same_mode_compares_wallclock(self):
        s = self._svc()
        rep = EvalReport()
        rep.mode = "fast"
        rep.wall_clock_ms = 600
        rep.pass_rate = 90.0
        c = s.compare(rep, {"id": "b1", "passRate": 100.0, "wallClockMs": 300, "mode": "fast"})
        assert c["sameMode"] is True
        assert c["wallClockDeltaMs"] == 300
        assert c["wallClockDeltaPct"] == 100.0

    def test_regressed_and_fixed_detection(self):
        s = self._svc()
        rep = EvalReport()
        rep.mode = "fast"
        rep.results = [CaseResult(id="a", passed=False), CaseResult(id="b", passed=True)]
        c = s.compare(rep, {"id": "b1", "passRate": 50.0, "wallClockMs": 1, "mode": "fast",
                            "results": [{"id": "a", "passed": True},
                                        {"id": "b", "passed": False}]})
        assert c["regressedCases"] == ["a"]
        assert c["fixedCases"] == ["b"]
        assert c["verdict"] == "REGRESSED"


# ------------------------------------------------------------------
# 模式与计数
# ------------------------------------------------------------------


class TestModes:
    @staticmethod
    def _svc(cases: List[Dict[str, Any]]) -> EvaluationService:
        s = EvaluationService(history=None, db=None)
        s._cases = [EvalCase.from_dict(c) for c in cases]
        return s

    def test_case_count_respects_mode(self):
        """进度条分母必须是「本模式实际会跑的条数」，
        否则 fast 模式下进度永远差一截到不了 100%。"""
        s = self._svc([{"id": "a"}, {"id": "b", "live": True}])
        assert s.case_count("fast") == 1
        assert s.case_count("live") == 1
        assert s.case_count("full") == 2


# ------------------------------------------------------------------
# 历史存档
# ------------------------------------------------------------------


class TestEvalHistory:
    def test_save_then_latest_then_delete(self):
        d = tempfile.mkdtemp()
        st = EvalHistoryStore(report_dir=os.path.join(d, "rep"))
        rid = st.save({"total": 3, "passed": 3, "passRate": 100.0,
                       "wallClockMs": 12, "model": "m", "generatedAt": "now",
                       "mode": "fast", "results": []})
        assert rid
        latest = st.latest_report()
        assert latest is not None
        # id 必须先写进对象再序列化，否则下一轮拿它当基线时报不出"比的是哪一份"
        assert latest["id"] == rid
        assert st.delete(rid) is True
        assert st.latest_report() is None

    def test_same_second_does_not_overwrite(self):
        """同一秒连跑两次会撞 id，后一份会覆盖前一份、基线也就丢了。"""
        d = tempfile.mkdtemp()
        st = EvalHistoryStore(report_dir=os.path.join(d, "rep"))
        a = st.save({"total": 1, "passed": 1, "passRate": 100.0, "wallClockMs": 1,
                     "model": "m", "generatedAt": "t1", "results": []})
        b = st.save({"total": 2, "passed": 1, "passRate": 50.0, "wallClockMs": 2,
                     "model": "m", "generatedAt": "t2", "results": []})
        assert a != b
        assert len(st.history()) == 2

    def test_max_history_prunes(self):
        d = tempfile.mkdtemp()
        st = EvalHistoryStore(report_dir=os.path.join(d, "rep"), max_history=2)
        for i in range(5):
            st.save({"total": i, "passed": i, "passRate": 0.0, "wallClockMs": i,
                     "model": "m", "generatedAt": f"t{i}", "results": []})
        assert len(st.history()) == 2

    def test_legacy_report_gets_id_backfilled(self):
        """旧报告里 id 是 null（只在文件名里）。按文件名回填，历史档案立刻可用。"""
        d = tempfile.mkdtemp()
        rdir = os.path.join(d, "rep")
        os.makedirs(rdir)
        with open(os.path.join(rdir, "report-20260101-000000.json"), "w", encoding="utf-8") as f:
            json.dump({"total": 1, "passed": 1}, f)
        st = EvalHistoryStore(report_dir=rdir)
        m = st.report("20260101-000000")
        assert m is not None
        assert m["id"] == "20260101-000000"


# ------------------------------------------------------------------
# 健康度（cost 维度的来源）
# ------------------------------------------------------------------


class TestHealth:
    def test_no_samples_is_unavailable_not_zero(self):
        """窗口内没有留痕数据时是「不适用」，不是 0 分。
        否则线上还没跑过就会把 cost 维度拉成 0，看起来像出了大事。"""
        class _Db:
            def fetch_one(self, q):
                return {"total": 0}

        h = OperationalHealthService(db=_Db())  # type: ignore[arg-type]
        r = h.health(24)
        assert r["available"] is False
        assert "score" not in r

    def test_score_ladder_degrades_with_degrade_rate(self):
        class _Db:
            def __init__(self, degraded: int) -> None:
                self.degraded = degraded

            def fetch_one(self, q):
                return {"total": 10, "degraded": self.degraded, "degradedFull": 0,
                        "withToolFailure": 0, "toolFailures": 0, "toolCalls": 10,
                        "llmCalls": 10, "avgDurationMs": 100.0, "maxDurationMs": 200}

        good = OperationalHealthService(db=_Db(0))  # type: ignore[arg-type]
        bad = OperationalHealthService(db=_Db(10))  # type: ignore[arg-type]
        assert good.health(24)["score"] > bad.health(24)["score"]


# ------------------------------------------------------------------
# 引用补齐（repair）的边界
# ------------------------------------------------------------------


class TestRepairBoundaries:
    """repair 的落点约束：补错了比不补更糟。"""

    def test_never_appends_into_reference_list(self):
        """「参考来源」清单本身不是正文 —— 往清单行尾补 [1] 会把 URL 撑坏。"""
        g = AnswerGroundingService()
        ans = ("一、结论（基于内部数据）\n毛利率下滑。\n\n"
               "二、结论（基于外部公开资料）\n行业合规趋严。\n\n"
               "参考来源（网页）\n[1] 区块链发票试点 https://example.com/chain")
        ev = [{"tool": "web_search", "sourceType": "web",
               "sources": [{"index": 1, "title": "区块链发票试点",
                            "url": "https://example.com/chain", "siteName": "example.com"}]}]
        rep = g.repair(ans, ev)
        assert "chain[1]" not in rep.answer          # URL 没被撑坏
        assert not [r for r in rep.repairs if r["type"] == "web"]

    def test_web_citation_only_lands_in_external_section(self):
        """内部结论里出现同名主题词，也不能把网页引用贴进去 —— 那是内外混写。"""
        g = AnswerGroundingService()
        ans = ("一、结论（基于内部知识库与经营数据）\n内部已上线客户健康度评分。\n\n"
               "二、结论（基于外部公开资料）\n本次未联网核实。")
        ev = [{"tool": "web_search", "sourceType": "web",
               "sources": [{"index": 1, "title": "客户健康度评分实践",
                            "url": "https://e.com/h", "siteName": "e.com"}]}]
        rep = g.repair(ans, ev)
        assert not [r for r in rep.repairs if r["type"] == "web"]
        assert "[1]" not in rep.answer

    def test_web_citation_can_land_in_external_section(self):
        g = AnswerGroundingService()
        ans = ("一、结论（基于内部知识库与经营数据）\n重复投诉占比上升。\n\n"
               "二、结论（基于外部公开资料）\n同业普遍把客户健康度评分纳入续费预警机制。")
        ev = [{"tool": "web_search", "sourceType": "web",
               "sources": [{"index": 1, "title": "客户健康度评分实践",
                            "url": "https://e.com/h", "siteName": "e.com"}]}]
        rep = g.repair(ans, ev)
        assert [r for r in rep.repairs if r["type"] == "web"], "该补没补"
        assert "[1]" in rep.answer

    def test_reference_list_appended_when_body_cites(self):
        g = AnswerGroundingService()
        ans = ("一、结论（基于内部数据）\n重复投诉上升。\n\n"
               "二、结论（基于外部公开资料）\n同业做法见 [1]。")
        ev = [{"tool": "web_search", "sourceType": "web",
               "sources": [{"index": 1, "title": "客户健康度评分实践",
                            "url": "https://e.com/h", "siteName": "e.com"}]}]
        rep = g.repair(ans, ev)
        assert any(r["type"] == "reference" for r in rep.repairs)
        assert "参考来源（网页）" in rep.answer
        assert "https://e.com/h" in rep.answer

    def test_data_basis_sentence_is_not_attributed_to_a_document(self):
        """已经用「（指标：」标明数据依据的句子不再挂文档 —— 那是替模型说话。"""
        g = AnswerGroundingService()
        ans = ("一、结论（基于内部知识库与经营数据）\n（指标：客户流失率）为 25%，续费处置要求见SOP。\n\n"
               "二、结论（基于外部公开资料）\n本次未联网核实。")
        ev = [{"tool": "search_knowledge", "sourceType": "knowledge",
               "sources": [{"sourceRef": "12", "title": "续费风险处置SOP"}]}]
        rep = g.repair(ans, ev)
        assert not [r for r in rep.repairs if r["type"] == "kb"]

    def test_empty_answer_is_untouched(self):
        g = AnswerGroundingService()
        rep = g.repair("", [{"tool": "web_search", "sourceType": "web",
                             "sources": [{"index": 1, "title": "T", "url": "u"}]}])
        assert rep.answer == ""
        assert rep.repairs == []


# ------------------------------------------------------------------
# 报告字段契约：界面靠这些键取值，错一个字母就是一整列空白
# ------------------------------------------------------------------

#: 前端 `AiAnalysis.vue` 回归表格真正读取的键。**改动前先 grep 前端**，
#: 这份清单是「报告字段」与「界面」之间唯一的契约。
FRONTEND_CASE_KEYS = {
    "id", "category", "companyId", "question", "passed",
    "webHits", "ragHits", "probeTool", "citationWebCited",
    "llmCalls", "singleRound", "answerChars",
    "ragTopScore", "ragTopRawScore", "live", "durationMs", "reasons",
}


class TestReportFieldContract:
    def test_case_keys_are_camel_case_and_cover_the_frontend(self):
        """用例字段必须是驼峰且覆盖前端读的每一个键。

        曾经这俩分叉过：`CaseResult.to_dict()` 输出 `company_id`/`rag_hits`/`llm_calls`，
        而前端读 `companyId`/`ragHits`/`llmCalls` —— 于是除 id/类别/问题/结果/原因外，
        「企业、证据、模型调用、成稿、Top分、耗时」六列在界面上**全是空白**，
        报告本身没有任何报错，属于最难发现的一类回归。
        """
        keys = set(CaseResult().to_dict().keys())
        missing = FRONTEND_CASE_KEYS - keys
        assert not missing, f"前端要读但报告里没有的键: {sorted(missing)}"
        snake = [k for k in keys if "_" in k]
        assert not snake, f"对外字段不该出现下划线: {snake}"

    def test_null_metrics_are_emitted_not_dropped(self):
        """「不适用」要显式给 null，不能把键删掉。

        键缺失与值为 null 在前端都渲染成「—」，但排查时前者会被误判成"后端漏填字段"，
        而后者能立刻读成"这条用例口径上就没有这个指标"。
        """
        d = CaseResult(id="EVAL-101", category="rag").to_dict()
        assert "llmCalls" in d and d["llmCalls"] is None
        assert "webHits" in d and d["webHits"] is None
        assert "ragTopScore" in d and d["ragTopScore"] is None
        assert d["reasons"] == []

    def test_duration_is_always_present(self):
        """耗时是 0 也要给（确定性用例确实耗时接近 0），不能被 None 化。"""
        d = CaseResult().to_dict()
        assert d["durationMs"] == 0


class TestTraceToolMatching:
    """轨迹装饰与机器断言的错配 —— 一次让 5 条端到端用例全部假失败的坑。"""

    def test_bare_tool_name_strips_latency_and_prefetch(self):
        assert bare_tool_name("get_metrics(5ms)") == "get_metrics"
        assert bare_tool_name("search_knowledge(预取 370ms)") == "search_knowledge"
        assert bare_tool_name("web_search") == "web_search"
        assert bare_tool_name(" propose_create_ticket(0ms) ") == "propose_create_ticket"
        assert bare_tool_name(None) == ""

    def test_live_case_passes_when_trace_is_decorated(self):
        """端到端用例：轨迹带 (ms) 装饰时，期望工具断言仍应判定通过。

        这是真实事故的回归：轨迹写成 `get_metrics(5ms)`，断言却拿 `get_metrics`
        做包含判断 → 恒不相等 → 明明工具调用得好好的，报告却说
        「模型未调用期望工具 ['get_metrics']，实际轨迹 ['get_metrics(5ms)']」。
        """
        svc = EvaluationService(agent_service=_FakeAgent(), rag=None,
                                grounding=AnswerGroundingService(),
                                guardrail=GuardrailService(),
                                history=None, cases_file=_CASES)
        case = EvalCase(id="EVAL-901", category="e2e", question="经营状况如何",
                        expect_tools=["get_metrics"], live=True)
        r = svc.run_case(case, 1, set(), {"done": 0, "web": 0}, [])
        assert r.passed, r.reasons
        assert r.trace_tools == ["search_knowledge", "get_metrics"]
        # 给人看的那份保留耗时，便于回答"这 14 秒花在哪了"
        assert r.trace_detail == ["search_knowledge(预取 370ms)", "get_metrics(5ms)"]

    def test_live_case_fails_when_tool_really_missing(self):
        """真没调用，必须如实判失败 —— 别把断言改成了永远通过。"""
        svc = EvaluationService(agent_service=_FakeAgent(), rag=None,
                                grounding=AnswerGroundingService(),
                                guardrail=GuardrailService(),
                                history=None, cases_file=_CASES)
        case = EvalCase(id="EVAL-902", category="e2e", question="有哪些高风险事件",
                        expect_tools=["get_risk_events"], live=True)
        r = svc.run_case(case, 1, set(), {"done": 0, "web": 0}, [])
        assert not r.passed
        assert any("get_risk_events" in x for x in r.reasons)

    def test_llm_calls_come_from_real_trace(self):
        """模型调用次数取真实留痕：它既上界面，也是 expectMaxLlmCalls 的判据。"""
        svc = EvaluationService(agent_service=_FakeAgent(), rag=None,
                                grounding=AnswerGroundingService(),
                                guardrail=GuardrailService(),
                                history=None, cases_file=_CASES)
        case = EvalCase(id="EVAL-903", category="e2e", question="q",
                        expect_tools=[], live=True, expect_max_llm_calls=1)
        r = svc.run_case(case, 1, set(), {"done": 0, "web": 0}, [])
        assert r.llm_calls == 3
        assert r.tool_calls == 2
        assert r.single_round is False
        assert not r.passed, "调用 3 次超过预算 1 次，必须判失败"


class TestLegacyReportReadback:
    def test_old_snake_case_report_is_normalized_on_read(self):
        """旧留档报告字段名是下划线 —— 读取时归一，别让历史报告整片空白。"""
        with tempfile.TemporaryDirectory() as d:
            st = EvalHistoryStore(report_dir=d, max_history=5)
            legacy = {"id": "20260921-201417", "total": 1, "passed": 1, "passRate": 100.0,
                      "results": [{"id": "EVAL-901", "category": "e2e", "passed": True,
                                   "company_id": 1, "llm_calls": 3, "duration_ms": 14000,
                                   "trace_tools": ["get_metrics(5ms)"],
                                   "rag_top_score": 70.1, "live": True}]}
            with open(os.path.join(d, "report-20260921-201417.json"), "w",
                      encoding="utf-8") as f:
                json.dump(legacy, f, ensure_ascii=False)
            m = st.report("20260921-201417")
            row = m["results"][0]
            assert row["companyId"] == 1
            assert row["llmCalls"] == 3
            assert row["durationMs"] == 14000
            assert row["ragTopScore"] == 70.1
            assert row["traceTools"] == ["get_metrics"], "旧轨迹的耗时装饰也要剥掉"
            # 旧键不残留，否则同一份报告里两套命名并存，后继代码会各读各的
            assert "company_id" not in row and "llm_calls" not in row

    def test_normalization_does_not_touch_the_file(self):
        """归一发生在内存里：留档文件是历史证据，读取不该改写它。"""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "report-20260921-201418.json")
            legacy = {"total": 1, "results": [{"id": "EVAL-101", "company_id": 3}]}
            with open(p, "w", encoding="utf-8") as f:
                json.dump(legacy, f, ensure_ascii=False)
            EvalHistoryStore(report_dir=d).report("20260921-201418")
            on_disk = json.loads(open(p, encoding="utf-8").read())
            assert "company_id" in on_disk["results"][0]
            assert "companyId" not in on_disk["results"][0]


# ------------------------------------------------------------------
# 夹具
# ------------------------------------------------------------------

_CASES = os.path.join(os.path.dirname(__file__), "..",
                      "..", "backend", "src", "main", "resources", "ai-eval-cases.json")


class _FakeOutcome:
    def __init__(self) -> None:
        self.analysis_id = None
        self.content = ("一、结论（基于内部数据）\n指标走弱。\n\n"
                        "二、结论（基于外部公开资料）\n本次未联网核实。\n\n"
                        "三、建议动作\n（一）内部：关注续费率。\n\n"
                        "四、不确定性\n样本期较短。")
        # 轨迹刻意带上耗时/预取装饰 —— 真实运行就是这样，断言必须能扛住
        self.tool_trace = ["search_knowledge(预取 370ms)", "get_metrics(5ms)"]
        self.evidence = [{"tool": "get_metrics", "sourceType": "internal", "sources": []}]
        self.degrade_level = "NONE"
        self.diagnostics = {"grounded": True,
                            "trace": {"llmCalls": 3, "toolCalls": 2, "iterations": 2}}


class _FakeAgent:
    def analyze(self, cid, question, top_k=5, **kw):
        return _FakeOutcome()
