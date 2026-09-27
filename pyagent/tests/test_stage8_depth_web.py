"""阶段 8：分析深度链路 + 强制联网 + 报告字段。

对应 2026-09-22 用户报的三个问题（前两个在这里锁，第三个在 Java 侧）：

1. **「评估与回归没有时间与调用模型等内容」**
   表面看是界面空了，实际有两层：
   a. 前端把确定性用例的耗时/模型调用渲染成「—」（口径解释，但用户读成"没数据"）；
   b. ``liveDepth`` 从来没传进 :class:`AgentBudget` —— 用例写着 ``quick``，
      实际跑的是 ``standard``（预算里 ``max_iterations`` 从 2 变 3、正文字数从 700 变 1500），
      于是 3 条 quick 用例拿 ``expectMaxLlmCalls=1`` 去撞真实的 3 次调用，**必然假失败**。

2. **「我问题都要求联网查询，结果还是不能联网」**
   路由把 ``web_search`` 放进白名单 ≠ 会联网：模型完全可以不调它，
   然后在第二节原样写「本次未联网核实」，格式上还完全合规。
   所以需要一条**硬要求**（force_web）把"调用"本身写成契约，
   并且评估要能判定"要求联网却没有外部来源"。

3. **「删除历史分析失败」**（Java 侧，见 PythonAgentForwardFilterTest）
   ``PythonAgentForwardFilter`` 只放行 GET/POST，DELETE 落到 Spring 静态资源兜底
   → ``500 No static resource api/ai/history/123``。

全部离线、零 token。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from app.agent.agent_service import AgentService
from app.agent.evaluation import CaseResult, EvalCase, EvaluationService
from app.agent.guardrail import GuardrailService
from app.agent.grounding import AnswerGroundingService
from app.agent.router import AgentRouter

_CASES = os.path.join(os.path.dirname(__file__), "..",
                      "..", "backend", "src", "main", "resources", "ai-eval-cases.json")


# ------------------------------------------------------------------
# 夹具
# ------------------------------------------------------------------

class _Outcome:
    """可控的 AgentOutcome 替身：这次要验的是「断言读什么」，不是模型写什么。"""

    def __init__(self, answer: str = "", with_web: bool = False,
                 llm_calls: int = 1, trace: Optional[List[str]] = None) -> None:
        self.analysis_id = None
        self.content = answer or (
            "一、结论（基于内部知识库与经营数据）\n指标走弱，续费率下滑。\n\n"
            "二、结论（基于外部公开资料）\n据公开资料[1]，同业均值约 85%。\n\n"
            "三、建议动作\n（一）内部：关注续费率。\n（二）外部：参照同业口径设阈值。\n\n"
            "四、不确定性\n样本期较短。"
        )
        self.tool_trace = list(trace or ["search_knowledge(预取 370ms)", "get_metrics(5ms)"])
        self.evidence: List[Dict[str, Any]] = [
            {"tool": "get_metrics", "sourceType": "internal", "sources": []}]
        if with_web:
            self.evidence.append({
                "tool": "web_search", "sourceType": "web",
                "sources": [{"index": 1, "title": "行业续费率报告",
                             "url": "https://example.com/renewal"}],
            })
        self.degrade_level = "NONE"
        self.diagnostics = {"grounded": True,
                            "trace": {"llmCalls": llm_calls, "toolCalls": 2, "iterations": 1}}


class _Agent:
    """记录 analyze 收到的关键字参数 —— depth 丢没丢就看这里。"""

    def __init__(self, outcome: Optional[_Outcome] = None) -> None:
        self.outcome = outcome or _Outcome()
        self.kwargs: Dict[str, Any] = {}
        self.calls = 0

    def analyze(self, cid, question, top_k=5, **kw):
        self.calls += 1
        self.kwargs = dict(kw)
        return self.outcome


def _service(agent: _Agent) -> EvaluationService:
    return EvaluationService(agent_service=agent, rag=None,
                             grounding=AnswerGroundingService(),
                             guardrail=GuardrailService(),
                             history=None, cases_file=_CASES)


# ------------------------------------------------------------------
# 一、深度档位必须真的走到预算
# ------------------------------------------------------------------

class TestDepthReachesTheBudget:
    def test_cache_key_is_sensitive_to_depth(self):
        """档位不进缓存键 → 先用「标准」问过一次，再选「快答」会秒回标准档的结果。

        这种缓存串味比缓存未命中难查得多：用户以为自己换了档位，实际上什么都没变。
        """
        a = AgentService._cache_key(1, 1, "q", True, 5, "quick")
        b = AgentService._cache_key(1, 1, "q", True, 5, "standard")
        assert a != b

    def test_cache_key_is_sensitive_to_force_web(self):
        """勾了「必须联网」却命中一条没联网的缓存 = 开关静默失效。"""
        a = AgentService._cache_key(1, 1, "q", True, 5, "standard", True)
        b = AgentService._cache_key(1, 1, "q", True, 5, "standard", False)
        assert a != b

    def test_single_flight_key_is_sensitive_to_depth_and_force_web(self):
        """并发的两个请求若档位不同，绝不能被"单飞"合并成同一个结果。"""
        base = AgentService._single_flight_key(1, 1, "q", "standard", False)
        assert base != AgentService._single_flight_key(1, 1, "q", "quick", False)
        assert base != AgentService._single_flight_key(1, 1, "q", "standard", True)

    def test_evaluation_passes_live_depth_to_the_agent(self):
        """用例写 quick 就必须把 quick 传下去 —— 这条断了那 3 条用例就永远假失败。"""
        agent = _Agent()
        svc = _service(agent)
        case = EvalCase(id="EVAL-901", category="e2e", question="经营状况如何",
                        live=True, live_depth="quick")
        svc.run_case(case, 1, set(), {"done": 0, "web": 0}, [])
        assert agent.kwargs.get("depth") == "quick"

    def test_empty_live_depth_means_server_default(self):
        """用例没写档位时不该硬塞一个值进去，交给服务端默认档。"""
        agent = _Agent()
        svc = _service(agent)
        case = EvalCase(id="EVAL-901", category="e2e", question="q", live=True, live_depth="")
        svc.run_case(case, 1, set(), {"done": 0, "web": 0}, [])
        assert agent.kwargs.get("depth") is None

    def test_report_records_the_effective_depth(self):
        """报告要记**实际生效**档位：光有耗时没有档位，解释不了"这条为什么慢"。"""
        svc = _service(_Agent())
        case = EvalCase(id="EVAL-901", category="e2e", question="q",
                        live=True, live_depth="quick")
        r = svc.run_case(case, 1, set(), {"done": 0, "web": 0}, [])
        assert r.depth == "quick"
        assert r.to_dict()["depth"] == "quick"

    def test_depth_falls_back_to_server_default_when_case_is_silent(self):
        svc = _service(_Agent())
        case = EvalCase(id="EVAL-901", category="e2e", question="q", live=True, live_depth="")
        r = svc.run_case(case, 1, set(), {"done": 0, "web": 0}, [])
        assert r.depth, "档位不能留空 —— 空值会让报告上那一列无从解释"

    def test_quick_case_with_single_llm_call_passes(self):
        """quick 档的真实产出是「单轮成稿」：1 次调用不该再被预算断言判失败。"""
        svc = _service(_Agent(_Outcome(llm_calls=1)))
        case = EvalCase(id="EVAL-901", category="e2e", question="q", live=True,
                        live_depth="quick", expect_max_llm_calls=1)
        r = svc.run_case(case, 1, set(), {"done": 0, "web": 0}, [])
        assert r.llm_calls == 1 and r.single_round is True
        assert not [x for x in r.reasons if "模型调用" in x]

    def test_standard_depth_keeps_three_calls_legal(self):
        """标准档允许 3 轮：同一份用例换个档位就不该再挑调用次数的毛病。"""
        svc = _service(_Agent(_Outcome(llm_calls=3)))
        case = EvalCase(id="EVAL-904", category="e2e", question="q", live=True,
                        live_depth="standard", expect_max_llm_calls=3)
        r = svc.run_case(case, 1, set(), {"done": 0, "web": 0}, [])
        assert r.llm_calls == 3
        assert not [x for x in r.reasons if "超过预期上限" in x]


# ------------------------------------------------------------------
# 二、强制联网
# ------------------------------------------------------------------

class TestForceWebRouting:
    @staticmethod
    def _route(q: str, allow_web: bool = True, requested: bool = False):
        return AgentRouter().route(q, allow_web, force_web_requested=requested)

    def test_explicit_wording_forces_web_and_opens_the_tool(self):
        r = self._route("请联网核实最近的行业监管政策与同业动向。")
        assert r.force_web is True
        assert "web_search" in r.tools

    def test_force_web_survives_even_when_specialist_slots_are_full(self):
        """专员名额上限是 2。强指令必须把工具直接塞进白名单，而不是靠"候补"捡漏。"""
        r = self._route("必须联网查一下公开资料：政策监管、行业行情、竞品动向、客户投诉、"
                        "经营指标、续费率、风险事件分别怎么看？")
        assert r.force_web is True
        assert "web_search" in r.tools

    def test_internal_question_does_not_pull_in_web_search(self):
        """纯内部问题不联网 —— 这是刻意的成本控制，不是缺陷。"""
        r = self._route("这家公司最近的经营状况怎么样？有哪些指标在恶化？")
        assert r.force_web is False
        assert "web_search" not in r.tools
        assert r.needs_web is False

    def test_frontend_switch_forces_web_without_any_keyword(self):
        """前端「必须联网」开关不依赖用户怎么写 —— 这是最可靠的一条路径。"""
        r = self._route("客户投诉集中在哪些方面？", requested=True)
        assert r.force_web is True
        assert "web_search" in r.tools

    def test_explicit_no_web_phrase_wins_over_keyword_matching(self):
        r = self._route("只用内部资料分析，不要联网，也不要查询外部资料。")
        assert r.force_web is False
        assert r.needs_web is False
        assert "web_search" not in r.tools

    def test_frontend_force_switch_overrides_no_web_wording(self):
        r = self._route("不要联网", requested=True)
        assert r.force_web is True
        assert "web_search" in r.tools

    def test_global_switch_still_wins_over_the_explicit_request(self):
        """全局联网关掉时，前端勾了也必须不给 —— 否则「关掉联网」就成了摆设。"""
        r = self._route("请联网核实政策", allow_web=False, requested=True)
        assert r.force_web is False
        assert "web_search" not in r.tools

    def test_asking_about_web_capability_is_not_a_command(self):
        """「你们能联网吗」这类元问题不该被当成下达了联网指令。

        词表刻意只用多字短语（"请联网"/"联网核实"…），不收单字"联网"。
        """
        r = self._route("你们能联网吗？")
        assert r.force_web is False

    def test_needs_web_reflects_either_source(self):
        """needs_web 曾经只被 getattr 读到、而 Route 上根本没有这个属性 ——
        留痕里的 needsWeb 永远是 false，"这次该不该联网"看不出来。"""
        assert self._route("请联网核实政策").needs_web is True
        assert self._route("经营状况如何").needs_web is False

    def test_route_needs_web_is_a_real_attribute(self):
        r = self._route("经营状况如何")
        assert "needs_web" in vars(r)


class TestGuardrailForcedWebPrompt:
    def test_forced_prompt_states_the_call_as_a_requirement(self):
        p = GuardrailService().build_system_prompt(None, True, force_web=True)
        assert "【本轮用户明确要求联网】" in p
        assert "必须" in p
        assert "web_search" in p

    def test_forced_prompt_keeps_the_escape_hatch(self):
        """硬要求不能堵死兜底出口：确实连不上时仍要如实说，而不是逼模型编造。"""
        p = GuardrailService().build_system_prompt(None, True, force_web=True)
        assert "本次未联网核实" in p
        assert "具体原因" in p

    def test_default_prompt_has_no_forced_block(self):
        p = GuardrailService().build_system_prompt(None, True)
        assert "【本轮用户明确要求联网】" not in p

    def test_forced_prompt_keeps_the_four_section_contract(self):
        p = GuardrailService().build_system_prompt(None, True, force_web=True)
        for sec in ("一、结论（基于内部知识库与经营数据）",
                    "二、结论（基于外部公开资料）",
                    "三、建议动作",
                    "四、不确定性 / 需人工确认"):
            assert sec in p


class TestExpectWebAssertion:
    """「要求联网就必须联网」的唯一可判定契约。

    报告头写着「联网搜索: 已启用」只说明**有能力**联网，
    不说明**这一次**联了 —— 用户抱怨的正是这个落差。
    """

    def _run(self, with_web: bool, answer: str = "", expect_web: bool = True):
        svc = _service(_Agent(_Outcome(answer=answer, with_web=with_web)))
        case = EvalCase(id="EVAL-905", category="e2e", question="请联网核实政策",
                        live=True, live_depth="standard", expect_web=expect_web)
        return svc.run_case(case, 1, set(), {"done": 0, "web": 0}, [])

    def test_missing_web_sources_fails_a_web_required_case(self):
        r = self._run(with_web=False)
        assert not r.passed
        assert any("联网" in x for x in r.reasons), r.reasons

    def test_web_required_case_with_sources_clears_the_assertion(self):
        r = self._run(with_web=True)
        assert not [x for x in r.reasons if "未联网核实" in x]
        assert r.web_hits == 1

    def test_saying_not_verified_fails_even_if_sources_exist(self):
        """调了工具却仍写「本次未联网核实」，等于把要求糊过去了。"""
        r = self._run(with_web=True, answer=(
            "一、结论（基于内部知识库与经营数据）\n指标走弱。\n\n"
            "二、结论（基于外部公开资料）\n本次未联网核实。\n\n"
            "三、建议动作\n（一）内部：关注续费率。\n\n"
            "四、不确定性\n无。"))
        assert not r.passed
        assert any("未联网核实" in x for x in r.reasons), r.reasons

    def test_non_web_case_is_not_affected(self):
        """没要求联网的用例不该被这条断言误伤。"""
        r = self._run(with_web=False, expect_web=False)
        assert not [x for x in r.reasons if "联网" in x], r.reasons


# ------------------------------------------------------------------
# 三、报告字段契约（前端读什么，报告就必须有什么）
# ------------------------------------------------------------------

class TestReportKeysCoverTheNewColumns:
    def test_depth_and_web_fields_are_exported(self):
        d = CaseResult(id="EVAL-905", category="e2e").to_dict()
        for k in ("depth", "llmCalls", "answerChars", "durationMs", "webHits"):
            assert k in d, f"前端要读的键缺失：{k}"

    def test_deterministic_case_keeps_null_metrics_visible(self):
        """确定性用例的「模型调用/成稿」是 null（不是没有键）——
        前端据此渲染成「未调用 / 不适用」，而不是一片空白的「—」。"""
        d = CaseResult(id="EVAL-101", category="rag").to_dict()
        assert d["llmCalls"] is None
        assert d["answerChars"] is None
        assert d["durationMs"] == 0


# ------------------------------------------------------------------
# 四、档位必须约束到「生成多少字」
# ------------------------------------------------------------------

class TestAnswerBudgetReachesThePrompt:
    """耗时的大头是"模型写多少字"。

    ``Plan.answer_max_chars`` 的字段注释写着「写进 system 提示约束模型，不是事后截断」，
    但**全项目从未使用过这个字段**。于是「快答」的 700 字预算对模型完全不可见，
    模型照写 1900 字 —— 档位只约束了轮数与工具配额，恰恰漏掉了最耗时的那个变量。
    """

    @staticmethod
    def _hint(depth: str) -> str:
        from types import SimpleNamespace

        from app.agent.budget import AgentBudget

        specs = [SimpleNamespace(name="search_knowledge"), SimpleNamespace(name="get_metrics")]
        return AgentService._tool_hint(specs, AgentBudget.start(depth))

    def test_quick_tells_the_model_seven_hundred_chars(self):
        h = self._hint("quick")
        assert "700" in h, "快答档必须把 700 字预算告诉模型"
        assert "quick" in h

    def test_standard_tells_the_model_its_own_budget(self):
        assert "1500" in self._hint("standard")

    def test_different_depths_give_different_hints(self):
        assert self._hint("quick") != self._hint("standard")

    def test_hint_still_lists_the_tools(self):
        """加了预算说明之后，工具清单不能丢。"""
        h = self._hint("standard")
        assert "search_knowledge" in h and "get_metrics" in h

    def test_missing_budget_does_not_crash(self):
        """调用方没给预算时也要能构造提示词（别为了加一行说明引入 NPE）。"""
        from types import SimpleNamespace

        h = AgentService._tool_hint([SimpleNamespace(name="get_metrics")], None)
        assert "get_metrics" in h
