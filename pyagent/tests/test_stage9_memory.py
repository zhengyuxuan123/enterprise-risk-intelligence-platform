"""阶段 9：长期记忆（跨会话）—— 抽取 / 冲突消解 / 召回 / 注入。

补的是八项能力盘点里唯一的短板：原先只有 ``ai_message`` 的会话内短期记忆，
换个会话就失忆。这里锁住长期记忆的四条底线：

1. **零 token**：抽取与召回全在本地（规则 + 本地哈希向量），不为"记住"多调一次模型；
2. **冲突消解而不是覆盖**：同主题的新口径出现时，旧条目置 ``SUPERSEDED`` 并指向新条目 ——
   "上次 8.3%、这次 12%"必须留下痕迹，那是口径漂移的早期信号；
3. **记忆不是证据**：不进 ``evidence``、不占 ``[n]`` / ``来源ID=x`` 编号，
   提示词里明确"历史数字必须重新核实后才能用"；
4. **独立预算**：与短期记忆分开计，互不挤占，也不因为"记忆坏了"而挡住分析。

全部离线、零 token。
"""

from __future__ import annotations

import pytest
from sqlalchemy import BigInteger, create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from app.agent.long_term_memory import LongTermMemory
from app.agent.memory_extract import (
    MemoryExtractor,
    content_fingerprint,
    normalize_topic_key,
    pick_topic,
)
from app.config import get_settings
from app.db import tables as T
from app.db.engine import Database


# ------------------------------------------------------------------
# 夹具：SQLite 内存库顶替 MySQL
# ------------------------------------------------------------------


@compiles(BigInteger, "sqlite")
def _bigint_as_integer(type_, compiler, **kw):  # noqa: ANN001
    """SQLite 里 ``BIGINT PRIMARY KEY`` 不是 rowid 别名 → 不自增。

    不映射的话插入会因为主键为 NULL 直接失败，而且报错信息与"自增"毫无关系。
    """
    return "INTEGER"


@pytest.fixture()
def db():
    import app.db.engine as E

    eng = create_engine("sqlite://", connect_args={"check_same_thread": False},
                        poolclass=StaticPool)
    old_build = E._build
    E._build = lambda: eng
    E.reset_engine()
    T.META.create_all(E.engine())

    s = get_settings()
    saved = (s.agent.long_term_memory, s.agent.long_term_write,
             s.agent.long_term_budget, s.agent.long_term_top_k,
             s.agent.long_term_min_score, s.agent.long_term_max_per_company)
    s.agent.long_term_memory = True
    s.agent.long_term_write = True
    s.agent.long_term_budget = 600
    s.agent.long_term_top_k = 5
    s.agent.long_term_min_score = 0.18
    s.agent.long_term_max_per_company = 500
    try:
        yield Database()
    finally:
        (s.agent.long_term_memory, s.agent.long_term_write,
         s.agent.long_term_budget, s.agent.long_term_top_k,
         s.agent.long_term_min_score, s.agent.long_term_max_per_company) = saved
        E._build = old_build
        E.reset_engine()


@pytest.fixture()
def mem(db):
    return LongTermMemory(db)


def _answer(conclusion: str, headline: str = "客户流失风险升高") -> dict:
    return {"conclusion_internal": conclusion, "headline": headline, "risk_level": "HIGH"}


# ------------------------------------------------------------------
# 1) 抽取：零 token、可复现
# ------------------------------------------------------------------


class TestExtract:

    def test_fact_needs_number_and_risk_word(self):
        """只有"带数字的风险断言"才是事实卡。

        否则"共 3 条记录""指标已同步"这类流水账会塞满记忆，
        真正有用的那几条被淹没，注入预算也被吃掉。
        """
        ex = MemoryExtractor()
        cs = ex.extract(
            company_id=3, user_id=1, question="客户流失怎么样？",
            answer_json=_answer("共 3 条记录。客户流失率 8.3%，高于阈值 5%，属于高风险。"),
            answer_text="", risk_level="HIGH", analysis_id=7, today="2026-09-22")
        facts = [c for c in cs if c.kind == T.MEM_KIND_FACT]
        assert len(facts) == 1
        assert "8.3" in facts[0].content

    def test_topic_prefers_long_word(self):
        """"流失率"必须先于"流失"命中，否则主题会被切碎成"流失"，
        下次问"续费率"也会因为主题相近而误覆盖。"""
        assert pick_topic("客户流失率 8.3% 上升") == "流失率"
        assert pick_topic("毛利率 12% 下降") == "毛利率"
        assert pick_topic("公司现金流紧张") == "现金流"

    def test_topic_key_ignores_numbers(self):
        """同主题不同口径 → 同一个 key（于是互相覆盖，而不是并列堆着）。"""
        a = normalize_topic_key("流失率", salt="fact")
        b = normalize_topic_key("流失率", salt="fact")
        assert a == b
        # 8.3 与 12 是同一主题的两次口径：归一化后必须一致
        assert normalize_topic_key("流失率 8.3%") == normalize_topic_key("流失率 12%")

    def test_content_fingerprint_keeps_numbers(self):
        """主题指纹抹数字、内容指纹留数字 —— 差别只有一处，但混用会把口径漂移抹平。

        抹数字的那把尺子量"是不是同一主题"（该覆盖），留数字的那把量"是不是同一件事"
        （8.3 与 12 是两件事，该判为"口径变了"，而不是"又说了一遍"）。
        """
        assert content_fingerprint("流失率 8.3%") != content_fingerprint("流失率 12%")
        # 日期不算变化：同一句话今天说、明天说仍然是同一件事
        assert content_fingerprint("【历史事实｜2026-09-22】流失率 8.3%") == \
            content_fingerprint("【历史事实｜2026-09-23】流失率 8.3%")

    def test_preference_clipped_to_the_preference_part(self):
        """"以后结论请先给风险等级"只记后半句，不把整句业务描述一起记下来。"""
        ex = MemoryExtractor()
        cs = ex.extract(
            company_id=3, user_id=1,
            question="请分析客户流失情况，以后结论请先用一句风险等级开头",
            answer_json={}, answer_text="", risk_level="LOW", today="2026-09-22")
        prefs = [c for c in cs if c.kind == T.MEM_KIND_PREFERENCE]
        assert len(prefs) == 1
        assert "以后结论请先用一句风险等级开头" in prefs[0].content
        assert "请分析客户流失情况" not in prefs[0].content

    def test_preference_needs_long_term_wording(self):
        """"请用表格给我"是一次性指令，不是偏好；只有"以后/记住/默认/一律"才是。"""
        ex = MemoryExtractor()
        one_off = ex.extract(company_id=3, user_id=1, question="请用表格给我结论",
                             answer_json={}, answer_text="")
        assert [c for c in one_off if c.kind == T.MEM_KIND_PREFERENCE] == []
        long_term = ex.extract(company_id=3, user_id=1, question="以后结论都用表格",
                               answer_json={}, answer_text="")
        assert len([c for c in long_term if c.kind == T.MEM_KIND_PREFERENCE]) == 1

    def test_citations_are_stripped_from_memory(self):
        """记忆里绝不能留上次的**证据编号**。

        这是"记忆污染证据链"最具体的形态：上次正文里写着「（来源ID=metric:12）」，
        原样存进记忆后，下次注入时模型会把它当成可照抄的依据，而本次证据里
        根本没有 metric:12 —— 要么产出悬空引用，要么让引用核对把历史编号验成本次证据。
        """
        ex = MemoryExtractor()
        cs = ex.extract(
            company_id=3, user_id=1, question="流失率多少",
            answer_json=_answer("客户流失率 8.3%，高于阈值 5%（来源ID=metric:12），属高风险。"),
            answer_text="", risk_level="LOW", today="2026-09-22")
        facts = [c for c in cs if c.kind == T.MEM_KIND_FACT]
        assert facts
        assert "来源ID" not in facts[0].content
        assert "metric:12" not in facts[0].content

    def test_episode_strips_citations_too(self):
        """处置经过最容易整段被照抄，里面同样不能留 [n] 与 来源ID。"""
        ex = MemoryExtractor()
        cs = ex.extract(
            company_id=3, user_id=1, question="流失为什么高？",
            answer_json={"headline": "流失率升至 25%（来源ID=event:1）；据公开资料[2]",
                         "conclusion_internal": "", "risk_level": "HIGH"},
            answer_text="", risk_level="HIGH", today="2026-09-22")
        eps = [c for c in cs if c.kind == T.MEM_KIND_EPISODE]
        assert eps
        assert "来源ID" not in eps[0].content and "[2]" not in eps[0].content

    def test_preference_clips_at_fullwidth_punctuation(self):
        """断句符必须收全角问号/感叹号。

        实测踩过：「客户流失率是多少？以后结论请先给风险等级」被整句记成偏好，
        因为切分字符集里漏了「？」—— 偏好卡于是变成"这个问题问过"的副本。
        """
        ex = MemoryExtractor()
        cs = ex.extract(company_id=3, user_id=1,
                        question="客户流失率是多少？以后结论请先给风险等级",
                        answer_json={}, answer_text="", risk_level="LOW")
        prefs = [c for c in cs if c.kind == T.MEM_KIND_PREFERENCE]
        assert prefs
        assert "客户流失率是多少" not in prefs[0].content
        assert "以后结论请先给风险等级" in prefs[0].content

    def test_episode_only_on_high_or_degraded(self):
        """每次都留档 = 记忆里全是流水账。只有高风险或降级交付才值得记。"""
        ex = MemoryExtractor()
        normal = ex.extract(company_id=3, user_id=1, question="看看指标",
                            answer_json=_answer("营收 100 万，增长 5%。"),
                            answer_text="", risk_level="LOW", degrade_level="NONE")
        assert [c for c in normal if c.kind == T.MEM_KIND_EPISODE] == []
        high = ex.extract(company_id=3, user_id=1, question="看看指标",
                          answer_json=_answer("营收 100 万，下降 5%。"),
                          answer_text="", risk_level="HIGH", degrade_level="NONE")
        assert len([c for c in high if c.kind == T.MEM_KIND_EPISODE]) == 1
        degraded = ex.extract(company_id=3, user_id=1, question="看看指标",
                              answer_json=_answer("营收 100 万。"), answer_text="",
                              risk_level="LOW", degrade_level="PARTIAL")
        eps = [c for c in degraded if c.kind == T.MEM_KIND_EPISODE]
        assert len(eps) == 1 and "降级" in eps[0].content


# ------------------------------------------------------------------
# 2) 写入与冲突消解
# ------------------------------------------------------------------


class TestWrite:

    def test_first_write_is_active(self, mem):
        r = mem.write(MemoryExtractor().extract(
            company_id=3, user_id=0, question="流失情况",
            answer_json=_answer("客户流失率 8.3%，高于阈值 5%，属于高风险。"),
            answer_text="", risk_level="HIGH"))
        assert r.written >= 1 and r.superseded == 0
        rows = mem.list_rows(company_id=3)
        assert rows and all(x["status"] == T.MEM_ACTIVE for x in rows)

    def test_new_value_supersedes_old_but_keeps_it(self, mem):
        """口径变了：新条目生效，旧条目**留档**而不是被抹掉。"""
        ex = MemoryExtractor()
        mem.write(ex.extract(company_id=3, user_id=0, question="流失率多少",
                             answer_json=_answer("客户流失率 8.3%，高于阈值 5%。"),
                             answer_text="", risk_level="HIGH"))
        r2 = mem.write(ex.extract(company_id=3, user_id=0, question="流失率多少",
                                  answer_json=_answer("客户流失率 12%，高于阈值 5%。"),
                                  answer_text="", risk_level="HIGH"))
        assert r2.superseded == 1, "同主题的新口径应当取代旧条目"

        active = [x for x in mem.list_rows(company_id=3)
                  if x["status"] == T.MEM_ACTIVE and x["kind"] == T.MEM_KIND_FACT]
        old = [x for x in mem.list_rows(company_id=3, include_superseded=True)
               if x["status"] == T.MEM_SUPERSEDED and x["kind"] == T.MEM_KIND_FACT]
        assert len(active) == 1 and "12" in active[0]["content"]
        assert old and "8.3" in old[0]["content"], "旧口径必须仍可查（口径漂移要看得见）"
        assert old[0]["supersededBy"] == active[0]["id"]

    def test_same_content_reinforces_instead_of_duplicating(self, mem):
        """同一件事又发生了一次 → 强化，不新增。否则记忆会被重复条目撑满。"""
        ex = MemoryExtractor()
        for _ in range(2):
            mem.write(ex.extract(company_id=3, user_id=0, question="流失率多少",
                                 answer_json=_answer("客户流失率 8.3%，高于阈值 5%。"),
                                 answer_text="", risk_level="LOW"))
        rows = mem.list_rows(company_id=3)
        facts = [x for x in rows if x["kind"] == T.MEM_KIND_FACT]
        assert len(facts) == 1

    def test_capacity_prunes_weakest(self, mem):
        """超容量时淘汰最弱的（置为失效，不物理删除 —— 记忆是审计材料）。"""
        s = get_settings()
        s.agent.long_term_max_per_company = 2
        try:
            for v in ("3.1", "4.2", "5.3"):
                mem.write(MemoryExtractor().extract(
                    company_id=3, user_id=0, question="流失率多少",
                    answer_json=_answer(f"客户流失率 {v}%，高于阈值 1%。"),
                    answer_text="", risk_level="LOW"))
            rows = mem.list_rows(company_id=3, include_superseded=True)
            assert sum(1 for x in rows if x["status"] == T.MEM_ACTIVE) <= 2
        finally:
            s.agent.long_term_max_per_company = 500

    def test_clear_and_delete_are_real(self, mem):
        """用户说"忘掉这条"时必须真删 —— 置为失效会把内容继续留在库里。"""
        ex = MemoryExtractor()
        mem.write(ex.extract(company_id=3, user_id=0, question="流失率多少",
                             answer_json=_answer("客户流失率 8.3%，高于阈值 5%。"),
                             answer_text="", risk_level="LOW"))
        rows = mem.list_rows(company_id=3)
        assert mem.delete(rows[0]["id"]) == 1
        assert mem.list_rows(company_id=3) == []


# ------------------------------------------------------------------
# 3) 召回与注入
# ------------------------------------------------------------------


class TestRecall:

    def _seed(self, mem):
        ex = MemoryExtractor()
        mem.write(ex.extract(company_id=3, user_id=0, question="客户流失情况如何",
                             answer_json=_answer("客户流失率 8.3%，高于阈值 5%，属于高风险。"),
                             answer_text="", risk_level="HIGH"))
        mem.write(ex.extract(company_id=3, user_id=1,
                             question="以后结论请先给风险等级",
                             answer_json={}, answer_text="", risk_level="LOW"))
        # 另一家企业的记忆，不该被召回
        mem.write(ex.extract(company_id=4, user_id=0, question="退款情况",
                             answer_json=_answer("退款额 30 万，上升 12%，属于高风险。"),
                             answer_text="", risk_level="HIGH"))

    def test_recall_is_scoped_by_company(self, mem):
        """租户隔离同样是记忆的红线：不能把 A 企业的事记到 B 企业头上。"""
        self._seed(mem)
        hits = mem.recall(3, 1, "客户流失率是多少")
        assert hits, "相关问题应当召回记忆"
        assert all("退款额" not in h.content for h in hits)

    def test_irrelevant_question_recalls_nothing(self, mem):
        self._seed(mem)
        hits = mem.recall(3, 1, "今天午饭吃什么")
        assert hits == [], "无关问题不该被塞进无关记忆"

    def test_preference_is_reused_across_sessions(self, mem):
        """这就是"跨会话"的意义：换个会话、换个问题，偏好仍然生效。"""
        self._seed(mem)
        hits = mem.recall(3, 1, "以后结论请先给风险等级，另外看下投诉")
        assert any(h.scope == T.MEM_SCOPE_USER for h in hits)

    def test_dedupe_same_topic(self, mem):
        """同主题只留一条：召回到两条互相矛盾的记忆比召回不到更糟。"""
        ex = MemoryExtractor()
        for v in ("8.3", "12"):
            mem.write(ex.extract(company_id=3, user_id=0, question="流失率多少",
                                 answer_json=_answer(f"客户流失率 {v}%，高于阈值 5%。"),
                                 answer_text="", risk_level="HIGH"))
        hits = mem.recall(3, 0, "客户流失率")
        keys = [(h.kind, h.topic) for h in hits]
        assert len(keys) == len(set(keys))

    def test_context_is_evidence_free(self, mem):
        """注入文本绝不能长得像证据：没有 [n]、没有"来源ID="。

        一旦记忆占了证据编号，引用核对就会把"记忆里的旧数字"验成本次证据 ——
        那等于给"编造数字"开了一条合法通道。
        """
        self._seed(mem)
        hits = mem.recall(3, 1, "客户流失率是多少")
        ctx = mem.build_context(hits)
        assert "长期记忆" in ctx
        assert "[1]" not in ctx and "[0]" not in ctx
        assert "来源ID=" not in ctx
        assert "记忆#" in ctx, "要带条目号，结论被质疑时能顺藤摸瓜查到出处"

    def test_budget_caps_injection(self, mem):
        self._seed(mem)
        hits = mem.recall(3, 1, "客户流失率是多少")
        assert len(hits) >= 2
        tiny = mem.build_context(hits, budget=80)
        assert len(tiny) <= 200, "预算很小时应当只注入少量条目，而不是全塞进去"

    def test_hits_counter_increases(self, mem):
        """命中计数是复盘"这条记忆到底有没有被用上"的唯一凭据。"""
        self._seed(mem)
        hits = mem.recall(3, 1, "客户流失率是多少")
        mem.mark_used(hits)
        rows = {x["id"]: x for x in mem.list_rows(company_id=3)}
        assert rows[hits[0].id]["hits"] == 1

    def test_diagnostics_explain_the_hit(self, mem):
        """留痕要说清楚"为什么想起了这条"，而不是只给个数。"""
        self._seed(mem)
        hits = mem.recall(3, 1, "客户流失率是多少")
        diag = mem.diagnostics(hits, mem.build_context(hits))
        assert diag["enabled"] is True and diag["hits"] == len(hits)
        assert diag["items"] and {"id", "kind", "score", "lexical", "vector"} <= set(diag["items"][0])


# ------------------------------------------------------------------
# 4) 开关与系统提示
# ------------------------------------------------------------------


class TestSwitchesAndPrompt:

    def test_master_switch_off_writes_nothing(self, mem):
        """总开关关掉就该真的什么都不写（页面上可一键回退旧行为）。"""
        s = get_settings()
        s.agent.long_term_memory = False
        try:
            r = mem.write(MemoryExtractor().extract(
                company_id=3, user_id=0, question="流失率多少",
                answer_json=_answer("客户流失率 8.3%，高于阈值 5%。"),
                answer_text="", risk_level="HIGH"))
            assert r.written == 0
            assert mem.recall(3, 0, "客户流失率") == []
        finally:
            s.agent.long_term_memory = True

    def test_write_switch_off_keeps_recall(self, mem):
        """只关写入时，已有记忆仍要能被召回（关的是"沉淀"，不是"使用"）。"""
        s = get_settings()
        mem.write(MemoryExtractor().extract(
            company_id=3, user_id=0, question="流失率多少",
            answer_json=_answer("客户流失率 8.3%，高于阈值 5%。"),
            answer_text="", risk_level="HIGH"))
        s.agent.long_term_write = False
        try:
            r = mem.write(MemoryExtractor().extract(
                company_id=3, user_id=0, question="流失率多少",
                answer_json=_answer("客户流失率 9.9%，高于阈值 5%。"),
                answer_text="", risk_level="HIGH"))
            assert r.written == 0
            assert mem.recall(3, 0, "客户流失率")
        finally:
            s.agent.long_term_write = True

    def test_system_prompt_forbids_memory_as_evidence(self):
        """护栏必须把"记忆不是证据"写死，否则模型会把历史数字写成本次结论。"""
        from app.agent.guardrail import GuardrailService

        prompt = GuardrailService().build_system_prompt(multi_agent=True)
        assert "长期记忆" in prompt
        assert "重新核实" in prompt
        assert "禁止为记忆内容标注" in prompt

    def test_rebuild_from_history_is_idempotent(self, db):
        """回填跑两遍不产生重复条目（线上"点了一下又点一下"是常态）。"""
        from datetime import datetime

        from sqlalchemy import insert

        s = get_settings()
        for i, v in enumerate(("8.3", "12")):
            db.insert_id(insert(T.ai_analysis).values(
                company_id=3, user_id=1, question="客户流失率多少",
                answer=f"一、结论\n客户流失率 {v}%，高于阈值 5%，属于高风险。",
                answer_json=f'{{"conclusion_internal":"客户流失率 {v}%，高于阈值 5%，属于高风险。",'
                           f'"headline":"流失升高","risk_level":"HIGH"}}',
                confidence="中", trace_id=f"t{i}", degrade_level="NONE",
                created_at=datetime.now()))
        s.agent.long_term_write = False  # 回填走 force，不受"实时写入"开关限制
        try:
            mem = LongTermMemory(db)
            first = mem.rebuild_from_history(3, limit=50)
            second = mem.rebuild_from_history(3, limit=50)
        finally:
            s.agent.long_term_write = True
        assert first["scanned"] == 2 and first["written"] >= 1
        assert second["written"] == 0, "重复回填不该新增条目"
        active = mem.list_rows(company_id=3)
        keys = [(x["kind"], x["topic"]) for x in active]
        assert len(keys) == len(set(keys))
