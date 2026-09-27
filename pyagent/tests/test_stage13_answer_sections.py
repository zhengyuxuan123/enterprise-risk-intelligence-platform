# -*- coding: utf-8 -*-
"""正文四节切分：模型怎么写，界面都得看得到结论（**全程离线、零 token**）。

为什么单开一个文件
------------------
这里的失败方式是本项目最隐蔽的一种：**接口 200、正文 1800 字、四个小节全空**。

界面上「一、结论（基于内部知识库与经营数据）」那一格是白的，用户看到的现象
就是「这次分析没有结论」—— 而 `answer` 字段里明明有完整的四节正文。
根因不在模型、不在提示词、不在网络，只在切分：

    模板要求：一、结论（基于内部知识库与经营数据）
    模型实写：**一、结论（基于内部知识库与经营数据）**      ← Markdown 加粗
    本地兜底：## 一、结论（基于内部数据）                  ← Markdown 标题

按裸前缀 ``line.startswith("一")`` 匹配的旧实现两种情况都切不出来，于是
``conclusion_internal`` / ``conclusion_web`` / ``uncertainties`` /
``recommendations_*`` 一起变空。真实事故记录见 ``ai_analysis`` #981。

另外两处顺带钉死：

* **「四」从来就没被填过**：旧判据要求标题里含「、结论」，而模板标题是
  「四、不确定性 / 需人工确认」，永远匹配不上；
* **正文里「一、xxx」开头的句子**不能当成新的小节起点，否则小节会被提前截断。

跑法::

    pytest tests/test_stage13_answer_sections.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.agent.agent_service import AgentService
from app.ai.llm_client import ChatResult

SECTION_KEYS = ("一", "二", "四")

#: 与模板一致的正文（普通写法）—— 对照组
PLAIN = """风险等级：HIGH
综合结论：客户流失率处于高位。

---

一、结论（基于内部知识库与经营数据）

1. 客户流失率处于高位（指标：客户流失率=15.8%）。
2. 续费率连续 2 个周期恶化（来源ID=知识库#2）。

---

二、结论（基于外部公开资料）

本次未联网核实。

---

三、建议动作

（一）基于内部证据的动作

1. 启动续费挽留专项 —— 依据：续费率恶化（来源ID=知识库#2）

（二）基于外部资料的动作

无

---

四、不确定性 / 需人工确认

1. 证据缺口：投诉与竞品明细未覆盖。
"""


def _bold(text: str) -> str:
    """模型真实输出的写法：把四个小节标题连同两处子标题都加粗。

    注意这是**真实记录 #981 的形态**，不是构造出来的极端输入 ——
    换个模型/换次采样，标题带上 ``**`` 是常态。
    """
    out = text
    for head in ("一、结论（基于内部知识库与经营数据）", "二、结论（基于外部公开资料）",
                 "三、建议动作", "四、不确定性 / 需人工确认",
                 "（一）基于内部证据的动作", "（二）基于外部资料的动作"):
        out = out.replace(head, "**" + head + "**")
    return out


class TestSectionParsing:
    """四节切分必须对「普通 / 加粗 / ## 标题」三种写法给出同一结果。"""

    def test_plain_headings_parse(self):
        """对照组：模板原样的标题当然要能切出来（否则是写反了）。"""
        assert "客户流失率处于高位" in AgentService._section(PLAIN, "一")
        assert "本次未联网核实" in AgentService._section(PLAIN, "二")
        assert AgentService._section(PLAIN, "四")

    def test_bold_headings_parse_like_plain(self):
        """★ 加粗标题必须与普通标题等价（#981 的直接根因）。"""
        bold = _bold(PLAIN)
        assert "**一、结论" in bold, "构造的输入没真的加粗，用例本身失效"
        for k in SECTION_KEYS:
            assert AgentService._section(bold, k) == AgentService._section(PLAIN, k), (
                f"第 {k} 节在加粗写法下切分结果不同："
                f"{AgentService._section(bold, k)!r} vs {AgentService._section(PLAIN, k)!r}")

    def test_hash_headings_parse(self):
        """本地兜底报告用的是 ``## 一、结论（基于内部数据）``，同样必须能切。

        它是"模型不可用时的最后一份可交付结论"，切不出来的话，
        系统越降级、界面越空白 —— 恰好是最需要看到内容的时刻。
        """
        md = ("# 风险分析（本地确定性报告）\n\n"
              "## 一、结论（基于内部数据）\n- 召回 3 篇（来源ID=6）\n\n"
              "## 二、结论（基于外部公开资料）\n- 本次未联网核实\n\n"
              "## 三、建议动作\n（一）基于内部数据的动作\n- 按上述证据逐条核对。\n\n"
              "## 四、不确定性\n- 本报告由本地规则生成。")
        assert "来源ID=6" in AgentService._section(md, "一")
        assert "未联网核实" in AgentService._section(md, "二")
        assert "本地规则" in AgentService._section(md, "四"), "「四、不确定性」从来切不出来的那个 bug"

    def test_uncertainties_section_is_extractable(self):
        """「四、不确定性 / 需人工确认」必须能取到 —— 旧判据要求含「、结论」，永远取不到。"""
        for text in (PLAIN, _bold(PLAIN)):
            body = AgentService._section(text, "四")
            assert "证据缺口" in body, f"不确定性小节没取到：{body!r}"

    def test_sections_do_not_leak_into_each_other(self):
        """内部结论里不能出现外部小节的内容，反之亦然（来源分层铁律）。"""
        for text in (PLAIN, _bold(PLAIN)):
            assert "未联网核实" not in AgentService._section(text, "一")
            assert "客户流失率处于高位" not in AgentService._section(text, "二")

    def test_separator_lines_are_not_part_of_the_body(self):
        """小节之间的 ``---`` 不属于任何一节正文，别带到结论里。"""
        for text in (PLAIN, _bold(PLAIN)):
            for k in ("一", "二", "四"):
                body = AgentService._section(text, k)
                assert not body.rstrip().endswith("---"), f"第 {k} 节尾部残留分隔线：{body!r}"

    def test_numbered_prose_is_not_a_heading(self):
        """正文里「一、客户流失率处于高位」这种句子不能把小节提前截断。"""
        text = ("一、结论（基于内部知识库与经营数据）\n"
                "1. 证据一：正常。\n"
                "一、此处是正文里的序号，不是新小节。\n"
                "2. 证据二：仍然属于本节。\n\n"
                "二、结论（基于外部公开资料）\n本次未联网核实。\n")
        body = AgentService._section(text, "一")
        assert "证据二" in body, f"小节被正文里的「一、」截断了：{body!r}"

    def test_empty_draft_is_empty_not_crash(self):
        assert AgentService._section("", "一") == ""
        assert AgentService._section(None, "一") == ""


class TestRecsParsing:
    """建议动作的两个子小节（一）/（二）同样要能顶住加粗写法。"""

    def test_internal_and_external_recs_split(self):
        for text in (PLAIN, _bold(PLAIN)):
            inner = AgentService._recs(text, [], True)
            outer = AgentService._recs(text, [], False)
            assert [r["action"] for r in inner] == ["启动续费挽留专项 —— 依据：续费率恶化（来源ID=知识库#2）"], inner
            assert [r["action"] for r in outer] == ["无"], outer

    def test_no_recs_when_section_missing(self):
        """抽不到就返回空 —— 绝不凭空编建议。"""
        assert AgentService._recs("一、结论（基于内部数据）\n- 无", [], True) == []


# --------------------------------------------------------------------------- #
# 端到端：走完整 analyze，落库的 answer_json 必须有结论
# --------------------------------------------------------------------------- #


class _FakeLlm:
    """按队列应答的假模型（与 test_stage5 同构，只为不跨文件 import 测试模块）。"""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: List[Dict[str, Any]] = []

    def chat_with_tools(self, tools, messages, temperature=0.2, max_tokens=0) -> ChatResult:
        self.calls.append({"tools": [t.name for t in (tools or [])], "n": len(messages)})
        return ChatResult(content=self.answer, finish_reason="stop")

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        self.calls.append({"kind": "complete"})
        return self.answer

    @property
    def model(self) -> str:
        return "fake-model"

    def is_configured(self) -> bool:
        return True

    def get_model(self) -> str:
        return "fake-model"

    def get_model_or_none(self) -> Optional[str]:  # pragma: no cover - 兼容旧接口
        return "fake-model"


def _svc(answer: str):
    from fake_langchain import adapt_factory

    llm = _FakeLlm(answer)
    svc = AgentService(llm=llm, lg_model_factory=adapt_factory(llm))
    svc.cache_enabled = False
    svc.cache.clear()
    return svc


class TestEndToEndAnswerJson:
    def test_bold_headings_still_produce_a_conclusion(self):
        """★ 落库的 ``answer_json`` 必须带上四节内容（#981 的回归）。"""
        out = _svc(_bold(PLAIN)).analyze(1, "客户流失率为何上升", 5, user_id=1)

        assert out.content.strip(), "正文本身应当非空"
        aj = out.answer_json if isinstance(out.answer_json, dict) else json.loads(out.answer_json)
        assert "客户流失率处于高位" in aj["conclusion_internal"], \
            f"内部结论为空 —— 界面就会显示「没有结论」：{aj['conclusion_internal']!r}"
        assert "未联网核实" in aj["conclusion_web"], aj["conclusion_web"]
        assert "证据缺口" in aj["uncertainties"], aj["uncertainties"]
        assert [r["action"] for r in aj["recommendations_internal"]], aj["recommendations_internal"]
        # 向后兼容键：旧前端读 conclusion，不能是空的
        assert aj["conclusion"], "conclusion 是旧前端的兜底键，不能为空"

    def test_plain_headings_behave_identically(self):
        """普通写法与加粗写法落库结果一致 —— 否则"换个模型结论就没了"。"""
        a = _svc(PLAIN).analyze(1, "客户流失率为何上升", 5, user_id=1)
        b = _svc(_bold(PLAIN)).analyze(1, "客户流失率为何上升", 5, user_id=1)
        aj_a = a.answer_json if isinstance(a.answer_json, dict) else json.loads(a.answer_json)
        aj_b = b.answer_json if isinstance(b.answer_json, dict) else json.loads(b.answer_json)
        for k in ("conclusion_internal", "conclusion_web", "uncertainties"):
            assert aj_a[k] == aj_b[k], f"{k} 两种写法不一致"
