"""长期记忆的抽取 —— 全部是本地规则，**一次模型调用都不发**。

为什么不"让模型总结这次分析"
------------------------------
那等于为记忆额外付一次推理（一次分析本来就要调好几轮模型），而且不可复现：
同一份分析两次跑出两条不同的记忆，从此每家企业的"事实卡"都不一样，
出了问题没法复盘。更现实的是成本 —— 记忆的价值在"长期可用"，
一旦被它拖成每轮都贵的功能，最后一定会被人关掉，关掉的记忆等于没有。

抽三类
------
* ``FACT``       企业事实：结论里**带数字的风险断言**（"客户流失率 8.3%，高于阈值"）；
* ``EPISODE``    处置经过：高风险或降级的那几次分析，记下"问过什么、当时结论是什么"；
* ``PREFERENCE`` 用户偏好：明确表达"以后/记住/默认/一律"这类长期要求的提问。

三条必须守住的红线
------------------
1. **数字只是历史快照**。content 里必须带日期，注入时提示词也会写明"不得作为本次结论依据"。
   否则模型会把记忆里的旧数字当成本次证据写进结论 —— 那正是最严重的"编造数字"。
2. **同主题覆盖而不是堆叠**。``topic_key`` 里数字被抹平（8.3 与 12 归成同一主题的两次口径），
   于是"口径变了"这件事在记忆里表现为"一条新记忆 + 一条被取代的旧记忆"，看得见、可追溯。
3. **不抽建议**。只抽可被复核的事实与经过；建议随情境变化，沉淀下来只会让下一轮照着旧建议抄。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from ..core.logging import get_logger
from ..db import tables as T

log = get_logger("pyagent.agent.memory_extract")

#: 句子切分（与 ``agent/memory.py`` 的压缩用同一套标点，避免同一段文字两边切出不同句）
_SENT = re.compile(r"[^。！？；\n]+[。！？；]?")
_NUM = re.compile(r"\d")
#: 归一化时抹掉的标点与空白：只影响指纹，不影响原文
_PUNCT = re.compile(
    r"[\s\u3000，。、；：？！,.;:?!（）()【】\[\]{}《》\"'“”‘’—+*/=~·|…]+"
)
#: 指纹里要抹掉的日期。不抹掉的话"同一句话今天说、明天说"会被判成两条不同内容，
#: 记忆会以每天一条的速度膨胀，而内容其实一模一样。
_DATE = re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?")

#: "带数字的风险断言"必须同时含风险词，否则"共 3 条记录"这种也会被当成事实卡。
_RISK_WORDS = (
    "风险", "下降", "下滑", "上升", "增长", "流失", "投诉", "逾期", "亏损", "低于", "高于",
    "超", "异常", "预警", "恶化", "减少", "偏离", "不足", "逾期", "退费", "流失率",
)

#: 主题词表：**长词必须排在短词前面**（"流失率"先于"流失"），否则先命中的短词会把主题切碎。
_TOPIC_WORDS = (
    "毛利率", "净利润", "净利", "利润率", "流失率", "续费率", "留存率", "满意度", "客单价",
    "应收账龄", "应收账款", "现金流", "投诉量", "重复投诉", "风险事件", "库存",
    "营收", "收入", "利润", "毛利", "成本", "费用", "流失", "续费", "留存", "回款",
    "投诉", "逾期", "SLA",
)

#: 高风险/严重信号：命中即提高重要度（这类事实过期得慢，值得反复提醒）
_HIGH_WORDS = ("高风险", "严重", "极高", "急剧", "恶化", "爆雷", "逾期", "触发")

#: 用户偏好信号。刻意只用**多字、且指向长期**的措辞：
#: "请用""不要"这种单词在一次性指令里天天出现，收进来会把每次提问都记成偏好。
_PREF_SIGNALS = (
    "以后", "记住", "每次都", "每次请", "下次", "默认", "一律", "统一用",
    "我习惯", "我的习惯", "长期", "持续关注", "prefer", "always",
)

#: 抽取时要从记忆里**剥掉**的引用编号。
#:
#: 为什么必须剥（这是"记忆污染证据链"最具体的形态）：
#: 上一次分析的正文里写着「（来源ID=event:1）」「据公开资料[2]」，如果原样存进记忆，
#: 下次注入时模型会把它当成可以照抄的依据，而**本次**证据里根本没有 event:1 →
#: 要么产出悬空引用，要么让引用核对把历史编号验成本次证据。
#: 记忆里该留的是"当时说了什么"，不是"当时引了哪一条"。
_CITE_KB = re.compile(
    r"[（(]?\s*(?:来源\s*ID|来源ID|知识库\s*ID|知识库ID|文档\s*ID|文档ID)\s*[=:：]\s*"
    r"[A-Za-z]{0,12}:?\d{1,7}\s*[)）]?",
    re.IGNORECASE,
)
_CITE_WEB = re.compile(r"\[(\d{1,2})]")
_CITE_DATA = re.compile(r"[（(]\s*[）)]")


@dataclass(frozen=True)
class Candidate:
    """一条待落库的记忆。``topic_key`` 相同即视为同一主题的新口径。"""

    scope: str
    kind: str
    topic: str
    topic_key: str
    content: str
    importance: float = 0.5
    company_id: int = 0
    user_id: int = 0
    source_analysis_id: Optional[int] = None
    source_trace_id: Optional[str] = None


def normalize_topic_key(text: str, salt: str = "") -> str:
    """主题指纹：抹掉标点与**数字**后取 md5。

    抹数字是有意的：同一指标这次 8.3%、下次 12%，它们的主题是同一个，
    应当互相覆盖而不是并列堆着 —— 否则问过三次就会有三条互相矛盾的"事实"同时被注入。
    """
    norm = _PUNCT.sub("", (text or "").lower())
    norm = re.sub(r"\d+(?:\.\d+)?%?", "#", norm)
    return hashlib.md5((salt + "|" + norm).encode("utf-8")).hexdigest()[:24]


def content_fingerprint(text: str) -> str:
    """内容指纹：抹标点与**日期**，但**保留数字**。

    和 :func:`normalize_topic_key` 的差别只有一处，但这一处是关键的：

    * 主题指纹抹数字 → "流失率 8.3%" 与 "流失率 12%" 是**同一主题**（该覆盖）；
    * 内容指纹留数字 → 它们是**两件事**（该判为"口径变了"，而不是"又说了一遍"）。

    两者混用会把每次口径变化都当成"同一件事又发生一次"，于是
    "8.3% → 12%" 这种漂移被静默抹平 —— 而那正是最该被看见的信号。
    """
    s = _PUNCT.sub("", (text or "").lower())
    s = _DATE.sub("#date", s)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:24]


def strip_citations(text: str) -> str:
    """剥掉句子里的证据编号（``来源ID=x``、``[n]``）与剥空后残留的空括号。

    记忆是"历史快照"，不是证据。带着上次的编号存进去，下次就会被照抄成
    "本次证据里并不存在的引用" —— 那比没有引用更危险，因为它看起来像有依据。
    """
    s = _CITE_KB.sub("", text or "")
    s = _CITE_WEB.sub("", s)
    s = _CITE_DATA.sub("", s)
    return re.sub(r"[，、；]\s*([。！？；])", r"\1", s).strip()


def pick_topic(sentence: str) -> str:
    """从句子里挑主题词；挑不到就返回空串（由调用方退化为句子指纹）。"""
    for w in _TOPIC_WORDS:
        if w in sentence:
            return w
    return ""


class MemoryExtractor:
    """无状态，可当单例。"""

    #: 每次分析最多沉淀几条事实（多了会把记忆变成"每次分析的副本"）
    MAX_FACTS = 2

    def extract(
        self,
        *,
        company_id: Optional[int],
        user_id: Optional[int],
        question: Optional[str],
        answer_json: Optional[Dict[str, Any]] = None,
        answer_text: Optional[str] = None,
        risk_level: Optional[str] = None,
        analysis_id: Optional[int] = None,
        trace_id: Optional[str] = None,
        degrade_level: Optional[str] = None,
        today: Optional[str] = None,
    ) -> List[Candidate]:
        """把一次分析的结果折算成记忆条目。任何一步失败都只影响这一条，不外抛。"""
        cid = int(company_id or 0)
        uid = int(user_id or 0)
        day = today or date.today().isoformat()
        out: List[Candidate] = []

        for c in self._preferences(question, cid, uid, day, analysis_id, trace_id):
            out.append(c)
        for c in self._facts(answer_json, answer_text, cid, day, analysis_id, trace_id):
            out.append(c)
        for c in self._episode(question, answer_json, answer_text, risk_level,
                               degrade_level, cid, day, analysis_id, trace_id):
            out.append(c)
        return out

    # ------------------------------------------------------------------ 用户偏好

    def _preferences(self, question, cid, uid, day, aid, tid) -> List[Candidate]:
        q = (question or "").strip()
        if not q or uid <= 0:
            return []
        hit = [w for w in _PREF_SIGNALS if w in q]
        if not hit:
            return []
        text = _clip_preference(q)
        # 太长的"偏好"其实是在描述需求，不是偏好；长文沉淀下来会占满注入预算。
        if not text or len(text) > 200:
            return []
        return [Candidate(
            scope=T.MEM_SCOPE_USER,
            kind=T.MEM_KIND_PREFERENCE,
            topic="用户偏好",
            # 偏好按**内容**指纹去重（不是按"偏好"一个大主题）：
            # "以后都用简体中文"和"以后结论别超过 500 字"是两条并存的偏好，不该互相覆盖。
            topic_key=normalize_topic_key(text, salt="pref"),
            content=f"【用户偏好｜{day}】{text}",
            importance=0.70,
            company_id=cid,
            user_id=uid,
            source_analysis_id=aid,
            source_trace_id=tid,
        )]

    # ------------------------------------------------------------------ 企业事实

    def _facts(self, answer_json, answer_text, cid, day, aid, tid) -> List[Candidate]:
        src = ""
        if isinstance(answer_json, dict):
            src = str(answer_json.get("conclusion_internal") or "")
        if not src.strip():
            src = _section(answer_text, "一")

        picked: List[tuple] = []
        for m in _SENT.finditer(src or ""):
            sent = m.group(0).strip()
            # 三个条件缺一不可：够长（不是标题）、有数字（是可核对的事实）、有风险词（不是流水账）
            if len(sent) < 10 or not _NUM.search(sent):
                continue
            if not any(w in sent for w in _RISK_WORDS):
                continue
            picked.append((sent, pick_topic(sent)))
            if len(picked) >= self.MAX_FACTS:
                break

        out: List[Candidate] = []
        for sent, topic_word in picked:
            body = strip_citations(sent)[:300]
            if not body:
                continue
            topic = topic_word or body[:24]
            out.append(Candidate(
                scope=T.MEM_SCOPE_COMPANY,
                kind=T.MEM_KIND_FACT,
                topic=topic,
                topic_key=normalize_topic_key(topic_word or body, salt="fact"),
                content=f"【历史事实｜{day}】{body}",
                importance=0.80 if any(w in body for w in _HIGH_WORDS) else 0.55,
                company_id=cid,
                user_id=0,
                source_analysis_id=aid,
                source_trace_id=tid,
            ))
        return out

    # ------------------------------------------------------------------ 处置经过

    def _episode(self, question, answer_json, answer_text, risk_level,
                 degrade_level, cid, day, aid, tid) -> List[Candidate]:
        """只在高风险或降级时留档。

        每次都记 = 记忆里全是"我问过什么"的流水账，真正有用的那几次被淹没。
        """
        high = str(risk_level or "").upper() == "HIGH"
        degraded = bool(degrade_level) and str(degrade_level).upper() != "NONE"
        if not high and not degraded:
            return []
        q = (question or "").strip()
        if not q:
            return []
        headline = ""
        if isinstance(answer_json, dict):
            headline = str(answer_json.get("headline") or "").strip()
        if not headline:
            headline = (answer_text or "").strip().splitlines()[0][:120] if answer_text else ""
        tag = "高风险分析" if high else "降级分析"
        # 题目与当时结论都要剥掉引用编号：EPISODE 是最容易被照抄进下一次正文的一段
        content = (
            f"【历史{tag}｜{day}】问题：{strip_citations(q)[:120]}；"
            f"当时结论：{strip_citations(headline)[:160] or '未留存结论摘要'}"
            f"{'；该次为降级交付（结论可信度有限）' if degraded else ''}"
        )
        return [Candidate(
            scope=T.MEM_SCOPE_COMPANY,
            kind=T.MEM_KIND_EPISODE,
            topic=q[:40],
            topic_key=normalize_topic_key(q, salt="episode"),
            content=content[:1000],
            importance=0.75,
            company_id=cid,
            user_id=0,
            source_analysis_id=aid,
            source_trace_id=tid,
        )]


# ---------------------------------------------------------------------- 工具


def _clip_preference(text: str) -> str:
    """只留"偏好"那一小段，别把整句提问一起记下来。

    "请分析客户流失情况，以后结论请先用一句风险等级开头" —— 要记的是后半句。
    记整句的后果有两个：偏好卡变成"这条问题问过"的副本，
    而且下次注入时模型会看到一堆与偏好无关的业务描述。
    """
    s = text.rstrip("？?。 　")
    pos = -1
    for w in _PREF_SIGNALS:
        i = s.find(w)
        if i >= 0 and (pos < 0 or i < pos):
            pos = i
    if pos <= 0:
        return s
    cut = -1
    # 断句符要收全：漏掉全角问号时，"…是多少？以后结论请…" 会被整句记下来
    # （实测踩过：偏好卡里混进了整段业务描述）。
    for ch in "，。；、,.;:：？！!?":
        c = s.rfind(ch, 0, pos)
        if c > cut:
            cut = c
    seg = s[cut + 1:] if cut >= 0 else s
    seg = seg.strip()
    # 切完只剩两三个字说明切坏了（比如"，每次"），退回原句更安全。
    return seg if len(seg) >= 4 else s


def _section(answer: Optional[str], marker: str) -> str:
    """从正文里取「一、结论」小节（与 ``AgentService._section`` 同语义）。

    正文没有小节标题（旧记录或降级交付）时返回空串，由调用方退化处理。
    """
    if not answer:
        return ""
    text = answer
    start = -1
    for m in ("一、结论", "一.结论", "一、内部", "1. 结论"):
        pos = text.find(m)
        if pos >= 0 and (start < 0 or pos < start):
            start = pos
    if start < 0:
        return ""
    tail = text[start:]
    for m in ("二、结论", "二.结论", "二、外部", "三、建议", "三.建议"):
        pos = tail.find(m, 1)
        if pos > 0:
            return tail[:pos]
    return tail
