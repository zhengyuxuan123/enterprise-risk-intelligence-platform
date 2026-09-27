"""答案引用核对（对应 Java ``AnswerGroundingService``）。

模型写下的「据 [3]」「来源ID=7」这类标注，必须能在证据里找到对应条目，
否则就是不可核对的引用。本服务做**确定性校验（不消耗 token）**，产出：

* 网页引用：正文用到的 ``[n]`` 是否都存在于本次联网来源中（悬空引用 dangling）；
* 网页覆盖率：本次拿到的来源有多少条被正文真正引用过（unused）；
* 内部引用：正文里的「来源ID=x」是否都存在于知识库召回结果中。

结果写入 ``answer_json.grounding_check``，同时作为评估回归里 ``citation`` 类的确定性用例。

引用标识 ≠ chunk_id
-------------------
知识库的 ref 是文档号 ``7``（chunk 是 ``k:7#0``）；结构化切片是 ``metric:12``；
聚合切片 ``agg:{企业}:metric:16`` 的 ref 剥掉企业前缀后也是 ``metric:16``。
所以内部引用的正则必须认得 ``{英文前缀}:{数字}`` 这种形式，
只写 ``\\d+`` 的话结构化引用一律匹配不到，会被误判成"没有引用"，失去核对意义。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from ..rag.textnorm import round3
from ..rag.text import bigrams as ragtext_bigrams
from ..rag.text import jaccard as ragtext_jaccard
from ..rag.text import lower as ragtext_lower
from ..rag.text import tokenize as ragtext_tokenize

#: 正文中的网页引用：[1] [12]（限 1~2 位，避免误吃 markdown 角标/年份）。
WEB_CITE = re.compile(r"\[(\d{1,2})]")
#: 正文中的内部引用：来源ID=7 / 知识库ID＝7 / 文档ID:7，且要认得 event:3 / metric:12。
KB_CITE = re.compile(
    r"(?:来源\s*ID|来源ID|知识库\s*ID|知识库ID|文档\s*ID|文档ID)\s*[=:：]?\s*"
    r"([A-Za-z]{3,12}:\d{1,7}|\d{1,7})",
    re.IGNORECASE,
)
#: 「参考来源」小节起点：其后的 [n] 属于来源清单，不应计入引用统计。
REF_SECTION = re.compile(r"(参考来源|参考文献|引用来源|Sources?)\s*[（(:：]?")

#: 「外部结论」小节起点：修补网页引用时只允许落在这个区间内。
EXTERNAL_HEADINGS = ["二、结论", "二.结论", "二、外部", "外部公开资料"]
#: 「内部结论」小节起点：内部引用不能落进外部小节，避免内外混写。
INTERNAL_HEADINGS = ["一、结论", "一.结论", "一、内部", "内部知识库"]
#: 建议动作里「基于外部资料的动作」小节：网页引用同样允许落在这里。
EXTERNAL_REC_HEADINGS = ["（二）基于外部", "(二)基于外部", "基于外部资料的动作"]
#: 句子里已有的内部数据依据标记：出现过就不再给它补知识库来源。
HAS_DATA_BASIS = re.compile(
    r"（指标[：:]|（风险事件[：:]|（投诉[：:]|（竞品[：:]|指标[：:]\s*\S|风险事件[：:]\s*\S"
)


@dataclass
class Check:
    """核对结果。``ok`` 为 False 时前端打徽标提示。"""

    web_sources: int = 0
    kb_sources: int = 0
    web_cited: List[int] = field(default_factory=list)
    web_dangling: List[int] = field(default_factory=list)
    web_unused: List[int] = field(default_factory=list)
    kb_cited: List[str] = field(default_factory=list)
    kb_dangling: List[str] = field(default_factory=list)
    web_cited_titles: List[Dict[str, Any]] = field(default_factory=list)
    kb_cited_titles: List[Dict[str, Any]] = field(default_factory=list)
    citation_coverage: float = 1.0
    issues: List[str] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "webSources": self.web_sources,
            "kbSources": self.kb_sources,
            "webCited": self.web_cited,
            "webDangling": self.web_dangling,
            "webUnused": self.web_unused,
            "kbCited": self.kb_cited,
            "kbDangling": self.kb_dangling,
            "webCitedTitles": self.web_cited_titles,
            "kbCitedTitles": self.kb_cited_titles,
            "citationCoverage": self.citation_coverage,
            "issues": self.issues,
            "ok": self.ok,
        }


@dataclass
class Repair:
    answer: str = ""
    repairs: List[Dict[str, Any]] = field(default_factory=list)


class AnswerGroundingService:
    """核对 + 自动补齐（护栏闭环）。"""

    def verify(self, answer: Optional[str], evidence: Optional[Sequence[Dict[str, Any]]]) -> Check:
        c = Check()
        text = answer or ""

        web_available: Set[int] = set()
        web_titles: Dict[int, str] = {}
        kb_available: Set[str] = set()
        kb_titles: Dict[str, str] = {}

        for e in evidence or []:
            if not isinstance(e, dict):
                continue
            srcs = e.get("sources")
            if not isinstance(srcs, list):
                continue
            if e.get("sourceType") == "web":
                for o in srcs:
                    if not isinstance(o, dict):
                        continue
                    idx = _int_of(o.get("index"))
                    if idx is None:
                        continue
                    web_available.add(idx)
                    web_titles[idx] = str(o.get("title") or "")
            elif e.get("sourceType") == "knowledge":
                for o in srcs:
                    if not isinstance(o, dict):
                        continue
                    did = str(o.get("sourceRef") or "").strip() or str(o.get("documentId") or "").strip()
                    if not did:
                        continue
                    kb_available.add(did)
                    kb_titles[did] = str(o.get("title") or "")

        c.web_sources = len(web_available)
        c.kb_sources = len(kb_available)

        # 只统计「参考来源」小节之前的正文，避免来源清单自身被当成引用
        m = REF_SECTION.search(text)
        body = text[: m.start()] if m else text

        cited_web: Set[int] = set()
        for mm in WEB_CITE.finditer(body):
            try:
                cited_web.add(int(mm.group(1)))
            except ValueError:
                pass
        cited_kb: Set[str] = {mm.group(1) for mm in KB_CITE.finditer(body)}

        c.web_cited = sorted(i for i in cited_web if i in web_available)
        c.web_dangling = sorted(i for i in cited_web if i not in web_available)
        c.web_unused = sorted(i for i in web_available if i not in cited_web)
        c.kb_cited = sorted(d for d in cited_kb if d in kb_available)
        c.kb_dangling = sorted(d for d in cited_kb if d not in kb_available)
        c.web_cited_titles = _titles(web_titles, c.web_cited, True)
        c.kb_cited_titles = _titles(kb_titles, c.kb_cited, False)

        if c.web_dangling:
            c.issues.append(f"正文引用了不存在的网页来源序号 {c.web_dangling}（可能是编造或序号错位）")
        if c.kb_dangling:
            c.issues.append(f"正文引用了不存在的知识库来源ID {c.kb_dangling}（该文档未被本次召回）")
        if web_available and not c.web_cited:
            c.issues.append(f"已联网检索到 {len(web_available)} 条来源，但正文没有任何 [n] 标注")

        if web_available:
            c.citation_coverage = _round2(len(c.web_cited) / len(web_available))
        elif kb_available:
            c.citation_coverage = _round2(len(c.kb_cited) / len(kb_available))
        else:
            c.citation_coverage = 1.0
        c.ok = not c.issues
        return c

    # -- 护栏闭环：把「检测到没标来源」升级为「自动补齐来源」 ----------

    def repair(self, answer: Optional[str],
               evidence: Optional[Sequence[Dict[str, Any]]]) -> Repair:
        """只做**证据能支撑**的补齐，改动都是确定性的（不调模型、不耗 token）。

        某条来源自始至终没被引用，且存在一个句子与它的标题/站点/主题词高度重合
        → 在该句句末补 ``[n]``（外部）或 ``来源ID=x``（内部）。

        三条边界（都是踩出来的）：

        * 所有改动都限制在 **正文** 内 —— 「参考来源（网页）」清单本身不是正文，
          往清单行尾补 ``[1]`` 会把 URL 撑坏（``…example.com/chain[1]``）。
        * 网页引用只能落进「外部结论」或「基于外部资料的动作」区间，
          内部引用只能落进内部区间 —— 混写等于把内部数据说成外部公开资料。
        * 已经有「（指标：」「（风险事件：」这类数据依据标记的句子不再补知识库来源，
          否则等于把一条数据型结论硬挂到某篇文档上 —— 那是替模型说话。
        """
        text = answer or ""
        if not text.strip():
            return Repair(text, [])

        # ---- 收集可用来源 ----
        web_pool: Dict[int, Dict[str, Any]] = {}
        kb_pool: Dict[str, Dict[str, Any]] = {}
        for e in evidence or []:
            if not isinstance(e, dict):
                continue
            srcs = e.get("sources")
            if not isinstance(srcs, list):
                continue
            if e.get("sourceType") == "web":
                for o in srcs:
                    if not isinstance(o, dict):
                        continue
                    idx = _int_of(o.get("index"))
                    if idx is not None:
                        web_pool.setdefault(idx, o)
            elif e.get("sourceType") == "knowledge":
                for o in srcs:
                    if not isinstance(o, dict):
                        continue
                    did = str(o.get("sourceRef") or "").strip() or str(o.get("documentId") or "").strip()
                    if did:
                        kb_pool.setdefault(did, o)
        if not web_pool and not kb_pool:
            return Repair(text, [])

        # ---- 已存在的位置信息 ----
        m = REF_SECTION.search(text)
        body_end = m.start() if m else len(text)
        has_ref_section = body_end < len(text)
        ext_start = _heading_start(text, body_end, EXTERNAL_HEADINGS)
        ext_end = body_end
        if ext_start >= 0:
            # 外部小节截止到下一个主章节（三、建议动作 / 四、不确定性）
            for h in ("三、", "四、", "不确定性"):
                p = text.find(h, ext_start)
                if 0 <= p < ext_end:
                    ext_end = p
        ext_rec_start = _heading_start(text, body_end, EXTERNAL_REC_HEADINGS)

        cited_web = {int(x) for x in WEB_CITE.findall(text[:body_end])}
        cited_kb = set(KB_CITE.findall(text[:body_end]))

        spans = _sentence_spans(text, body_end)
        inserts: Dict[int, str] = {}
        repairs: List[Dict[str, Any]] = []
        per_sentence: Dict[int, int] = {}

        # ---- ① 网页来源 ----
        if ext_start >= 0 or ext_rec_start >= 0:
            for idx, src in web_pool.items():
                if idx in cited_web:
                    continue
                title = str(src.get("title") or "")
                site = str(src.get("siteName") or "")
                lt, ls = ragtext_lower(title), ragtext_lower(site)
                terms = ragtext_tokenize(f"{title} {site}")
                best_i, best = -1, -1
                for i, sp in enumerate(spans):
                    if not _in_external_range(sp[0], ext_start, ext_end, ext_rec_start, text, body_end):
                        continue
                    sentence = text[sp[0]:sp[1]]
                    if WEB_CITE.search(sentence):
                        continue
                    if per_sentence.get(i, 0) >= 2:
                        continue
                    score = _title_score(ragtext_lower(sentence), lt, ls, terms)
                    if score > best:
                        best, best_i = score, i
                if best_i >= 0 and best >= 2:
                    sp = spans[best_i]
                    pos = _insert_pos_at_sentence_end(text, sp)
                    inserts[pos] = inserts.get(pos, "") + f"[{idx}]"
                    per_sentence[best_i] = per_sentence.get(best_i, 0) + 1
                    repairs.append(_rec("web", f"[{idx}]", title,
                                        text[sp[0]:min(sp[1], sp[0] + 60)]))

        # ---- ② 知识库来源 ----
        for doc_id, src in kb_pool.items():
            if doc_id in cited_kb:
                continue
            title = str(src.get("title") or "")
            if not title.strip():
                continue
            lt = ragtext_lower(title)
            terms = ragtext_tokenize(title)
            title_grams = set(ragtext_bigrams(title))
            best_i, best = -1, 0.0
            for i, sp in enumerate(spans):
                # 别把内部引用写进外部小节
                if ext_start >= 0 and sp[0] >= ext_start and sp[1] <= ext_end:
                    continue
                sentence = text[sp[0]:sp[1]]
                if KB_CITE.search(sentence):
                    continue
                if HAS_DATA_BASIS.search(sentence):
                    continue
                score = _kb_score(ragtext_lower(sentence), lt, title_grams, terms)
                if score >= 0.62 and score > best:
                    best, best_i = score, i
            if best_i >= 0:
                sp = spans[best_i]
                pos = _insert_pos_at_sentence_end(text, sp)
                inserts[pos] = inserts.get(pos, "") + f"（来源ID={doc_id}）"
                repairs.append(_rec("kb", f"来源ID={doc_id}", title,
                                    text[sp[0]:min(sp[1], sp[0] + 60)]))

        # ---- 应用所有插入（倒序，防止偏移漂移） ----
        out = text
        for pos in sorted(inserts, reverse=True):
            out = out[:pos] + inserts[pos] + out[pos:]

        # ---- 补齐参考来源清单 ----
        has_cite = bool(WEB_CITE.search(out[:min(len(out), body_end + 400)]))
        if has_cite and not has_ref_section and web_pool:
            lines = ["", "", "参考来源（网页）：", ""]
            for idx in sorted(web_pool):
                s = web_pool[idx]
                lines.append(f"[{idx}] {str(s.get('title') or '')}")
                url = str(s.get("url") or "")
                if url.strip():
                    lines.append(f"    URL: {url}")
            out = out.rstrip() + "\n".join(lines) + "\n"
            repairs.append(_rec("reference", "参考来源（网页）", "自动补齐来源清单", ""))
        return Repair(out, repairs)

    def summarize(self, c: Check) -> str:
        if c.ok:
            return "引用核对通过"
        return "；".join(c.issues)


# ------------------------------------------------------------------
# 小工具
# ------------------------------------------------------------------


def _sentence_spans(text: str, limit: int):
    """按句末标点切句，返回每个句子在原文中的 ``[start, end)``（**含**句末标点）。

    与 Java ``sentenceSpans`` 一致：不做最小长度过滤。曾经加了「长度 ≥ 4 才成句」，
    短句（尤其是小节标题行）会被跳过，标题行反而成了最该补引用的位置。
    """
    spans = []
    i = 0
    while i < limit:
        j = i
        while j < limit and text[j] not in "。！？；\n":
            j += 1
        if j < limit:
            j += 1
        spans.append((i, j))
        i = j
    return spans


def _insert_pos_at_sentence_end(text: str, span) -> int:
    """插在句末标点**之前**；没有标点就插在句尾。

    只退**一个**字符（与 Java 一致）。退多了会把 `[n]` 插到上一句的句号前面。
    """
    end = span[1]
    if end > span[0] and text[end - 1] in "。！？；\n":
        return end - 1
    return end


def _heading_start(text: str, limit: int, headings: Sequence[str]) -> int:
    """小节起点；``limit`` 之后的不算（正文止于「参考来源」清单）。"""
    best = -1
    for h in headings:
        i = text.find(h)
        if 0 <= i < limit and (best < 0 or i < best):
            best = i
    return best


def _in_external_range(pos: int, ext_start: int, ext_end: int,
                       ext_rec_start: int, text: str, body_end: int) -> bool:
    """网页引用只允许落在「外部结论」或「基于外部资料的动作」两个区间里。"""
    if ext_start >= 0 and ext_start <= pos < ext_end:
        return True
    if ext_rec_start >= 0 and pos >= ext_rec_start:
        end = body_end
        for h in ("四、", "不确定性"):
            p = text.find(h, ext_rec_start)
            if 0 <= p < end:
                end = p
        return pos < end
    return False


def _title_score(lower_sentence: str, lower_title: str, lower_site: str, terms: Sequence[str]) -> int:
    """句子与「网页来源标题 + 站点」的重合度。"""
    hits = 0
    counted: Set[str] = set()
    for t in terms:
        if t in counted:
            continue
        if t in lower_sentence:
            counted.add(t)
            hits += 1
    title_hit = len(lower_title) >= 3 and lower_title in lower_sentence
    gram_hit = ragtext_jaccard(set(ragtext_bigrams(lower_title)),
                               set(ragtext_bigrams(lower_sentence))) >= 0.2
    site_hit = bool(lower_site) and lower_site in lower_sentence
    if title_hit or gram_hit:
        return hits + 3
    if site_hit:
        return hits + 1
    return hits


def _kb_score(lower_sentence: str, lower_title: str, title_grams: Set[str],
              terms: Sequence[str]) -> float:
    """句子与「知识库文档标题」的重合度（0~1）。阈值 0.62 才补引用。"""
    if len(lower_title) >= 3 and lower_title in lower_sentence:
        return 1.0
    if not title_grams:
        return 0.0
    sent_grams = set(ragtext_bigrams(lower_sentence))
    inter = sum(1 for g in title_grams if g in sent_grams)
    coverage = inter / len(title_grams)
    hits = sum(1 for t in terms if t in lower_sentence)
    return min(1.0, coverage * 0.6 + (0.3 if hits >= 2 else 0.1))


def _titles(pool: Dict, keys: Sequence[Any], numeric: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for k in keys:
        key = int(k) if numeric else str(k)
        out.append({"index" if numeric else "ref": key, "title": pool.get(key, "")})
    return out


def _int_of(v: Any) -> Optional[int]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _rec(kind: str, ref: str, title: str, sentence: str) -> Dict[str, Any]:
    return {"type": kind, "ref": ref, "title": title, "sentence": sentence[:120]}


def _round2(v: float) -> float:
    return round3(v * 100.0) / 100.0
