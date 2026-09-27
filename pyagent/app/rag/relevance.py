"""网页来源的相关性回检，对应 Java ``SourceRelevanceFilter``。

**为什么必须回检**：搜索引擎给了 10 条，不代表这 10 条都配进"参考来源"。
直连检索（Bing/360）只会按它自己的排序取前 N 条返回，
**不会替你判断和本次问题有没有关系**。原实现把这 N 条原封不动递给模型、
又原封不动列进「参考来源」，于是出现了「问的是客户流失，参考来源列的是
2024 年日历、汉语词典、某师范大学招生网」。

**怎么打分（零 token，不许调模型）**：两个确定性信号加权 —— 词面覆盖率（标题权重更高）
+ 本地向量余弦。任一信号都不灵时才容易被拦，因此额外设一道"硬性下限"：
词面完全没命中且余弦低于 ``min_cosine`` 的直接丢弃（这正是词典/日历/学校主页的特征）。

**「问题」与「检索词」必须分开赋权（2026-09-22）**：覆盖率是个**比例**，
把问题整句和检索词混成一个词集，分母就会随问题长度膨胀——问题写得越啰嗦，
同一条相关网页的分数越低。实测同一批明显相关的来源：
长问题+检索词 → 最高 0.15（5 条全被丢弃，联网等于白搜）；
只用检索词 → 0.40（正确保留 3 条、丢弃 2 条）。
因此：**检索词是判据主体，问题只是低权背景**（``CONTEXT_WEIGHT``），
没有检索词时问题才顶上当主体。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .embedding import LocalEmbedding, cosine, tokenize
from .textnorm import fmt2, lower, round3, trim

STOP = frozenset(
    {
        "de", "的", "了", "是", "在", "和", "与", "对", "有", "为", "把", "被", "给", "从", "向",
        "怎么", "如何", "什么", "哪些", "是否", "以及", "这个", "那个", "我们", "他们", "可以", "需要",
    }
)


@dataclass
class WebSource:
    """检索来源（与 Java ``WebSearchService.WebSource`` 字段顺序一致）。"""

    index: int = 0
    title: str = ""
    url: str = ""
    site_name: str = ""
    snippet: str = ""


@dataclass
class Scored:
    """一条来源的打分明细。"""

    source: WebSource
    score: float
    coverage: float
    cosine: float
    keep: bool
    why: str


@dataclass
class FilterResult:
    """过滤结果。

    .. IMPORTANT::
       ``kept`` 与 ``strict_kept`` 不是一回事，用错会让整道防线失效：

       * ``strict_kept`` —— **真正通过阈值**的来源（``Scored.keep is True``）。
         凡是需要回答"到底有没有相关结果"的地方（是否联网成功、是否丢弃来源、
         dropped 计数）都必须用它。
       * ``kept`` —— 用于**来源区展示**的集合：全部不相关时会补一条最高分的，
         免得界面上一片空白。它里面可能含一条弱相关来源。

       2026-09-22 之前不存在 ``strict_kept``，而 ``kept`` 在全部不相关时也会被塞
       一条，于是 ``if not kept: 全部不相关，已丢弃`` 这条分支**永远不成立**——
       从 Bing 抓回来的无关网页（日历网 / 随机门户）就这样被当成来源递给了模型，
       还被标上了 ``[1]`` 编号。
    """

    kept: list[WebSource] = field(default_factory=list)
    strict_kept: list[WebSource] = field(default_factory=list)
    detail: list[Scored] = field(default_factory=list)
    threshold: float = 0.16
    total: int = 0

    def dropped_count(self) -> int:
        """按严格口径算丢弃数：被展示兜底留下来的那条也算"没通过"。"""
        return self.total - len(self.strict_kept)

    def has_relevant(self) -> bool:
        """这批来源里到底有没有一条是与问题相关的。"""
        return bool(self.strict_kept)


class SourceRelevanceFilter:
    def __init__(
        self,
        local: LocalEmbedding,
        min_relevance: float = 0.16,
        min_cosine: float = 0.18,
        max_keep: int = 5,
    ) -> None:
        self.local = local
        self.min_relevance = min_relevance
        self.min_cosine = min_cosine
        self.max_keep = max_keep

    def filter(  # noqa: A003 - 与 Java 同名，保持对账时 grep 一致
        self, question: str | None, query: str | None, sources: list[WebSource] | None
    ) -> FilterResult:
        detail: list[Scored] = []
        if not sources:
            return FilterResult([], [], detail, self.min_relevance, 0)

        primary = self._terms(query)
        context = self._terms(question)
        if not primary:
            # 没有检索词时才让问题顶上当主体（直连通道与质量门就是只传 query/只传 None）
            primary, context = context, {}
        # 向量余弦用「问题+检索词」整段：语义相似度不会被长文本稀释（实测长问题下余弦反而不低），
        # 会被稀释的只有覆盖率那种**比例**指标，所以只在那里分开赋权。
        q_vec = self.local.embed(_nz_trim(question) + " " + _nz_trim(query))

        for s in sources:
            if s is None or not s.url or s.url.strip() == "":
                continue
            title = _nz(s.title)
            text = title + " " + _nz(s.snippet) + " " + _nz(s.site_name)

            coverage = _coverage(primary, text, context)
            cos = max(0.0, cosine(q_vec, self.local.embed(text))) if q_vec is not None else 0.0
            # 标题命中占比单独算：标题命中往往意味着整篇就是讲这个的
            title_cov = _coverage(primary, title, context)
            score = 0.45 * coverage + 0.35 * title_cov + 0.20 * cos

            if not primary and not context:
                keep = True
                why = "无可用比对词，放行"  # 判据本身没东西可比，不做无依据的拦截
            elif coverage <= 0 and cos < self.min_cosine:
                keep = False
                why = "词面零命中且余弦 " + fmt2(cos) + " < " + fmt2(self.min_cosine)
            elif score < self.min_relevance:
                keep = False
                why = (
                    "相关度 " + fmt2(score) + " < " + fmt2(self.min_relevance)
                    + "（覆盖 " + fmt2(coverage) + " / 余弦 " + fmt2(cos) + "）"
                )
            else:
                keep = True
                why = "相关度 " + fmt2(score)
            detail.append(
                Scored(s, round3(score), round3(coverage), round3(cos), keep, why)
            )

        # 稳定排序：同分时保持原来的先后顺序（Java List.sort 与 Python sorted 都是稳定的）
        detail.sort(key=lambda x: x.score, reverse=True)
        limit = max(1, self.max_keep)
        #! 严格集合必须先算出来：它是"有没有相关结果"的唯一真实依据。
        #! 先做 UI 兜底再让 strict_kept 跟着 kept 走，就等于把兜底那条也判成相关了。
        strict_kept = [sc.source for sc in detail if sc.keep][:limit]

        kept: list[WebSource] = list(strict_kept)
        if not kept and detail:
            # 全部被判无关：展示层保留分最高的 1 条，但**不改写 keep 标记**
            # （strict_kept 仍为空），这样"界面不留白"与"判定不失真"可以同时成立。
            best = detail[0]
            kept.append(best.source)
        return FilterResult(kept, strict_kept, detail, self.min_relevance, len(detail))

    # ---------------------------------------------------------------

    def _terms(self, text: str) -> dict[str, None]:
        """问题切词：中文 bigram、英文按词；过短与停用词剔除。"""
        out: dict[str, None] = {}
        #! 局部变量**不能**叫 ``lower``：那会把导入的 ``lower`` 变成局部变量，
        #! 同一行里右边那个 ``lower(text)`` 就成了未绑定引用（``UnboundLocalError``）。
        low = lower(text)
        for t in tokenize(low):
            if len(t) < 2 or t in STOP:
                continue
            out[t] = None
        return out


#: 「问题」相对「检索词」的权重。取 0.3 而非 1.0：问题的长度由用户怎么问决定，
#: 不该影响"这条网页相不相关"的判定。降到 0 又会丢掉"问题里那个检索词没表达出来的
#: 限定条件"，所以保留一个低权补充。
CONTEXT_WEIGHT = 0.3


def _coverage(q_terms, text: str | None, extra=None) -> float:
    """检索词命中占比；``extra``（问题背景词）按 :data:`CONTEXT_WEIGHT` 折算进分子分母。

    分母里带上 ``extra`` 而不是直接丢掉，是为了让"检索词没覆盖、但问题里明确提到的限定"
    仍能加分；权重放在 0.3 是让它加分而不喧宾夺主。
    """
    if not q_terms:
        return 0.0
    low = lower(text)  # 同上：不能叫 ``lower``，否则右边那个引用变成未绑定
    hit = float(sum(1 for t in q_terms if t in low))
    denom = float(len(q_terms))
    if extra:
        hit += CONTEXT_WEIGHT * sum(1 for t in extra if t in low)
        denom += CONTEXT_WEIGHT * len(extra)
    return hit / denom if denom > 0 else 0.0


def _nz(s: str | None) -> str:
    return "" if s is None else s


def _nz_trim(s: str | None) -> str:
    return "" if s is None else trim(s)
