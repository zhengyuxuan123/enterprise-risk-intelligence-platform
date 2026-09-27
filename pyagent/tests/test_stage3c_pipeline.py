"""阶段 3c 回归：结构化语料 / 语料源 / 写入即索引 / 四阶段检索流水线。

全部**不连数据库**（除一个显式跳过的集成用例），语料源用固定 dict 渲染，
索引落在临时目录。这样这套测试能在没有 MySQL 的机器上跑。
"""

from __future__ import annotations

import math

import pytest

from app.rag.corpus import CorpusChunk, T as CT, date, num, nz, trim, blank, yes
from app.rag.index_store import ChunkInput, RagIndexStore
from app.rag.embedding import LocalEmbedding
from app.rag.rag_service import (
    RagHit,
    RagService,
    _after,
    _best_term,
    _document_id_of,
    _positions_of,
    _ref_of,
)
from app.rag.sources import (
    ComplaintSource,
    CompanySource,
    CompetitorSource,
    EventSource,
    KnowledgeSource,
    MetricSource,
    RiskRuleSource,
    _op,
)


class _FakeDb:
    """不连库的替身：按预设数据应答，并记录被打过的查询。"""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.calls = 0

    def fetch_all(self, stmt, params=None):
        self.calls += 1
        return list(self.rows)

    def fetch_one(self, stmt, params=None):
        self.calls += 1
        return self.rows[0] if self.rows else None

    def count(self, table, where=None):
        return len(self.rows)

    def execute(self, stmt, params=None):
        return 1


# ------------------------------------------------------------------
# Render / CorpusChunk
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, want",
    [
        (None, "—"),
        ("0.00", "0"),
        ("1.50", "1.5"),
        ("100", "100"),
        ("-12.300", "-12.3"),
        ("0.0001", "0.0001"),
    ],
)
def test_num_matches_java_strip_trailing_zeros(raw, want):
    """``BigDecimal.stripTrailingZeros().toPlainString()`` 的等价物。

    Python 的 ``Decimal.normalize()`` 会把 100 变成 ``1E+2``，
    不还原的话切片里就会出现科学计数法。
    """
    assert num(raw) == want


def test_num_of_decimal_and_bad_input():
    from decimal import Decimal

    assert num(Decimal("2.500")) == "2.5"
    assert num("abc") == "—"


def test_nz_and_blank_distinction():
    assert nz(None) == "未填写"
    assert nz("   ") == "未填写"
    assert nz(" 收入 ") == "收入"
    assert blank(None) == ""
    assert blank(" x ") == "x"


def test_date_placeholder():
    import datetime

    assert date(None) == "未知日期"
    assert date(datetime.date(2026, 9, 20)) == "2026-09-20"


def test_trim_collapses_unicode_whitespace():
    """空白集是 **Unicode**（Python 原生），U+00A0 / U+3000 **也会**被折叠。

    旧实现为了对齐 Java 只认 ASCII 那 6 个空白，于是中文语料里很常见的全角空格
    被原样留在切片里 —— 分片起点因此比"看起来"靠后一个字符，不报错，只是悄悄偏。
    """
    assert trim("a　　b", 10) == "a b"      # U+3000 被折叠成一个空格
    assert trim("a \n\t b", 10) == "a b"   # ASCII 空白同样折叠
    assert trim("x" * 20, 5) == "xxxxx…"
    assert trim(None, 5) == ""


def test_yes_only_matches_one():
    assert yes(1) is True
    assert yes(0) is False
    assert yes(None) is False
    assert yes("1") is True


def test_corpus_chunk_hash_is_first_8_bytes_sha256():
    import hashlib

    c = CorpusChunk("k:1#0", CT.KNOWLEDGE, "1", 1, "标题", "正文", 2, None)
    want = hashlib.sha256("标题\x01正文".encode("utf-8")).hexdigest()[:16]
    assert c.hash() == want
    # 换正文必须换 hash —— 这是"内容没变就不重写"的唯一依据
    assert CorpusChunk("k:1#0", CT.KNOWLEDGE, "1", 1, "标题", "正文2", 2, None).hash() != want


# ------------------------------------------------------------------
# 语料源渲染（不查库）
# ------------------------------------------------------------------


def test_metric_source_render():
    out = MetricSource(db=_FakeDb()).render(
        {"id": 12, "company_id": 3, "metric_date": None, "metric_code": "GROSS_MARGIN",
         "metric_name": "毛利率", "metric_value": "18.5000", "unit": "%", "source_type": "ERP"}
    )
    assert len(out) == 1
    c = out[0]
    assert c.chunk_id == "m:12"
    assert c.source_type == CT.METRIC
    assert c.company_id == 3
    assert c.text == "经营指标【毛利率】未知日期 为 18.5%。指标代码 GROSS_MARGIN。数据来源 ERP。"


def test_metric_source_falls_back_to_code_when_name_missing():
    out = MetricSource(db=_FakeDb()).render(
        {"id": 1, "company_id": 3, "metric_code": "CHURN", "metric_name": "  ",
         "metric_value": "1", "unit": None, "source_type": None}
    )
    assert "经营指标【CHURN】" in out[0].text


def test_event_source_render_includes_handle_and_review():
    out = EventSource(db=_FakeDb()).render(
        {"id": 5, "company_id": 1, "risk_title": "客户流失率超过20%", "risk_type": "客户流失风险",
         "risk_level": "HIGH", "status": "OPEN", "trigger_value": "25", "threshold_value": "20",
         "metric_date": None, "handle_result": "已 联系 客户", "review_comment": None}
    )
    t = out[0].text
    assert out[0].chunk_id == "e:5"
    assert "触发值 25（阈值 20）" in t
    assert "处置情况：已 联系 客户。" in t
    assert "复核意见" not in t


def test_complaint_source_omits_customer_name():
    """刻意不写客户姓名：切片会进 prompt，没必要让真名出现在每段上下文里。"""
    out = ComplaintSource(db=_FakeDb()).render(
        {"id": 847, "company_id": 4, "complaint_date": None, "customer_name": "张三",
         "product_name": "CRM", "category": "交付延迟", "severity": "HIGH",
         "repeat_flag": 1, "sla_exceeded": 1, "solve_hours": "72.00", "description": "拖 了 很久",
         "root_cause": "产能", "churn_risk": "HIGH", "status": "OPEN"}
    )
    t = out[0].text
    assert "张三" not in t
    assert "该投诉为重复投诉。" in t
    assert "处理已超出 SLA。" in t
    assert "解决耗时 72 小时。" in t


def test_competitor_source_render():
    out = CompetitorSource(db=_FakeDb()).render(
        {"id": 2, "company_id": 1, "competitor_name": "竞品A", "product_name": "X1",
         "price": "9800.00", "price_unit": "元/年", "risk_level": "HIGH",
         "selling_point": "便宜", "weakness": None, "promotion": None,
         "delivery_cycle": None, "service_commitment": None, "target_customer": None}
    )
    assert out[0].chunk_id == "p:2"
    assert "标准价 9800元/年" in out[0].text
    assert "其卖点：便宜。" in out[0].text
    assert "短板" not in out[0].text


def test_company_source_render():
    out = CompanySource(db=_FakeDb()).render(
        {"id": 1, "company_code": "C001", "company_name": "云帆科技", "industry": "SaaS",
         "region": "华东", "customer_level": "A", "company_scale": "中型", "dept_id": 7}
    )
    assert out[0].chunk_id == "co:1"
    assert out[0].dept_id == 7
    assert "企业档案【云帆科技】：编码 C001" in out[0].text


def test_risk_rule_source_skips_disabled_and_translates_operator():
    db = _FakeDb()
    assert RiskRuleSource(db=db).render({"id": 1, "enabled": 0, "rule_name": "x"}) == []
    out = RiskRuleSource(db=db).render(
        {"id": 3, "rule_name": "毛利率红线", "metric_code": "GROSS_MARGIN",
         "operator_code": "lt", "threshold_value": "20", "risk_level": "HIGH",
         "risk_type": "经营风险", "enabled": 1, "description": None}
    )
    assert out[0].company_id is None  # 全局语料
    assert "当指标 GROSS_MARGIN 低于 20 时" in out[0].text
    assert out[0].chunk_id == "r:3"


@pytest.mark.parametrize(
    "code, want",
    [("GT", "高于"), ("<", "低于"), ("ge", "不低于"), ("LE", "不高于"),
     ("=", "等于"), ("!=", "不等于"), ("??", "??"), (None, "满足")],
)
def test_operator_translation(code, want):
    assert _op(code) == want


def test_knowledge_source_splits_and_prefixes_doc_type():
    db = _FakeDb()
    body = "甲" * 2000
    out = KnowledgeSource(db=db).render(
        {"id": 9, "company_id": 1, "title": "制度", "doc_type": "管理办法",
         "content": body, "security_level": 3, "dept_id": 5}
    )
    assert len(out) > 1  # 700 字一片，2000 字必然多片
    assert out[0].chunk_id == "k:9#0"
    assert out[0].text.startswith("【管理办法】")
    assert all(c.level == 3 for c in out)
    assert all(c.dept_id == 5 for c in out)
    assert "第 1/" in out[0].title


def test_knowledge_source_empty_body_yields_nothing():
    db = _FakeDb()
    assert KnowledgeSource(db=db).render({"id": 1, "content": "   ", "title": "t"}) == []
    assert KnowledgeSource(db=db).render({"id": 1, "content": None, "title": "t"}) == []


def test_row_source_of_all_of_ids_of_use_fake_db():
    db = _FakeDb([{"id": 12, "company_id": 3, "metric_code": "A", "metric_name": "甲",
                   "metric_value": "1", "unit": None, "source_type": None,
                   "metric_date": None}])
    s = MetricSource(db=db)
    assert [c.chunk_id for c in s.of(12)] == ["m:12"]
    assert [c.chunk_id for c in s.all_of(3)] == ["m:12"]
    assert s.ids_of(3) == [12]
    assert db.calls == 3


def test_row_source_of_returns_empty_for_missing_row():
    assert MetricSource(db=_FakeDb([])).of(999) == []


# ------------------------------------------------------------------
# RagService 的纯函数
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_id, source_type, want",
    [
        ("k:12#0", CT.KNOWLEDGE, "12"),       # 文档沿用纯数字
        ("k:12", CT.KNOWLEDGE, "12"),
        ("m:57", CT.METRIC, "metric:57"),
        ("e:3", CT.EVENT, "event:3"),
        ("agg:4:complaint:847", CT.AGGREGATE, "complaint:847"),  # 聚合切片剥掉企业前缀
        ("x:1", None, "x:1"),
        (None, CT.METRIC, None),
    ],
)
def test_ref_of(chunk_id, source_type, want):
    assert _ref_of(chunk_id, source_type) == want


def test_document_id_only_for_knowledge():
    assert _document_id_of("k:12#3", CT.KNOWLEDGE) == 12
    assert _document_id_of("m:12", CT.METRIC) is None
    assert _document_id_of("k:abc", CT.KNOWLEDGE) is None


def test_after_helper():
    assert _after("m:57", ":") == "57"
    assert _after("noSep", ":") == "noSep"


def test_best_term_prefers_title_and_longer():
    terms = ["毛利", "毛利率"]
    assert _best_term("毛利率下降", "毛利率 报表", terms) == "毛利率"
    assert _best_term("无命中", "", terms) is None


def test_positions_of_filters_unknown_ids():
    class C:
        def __init__(self, i):
            self.id = i

    assert _positions_of([C("a"), C("zzz"), C("b")], {"a": 0, "b": 1}) == [0, 1]
    assert _positions_of(None, {"a": 0}) == []


def test_hit_ref_falls_back_to_document_id():
    assert RagHit(document_id=7).ref() == "7"
    assert RagHit(source_ref="metric:9").ref() == "metric:9"
    assert RagHit().ref() is None  # 两者都没有 → 不可用引用


class _Idx:
    """索引替身：直接给出候选与回读内容。"""

    def __init__(self, data):
        self.data = data
        self.available = lambda: True
        self.stats = lambda: {"docs": len(data)}

    def keyword(self, company_id, terms, limit, max_level=None, visible_depts=None, source_types=None):
        hits = [(k, v) for k, v in self.data.items() if any(t in (v[0] + v[1]) for t in terms)]
        if source_types is not None:
            hits = [(k, v) for k, v in hits if v[2] in source_types]
        return [type("Cand", (), {"id": k, "score": 1.0})() for k, v in hits][:limit]

    def vector(self, *a, **kw):
        return []

    def read_chunks(self, ids):
        out = {}
        for i in ids:
            if i in self.data:
                t, x, st = self.data[i]
                out[i] = type("ChunkData", (), {
                    "title": t, "text": x, "source_type": st, "company_id": 1,
                    "level": 1, "dept_id": None})()
        return out

    def read_vectors(self, ids):
        return {}


def test_search_by_index_dedups_by_ref():
    """同一 ref 只保留一条：行级切片与聚合切片会撞车，
    模型看到两条同 ID、内容却不同的证据会无所适从。"""
    idx = _Idx({
        "c:847": ("客户投诉 · 交付延迟", "客户投诉 类别 交付延迟 的内容", CT.COMPLAINT),
        "agg:4:complaint:847": ("客户投诉 · 交付延迟", "聚合视角的投诉分布", CT.AGGREGATE),
    })
    svc = RagService(index=idx, embedder=None)
    r = svc.search_by_index(4, "交付延迟", 5)
    refs = [h.ref() for h in r.hits]
    assert refs.count("complaint:847") == 1
    assert len(r.hits) == 1


def test_search_by_index_rejects_empty_input():
    svc = RagService(index=_Idx({}), embedder=None)
    assert svc.search_by_index(None, "x", 5).hits == []
    assert svc.search_by_index(1, "  ", 5).diagnostics["mode"] == "空企业或空查询"


def test_precision_score_semantic_weight_redistribution():
    """语义不可用时把权重让给词汇信号（除以 0.76 而不是 1.0），否则分数会系统性偏低。

    注意：语义**满分**时两种算法结果相同（都是 1.0），只有语义打不满时才看得出差别——
    所以用 0.5 而不是 1.0 来对照。
    """
    svc = RagService(index=None, embedder=None)
    terms = ["毛利", "下滑"]
    body = "毛利率 下滑 严重"
    idf = {"毛利": 1.0, "下滑": 1.0}
    half = svc._precision_score(terms, set(terms), idf, 2.0, body, "毛利", 0.5, body)
    none_ = svc._precision_score(terms, set(terms), idf, 2.0, body, "毛利", 0.0, body)
    assert none_ > half          # 词汇全命中时，摊回权重后更高
    assert none_ <= 1.0
    # 逐项核对：cover=1 / titleHit=0.5 / phrase=1 / window=1
    lf = 1.0 - math.log(max(1.0, len(body) / 1200.0) + 1.0) * 0.06
    assert half == pytest.approx((0.34 + 0.16 * 0.5 + 0.14 + 0.12 + 0.24 * 0.5) * lf, rel=1e-9)
    assert none_ == pytest.approx((0.34 + 0.16 * 0.5 + 0.14 + 0.12) / 0.76 * lf, rel=1e-9)


def test_precision_score_length_penalty_has_floor():
    """长度惩罚有下限 0.86：超长文档会被降权，但不会被一棒子打死。

    对照组刻意让 ``body`` / ``title`` 完全相同，只有 ``raw_text`` 的长度不同，
    这样差值就精确等于惩罚系数本身。
    """
    svc = RagService(index=None, embedder=None)
    terms = ["毛利"]
    args = (terms, set(terms), {"毛利": 1.0}, 1.0, "毛利", "毛利", 0.0)
    short = svc._precision_score(*args, raw_text="毛利")
    long_ = svc._precision_score(*args, raw_text="毛利" + "填" * 200000)
    assert long_ < short
    # 再长也不会更低：两个都撞到 0.86 这个下限，说明惩罚是"有底"的
    longer = svc._precision_score(*args, raw_text="毛利" + "填" * 2000000)
    assert long_ == pytest.approx(longer, rel=1e-12)
    assert long_ == pytest.approx(short * 0.86 / (1.0 - math.log(2.0) * 0.06), rel=1e-9)


def test_keyword_score_title_boost():
    svc = RagService(index=None, embedder=None)
    bodies = ["毛利 下滑", "毛利 下滑"]
    s = svc._keyword_score(["毛利", "下滑"], bodies, ["毛利", ""])
    assert s[0] > s[1]  # 标题命中那条更高（×1.6）


def test_idf_skips_absent_terms():
    svc = RagService(index=None, embedder=None)
    assert svc._idf_of(["不存在"], ["abc"]) == {}
    assert set(svc._idf_of(["毛利"], ["毛利", "abc"])) == {"毛利"}


def test_order_desc_drops_zero():
    svc = RagService(index=None, embedder=None)
    assert svc._order_desc([0.0, 3.0, 1.0, 0.0]) == [1, 2]


def test_phrases_of_removes_stopwords_and_short_runs():
    svc = RagService(index=None, embedder=None)
    p = svc._phrases_of("毛利率 为什么 下滑", ["毛利", "下滑"])
    assert "毛利" in p and "下滑" in p
    assert all(len(x) >= 2 for x in p)


def test_snippet_centers_on_best_term():
    svc = RagService(index=None, embedder=None)
    body = "前" * 300 + "关键内容" + "后" * 500
    s = svc._snippet(body, "关键内容", ["关键内容"])
    assert "关键内容" in s
    assert s.startswith("…") and s.endswith("…")
    # 短文本不截断 → 两端都不加省略号
    assert svc._snippet("就是这句话", "这句", ["这句"]) == "就是这句话"
    assert svc._snippet(None, None, []) == ""
    assert svc._snippet("   ", None, []) == ""


# ------------------------------------------------------------------
# 索引 + 检索 端到端（临时目录，不连库）
# ------------------------------------------------------------------


@pytest.fixture()
def idx(tmp_path):
    s = RagIndexStore(index_dir=str(tmp_path / "ix"))
    s.init()
    yield s
    s.close()


def test_index_roundtrip_then_search(idx):
    le = LocalEmbedding(64)
    idx.upsert(
        [
            ChunkInput("m:16", "经营指标 · 客户流失率", "经营指标【客户流失率】最新一期 25%", 1, None, 1, CT.METRIC),
            ChunkInput("k:7#0", "历史经营分析报告", "续费率下降与客户成功体系优化", 1, None, 1, CT.KNOWLEDGE),
        ],
        lambda texts: [le.embed(t) for t in texts],
        "local:64",
    )
    svc = RagService(index=idx, embedder=None)
    r = svc.search_by_index(1, "客户流失率", 5)
    assert r.hits
    assert r.hits[0].ref() == "metric:16"
    assert r.diagnostics["corpus"] >= 1


def test_indexing_service_upsert_and_remove_by_source(tmp_path, idx):
    from app.rag.indexing import CorpusChangeEvent, RagIndexingService

    class _Src:
        def __init__(self):
            self.alive = {"m:1": CorpusChunk("m:1", CT.METRIC, "1", 1, "甲", "内容甲", 1, None)}

        def type(self):
            return CT.METRIC

        def of(self, sid):
            return [self.alive["m:1"]] if (sid == 1 and "m:1" in self.alive) else []

        def all_of(self, cid):
            return list(self.alive.values())

        def ids_of(self, cid):
            return [1]

        def per_company(self):
            return True

    src = _Src()
    fake = _FakeDb([])
    svc = RagIndexingService(index=idx, db=fake, sources=[src], local=LocalEmbedding(64))
    svc.on_corpus_change(CorpusChangeEvent.upsert(CT.METRIC, 1, 1))
    assert svc.flush() == 1
    assert idx.stats()["docs"] == 1

    # 第二次 flush 同内容 → hash 未变，不重写（saveState 不会被再次调用）
    svc.on_corpus_change(CorpusChangeEvent.upsert(CT.METRIC, 1, 1))
    assert svc.flush() == 1
    assert idx.stats()["docs"] == 1

    # 删除 → 记录已不存在，走 removeBySource 分支。
    # 物理删除后业务表已查不到，索引只能靠**记账表**反查切片 —— 所以这里必须让记账表有记录。
    fake.rows = [{"id": 1, "chunk_id": "m:1", "source_type": CT.METRIC,
                  "source_id": "1", "company_id": 1}]
    src.alive.clear()
    svc.on_corpus_change(CorpusChangeEvent.upsert(CT.METRIC, 1, 1))
    svc.flush()
    assert idx.stats()["docs"] == 0

    # 记账表里也没有 → 什么都不该发生（不能因为查不到就误删别的语料）
    idx.upsert([ChunkInput("m:2", "乙", "内容乙", 1, None, 1, CT.METRIC)], None, None)
    before = idx.stats()["docs"]
    fake.rows = []
    svc.on_corpus_change(CorpusChangeEvent.delete(CT.METRIC, 1, 1))
    svc.flush()
    assert idx.stats()["docs"] == before


def test_indexing_merge_window_keeps_last_event_only():
    from app.rag.indexing import CorpusChangeEvent, RagIndexingService

    svc = RagIndexingService(index=None, db=_FakeDb([]), sources=[], local=None)
    svc.on_corpus_change(CorpusChangeEvent.upsert(CT.METRIC, 1, 1))
    svc.on_corpus_change(CorpusChangeEvent.delete(CT.METRIC, 1, 1))
    # 同一 key 只保留最后一次：后发的 DELETE 覆盖先发的 UPSERT
    assert svc.pending == 1


def test_indexing_disabled_drops_events():
    from app.rag.indexing import CorpusChangeEvent, RagIndexingService

    svc = RagIndexingService(index=None, db=_FakeDb([]), sources=[], local=None)
    svc.enabled = False
    svc.on_corpus_change(CorpusChangeEvent.upsert(CT.METRIC, 1, 1))
    assert svc.pending == 0


# ---------------------------------------------------------------- 向量开关闸门


class _FakeEmbedder:
    """按 ``enabled`` 决定是否产向量，行为与 ``EmbeddingClient`` 一致。

    真实实现里 ``embed_cached`` 在 ``enabled=False`` 时**返回空列表**而不是抛错，
    这正是当初写入路径被它骗过去的原因。
    """

    def __init__(self, enabled: bool, mode: str = "local") -> None:
        self.enabled = enabled
        self.mode = mode
        self.calls = 0

    def is_enabled(self) -> bool:
        return self.enabled

    def embed_cached(self, texts):
        self.calls += 1
        if not self.enabled:
            return []
        le = LocalEmbedding(64)
        return [le.embed(t) for t in texts]


class _MetricSource:
    """只提供一条指标语料的假语料源。"""

    def __init__(self) -> None:
        self.alive = {"m:1": CorpusChunk("m:1", CT.METRIC, "1", 1, "甲", "内容甲", 1, None)}

    def type(self):
        return CT.METRIC

    def of(self, sid):
        return [self.alive["m:1"]] if (sid == 1 and "m:1" in self.alive) else []

    def all_of(self, cid):
        return list(self.alive.values())

    def ids_of(self, cid):
        return [1]

    def per_company(self):
        return True


def _indexing_with_embedder(em, idx):
    from app.rag.indexing import RagIndexingService

    return RagIndexingService(index=idx, db=_FakeDb([]), sources=[_MetricSource()],
                              local=LocalEmbedding(64), embedder=em)


def test_disabled_embedding_must_not_claim_a_vector_space(idx):
    """总开关关着时，写入路径**不能**宣称自己做了向量化。

    真实事故：``_embedder_for_write()`` 只看 ``mode == 'local'``（不问开关），
    于是它返回了可调用的 ``embed_cached``，后者因开关关着直接返回空列表，
    索引把 ``vectorKey`` 记成 ``local:512`` 并回报成功。
    最终 1070 条切片一条向量都没有 —— 而"缺向量"这件事在索引元数据上完全看不出来。
    """
    from app.rag.indexing import CorpusChangeEvent

    em = _FakeEmbedder(enabled=False)
    svc = _indexing_with_embedder(em, idx)
    svc.on_corpus_change(CorpusChangeEvent.upsert(CT.METRIC, 1, 1))
    assert svc.flush() == 1

    assert em.calls == 0, "开关关着就不该去调向量化通道"
    s = idx.stats()
    assert s["docs"] == 1
    assert s["vectorKey"] is None, "没算向量就不能记 vectorKey"
    assert s["vectorsAbsent"] is False, "从没宣称过向量空间，不算异常状态"


def test_enabled_embedding_records_vectors_and_space(idx):
    """开关开着：真算、真写、真记账。"""
    from app.rag.indexing import CorpusChangeEvent

    em = _FakeEmbedder(enabled=True)
    svc = _indexing_with_embedder(em, idx)
    svc.on_corpus_change(CorpusChangeEvent.upsert(CT.METRIC, 1, 1))
    assert svc.flush() == 1

    s = idx.stats()
    assert em.calls == 1
    assert s["vectorKey"] == "local:64"
    assert s["vectorsAbsent"] is False
    assert s["vectorRows"] == 1 and s["vectorCoveragePct"] == 100.0
