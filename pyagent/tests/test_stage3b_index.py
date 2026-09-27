"""阶段 3b：索引层的离线回归 + 与 Lucene 真身的对账。

基准文件 ``tests/data/java_ground_index.txt`` 由 ``qa/_java_ground_index.jsh`` 生成：
jshell 加载 **fat jar 里抽出的 lucene 9.11 jar**，按 ``RagIndexStore`` 同样的建文档与
查询构造真跑出来的。它不是手抄的期望值。

已知并接受的差异（**不要偷偷改成"通过"**）
---------------------------------------
``Q2 客户集中度风险``：Lucene 给 k:1 / k:5 的分数是 0.693125 / 0.687222，
**相差 0.9%**，属于并列区；FTS5 的 bm25 与 Lucene 的 idf 公式不同
（``log(x)`` vs ``log(1+x)``），并列区内会换位。
上层 RRF 只吃排名（rank3=1/64、rank4=1/65），且最终顺序由精排决定，
影响可以忽略。测试里对这条**只断言集合相等**，其余 26 条断言顺序完全相同。
"""

from __future__ import annotations

import math
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.embedding import LocalEmbedding  # noqa: E402
from app.rag.index_store import (  # noqa: E402
    GLOBAL_COMPANY,
    SCHEMA,
    ChunkInput,
    RagIndexStore,
    _hash,
    _index_tokens,
)
from app.rag.text import tokenize as ragtext_tokenize  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_GROUND = Path(__file__).resolve().parent / "data" / "java_ground_index.txt"
_CORPUS = _ROOT / "qa" / "_rag_corpus.txt"
_QUERIES = _ROOT / "qa" / "_rag_queries.txt"

#: 只比对集合、不比对顺序的查询（并列区换位，见模块头说明）
_RANK_ORDER_EXEMPT = {2}


# ---------------------------------------------------------------- 基准解析


def _load_corpus():
    rows = []
    for line in _CORPUS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cid, company, ty, level, dept, title, text = line.split("|")
        rows.append(
            ChunkInput(
                id=cid,
                title=title,
                text=text,
                level=int(level),
                dept_id=None if dept == "-" else int(dept),
                company_id=None if company == "*" else int(company),
                source_type=ty,
            )
        )
    return rows


def _load_queries():
    out = []
    for line in _QUERIES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        company, level, depts, types, query = line.split("|")
        out.append(
            (
                int(company),
                None if level == "-" else int(level),
                None if depts == "-" else {int(x) for x in depts.split(",") if x.strip()},
                None if types == "-" else {x.strip() for x in types.split(",") if x.strip()},
                query,
            )
        )
    return out


def _parse_ground():
    tokens, hashes, queries = {}, {}, []
    cur = None
    for line in _GROUND.read_text(encoding="utf-8").splitlines():
        if line.startswith("T "):
            _, cid, tok = line.split(" ", 2)
            tokens[cid] = tok
        elif line.startswith("H "):
            _, cid, h = line.split(" ", 2)
            hashes[cid] = h
        elif line.startswith("Q "):
            _, qi, terms = line.split(" ", 2)
            cur = (int(qi), terms.split(",") if terms else [], [])
            queries.append(cur)
        elif line.startswith("  ") and cur is not None:
            cur[2].append(line.strip().split(" ")[0])
    return tokens, hashes, queries


@pytest.fixture(scope="module")
def ground():
    if not _GROUND.exists():
        pytest.skip("缺少 Java 基准，先跑 qa/_java_ground_index.jsh")
    return _parse_ground()


@pytest.fixture()
def store():
    s = RagIndexStore(index_dir=tempfile.mkdtemp(prefix="rag-test-"))
    s.init()
    assert s.available(), s.stats().get("error")
    yield s
    s.close()


# ---------------------------------------------------------------- 对账


def test_token_string_matches_java(ground):
    g_tokens, _, _ = ground
    chunks = {c.id: c for c in _load_corpus()}
    for cid, expect in g_tokens.items():
        assert _index_tokens(chunks[cid].title, chunks[cid].text) == expect, cid


def test_hash_matches_java(ground):
    _, g_hashes, _ = ground
    chunks = {c.id: c for c in _load_corpus()}
    for cid, expect in g_hashes.items():
        assert len(expect) == 16, f"hash 宽度变了：{expect}"
        assert _hash(chunks[cid].title, chunks[cid].text) == expect, cid


def test_keyword_ranking_matches_lucene(ground, store):
    _, _, g_queries = ground
    store.upsert(_load_corpus(), None, None)
    queries = _load_queries()
    assert len(queries) == len(g_queries)

    for (qi, java_terms, java_ids), (company, level, depts, types, query) in zip(g_queries, queries):
        terms = ragtext_tokenize(query)
        assert terms == java_terms, f"Q{qi} 分词与 Java 不一致"
        got = [c.id for c in store.keyword(company, terms, 50, level, depts, types)]
        if qi in _RANK_ORDER_EXEMPT:
            assert set(got) == set(java_ids), f"Q{qi} 候选集合都不一致"
        else:
            assert got == java_ids, f"Q{qi} 排名与 Lucene 不一致"


def test_empty_result_when_no_term_matches(store):
    """查不到的词必须给空列表，不能退化为"返回全部"。"""
    store.upsert(_load_corpus(), None, None)
    assert store.keyword(3, ragtext_tokenize("网络安全等级保护测评"), 50) == []
    assert store.keyword(3, [], 50) == []
    assert store.keyword(None, ragtext_tokenize("风险"), 50) == []


# ---------------------------------------------------------------- 增量与同步


def _emb(dim: int = 64):
    le = LocalEmbedding(dim)
    return lambda texts: [le.embed(t) for t in texts]


def test_upsert_is_incremental_by_content_hash(store):
    c = ChunkInput("a", "标题", "正文内容", 1, None, 3, "knowledge")
    r1 = store.upsert([c], _emb(), "local:64")
    assert (r1.total, r1.added, r1.updated, r1.skipped, r1.ok) == (1, 1, 0, 0, True)

    # 内容没变 → 连向量都不重算
    r2 = store.upsert([c], _emb(), "local:64")
    assert (r2.added, r2.updated, r2.skipped) == (0, 0, 1)

    # 内容变了 → 更新
    c2 = ChunkInput("a", "标题", "正文内容（改）", 1, None, 3, "knowledge")
    r3 = store.upsert([c2], _emb(), "local:64")
    assert (r3.added, r3.updated, r3.skipped) == (0, 1, 0)

    # 换向量空间 → 全量重建，原来那条算新增
    r4 = store.upsert([c2], _emb(128), "local:128")
    assert (r4.added, r4.updated) == (1, 0)
    assert store.stats()["vectorKey"] == "local:128"


def test_upsert_does_not_delete_untouched_chunks(store):
    """写入即索引的语义：一批事件只代表"这几条变了"，不能顺手判别人已删除。"""
    store.upsert([ChunkInput("x", "t1", "c1", 1, None, 3, "metric"),
                  ChunkInput("y", "t2", "c2", 1, None, 3, "metric")], None, None)
    store.upsert([ChunkInput("x", "t1", "c1-改", 1, None, 3, "metric")], None, None)
    assert set(store.read_chunks(["x", "y"])) == {"x", "y"}
    assert store.read_chunks(["x"])["x"].text == "c1-改"


def test_sync_deletes_stale_chunks_of_same_company(store):
    chunks = [ChunkInput(f"s:{i}", f"t{i}", f"正文{i}", 1, None, 3, "metric") for i in range(4)]
    r1 = store.sync(3, chunks, None, None)
    assert (r1.total, r1.added, r1.removed) == (4, 4, 0)

    r2 = store.sync(3, chunks[:2], None, None)
    assert (r2.total, r2.added, r2.unchanged, r2.removed) == (2, 0, 2, 2)
    assert set(store.read_chunks([c.id for c in chunks])) == {"s:0", "s:1"}


def test_sync_does_not_touch_other_companies(store):
    store.upsert([ChunkInput("a", "t", "c", 1, None, 3, "metric"),
                  ChunkInput("b", "t", "c", 1, None, 4, "metric")], None, None)
    store.sync(3, [], None, None)
    got = store.read_chunks(["a", "b"])
    assert "a" not in got and "b" in got


def test_remove(store):
    store.upsert([ChunkInput("a", "t", "c", 1, None, 3, "metric")], None, None)
    assert store.remove(["a", "", None]) == 1
    assert store.read_chunks(["a"]) == {}
    assert store.remove([]) == 0


def test_stats(store):
    store.upsert([ChunkInput("a", "t", "c", 1, None, 3, "metric")], _emb(), "local:64")
    s = store.stats()
    assert s["available"] is True and s["docs"] == 1 and s["vectorKey"] == "local:64"
    assert s["engine"] == "sqlite-fts5+numpy"


# ---------------------------------------------------------------- 权限


def _seed(store):
    store.upsert(
        [
            ChunkInput("c3", "本企业公开", "现金流紧张", 1, None, 3, "knowledge"),
            ChunkInput("c3hi", "本企业机密", "现金流紧张", 3, None, 3, "knowledge"),
            ChunkInput("c3d7", "本企业七部", "现金流紧张", 1, 7, 3, "knowledge"),
            ChunkInput("c4", "别家企业", "现金流紧张", 1, None, 4, "knowledge"),
            ChunkInput("g", "全局规则", "现金流紧张", 1, None, None, "knowledge"),
        ],
        None,
        None,
    )


def test_acl_company_scope(store):
    _seed(store)
    terms = ragtext_tokenize("现金流")
    ids = {c.id for c in store.keyword(3, terms, 50)}
    assert ids == {"c3", "c3hi", "c3d7", "g"}, ids


def test_acl_level_and_dept_and_type(store):
    _seed(store)
    terms = ragtext_tokenize("现金流")
    assert {c.id for c in store.keyword(3, terms, 50, 1)} == {"c3", "c3d7", "g"}
    # dept="-"（不属于任何部门）对所有可见部门集合都放行，这是 Java 的行为
    assert {c.id for c in store.keyword(3, terms, 50, None, {7})} == {"c3", "c3d7", "c3hi", "g"}
    assert {c.id for c in store.keyword(3, terms, 50, None, {9})} == {"c3", "c3hi", "g"}
    assert {c.id for c in store.keyword(3, terms, 50, None, None, {"metric"})} == set()


def test_global_chunk_is_visible_to_every_company(store):
    _seed(store)
    terms = ragtext_tokenize("现金流")
    assert [c.id for c in store.keyword(99, terms, 50)] == ["g"]


# ---------------------------------------------------------------- 向量通道


def test_vector_channel_ranks_by_cosine(store):
    le = LocalEmbedding(64)
    q = le.embed("资金链断裂")
    store.upsert(
        [
            ChunkInput("v1", "资金链", "资金链断裂风险处置", 1, None, 3, "knowledge"),
            ChunkInput("v2", "安全", "安全生产管理制度", 1, None, 3, "knowledge"),
            ChunkInput("v3", "他企", "资金链断裂风险处置", 1, None, 4, "knowledge"),
        ],
        lambda texts: [le.embed(t) for t in texts],
        "local:64",
    )
    cands = store.vector(3, q, 10)
    # 与 Java 的 vector() 一致：**不设相关度下限**，返回的就是 top-k（分数低也照返）
    assert cands[0].id == "v1"
    assert [c.id for c in cands] == sorted([c.id for c in cands], key=lambda i: 0)
    assert "v3" not in {c.id for c in cands}, "跨企业语料必须被 ACL 挡掉"
    assert [c.score for c in cands] == sorted((c.score for c in cands), reverse=True)
    assert 0 < cands[0].score <= 1.0


def test_vector_skips_other_dimension(store):
    """维度不同的行直接跳过 —— 等价于 Java 里"维度冲突 → 重建后只剩一种维度"。

    .. NOTE::
       ``LocalEmbedding`` 会把维度夹到 ``[64, 4096]``，所以这里必须用 64 / 128，
       写 32 / 16 会被夹成同一个维度，测试就变成假通过了。
    """
    lo, hi = LocalEmbedding(64), LocalEmbedding(128)
    assert (lo.dim, hi.dim) == (64, 128)
    store.upsert([ChunkInput("d64", "t", "内容", 1, None, 3, "knowledge")],
                 lambda t: [lo.embed(x) for x in t], "local:64")
    store.upsert([ChunkInput("d128", "t", "内容", 1, None, 3, "knowledge")],
                 lambda t: [hi.embed(x) for x in t], "local:128")
    # 第二次 upsert 触发了向量空间变化 → 全量重建，只剩下 d128
    assert store.stats()["docs"] == 1
    assert [c.id for c in store.vector(3, lo.embed("内容"), 10)] == []
    assert [c.id for c in store.vector(3, hi.embed("内容"), 10)] == ["d128"]


def test_read_vectors_roundtrip_is_bit_exact(store):
    le = LocalEmbedding(64)
    v = le.embed("资金链断裂")
    store.upsert([ChunkInput("a", "t", "c", 1, None, 3, "knowledge")],
                 lambda t: [v], "local:64")
    got = store.read_vectors(["a"])["a"]
    assert got.dtype == np.float32
    assert got.tobytes() == np.asarray(v, dtype="<f4").tobytes()


def test_read_chunks_returns_original_text(store):
    store.upsert([ChunkInput("k:7", "标题", "正文", 2, 7, 3, "knowledge")], None, None)
    d = store.read_chunks(["k:7"])["k:7"]
    assert (d.title, d.text, d.level, d.dept, d.company, d.source_type) == (
        "标题", "正文", 2, "7", "3", "knowledge")


def test_level_is_floored_at_one(store):
    store.upsert([ChunkInput("a", "t", "c", 0, None, 3, "knowledge")], None, None)
    assert store.read_chunks(["a"])["a"].level == 1


def test_null_company_becomes_global(store):
    store.upsert([ChunkInput("a", "t", "c", 1, None, None, None)], None, None)
    assert store.read_chunks(["a"])["a"].company == GLOBAL_COMPANY
    assert store.read_chunks(["a"])["a"].source_type == "-"


# ---------------------------------------------------------------- 结构版本


def test_schema_change_triggers_rebuild():
    d = tempfile.mkdtemp(prefix="rag-schema-")
    s = RagIndexStore(index_dir=d)
    s.init()
    s.upsert([ChunkInput("a", "t", "c", 1, None, 3, "knowledge")], None, None)
    assert s.stats()["docs"] == 1
    s.close()

    # 模拟"旧版本索引"：把 schema 改回 v1
    con = sqlite3.connect(Path(d) / "rag-index.sqlite3")
    con.execute("UPDATE rag_chunk SET schema='v1'")
    con.commit()
    con.close()

    s2 = RagIndexStore(index_dir=d)
    s2.init()
    assert s2.stats()["docs"] == 0, "结构版本不一致必须清空重建"
    assert SCHEMA == "v2"
    s2.close()


def test_rebuild_on_boot():
    d = tempfile.mkdtemp(prefix="rag-boot-")
    s = RagIndexStore(index_dir=d)
    s.init()
    s.upsert([ChunkInput("a", "t", "c", 1, None, 3, "knowledge")], None, None)
    s.close()
    s2 = RagIndexStore(index_dir=d, rebuild_on_boot=True)
    s2.init()
    assert s2.stats()["docs"] == 0
    s2.close()


def test_failure_degrades_instead_of_raising():
    """目录不可写时只降级，不抛出 —— 检索会退回内存路径。"""
    s = RagIndexStore(index_dir="Z:/no-such-drive/rag-index")
    s.init()
    assert s.available() is False
    assert s.keyword(3, ["风险"], 10) == []
    assert s.upsert([ChunkInput("a", "t", "c", 1)], None, None).ok is False
    assert s.sync(3, [], None, None).error == "索引不可用"
    assert "error" in s.stats()


# ---------------------------------------------------------------- 向量覆盖自愈


class TestVectorCoverageSelfHeal:
    """「记录着向量空间、实际一条向量都没有」的状态必须能自愈。

    真实事故：embedding 总开关关着，写入路径却照样把 ``vectorKey`` 记成
    ``local:512`` 并回报成功。1070 条切片的 ``vbin`` 全是 NULL，
    检索只剩 BM25，``ragTopRawScore`` 恒 0，评估里的负样本绝对阈值判据
    全部作废 —— 而所有接口都显示"一切正常"。
    更糟的是它**不会自己好**：增量判定看内容 hash，内容没变就永远跳过。
    """

    def test_vectors_absent_is_detectable(self, store):
        """只写倒排不写向量时，必须能识别出"缺向量"，而不是只看 vectorKey。"""
        store.upsert([ChunkInput("a", "标题", "正文", 1, None, 3, "knowledge")], None, None)
        s = store.stats()
        assert s["docs"] == 1
        assert s["vectorRows"] == 0
        assert s["vectorCoveragePct"] == 0.0
        # vectorKey 为 None（这次压根没打算算向量）→ 不算异常状态
        assert s["vectorsAbsent"] is False

    def test_declared_vector_space_without_vectors_is_flagged(self, store):
        """曾经的状态：vectorKey 记着 local:64，向量却一条都没有。"""
        store.upsert([ChunkInput("a", "标题", "正文", 1, None, 3, "knowledge")], None, None)
        store._index_vector_key = "local:64"   # 模拟旧索引留下来的元数据
        store._refresh_vector_rows()
        s = store.stats()
        assert s["vectorRows"] == 0
        assert s["vectorsAbsent"] is True
        assert s["vectorMissing"] is True
        assert "vectorWarning" in s
        # 关键：这一行必须被判为"不能跳过"，否则永远补不回向量
        assert store.has_vector("a") is False
        assert store._can_skip("a", _hash("标题", "正文"), "local:64") is False

    def test_partial_vector_coverage_is_also_flagged(self, store):
        """缺口只要存在就要报，不必等到"一条都没有" —— 实测的损坏正是只差一点点。"""
        store.upsert(
            [
                ChunkInput("a", "标题A", "正文A", 1, None, 3, "knowledge"),
                ChunkInput("b", "标题B", "正文B", 1, None, 3, "knowledge"),
            ],
            _emb(), "local:64",
        )
        store._ids_with_vec.discard("b")   # 模拟其中一条的向量丢了
        s = store.stats()
        assert (s["vectorRows"], s["vectorCoveragePct"]) == (1, 50.0)
        assert s["vectorMissing"] is True and s["vectorsAbsent"] is False
        # 有向量的那条继续走增量，缺的那条被补写 —— 不做无谓的全量重算
        assert store._can_skip("a", _hash("标题A", "正文A"), "local:64") is True
        assert store._can_skip("b", _hash("标题B", "正文B"), "local:64") is False

    def test_write_heals_missing_vectors_without_deleting(self, store):
        """补算靠"重写"而不是"先删库"：倒排与内容都在，重算即可。"""
        c = ChunkInput("a", "标题", "正文内容", 1, None, 3, "knowledge")
        store.upsert([c], None, None)
        store._index_vector_key = "local:64"
        store._refresh_vector_rows()
        assert store.vectors_absent() is True

        r = store.upsert([c], _emb(), "local:64")
        assert r.ok is True
        assert r.updated == 1 and r.added == 0, "内容没变也要重写，因为向量丢了"
        s = store.stats()
        assert s["vectorRows"] == 1 and s["vectorCoveragePct"] == 100.0
        assert s["vectorsAbsent"] is False
        assert store.keyword(3, ["正文"], 10), "倒排没有被破坏"

    def test_once_healed_it_goes_back_to_incremental(self, store):
        """补完之后必须回到增量：不能每次写入都全量重算。"""
        c = ChunkInput("a", "标题", "正文内容", 1, None, 3, "knowledge")
        store.upsert([c], _emb(), "local:64")
        r = store.upsert([c], _emb(), "local:64")
        assert (r.added, r.updated, r.skipped) == (0, 0, 1)

    def test_declared_vector_space_with_zero_vectors_reports_failure(self, store):
        """说了要算向量却一条没算出来 → 必须如实报失败，而不是安静地成功。"""
        empty_embedder = lambda texts: []          # noqa: E731 - 模拟"开关关着"
        r = store.upsert([ChunkInput("a", "t", "c", 1, None, 3, "knowledge")],
                         empty_embedder, "local:64")
        assert r.ok is False
        assert "向量化未产出任何向量" in (r.error or "")
        assert "关键词通道" in (r.error or "")

    def test_rows_without_vectors_do_not_claim_one(self, store):
        """逐行如实记账：没算出向量的那一行不写 vectorKey。"""
        store.upsert([ChunkInput("a", "t", "c", 1, None, 3, "knowledge")], None, None)
        row = store._conn.execute("SELECT vkey, dim FROM rag_chunk WHERE id='a'").fetchone()
        assert row[0] is None and row[1] == 0

    def test_stats_key_names_stay_java_compatible(self, store):
        """vectorKey / docs / engine 是与 Java 对账的既有键，新增项只能追加。"""
        store.upsert([ChunkInput("a", "t", "c", 1, None, 3, "knowledge")], _emb(), "local:64")
        s = store.stats()
        for k in ("available", "dir", "docs", "vectorKey", "engine"):
            assert k in s, f"既有键丢了：{k}"
        for k in ("vectorRows", "vectorCoveragePct", "vectorsAbsent"):
            assert k in s, f"新增的向量覆盖项缺失：{k}"
