"""四阶段检索流水线（对应 Java ``RagService``）。

    查询改写 → 多路召回 → RRF 融合 → 精排 + MMR → 阈值与保底

两条路径
--------
* :meth:`search_by_index`（**主路径**）：语料已在索引里（写入时就进去了），
  检索只召回候选并回读原文做精排。
* :meth:`search_by_memory`（兜底）：索引还没建起来 / 被关掉 / 一条都没召回时，
  查库拼一份临时语料再走同一套流水线。它每次都要把全量文档与结构化切片重新捞一遍，
  所以不再是主路径。

权限两道闸
----------
召回时按「企业 + 密级 + 部门」过滤（在索引里做），回读原文后不再二次过滤——
索引里的 level/dept 是入库时按实体字段写死的，不随查询者变化。
"""

from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import regex as _regex

from ..config import get_settings
from ..db import tables as T
from ..db.engine import Database, get_db
from .corpus import T as CT
from .index_store import RagIndexStore
from .textnorm import round3
from .query_rewrite import variants as rewrite_variants
from .structured import StructuredCorpus
from .text import STOPWORDS, bigrams, effective_terms, jaccard, lower, occurrences, tokenize

log = logging.getLogger(__name__)

RRF_K = 60.0

#: 结构化来源类型：这些是"最硬的证据"，值得保底。
STRUCTURAL_TYPES: frozenset = frozenset(
    {CT.METRIC, CT.EVENT, CT.COMPLAINT, CT.AGGREGATE}
)

_SPLIT_PHRASE = _regex.compile(r"[^\p{Han}a-z0-9]+")


# ------------------------------------------------------------------
# 结果结构
# ------------------------------------------------------------------


@dataclass
class RagHit:
    """一条检索命中。

    ``score`` 是 query 内归一后的分（Top1 恒接近 100），``raw_score`` 是**真实余弦 ×100**——
    跨 query 可比的是后者：想给门禁设绝对阈值，只能用 ``raw_score``。
    """

    document_id: Optional[int] = None
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    rerank_score: Optional[float] = None
    recall_paths: List[str] = field(default_factory=list)
    source_ref: Optional[str] = None
    raw_score: Optional[float] = None

    def ref(self) -> Optional[str]:
        """引用标识：知识库文档用纯数字 ID，结构化切片用 ``metric:id`` 这类。"""
        if self.source_ref:
            return self.source_ref
        return None if self.document_id is None else str(self.document_id)


@dataclass
class RagResult:
    hits: List[RagHit] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)


# ------------------------------------------------------------------
# 权限闸门
# ------------------------------------------------------------------


class AclGuard:
    """文档级访问闸门（对应 Java ``DocumentAccessGuard`` 的用法子集）。

    Python 侧只用到三个能力：当前用户密级上限、可见部门集合、对文档列表做过滤。
    没有登录上下文时（批处理 / 评估脚本）三者都返回 ``None``，表示**不过滤**。
    """

    def __init__(
        self,
        level: Optional[int] = None,
        visible_depts: Optional[Set[int]] = None,
    ) -> None:
        self._level = level
        self._depts = visible_depts

    def current_level(self) -> Optional[int]:
        return self._level

    def current_visible_depts(self) -> Optional[Set[int]]:
        return self._depts

    def filter(self, docs: List[dict]) -> Tuple[List[dict], dict]:
        """按密级 + 部门过滤。返回 ``(保留的, 诊断)``。

        无条件写诊断：被拦 0 条也是重要信息，能证明这道闸真的跑了。
        """
        diag: Dict[str, object] = {"aclCorpusBefore": len(docs)}
        blocked_level = 0
        blocked_dept = 0
        kept: List[dict] = []
        for d in docs:
            lv = d.get("security_level")
            lv = 1 if lv is None else int(lv)
            if self._level is not None and lv > int(self._level):
                blocked_level += 1
                continue
            dept = d.get("dept_id")
            if self._depts is not None and dept is not None and int(dept) not in self._depts:
                blocked_dept += 1
                continue
            kept.append(d)
        diag["aclMode"] = "level+dept" if self._level is not None else "none"
        diag["aclBlocked"] = blocked_level + blocked_dept
        diag["aclBlockedByLevel"] = blocked_level
        diag["aclBlockedByDept"] = blocked_dept
        return kept, diag


# ------------------------------------------------------------------
# 服务
# ------------------------------------------------------------------


class RagService:
    """检索服务。构造参数都可注入，便于测试里换成内存索引。"""

    def __init__(
        self,
        index: Optional[RagIndexStore] = None,
        db: Optional[Database] = None,
        corpus: Optional[StructuredCorpus] = None,
        embedder=None,
        acl: Optional[AclGuard] = None,
        index_enabled: Optional[bool] = None,
    ) -> None:
        s = get_settings()
        r = s.rag
        self.index = index
        self.db = db or get_db()
        self.corpus = corpus or StructuredCorpus(db=self.db)
        self.embedder = embedder
        self.acl = acl or AclGuard()
        self.index_enabled = bool(r.index_enabled) if index_enabled is None else index_enabled

        self.vector_weight = float(r.vector_weight)
        self.query_rewrite_enabled = bool(r.query_rewrite)
        self.multi_recall_enabled = bool(r.multi_recall)
        self.rerank_enabled = bool(r.rerank)
        self.rerank_pool = int(r.rerank_pool)
        self.rerank_weight = float(r.rerank_weight)
        self.diversity = float(r.diversity)
        self.min_score = float(r.min_score)
        self.relative_floor = float(r.relative_floor)

        self.structured_enabled = bool(r.structured_corpus)
        self.structured_max = int(r.structured_max)
        self.structured_guarantee = int(r.structured_guarantee)
        self.sync_throttle_ms = int(r.sync_throttle_ms)

        self._last_sync_at: Dict[int, float] = {}
        self._sync_lock = threading.Lock()

    # -- 对外 API ----------------------------------------------------

    def search(self, company_id: Optional[int], query: Optional[str], top_k: int) -> List[RagHit]:
        return self.search_detailed(company_id, query, top_k).hits

    def search_detailed(self, company_id: Optional[int], query: Optional[str], top_k: int) -> RagResult:
        """索引优先；索引不可用或一条都没召回才退回实时语料路径。"""
        if self.index_enabled and self.index is not None and self.index.available():
            try:
                r = self.search_by_index(company_id, query, top_k)
                if r is not None and r.hits:
                    return r
            except Exception as e:  # noqa: BLE001
                log.warning("索引检索失败，本次回退实时语料检索: %s", e)
        return self.search_by_memory(company_id, query, top_k)

    def index_stats(self) -> Dict[str, object]:
        m: Dict[str, object] = dict(self.index.stats()) if self.index is not None else {}
        m["enabled"] = self.index_enabled
        return m

    # -- 主路径：索引检索 --------------------------------------------

    def search_by_index(self, company_id: Optional[int], query: Optional[str], top_k: int) -> RagResult:
        diag: Dict[str, object] = {"indexMode": "sqlite-fts5"}
        if company_id is None or query is None or not query.strip():
            diag["mode"] = "空企业或空查询"
            return RagResult([], diag)

        user_level = self.acl.current_level() if self.acl else None
        visible_depts = self.acl.current_visible_depts() if self.acl else None

        vs = (rewrite_variants(query)
              if (self.query_rewrite_enabled and self.multi_recall_enabled)
              else [query.strip()])
        diag["queryVariants"] = len(vs)

        q_emb = self._embed_query(query)
        recall_limit = max(self.rerank_pool, max(1, top_k) * 4)

        # ---- 多路召回 + RRF 融合 ----
        fused: Dict[str, float] = {}
        path_ids: List[Set[str]] = []
        path_names: List[str] = []
        main_terms: List[str] = []

        for i, v in enumerate(vs):
            terms = tokenize(v)
            if not terms:
                continue
            if i == 0:
                main_terms = terms
            cands = self.index.keyword(company_id, terms, recall_limit,
                                       user_level, visible_depts)
            if not cands:
                continue
            path_ids.append({c.id for c in cands if c.id})
            path_names.append("index:bm25" if i == 0 else f"index:bm25:rewrite{i}")
            _rrf(cands, fused, 1.0 if i == 0 else 0.65)

        if q_emb is not None:
            try:
                knn = self.index.vector(company_id, q_emb, recall_limit, user_level, visible_depts)
                if knn:
                    path_ids.append({c.id for c in knn if c.id})
                    path_names.append("index:knn")
                    _rrf(knn, fused, 0.5 + self.vector_weight)
            except Exception as e:  # noqa: BLE001
                log.warning("向量召回失败，退化为多路关键词: %s", e)

        if not fused:
            diag["mode"] = "索引无召回（语料可能尚未回填）"
            return RagResult([], diag)
        diag["paths"] = path_names

        max_fused = max(fused.values()) if fused else 0.0
        pool = [k for k, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]
        pool_size = max(self.rerank_pool, max(1, top_k) * 4)
        if len(pool) > pool_size:
            pool = pool[:pool_size]
        diag["corpus"] = len(fused)
        diag["candidates"] = len(pool)

        # ---- 回读原文，做精排 ----
        data = self.index.read_chunks(pool)
        ids: List[str] = []
        titles: List[str] = []
        bodies: List[str] = []
        raws: List[str] = []
        for cid in pool:
            d = data.get(cid)
            if d is None:
                continue
            ids.append(cid)
            t = lower(d.title or "")
            titles.append(t)
            bodies.append(t + "\n" + lower(d.text or ""))
            raws.append((d.title or "") + "\n" + (d.text or ""))
        if not ids:
            diag["mode"] = "索引回读为空"
            return RagResult([], diag)

        if not main_terms:
            main_terms = tokenize(query)
        idf = self._idf_of(main_terms, bodies)
        total_idf = sum(idf.values())
        phrases = self._phrases_of(query, main_terms)

        prior = [(fused.get(i, 0.0) / max_fused if max_fused > 0 else 0.0) for i in ids]
        vec_sim = [0.0] * len(ids)
        if q_emb is not None:
            stored = self.index.read_vectors(ids)
            for i, cid in enumerate(ids):
                v = stored.get(cid)
                vec_sim[i] = max(0.0, _cosine(v, q_emb)) if v is not None else 0.0
        max_vec = max(vec_sim) if vec_sim else 0.0

        precision = [
            self._precision_score(main_terms, phrases, idf, total_idf, bodies[i], titles[i],
                                  (vec_sim[i] / max_vec) if max_vec > 0 else 0.0, raws[i])
            for i in range(len(ids))
        ]
        relevance = [
            ((1 - self.rerank_weight) * prior[i] + self.rerank_weight * precision[i])
            if self.rerank_enabled else min(1.0, prior[i])
            for i in range(len(ids))
        ]

        selected = self._mmr(relevance, bodies, max(1, min(top_k, 10)) * 3)

        # 同一引用标识只保留分数最高的一条：
        # 一条投诉既有行级切片（c:847）又有类别聚合切片（agg:4:complaint:847），
        # 两者 ref 都是 complaint:847。模型看到两条同 ID、内容却不同的证据会无所适从。
        ordered: List[RagHit] = []
        seen_refs: Set[str] = set()
        for i in selected:
            d = data.get(ids[i])
            ref = _ref_of(ids[i], d.source_type if d else None)
            if ref is not None and ref in seen_refs:
                continue
            if ref is not None:
                seen_refs.add(ref)
            paths = [path_names[p] for p in range(len(path_ids)) if ids[i] in path_ids[p]]
            ordered.append(RagHit(
                document_id=_document_id_of(ids[i], d.source_type if d else None),
                title=(d.title or "") if d else "",
                snippet=self._snippet((d.text or "") if d else "",
                                      _best_term(bodies[i], titles[i], main_terms), main_terms),
                score=_round2(100 * max(0.0, min(1.0, relevance[i]))),
                rerank_score=_round2(100 * precision[i]),
                recall_paths=paths,
                source_ref=ref,
                raw_score=None if q_emb is None else _round2(100 * vec_sim[i]),
            ))

        top = ordered[0].score if ordered else 0.0
        floor = max(self.min_score, top * self.relative_floor)
        hits = [h for h in ordered if h.score >= floor]
        if not hits and ordered:
            hits.append(ordered[0])
        hits = hits[: max(1, min(top_k, 10))]

        # ---- 结构化证据保底 ----
        # 指标 / 事件 / 投诉 / 聚合切片文本短，精排的长度惩罚和相对阈值容易把它们整批砍掉。
        # 这里定向召回一次，只要沾一点边就补一条可核对的硬证据。
        if not any(h.source_ref for h in hits) and main_terms:
            c = self.index.keyword(company_id, main_terms, 5, user_level, visible_depts,
                                   STRUCTURAL_TYPES)
            if c:
                d = self.index.read_chunks([c[0].id]).get(c[0].id)
                if d is not None:
                    body = d.text or ""
                    slice_hit = RagHit(
                        document_id=None,
                        title=d.title or "",
                        snippet=self._snippet(body, _best_term(lower(body), "", main_terms), main_terms),
                        score=_round2(100 * max(0.0, min(1.0, top * 0.5 if top > 0 else 0.01))),
                        rerank_score=_round2(100 * (top * 0.5 if top > 0 else 0.01)),
                        recall_paths=["index:bm25:structural"],
                        source_ref=_ref_of(c[0].id, d.source_type),
                    )
                    cap = max(1, min(top_k, 10))
                    if len(hits) >= cap:
                        hits[cap - 1] = slice_hit
                    else:
                        hits.append(slice_hit)
                    diag["structuralGuaranteed"] = c[0].id

        diag["mode"] = "+".join(path_names) + ("+rerank+mmr" if self.rerank_enabled else "")
        diag["topScore"] = hits[0].score if hits else 0
        if hits and hits[0].raw_score is not None:
            diag["topRawScore"] = hits[0].raw_score
        diag["floor"] = _round2(floor)
        return RagResult(hits, diag)

    # -- 兜底路径：查库拼语料 ----------------------------------------

    def search_by_memory(self, company_id: Optional[int], query: Optional[str], top_k: int) -> RagResult:
        diag: Dict[str, object] = {}
        docs = self._list_knowledge(company_id)

        # 权限过滤必须发生在「召回之前」：被拦下的文档不能进任何一路召回，
        # 否则向量库的缓存里仍留着它的指纹，等于过滤形同虚设。
        before = len(docs)
        docs, acl_diag = self.acl.filter(docs) if self.acl else (docs, {})
        diag.update(acl_diag)
        diag.setdefault("aclCorpusBefore", before)

        if not docs or query is None or not query.strip():
            blocked_all = bool(acl_diag.get("aclBlocked")) and bool(query and query.strip())
            diag["mode"] = (
                f"知识库文档被文档级权限全部过滤（密级 {acl_diag.get('aclBlockedByLevel')}"
                f" / 部门 {acl_diag.get('aclBlockedByDept')}）"
                if blocked_all else "空语料或空查询"
            )
            return RagResult([], diag)

        # ---- 结构化切片并入语料 ----
        # 切片必须在权限过滤之后加入 —— 它们来自本企业自己的业务数据，本就在可见范围内。
        refs: List[Optional[str]] = [None] * len(docs)
        if self.structured_enabled and self.corpus is not None:
            try:
                chunks = self.corpus.chunks(company_id)
                if chunks:
                    take = min(len(chunks), max(0, self.structured_max))
                    docs = list(docs)
                    for i in range(take):
                        ch = chunks[i]
                        docs.append({"id": None, "company_id": company_id, "title": ch.title,
                                     "content": ch.text, "security_level": 1, "dept_id": None})
                        refs.append(ch.source_id)
                    diag["structuredChunks"] = take
            except Exception as e:  # noqa: BLE001
                log.warning("结构化切片并入语料失败（本次跳过）：%s", e)

        n = len(docs)
        titles = [lower(d.get("title") or "") for d in docs]
        bodies = [titles[i] + "\n" + lower(d.get("content") or "") for i, d in enumerate(docs)]
        index_texts = [(d.get("title") or "") + "\n" + (d.get("content") or "") for d in docs]

        # ============ 阶段 0：查询改写 ============
        vs = (rewrite_variants(query)
              if (self.query_rewrite_enabled and self.multi_recall_enabled) else [query.strip()])
        diag["queryVariants"] = len(vs)

        use_index = self.index_enabled and self.index is not None and self.index.available()
        chunk_ids: List[str] = []
        pos_by_id: Dict[str, int] = {}
        for i, d in enumerate(docs):
            key = _chunk_key(d, refs[i] if i < len(refs) else None, i)
            chunk_ids.append(key)
            pos_by_id[key] = i

        user_level = self.acl.current_level() if self.acl else None
        visible_depts = self.acl.current_visible_depts() if self.acl else None
        q_emb = self._embed_query(query)

        if use_index:
            try:
                # 节流：这条是兜底路径，一旦索引不可用就会被反复调用，
                # 而 sync 会触发全量向量化——每个查询都做一次，延迟直接跳到秒级。
                if self.sync_throttle_ms > 0 and not self._should_sync(company_id):
                    diag["indexSync"] = f"节流跳过（{self.sync_throttle_ms}ms 内已同步过）"
                else:
                    from .index_store import ChunkInput  # 局部导入避免顶层循环

                    inputs = [ChunkInput(chunk_ids[i], docs[i].get("title") or "",
                                         docs[i].get("content") or "",
                                         int(docs[i].get("security_level") or 1),
                                         docs[i].get("dept_id"), company_id, CT.KNOWLEDGE)
                              for i in range(n)]
                    embed_fn = (lambda texts: self.embedder.embed_cached(texts)) if self.embedder else None
                    vkey = None if (self.embedder is None or q_emb is None) else f"{self.embedder.mode}:{len(q_emb)}"
                    # ⚠️ 这里是 **upsert**，不是 sync。
                    # sync 是"这一整批就是全部"的对账语义，会把本次没提到的语料删掉。
                    # 但兜底路径的语料并不完整 —— 结构化切片被 structured_max 截断（默认 12 条），
                    # 于是一次"索引没召回到 → 兜底"就会把索引里其余几百条结构化切片删光，
                    # 之后的检索越搜越少（实测 217 → 22）。这种"越用越差"最难查：
                    # 没有任何报错，只是召回数悄悄掉下去。
                    ur = self.index.upsert(inputs, embed_fn, vkey)
                    diag["indexSync"] = _sync_info(ur)
                    if not ur.ok:
                        use_index = False
                        diag["indexFallback"] = ur.error or "未知原因"
            except Exception as e:  # noqa: BLE001
                use_index = False
                log.warning("向量索引同步失败，本次退化为内存检索: %s", e)
                diag["indexFallback"] = str(e)
        diag["indexMode"] = "sqlite-fts5" if use_index else "memory"
        if not self.index_enabled:
            diag["indexDisabled"] = True

        # ============ 阶段 1：多路召回 ============
        rankings: List[List[int]] = []
        weights: List[float] = []
        path_names: List[str] = []
        recall_limit = max(self.rerank_pool, max(1, top_k) * 4)

        variant_terms: Dict[str, List[str]] = {}
        for i, v in enumerate(vs):
            terms = effective_terms(tokenize(v), bodies)
            if not terms:
                continue
            variant_terms[v] = terms
            if use_index:
                ranking = _positions_of(
                    self.index.keyword(company_id, terms, recall_limit, user_level, visible_depts),
                    pos_by_id)
            else:
                ranking = self._order_desc(self._keyword_score(terms, bodies, titles))
            if not ranking:
                continue
            rankings.append(ranking)
            # 原文路径权重最高；改写/扩展路径作为补充，避免"改写坏了就全崩"
            weights.append(1.0 if i == 0 else 0.65)
            path_names.append(
                ("index:bm25" if i == 0 else f"index:bm25:rewrite{i}") if use_index
                else ("keyword" if i == 0 else f"keyword:rewrite{i}")
            )

        vec_sim: Optional[List[float]] = None
        if q_emb is not None:
            try:
                if use_index:
                    knn = self.index.vector(company_id, q_emb, recall_limit, user_level, visible_depts)
                    ranking = _positions_of(knn, pos_by_id)
                    if ranking:
                        rankings.append(ranking)
                        weights.append(0.5 + self.vector_weight)
                        path_names.append("index:knn")
                    # 精排要的是"真实余弦"，不是索引的距离分数：把入库时存的向量读回来算
                    vec_sim = [0.0] * n
                    stored = self.index.read_vectors(chunk_ids)
                    for cid, v in stored.items():
                        pos = pos_by_id.get(cid)
                        if pos is None:
                            continue
                        vec_sim[pos] = max(0.0, _cosine(v, q_emb))
                else:
                    d_emb = self.embedder.embed_cached(index_texts) if self.embedder else []
                    vec_sim = [0.0] * n
                    for i in range(n):
                        v = d_emb[i] if i < len(d_emb) else None
                        vec_sim[i] = max(0.0, _cosine(v, q_emb)) if v is not None else 0.0
                    ranking = self._order_desc(vec_sim)
                    if ranking:
                        rankings.append(ranking)
                        weights.append(0.5 + self.vector_weight)
                        path_names.append("vector")
            except Exception as e:  # noqa: BLE001
                log.warning("向量召回失败，退化为多路关键词: %s", e)

        if not rankings:
            diag["mode"] = "无可用召回信号"
            return RagResult([], diag)
        diag["paths"] = path_names

        # ============ 阶段 2：RRF 融合 ============
        fused_arr = [0.0] * n
        for p, ranking in enumerate(rankings):
            w = weights[p]
            for r, idx in enumerate(ranking):
                fused_arr[idx] += w / (RRF_K + r + 1)
        max_fused = max(fused_arr) if fused_arr else 0.0
        prior = [(v / max_fused if max_fused > 0 else 0.0) for v in fused_arr]

        pool_size = max(self.rerank_pool, max(1, top_k) * 4)
        pool = self._order_desc(prior)
        pool = [i for i in pool if fused_arr[i] > 0][:pool_size]
        diag["candidates"] = len(pool)
        diag["corpus"] = n

        # ============ 阶段 3：精排 ============
        main_terms = variant_terms.get(query.strip()) or effective_terms(tokenize(query), bodies)
        if not main_terms and variant_terms:
            main_terms = next(iter(variant_terms.values()))
        phrases = self._phrases_of(query, main_terms)

        max_vec_in_pool = 0.0
        if vec_sim is not None:
            for i in pool:
                max_vec_in_pool = max(max_vec_in_pool, vec_sim[i])

        idf = self._idf_of(main_terms, bodies)
        total_idf = sum(idf.values())
        precision = [0.0] * n
        for i in pool:
            precision[i] = self._precision_score(
                main_terms, phrases, idf, total_idf, bodies[i], titles[i],
                0.0 if vec_sim is None else (vec_sim[i] / max_vec_in_pool if max_vec_in_pool > 0 else 0.0),
                index_texts[i],
            )
        relevance = [0.0] * n
        for i in pool:
            relevance[i] = (
                (1 - self.rerank_weight) * prior[i] + self.rerank_weight * precision[i]
                if self.rerank_enabled else min(1.0, prior[i])
            )

        selected = self._mmr([relevance[i] for i in range(n)], bodies,
                             max(1, min(top_k, 10)) * 3, candidates=pool)

        ordered: List[RagHit] = []
        for i in selected:
            paths: List[str] = []
            for p in range(len(rankings)):
                if i in rankings[p] and path_names[p] not in paths:
                    paths.append(path_names[p])
            bt = _best_term(bodies[i], titles[i], main_terms)
            ordered.append(RagHit(
                document_id=docs[i].get("id"),
                title=docs[i].get("title") or "",
                snippet=self._snippet(docs[i].get("content") or "", bt, main_terms),
                score=_round2(100 * max(0.0, min(1.0, relevance[i]))),
                rerank_score=_round2(100 * precision[i]),
                recall_paths=paths,
                source_ref=refs[i] if i < len(refs) else None,
                raw_score=None if vec_sim is None else _round2(100 * vec_sim[i]),
            ))

        top = ordered[0].score if ordered else 0.0
        floor = max(self.min_score, top * self.relative_floor)
        hits = [h for h in ordered if h.score >= floor]
        if not hits and ordered:
            hits.append(ordered[0])
        hits = hits[: max(1, min(top_k, 10))]

        # ---------- 结构化切片保底 ----------
        # 指标 / 风险事件 / 投诉是最硬的证据，但它们文本短、不像"文档"，
        # 精排的长度惩罚与相对阈值会把它们整批砍掉。所以给它们留一个保底名额。
        # 在「全部语料」里挑分数最高的切片，而不是只看候选池：切片文本短、
        # 常常连候选池都进不去，只在池子里找的话保底永远不生效。
        best: Optional[int] = None
        best_score = -1.0
        for i in range(n):
            if i < len(refs) and refs[i] is not None:
                sc = max(relevance[i], prior[i])
                if sc > best_score:
                    best_score = sc
                    best = i
        # 无论保底是否生效都记下来 —— 看不到切片分数的话，
        # "切片进了语料却从不命中"根本无从排查
        if best is not None:
            diag["structuredBestScore"] = _round2(100 * best_score)
            diag["structuredBestRef"] = refs[best]
            diag["structuredBestTitle"] = docs[best].get("title")
        if self.structured_enabled and self.corpus is not None and self.structured_guarantee > 0:
            have = sum(1 for h in hits if h.source_ref is not None)
            if have < self.structured_guarantee and best is not None and best_score > 0.01:
                ref = refs[best]
                if not any(h.ref() == ref for h in hits):
                    bt = _best_term(bodies[best], titles[best], main_terms)
                    slice_hit = RagHit(
                        document_id=None,
                        title=docs[best].get("title") or "",
                        snippet=self._snippet(docs[best].get("content") or "", bt, main_terms),
                        score=_round2(100 * max(0.0, min(1.0, best_score))),
                        rerank_score=_round2(100 * best_score),
                        recall_paths=["structured"],
                        source_ref=ref,
                    )
                    cap = max(1, min(top_k, 10))
                    if len(hits) >= cap:
                        hits[cap - 1] = slice_hit
                    else:
                        hits.append(slice_hit)
                    diag["structuredGuaranteed"] = ref

        diag["mode"] = "+".join(path_names) + ("+rerank+mmr" if self.rerank_enabled else "")
        diag["rerankEnabled"] = self.rerank_enabled
        diag["diversity"] = self.diversity
        diag["topScore"] = hits[0].score if hits else 0
        if hits and hits[0].raw_score is not None:
            diag["topRawScore"] = hits[0].raw_score
        diag["floor"] = _round2(floor)
        return RagResult(hits, diag)

    # -- 内部：向量与语料 --------------------------------------------

    def _embed_query(self, query: str):
        if self.embedder is None:
            return None
        if not getattr(self.embedder, "is_enabled", lambda: True)():
            return None
        if self.vector_weight <= 0:
            return None
        try:
            return self.embedder.embed_one(query)
        except Exception as e:  # noqa: BLE001
            log.warning("查询向量化失败，向量通道本次停用: %s", e)
            return None

    def _list_knowledge(self, company_id: Optional[int]) -> List[dict]:
        """文档列表：兜底路径与旧实现一致，按企业取全量。"""
        from sqlalchemy import select as _select

        stmt = _select(T.knowledge_document)
        if company_id is None:
            stmt = stmt.where(T.knowledge_document.c.company_id.is_(None))
        else:
            stmt = stmt.where(T.knowledge_document.c.company_id == company_id)
        return self.db.fetch_all(stmt)

    def _should_sync(self, company_id: Optional[int]) -> bool:
        import time

        now = time.time() * 1000.0
        key = company_id if company_id is not None else -1
        with self._sync_lock:
            prev = self._last_sync_at.get(key)
            if prev is not None and now - prev < self.sync_throttle_ms:
                return False
            self._last_sync_at[key] = now
            return True

    # -- 内部：评分 --------------------------------------------------

    def _keyword_score(self, terms: List[str], bodies: List[str], titles: List[str]) -> List[float]:
        """阶段 1：IDF + 词频 + 标题加权 + 覆盖率。"""
        n = len(bodies)
        scores = [0.0] * n
        if not terms:
            return scores
        idf = self._idf_of(terms, bodies)
        total_idf = sum(idf.values())
        if total_idf <= 0:
            return scores
        raws = [0.0] * n
        covers = [0.0] * n
        max_raw = 0.0
        for i in range(n):
            body, title = bodies[i], titles[i]
            raw = 0.0
            cover = 0.0
            for t in terms:
                cnt = occurrences(body, t)
                if cnt == 0:
                    continue
                v = idf.get(t, 0.0)
                raw += v * (1.0 + math.log(cnt)) * (1.6 if t in title else 1.0)
                cover += v
            if raw <= 0:
                continue
            raws[i] = raw
            covers[i] = cover / total_idf
            max_raw = max(max_raw, raw)
        if max_raw <= 0:
            return scores
        for i in range(n):
            if raws[i] > 0:
                scores[i] = 100.0 * (raws[i] / max_raw) * covers[i]
        return scores

    def _idf_of(self, terms: List[str], bodies: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        n = len(bodies)
        for t in terms:
            df = sum(1 for b in bodies if t in b)
            if df == 0:
                continue
            out[t] = math.log(1.0 + n / (1.0 + df))
        return out

    def _order_desc(self, scores: Sequence[float]) -> List[int]:
        """按分数降序排列下标，丢弃 0 分项。"""
        idx = [i for i, v in enumerate(scores) if v > 0]
        idx.sort(key=lambda i: scores[i], reverse=True)
        return idx

    def _phrases_of(self, query: Optional[str], main_terms: List[str]) -> Set[str]:
        phrases: Set[str] = set(main_terms)
        for run in _SPLIT_PHRASE.split(lower(query)):
            if len(run) < 2:
                continue
            for i in range(0, len(run) - 1):
                phrases.add(run[i:i + 2])
        phrases -= set(STOPWORDS)
        return phrases

    def _precision_score(self, terms: List[str], phrases: Set[str], idf: Dict[str, float],
                         total_idf: float, body: str, title: str, vec_norm: float,
                         raw_text: str) -> float:
        """精排总分（0~1）：覆盖率 / 标题命中 / 邻接短语 / 窗口密度 / 语义余弦 / 长度惩罚。

        语义不可用时自动把权重让给词汇信号，保证分数不会偏低。
        """
        if not terms and not phrases:
            return 0.0

        cover = 0.0
        if total_idf > 0:
            hit = sum(v for k, v in idf.items() if k in body)
            cover = max(0.0, min(1.0, hit / total_idf))

        title_hit = 0.0
        if total_idf > 0:
            hit = sum(v for k, v in idf.items() if k in title)
            title_hit = max(0.0, min(1.0, hit / total_idf))

        phrase = 0.0
        if phrases:
            phrase = sum(1 for p in phrases if p in body) / len(phrases)

        window = self._best_window_coverage(body, terms, 200, 50)

        length = len(raw_text or "")
        len_factor = max(0.86, 1.0 - math.log(max(1.0, length / 1200.0) + 1.0) * 0.06)

        if vec_norm > 0:
            weighted = (0.34 * cover + 0.16 * title_hit + 0.14 * phrase
                        + 0.12 * window + 0.24 * vec_norm)
        else:
            # 语义缺失时重新归一化：0.34+0.16+0.14+0.12=0.76
            weighted = (0.34 * cover + 0.16 * title_hit + 0.14 * phrase + 0.12 * window) / 0.76
        return max(0.0, min(1.0, weighted * len_factor))

    @staticmethod
    def _best_window_coverage(body: Optional[str], terms: List[str], win_size: int, step: int) -> float:
        """最佳 200 字窗口内的覆盖情况，惩罚"关键词散落在超长文档里"。"""
        if not body or not terms:
            return 0.0
        best = 0.0
        n = len(body)
        for start in range(0, n, step):
            end = min(n, start + win_size)
            win = body[start:end]
            hit = sum(1 for t in terms if t in win)
            best = max(best, hit / len(terms))
            if end >= n:
                break
        return max(0.0, min(1.0, best))

    def _mmr(self, relevance: Sequence[float], bodies: List[str], want: int,
             candidates: Optional[List[int]] = None) -> List[int]:
        """MMR：保持相关性的同时抑制重复文档。"""
        rest = list(range(len(bodies))) if candidates is None else list(candidates)
        selected: List[int] = []
        cache: Dict[int, frozenset] = {}
        while rest and len(selected) < want:
            best = -1
            best_score = -1.0
            for cand in rest:
                redundancy = 0.0
                cb = cache.setdefault(cand, bigrams(bodies[cand]))
                for s in selected:
                    redundancy = max(redundancy, jaccard(cb, cache.setdefault(s, bigrams(bodies[s]))))
                score = self.diversity * relevance[cand] - (1 - self.diversity) * redundancy
                if score > best_score:
                    best_score = score
                    best = cand
            if best < 0:
                break
            selected.append(best)
            rest.remove(best)
        return selected

    def _snippet(self, content: Optional[str], best_term: Optional[str], terms: List[str]) -> str:
        if not content or not content.strip():
            return ""
        lc = lower(content)
        index = lc.find(best_term) if best_term else -1
        if index < 0:
            for t in terms:
                i = lc.find(t)
                if i >= 0 and (index < 0 or i < index):
                    index = i
        if index < 0:
            index = 0
        start = max(0, index - 80)
        end = min(len(content), start + 400)
        text = re.sub(r"\s+", " ", content[start:end]).strip()
        return ("…" if start > 0 else "") + text + ("…" if end < len(content) else "")


# ------------------------------------------------------------------
# 模块级小工具
# ------------------------------------------------------------------


def _rrf(cands, fused: Dict[str, float], weight: float) -> None:
    for r, c in enumerate(cands):
        if c is None or not c.id:
            continue
        fused[c.id] = fused.get(c.id, 0.0) + weight / (RRF_K + r + 1)


def _positions_of(cands, pos_by_id: Dict[str, int]) -> List[int]:
    """索引返回的 id 序列 → 语料下标序列（过滤掉索引里残留、但本轮语料没有的脏数据）。"""
    out: List[int] = []
    if not cands:
        return out
    for c in cands:
        p = pos_by_id.get(c.id) if c is not None else None
        if p is not None:
            out.append(p)
    return out


def _best_term(body: str, title: str, terms: List[str]) -> Optional[str]:
    best = None
    best_w = -1.0
    for t in terms:
        if t not in body:
            continue
        w = math.log(1.0 + len(t)) * (1.6 if (title and t in title) else 1.0)
        if w > best_w:
            best_w = w
            best = t
    return best


def _ref_of(chunk_id: Optional[str], source_type: Optional[str]) -> Optional[str]:
    """引用标识：知识库文档沿用纯数字，其余写成 ``metric:12 / event:3``。"""
    if chunk_id is None:
        return None
    if source_type == CT.KNOWLEDGE:
        v = _after(chunk_id, ":")
        h = v.find("#")
        return v[:h] if h > 0 else v
    if source_type == CT.AGGREGATE:
        # agg:{企业}:{原始ref}
        second = chunk_id.find(":", 4)
        return chunk_id[second + 1:] if second > 0 else _after(chunk_id, ":")
    if source_type is None or source_type == "-":
        return chunk_id
    return f"{source_type}:{_after(chunk_id, ':')}"


def _document_id_of(chunk_id: Optional[str], source_type: Optional[str]) -> Optional[int]:
    if source_type == CT.KNOWLEDGE and chunk_id:
        v = _after(chunk_id, ":")
        h = v.find("#")
        try:
            return int(v[:h] if h > 0 else v)
        except ValueError:
            return None
    return None


def _after(s: str, c: str) -> str:
    i = s.find(c)
    return s if i < 0 else s[i + 1:]


def _chunk_key(d: dict, ref: Optional[str], pos: int) -> str:
    """语料块在索引里的主键：保证同一份语料每次算出来的主键稳定，
    增量同步才不会把同一篇反复当成新文档重写一遍。"""
    if d is not None and d.get("id") is not None:
        return f"d:{d['id']}"
    if ref:
        return f"s:{ref}"
    return f"p:{pos}"


def _sync_info(sr) -> Dict[str, object]:
    """兼容 ``SyncResult`` 与 ``UpsertResult``（后者没有 removed/unchanged/rebuilt）。"""
    return {
        "docs": sr.total,
        "added": sr.added,
        "updated": sr.updated,
        "removed": getattr(sr, "removed", 0),
        "unchanged": getattr(sr, "unchanged", getattr(sr, "skipped", 0)),
        "rebuilt": getattr(sr, "rebuilt", False),
    }


def _cosine(a, b) -> float:
    """真实余弦（与 Java 版逐项一致：长度不等或零向量给 0）。"""
    if a is None or b is None:
        return 0.0
    la, lb = len(a), len(b)
    if la != lb or la == 0:
        return 0.0
    dot = na = nb = 0.0
    for i in range(la):
        dot += float(a[i]) * float(b[i])
        na += float(a[i]) * float(a[i])
        nb += float(b[i]) * float(b[i])
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _round2(v: float) -> float:
    """``Math.round(v * 100.0) / 100.0`` —— 注意是取整（HALF_UP），不是格式化。"""
    return round3(v * 100.0) / 100.0
