"""写入即索引的服务端（对应 Java ``RagIndexingService``）。

业务数据写完 → 发一条 :class:`CorpusChangeEvent` → 这里把切片增量写进索引。

合并窗口
--------
导入是逐行调 create 的，一千行就是一千个事件。这里按 ``类型:主键`` 去重，
每隔 ``merge-window-ms`` 统一 flush 一次——写放大从 O(行数) 降到 O(1)。
代价是新数据最多延迟一个窗口才可检索（默认 3 秒），这是刻意选择的最终一致。

关于 Java 的 ``@TransactionalEventListener(fallbackExecution = true)``
-------------------------------------------------------------------
Java 侧业务 Service 全部没有 ``@Transactional``，而该注解在没有事务时**默认不触发**，
必须靠 ``fallbackExecution=true`` 兜住。Python 侧没有这套机制，
事件直接投递进 ``pending``——行为上等价于「始终 fallback」，不会出现"事件丢了但不报错"。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

from sqlalchemy import delete, func, insert, select, update

from ..config import get_settings
from ..db import tables as T
from ..db.engine import Database, get_db
from .corpus import CorpusChunk, T as CT
from .embedding import LocalEmbedding
from .index_store import ChunkInput, RagIndexStore
from .sources import CorpusSource, build_sources

log = logging.getLogger(__name__)


class Kind(str, Enum):
    UPSERT = "UPSERT"
    DELETE = "DELETE"


@dataclass(frozen=True)
class CorpusChangeEvent:
    """语料变更事件。业务侧必须**在写库之后**发布。"""

    source_type: str
    source_id: Optional[int]
    company_id: Optional[int]
    kind: Kind = Kind.UPSERT

    @staticmethod
    def upsert(source_type: str, source_id: Optional[int], company_id: Optional[int]) -> "CorpusChangeEvent":
        return CorpusChangeEvent(source_type, source_id, company_id, Kind.UPSERT)

    @staticmethod
    def delete(source_type: str, source_id: Optional[int], company_id: Optional[int]) -> "CorpusChangeEvent":
        return CorpusChangeEvent(source_type, source_id, company_id, Kind.DELETE)

    def key(self) -> str:
        """合并窗口用的去重键：同一条记录的连续多次变更只保留最后一次处理。"""
        return f"{self.source_type}:{self.source_id}"


class RagIndexingService:
    """接住变更事件，把切片增量写进本地索引。"""

    def __init__(
        self,
        index: Optional[RagIndexStore] = None,
        db: Optional[Database] = None,
        sources: Optional[List[CorpusSource]] = None,
        local: Optional[LocalEmbedding] = None,
        embedder=None,
        embedder_factory: Optional[Callable[[], Optional[object]]] = None,
    ) -> None:
        s = get_settings()
        self.index = index
        self.db = db or get_db()
        self.sources = sources if sources is not None else build_sources(self.db)
        self.local = local
        self._embedder = embedder
        self._embedder_factory = embedder_factory
        self._embedder_probe: Optional[bool] = None

        r = s.rag
        self.enabled = bool(r.write_through_enabled)
        self.merge_window_ms = max(200, int(r.merge_window_ms))
        self.embed_on_write_raw = (r.embed_on_write or "auto").strip()
        self.reconcile_enabled = bool(r.reconcile_enabled)
        self.reconcile_interval_min = max(5, int(r.reconcile_interval_minutes))

        self._pending: Dict[str, CorpusChangeEvent] = {}
        self._lock = threading.RLock()
        self._timer: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self.last_flush_at: Optional[datetime] = None
        self.flushed = 0
        self.last_failed = 0
        self._remote_dim: Optional[int] = None

    # -- 生命周期 ----------------------------------------------------

    def start(self, backfill: bool = True) -> None:
        """启动 flush / 巡检两个后台循环。

        ``backfill``：索引为空（全新部署、或本次升级把旧结构清掉了）时静默回填一次存量数据。
        只在"真的空"时才做，否则每次重启都要全量重算一遍，纯属浪费。
        """
        if self._started or not self.enabled:
            return
        self._started = True
        self._stop.clear()
        self._spawn(self._flush_loop, "rag-index-flush")
        if self.reconcile_enabled:
            self._spawn(self._reconcile_loop, "rag-index-reconcile")
        if backfill and self.index is not None and self.index.available() and self._indexed_docs() == 0:
            self._spawn(self._startup_backfill, "rag-index-backfill")
        log.info(
            "[RagIndex] 写入即索引：enabled=%s 合并窗口=%sms 写入时向量化=%s 巡检=%s(%s分钟)",
            self.enabled, self.merge_window_ms, self.embed_on_write_raw,
            self.reconcile_enabled, self.reconcile_interval_min,
        )

    def stop(self) -> None:
        self._stop.set()
        self._started = False

    def _spawn(self, fn, name: str) -> None:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()

    def _sleep(self, seconds: float) -> bool:
        """可被 stop() 打断的 sleep；返回 True 表示应当继续循环。"""
        return not self._stop.wait(seconds)

    def _flush_loop(self) -> None:
        # 先等一个窗口，让启动期的写入攒一批
        if not self._sleep(self.merge_window_ms / 1000.0):
            return
        while not self._stop.is_set():
            try:
                self.flush()
            except Exception as e:  # noqa: BLE001 - 后台循环绝不能因为一次异常退出
                log.warning("[RagIndex] flush 异常：%s", e)
            if not self._sleep(self.merge_window_ms / 1000.0):
                return

    def _reconcile_loop(self) -> None:
        if not self._sleep(self.reconcile_interval_min * 60.0):
            return
        while not self._stop.is_set():
            try:
                log.info("[RagIndex] 巡检完成：%s", self.reconcile())
            except Exception as e:  # noqa: BLE001
                log.warning("[RagIndex] 巡检异常：%s", e)
            if not self._sleep(self.reconcile_interval_min * 60.0):
                return

    def _startup_backfill(self) -> None:
        if not self._sleep(5.0):
            return
        try:
            n = self.reindex(None)
            log.info("[RagIndex] 启动回填完成，共 %s 条语料", n)
        except Exception as e:  # noqa: BLE001
            log.warning("[RagIndex] 启动回填失败：%s", e)

    # -- 事件入口 ----------------------------------------------------

    def on_corpus_change(self, e: Optional[CorpusChangeEvent]) -> None:
        """同一条记录的多次变更只保留最后一次：取到的内容永远是最新的。"""
        if e is None or not self.enabled:
            return
        with self._lock:
            self._pending[e.key()] = e

    submit = on_corpus_change

    def apply_now(self, e: CorpusChangeEvent) -> dict:
        """同步处理一条可靠事件，供 Java Outbox 获得明确成功确认。"""
        if not self.enabled:
            raise RuntimeError("写入即索引服务未启用")
        if self._source_of(e.source_type) is None:
            raise ValueError(f"未注册的语料类型: {e.source_type}")
        self._apply(e)
        chunks = 0 if e.kind == Kind.DELETE else len(self._states_of(e.source_type, e.source_id))
        self.last_flush_at = datetime.now()
        self.flushed += 1
        self.last_failed = 0
        return {"accepted": True, "chunks": chunks, "sourceType": e.source_type,
                "sourceId": e.source_id, "kind": e.kind.value}

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- flush -------------------------------------------------------

    def flush(self) -> int:
        """处理一批待办事件。也供运维端点手工触发。返回成功条数。"""
        with self._lock:
            if not self._pending:
                return 0
            batch = list(self._pending.values())
            self._pending.clear()
        ok = 0
        bad = 0
        for e in batch:
            try:
                self._apply(e)
                ok += 1
            except Exception as ex:  # noqa: BLE001
                bad += 1
                self._mark_failed(e, ex)
        self.last_flush_at = datetime.now()
        self.flushed += len(batch)
        self.last_failed = bad
        if bad > 0:
            log.warning("[RagIndex] flush %s 条，失败 %s 条（已记 FAILED，巡检会重试）", len(batch), bad)
        return ok

    def _apply(self, e: CorpusChangeEvent) -> None:
        src = self._source_of(e.source_type)
        if src is None:
            log.debug("[RagIndex] 未注册的语料类型：%s", e.source_type)
            return
        if e.kind == Kind.DELETE:
            self._remove_by_source(e.source_type, e.source_id)
            return

        chunks = src.of(e.source_id)
        if not chunks:
            # 记录已不存在（刚被删、或本来就不该入库）→ 顺手清掉可能残留的旧切片
            self._remove_by_source(e.source_type, e.source_id)
            return

        old = self._states_of(e.source_type, e.source_id)
        changed: List[CorpusChunk] = []
        for c in chunks:
            s = old.get(c.chunk_id)
            if s is not None and s.get("status") == T.ST_INDEXED and c.hash() == s.get("content_hash"):
                continue  # 内容没变，不重写
            changed.append(c)

        # 文档改短了：旧的第 4、5 片现在已经不存在，必须剔除，否则变成永远召不到的幽灵
        live = {c.chunk_id for c in chunks}
        stale = [i for i in old.keys() if i not in live]
        if stale:
            self._index_remove(stale)
            for i in stale:
                self._delete_state(i)
        if not changed:
            return

        inputs = [ChunkInput(c.chunk_id, c.title, c.text, c.level, c.dept_id,
                             c.company_id, c.source_type) for c in changed]
        r = self._index_upsert(inputs)
        if r is not None and not r.ok:
            raise RuntimeError(r.error or "索引写入失败")
        for c in changed:
            self._save_state(c, T.ST_INDEXED, None)

    # -- 回填 / 巡检 -------------------------------------------------

    def reindex(self, company_id: Optional[int]) -> int:
        """全量回填。``company_id`` 为 None 表示**所有企业 + 全局语料**。"""
        if company_id is None:
            n = 0
            rows = self.db.fetch_all(select(T.company.c.id).order_by(T.company.c.id.asc()))
            for r in rows:
                cid = r.get("id")
                if cid is not None:
                    n += self._reindex_one(cid)
            n += self._reindex_one(None)
            # 全量之后才做这件事：只有这时"记账表是完整的"才成立。
            # 单企业回填时清无主切片，会把别家还没回填的切片误判成僵尸。
            self.prune_untracked()
            return n
        return self._reindex_one(company_id)

    def prune_untracked(self, max_ratio: float = 0.5) -> dict:
        """清掉「索引里有、``rag_chunk_state`` 里没有」的无主切片。

        为什么非有它不可：语料源的 ``chunk_id`` 规则变过一次（知识库切片从
        ``d:{id}`` 改成 ``{id}``），老切片既不会被 upsert 覆盖，**也不在记账表里**
        —— ``_clean_orphans`` 是拿记账表去比对业务表的，看不见它们。
        于是索引里永远留着一批"能被召回、但追溯不到任何来源"的幽灵，
        表现为：Python 侧 docs 比 Java 侧多出几十条，且检索结果里混进旧版文档。

        安全闸门：记账表一旦被误清空，这里的差集就是整个索引。
        僵尸数超过索引总量一半时**宁可不删**，只告警——
        删错索引可以重建，但"静默删光"会让人以为迁移没问题。
        """
        if self.index is None:
            return {"pruned": 0, "skipped": "索引未装配"}
        try:
            idx_ids = set(self.index.all_ids())
        except Exception as e:  # noqa: BLE001
            return {"pruned": 0, "skipped": str(e)}
        if not idx_ids:
            return {"pruned": 0, "indexed": 0, "stateRows": 0}
        try:
            rows = self.db.fetch_all(select(T.rag_chunk_state.c.chunk_id))
        except Exception as e:  # noqa: BLE001
            return {"pruned": 0, "skipped": f"记账表不可读：{e}"}
        st_ids = {str(r.get("chunk_id")) for r in rows}
        dead = sorted(idx_ids - st_ids)
        out = {"indexed": len(idx_ids), "stateRows": len(st_ids), "untracked": len(dead)}
        if not st_ids or not dead:
            out["pruned"] = 0
            return out
        if len(dead) > len(idx_ids) * max_ratio:
            log.warning("[RagIndex] 僵尸 %s 条 > 索引 %s 条的一半，疑似记账表异常，本次不清理",
                        len(dead), len(idx_ids))
            out["pruned"] = 0
            out["skipped"] = f"僵尸占比异常（{len(dead)}/{len(idx_ids)}），疑似记账表被清空，已跳过"
            return out
        self._index_remove(dead)
        out["pruned"] = len(dead)
        log.warning("[RagIndex] 清理无主切片 %s 条（多为旧版 chunk_id 规则遗留）", len(dead))
        return out

    def _reindex_one(self, company_id: Optional[int]) -> int:
        n = 0
        for src in self.sources:
            try:
                chunks = src.all_of(company_id)
                if not chunks:
                    continue
                inputs = [ChunkInput(c.chunk_id, c.title, c.text, c.level, c.dept_id,
                                     c.company_id, c.source_type) for c in chunks]
                r = self._index_upsert(inputs)
                if r is not None and not r.ok:
                    log.warning("[RagIndex] 回填 %s 失败：%s", src.type(), r.error)
                    continue
                for c in chunks:
                    self._save_state(c, T.ST_INDEXED, None)
                n += len(chunks)
            except Exception as e:  # noqa: BLE001
                log.warning("[RagIndex] 回填 %s 异常：%s", src.type(), e)
        self._clean_orphans(company_id)
        return n

    def reconcile(self) -> dict:
        """巡检：重跑 FAILED，并报告「state 里有、业务表里没了」的僵尸切片。"""
        failed = self.db.fetch_all(
            select(T.rag_chunk_state)
            .where(T.rag_chunk_state.c.status == T.ST_FAILED)
            .where(T.rag_chunk_state.c.retry_count < 5)
        )
        fixed = 0
        for s in failed:
            try:
                src = self._source_of(s.get("source_type"))
                if src is None:
                    continue
                sid = _parse_long(s.get("source_id"))
                if sid is None:
                    continue
                self._apply(CorpusChangeEvent.upsert(s["source_type"], sid, s.get("company_id")))
                fixed += 1
            except Exception as e:  # noqa: BLE001
                log.warning("[RagIndex] 重试 %s 仍失败：%s", s.get("chunk_id"), e)
        out = {
            "retried": len(failed),
            "fixed": fixed,
            "orphans": self._count_orphans(),
            "at": datetime.now().isoformat(),
        }
        # 巡检顺手把无主切片清掉：它是"索引里有、记账表里没有"的那半边，
        # 而 orphans 是"记账表里有、业务表里没有"的那半边，两边都要有出口。
        out["untracked"] = self.prune_untracked()
        return out

    def status(self) -> dict:
        m = {
            "enabled": self.enabled,
            "pending": self.pending,
            "mergeWindowMs": self.merge_window_ms,
            "flushed": self.flushed,
            "lastFailed": self.last_failed,
            "lastFlushAt": self.last_flush_at.isoformat() if self.last_flush_at else None,
            "embedOnWrite": self.embed_on_write_raw,
        }
        by_type: Dict[str, dict] = {}
        try:
            rows = self.db.fetch_all(
                select(T.rag_chunk_state.c.source_type, func.count().label("cnt"))
                .group_by(T.rag_chunk_state.c.source_type)
            )
            for r in rows:
                st = str(r.get("source_type"))
                by_type[st] = {
                    "chunks": r.get("cnt"),
                    "failed": self.db.count(
                        T.rag_chunk_state,
                        (T.rag_chunk_state.c.source_type == st)
                        & (T.rag_chunk_state.c.status == T.ST_FAILED),
                    ),
                }
        except Exception as e:  # noqa: BLE001
            by_type["error"] = str(e)
        m["byType"] = by_type
        m["index"] = self.index.stats() if self.index is not None else {}
        return m

    # -- 内部：索引 --------------------------------------------------

    def _index_upsert(self, inputs):
        if self.index is None:
            return None
        return self.index.upsert(inputs, self._embedder_for_write(), self._vector_key_for_write())

    def _index_remove(self, ids) -> None:
        if self.index is not None:
            self.index.remove(ids)

    def _indexed_docs(self) -> int:
        try:
            v = self.index.stats().get("docs")
            return int(v) if isinstance(v, (int, float)) else 0
        except Exception:  # noqa: BLE001
            return 0

    def embed_client(self):
        """第三方向量通道，懒加载一次并缓存结论（它可能是"没配"，那就返回 None）。

        ! 方法名**不能**叫 ``_embedder``：``__init__`` 里往 ``self._embedder`` 上存了同一个
          名字的实例属性，它会遮蔽掉类上的同名方法，调用时得到
          ``TypeError: 'NoneType' object is not callable`` —— 而且因为外层有兜底日志，
          只会表现为"每个语料源都回填失败"，很难定位。
        """
        if self._embedder is not None:
            return self._embedder
        if self._embedder_probe is None:
            self._embedder_probe = False
            if self._embedder_factory is not None:
                try:
                    got = self._embedder_factory()
                    if got is not None and getattr(got, "is_enabled", lambda: False)():
                        self._embedder = got
                except Exception as e:  # noqa: BLE001
                    log.debug("[RagIndex] 第三方向量不可用：%s", e)
        return self._embedder

    def _embedder_for_write(self):
        """写入时的向量化通道。

        本地向量零成本，永远算；第三方向量默认**不算**——一次导入几百行数据
        会变成几百次计费请求，而用户并没有要求立刻能语义检索到它们。
        落下的切片照样进倒排（关键词可召回），向量留到巡检批量补。
        """
        em = self.embed_client()
        if em is None:
            return None
        # ‼️ 必须先看总开关。``EmbeddingClient.embed_cached`` 在
        # ``enabled=False`` 时**直接返回空列表**（而不是抛错），
        # 而 ``_vector_key_for_write()`` 只看 `mode` 是不是 local。
        # 两者一错位，就会出现："开关关着、向量一条没算出来"，
        # 但索引把 vectorKey 记成 local:512 并对外报成功 ——
        # 1070 条切片 vbin 全 NULL，检索只剩 BM25，原始分恒 0，
        # 而所有接口都显示一切正常。
        if not getattr(em, "is_enabled", lambda: True)():
            return None
        if not self._embed_on_write():
            return None
        fn = getattr(em, "embed_cached", None) or getattr(em, "embed", None)
        return fn

    def _embed_on_write(self) -> bool:
        v = (self.embed_on_write_raw or "").strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        # auto：本地向量恒算，第三方留到巡检
        return getattr(self.embed_client(), "mode", "") == "local"

    def _vector_key_for_write(self) -> Optional[str]:
        em = self.embed_client()
        if em is None or self._embedder_for_write() is None:
            return None
        mode = getattr(em, "mode", "")
        dim = self._dim_of(mode)
        return None if dim <= 0 else f"{mode}:{dim}"

    def _dim_of(self, mode: str) -> int:
        if mode == "local":
            return self.local.dim if self.local is not None else 0
        if self._remote_dim is not None:
            return self._remote_dim
        em = self.embed_client()
        try:
            v = em.embed_one("维度探测")
            self._remote_dim = 0 if v is None else len(v)
        except Exception:  # noqa: BLE001
            self._remote_dim = 0
        return self._remote_dim or 0

    # -- 内部：记账表 ------------------------------------------------

    def _source_of(self, source_type: Optional[str]) -> Optional[CorpusSource]:
        if not source_type:
            return None
        for s in self.sources:
            if source_type == s.type():
                return s
        return None

    def _states_of(self, source_type: str, source_id: Optional[int]) -> Dict[str, dict]:
        rows = self.db.fetch_all(
            select(T.rag_chunk_state)
            .where(T.rag_chunk_state.c.source_type == source_type)
            .where(T.rag_chunk_state.c.source_id == str(source_id))
        )
        return {r["chunk_id"]: r for r in rows}

    def _save_state(self, c: CorpusChunk, status: str, error: Optional[str]) -> None:
        tbl = T.rag_chunk_state
        exist = self.db.fetch_one(select(tbl).where(tbl.c.chunk_id == c.chunk_id).limit(1))
        try:
            if exist is None:
                self.db.execute(insert(tbl).values(
                    chunk_id=c.chunk_id, source_type=c.source_type, source_id=c.source_id,
                    company_id=c.company_id, content_hash=c.hash(), status=status,
                    retry_count=1 if status == T.ST_FAILED else 0, error=error,
                    updated_at=datetime.now(),
                ))
            else:
                self.db.execute(
                    update(tbl).where(tbl.c.id == exist["id"]).values(
                        content_hash=c.hash(), status=status, company_id=c.company_id,
                        retry_count=0 if status != T.ST_FAILED
                        else (int(exist.get("retry_count") or 0) + 1),
                        error=error, updated_at=datetime.now(),
                    )
                )
        except Exception as e:  # noqa: BLE001
            log.warning("[RagIndex] 写 state 失败 %s: %s", c.chunk_id, e)

    def _delete_state(self, chunk_id: str) -> None:
        self.db.execute(delete(T.rag_chunk_state).where(T.rag_chunk_state.c.chunk_id == chunk_id))

    def _remove_by_source(self, source_type: str, source_id: Optional[int]) -> None:
        tbl = T.rag_chunk_state
        rows = self.db.fetch_all(
            select(tbl).where(tbl.c.source_type == source_type)
            .where(tbl.c.source_id == str(source_id))
        )
        if not rows:
            return
        self._index_remove([r["chunk_id"] for r in rows])
        for r in rows:
            self.db.execute(delete(tbl).where(tbl.c.id == r["id"]))

    def _mark_failed(self, e: CorpusChangeEvent, ex: Exception) -> None:
        msg = str(ex) or ex.__class__.__name__
        log.warning("[RagIndex] 索引 %s 失败：%s", e.key(), msg)
        tbl = T.rag_chunk_state
        rows = self.db.fetch_all(
            select(tbl).where(tbl.c.source_type == e.source_type)
            .where(tbl.c.source_id == str(e.source_id))
        )
        if not rows:
            # 连 state 都没有：插一条 FAILED，让巡检知道这里欠着一笔
            self.db.execute(insert(tbl).values(
                chunk_id=e.key(), source_type=e.source_type, source_id=str(e.source_id),
                company_id=e.company_id, content_hash="", status=T.ST_FAILED,
                retry_count=1, error=_trunc(msg, 500), updated_at=datetime.now(),
            ))
            return
        for r in rows:
            self.db.execute(
                update(tbl).where(tbl.c.id == r["id"]).values(
                    status=T.ST_FAILED,
                    retry_count=int(r.get("retry_count") or 0) + 1,
                    error=_trunc(msg, 500), updated_at=datetime.now(),
                )
            )

    def _count_orphans(self) -> int:
        """业务表已经没有、state 里还留着的切片数量（只报告不自动删，避免误伤）。"""
        n = 0
        for src in self.sources:
            try:
                rows = self.db.fetch_all(
                    select(T.rag_chunk_state).where(T.rag_chunk_state.c.source_type == src.type())
                )
                if not rows:
                    continue
                by_company: Dict[Optional[int], list] = {}
                for s in rows:
                    by_company.setdefault(s.get("company_id"), []).append(s)
                for cid, group in by_company.items():
                    alive = set(src.ids_of(cid))
                    for s in group:
                        sid = _parse_long(s.get("source_id"))
                        if sid is not None and sid not in alive:
                            n += 1
            except Exception:  # noqa: BLE001
                continue
        return n

    def _clean_orphans(self, company_id: Optional[int]) -> None:
        """回填后清掉该企业名下"已经不存在的切片"。"""
        for src in self.sources:
            try:
                tbl = T.rag_chunk_state
                cond = tbl.c.source_type == src.type()
                cond = cond & (tbl.c.company_id.is_(None) if company_id is None
                               else (tbl.c.company_id == company_id))
                rows = self.db.fetch_all(select(tbl).where(cond))
                if not rows:
                    continue
                alive = set(src.ids_of(company_id) or [])
                dead = []
                for s in rows:
                    sid = _parse_long(s.get("source_id"))
                    if sid is None or sid in alive:
                        continue
                    dead.append(s["chunk_id"])
                    self.db.execute(delete(tbl).where(tbl.c.id == s["id"]))
                if dead:
                    self._index_remove(dead)
            except Exception as e:  # noqa: BLE001
                log.warning("[RagIndex] 清理 %s 残留切片失败：%s", src.type(), e)


def _parse_long(s) -> Optional[int]:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    try:
        return int(t)
    except ValueError:
        return None


def _trunc(s: Optional[str], max_len: int) -> Optional[str]:
    return None if s is None else (s if len(s) <= max_len else s[:max_len])


__all__ = ["RagIndexingService", "CorpusChangeEvent", "Kind", "CT"]
