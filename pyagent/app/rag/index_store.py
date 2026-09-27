"""本地资料的**持久化索引**（对应 Java ``RagIndexStore``）。

Java 版用的是 Lucene 9.11（倒排 + HNSW 向量），Python 版按既定方案换成
**SQLite FTS5（词法通道）+ numpy（向量通道）**。之所以可以换，是因为上层
``RagService`` 只消费「候选的**排名**」，不消费 Lucene 的绝对分数
（RRF 融合用的是 ``1/(60+rank)``）。所以只要保证**排序与 ACL 语义一致**，
检索行为就一致；分数量纲不同是有意为之，不是缺陷。

对齐说明（刻意保留的差异，改之前先读）
-------------------------------------
1. **分词不在数据库里做**。语料先由 :func:`embedding.tokenize` 切成 bigram /
   ASCII 词，再以空格拼成一列交给 FTS5。FTS5 的 ``unicode61`` 只负责按空格切，
   不会自作主张再做一次中文分词 —— 这一点已用探针验证（见 ``qa/_probe_fts5.py``）。
2. **倒排的排序方向**：SQLite 的 ``bm25()`` 返回**负值**且**越小越相关**，
   Lucene 是正值越大越相关。这里对外统一翻成正数（``-bm25``），
   让 :class:`Cand` 的语义与 Java 一致（大 = 好）。
3. **向量通道改成精确余弦**。Java 走 HNSW **近似**最近邻，这里用 numpy 暴力点积，
   因为语料规模（几百到几万片）下暴力反而更快，而且**没有召回抖动**。
   入库向量已归一化时，精确余弦与 HNSW 欧氏距离的排序一致。
4. **没有 Lucene 的"同索引必须同一维度"约束**：SQLite 不限制 BLOB 长度，
   因此 Java 里那段「维度冲突 → 全量重建后重试」的分支在 Python 侧不复存在。
   维度不一致由 :meth:`RagIndexStore.vector` 过滤（只比同维度的行），
   效果等价于 Java 里重建之后"只剩一种维度"。

**权限必须双闸**：入库前只写 ACL 放行的语料，查询时再按 company / 密级 /
可见部门过滤一次 —— 索引是累积的，不能假设"写进去的都是本次可见的"。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Callable, Collection, Iterable, List, Optional, Sequence, Set

import numpy as np

from .embedding import tokenize
from .textnorm import is_blank, lower

# ---------------------------------------------------------------- 常量

#: 索引结构版本。语料块编号规则或字段语义变了就 +1，启动发现不一致自动清空重建。
SCHEMA = "v2"

#: 全局语料（不属于任何企业）在索引里的 company 取值。
GLOBAL_COMPANY = "*"

_DB_NAME = "rag-index.sqlite3"


# ---------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class ChunkInput:
    """待入库的一个语料块（对应 Java ``RagIndexStore.ChunkInput``）。"""

    id: str
    title: Optional[str]
    text: Optional[str]
    level: int
    dept_id: Optional[int] = None
    company_id: Optional[int] = None
    source_type: Optional[str] = None


@dataclass(frozen=True)
class ChunkData:
    """从索引读回的一个语料块（对应 Java ``ChunkData``）。"""

    id: str
    title: Optional[str]
    text: Optional[str]
    level: int
    dept: Optional[str]
    company: Optional[str]
    source_type: Optional[str]


@dataclass(frozen=True)
class UpsertResult:
    total: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    ok: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class SyncResult:
    total: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    rebuilt: bool = False
    ok: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class Cand:
    id: str
    score: float


#: 批量向量化回调：交给调用方决定用本地还是第三方 embedding。
Embedder = Callable[[List[str]], List[Optional[np.ndarray]]]


# ---------------------------------------------------------------- 工具


def _hash(title: Optional[str], text: Optional[str]) -> str:
    """内容 hash（前 8 字节 hex），增量判定用。

    分隔符是 ``\\u0001``：它几乎不可能出现在正文里，用它拼接可以把
    「标题短 + 正文长」与「标题长 + 正文短」区分开，避免 hash 撞车被判成"未变更"。
    """
    payload = ((title or "") + "\u0001" + (text or "")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _index_tokens(title: Optional[str], text: Optional[str]) -> str:
    """拼索引用的 token 串。

    .. WARNING::
       Java 侧是 ``(title == null ? "" : title) + " " + c.text()``。
       **text 为 null 时 Java 的字符串拼接会得到字面量 "null"**（不是空串）。
       这是一处历史行为，照抄而不是"顺手修正"——改了会让已有的索引 hash 判定
       与旧索引不一致。空正文在真实链路里不会出现，保留只为对账。
    """
    raw = (title or "") + " " + ("null" if text is None else text)
    return " ".join(tokenize(raw))


def _norm_level(level: Optional[int]) -> int:
    return max(1, int(level or 0))


def _dept_of(dept_id: Optional[int]) -> str:
    return "-" if dept_id is None else str(dept_id)


def _to_bytes(v: np.ndarray) -> bytes:
    """float32 小端序列化 —— 与 Java ``ByteBuffer.order(LITTLE_ENDIAN)`` 一致。"""
    return np.asarray(v, dtype="<f4").tobytes()


def _from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<f4").copy()


def _quote(term: str) -> str:
    """FTS5 查询词加引号：原样按整词匹配，等价于 Lucene 的 ``TermQuery``。"""
    t = term.replace('"', "").replace("*", "")
    return f'"{t}"'


class _Rows:
    """已物化的查询结果：``execute`` 在锁内就把行取完，锁外随便读。"""

    __slots__ = ("rows", "lastrowid", "rowcount")

    def __init__(self, rows: list, lastrowid: Optional[int], rowcount: int) -> None:
        self.rows = rows
        self.lastrowid = lastrowid
        self.rowcount = rowcount

    def fetchall(self) -> list:
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def __iter__(self):
        return iter(self.rows)


class _SerializedConn:
    """把一条 sqlite3 连接的所有操作串行化。

    为什么必须这样：``sqlite3`` 的**连接**不是线程安全的（模块文档的
    ``check_same_thread=False`` 只是允许跨线程调用，不等于并发安全）。
    用例并行执行时会同时检索，同一条查询会间歇返回 0/1/3 条而不是稳定的 5 条，
    且**不抛任何异常** —— 这种"静默少召回"比报错难查得多。

    行在锁内就 ``fetchall`` 完，游标状态不会在锁外被别的语句踩掉。
    """

    __slots__ = ("_raw", "_lock")

    def __init__(self, raw: "sqlite3.Connection", lock: threading.RLock) -> None:
        self._raw = raw
        self._lock = lock

    def execute(self, sql: str, args=()):
        with self._lock:
            cur = self._raw.execute(sql, args)
            try:
                rows = cur.fetchall()
            except sqlite3.ProgrammingError:
                rows = []          # INSERT/DELETE 等无结果集的语句
            finally:
                cur.close()
            return _Rows(rows, self._raw.lastrowid if hasattr(self._raw, "lastrowid") else None,
                         cur.rowcount)

    def executemany(self, sql: str, seq):
        with self._lock:
            cur = self._raw.executemany(sql, seq)
            cur.close()
            return _Rows([], None, cur.rowcount)

    def executescript(self, sql: str):
        with self._lock:
            return self._raw.executescript(sql)

    def commit(self) -> None:
        with self._lock:
            self._raw.commit()

    def close(self) -> None:
        with self._lock:
            self._raw.close()


# ---------------------------------------------------------------- 主类


class RagIndexStore:
    """SQLite FTS5 + numpy 的语料索引。

    对外 API 与 Java ``RagIndexStore`` 一一对应，方法名采用 Python 下划线风格。
    """

    def __init__(self, index_dir: str = "./data/rag-index",
                 rebuild_on_boot: bool = False) -> None:
        self.index_dir = index_dir
        self.rebuild_on_boot = rebuild_on_boot

        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._failure: Optional[str] = None

        # id -> 内容 hash（增量判定用；启动时从库里读回，不在别处偷存一份）
        self._id_hash: dict[str, str] = {}
        # id -> 企业 ID（删除"该企业已不再可见的文档"时用）
        self._id_company: dict[str, Optional[int]] = {}
        # 当前索引的向量空间标识（local:512 / remote:BAAI/bge-m3:1024）
        self._index_vector_key: Optional[str] = None
        self._index_schema: Optional[str] = None
        #: 真正带向量的行数（``dim > 0``）。
        #:
        #: 必须单独记账：``_index_vector_key`` 只说明"用的是哪套向量空间"，
        #: 它曾经在**一条向量都没写进去**的情况下照样被记成 ``local:512``。
        #: 只看它就会得出"有向量"的结论，而检索实际只有 BM25、原始分恒 0。
        self._ids_with_vec: set = set()

    # ---------------- 生命周期 ----------------

    def init(self) -> None:
        """对应 ``@PostConstruct init()``。任何失败都只降级、不抛出。"""
        try:
            os.makedirs(self.index_dir, exist_ok=True)
            path = os.path.abspath(os.path.join(self.index_dir, _DB_NAME))
            raw = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            # sqlite3 的**连接不是线程安全的**：即便开了 check_same_thread=False，
            # 两条线程同时 execute 也会互相踩掉语句状态。实测表现极具迷惑性 ——
            # 同一条查询串行跑稳定返回 5 条，并发跑会返回 0/1/3 条，且没有报错。
            # 评估用例是并行的，这个坑会直接把检索质量算成"间歇性失效"。
            self._conn = _SerializedConn(raw, self._lock)
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._create_tables()
            if self.rebuild_on_boot:
                self._delete_all()
            self._load_state()
            if self._index_schema is not None and self._index_schema != SCHEMA:
                # 新旧两套 chunkId 会并存，检索会召回出已废弃的"幽灵语料"
                self._delete_all()
                self._id_hash.clear()
                self._id_company.clear()
                self._index_vector_key = None
        except Exception as e:  # noqa: BLE001 - 与 Java 一致：一律降级不抛出
            self._failure = str(e)

    def close(self) -> None:
        """对应 ``@PreDestroy close()``。"""
        try:
            if self._conn is not None:
                self._conn.commit()
                self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._conn = None

    def available(self) -> bool:
        return self._failure is None and self._conn is not None

    def all_ids(self) -> list:
        """索引内全部 chunk_id。用于「索引 vs 记账表」对账，找出无主切片。"""
        return list(self._id_hash.keys())

    def stats(self) -> dict:
        """索引状态，进「能力体检」卡片（键名与 Java 一致）。"""
        m: dict = {
            "available": self.available(),
            "dir": self.index_dir,
            "docs": len(self._id_hash),
            "vectorKey": self._index_vector_key,
            "engine": "sqlite-fts5+numpy",
        }
        # 向量覆盖数是**新增**的：Java 那版只有 vectorKey，
        # 而这一项恰恰是"有倒排、没向量"能瞒过所有人的原因。
        m.update(self.vector_stats())
        if self._failure is not None:
            m["error"] = self._failure
        return m

    # ---------------- 建表 ----------------

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rag_chunk (
                id       TEXT PRIMARY KEY,
                company  TEXT NOT NULL,
                type     TEXT NOT NULL,
                schema   TEXT,
                level    INTEGER NOT NULL,
                dept     TEXT NOT NULL,
                title    TEXT,
                body     TEXT,
                hash     TEXT,
                vkey     TEXT,
                vbin     BLOB,
                dim      INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS ix_rag_chunk_scope ON rag_chunk(company, level, dept, type);
            """
        )
        # 独立（非 external-content）FTS5：写入/删除就是普通 SQL，
        # 省掉 external content 那套"删除时必须回填原文"的隐性约束。
        # 代价是 tokens 列存两份，规模在万级语料下完全可接受。
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts USING fts5(id UNINDEXED, tokens)"
        )

    def _delete_all(self) -> None:
        assert self._conn is not None
        self._conn.execute("DELETE FROM rag_fts")
        self._conn.execute("DELETE FROM rag_chunk")
        self._ids_with_vec.clear()

    # ---------------- 状态回读 ----------------

    def _load_state(self) -> None:
        """启动时把 id / hash / company / vectorKey / schema 读回内存。"""
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT id, hash, company, vkey, schema FROM rag_chunk ORDER BY rowid"
        )
        for cid, h, company, vkey, schema in cur.fetchall():
            self._id_hash[cid] = h
            if company is not None:
                try:
                    self._id_company[cid] = int(company)
                except (TypeError, ValueError):
                    # "*"（全局语料）解析失败 → 不入表，与 Java NumberFormatException 一致
                    pass
            if self._index_vector_key is None:
                self._index_vector_key = vkey
            if self._index_schema is None:
                self._index_schema = schema
        self._refresh_vector_rows()

    def _refresh_vector_rows(self) -> None:
        """重新读一遍"哪些行真带向量"（``dim`` 在无向量时写 0）。"""
        assert self._conn is not None
        try:
            cur = self._conn.execute("SELECT id FROM rag_chunk WHERE dim > 0")
            self._ids_with_vec = {r[0] for r in cur.fetchall()}
        except Exception:  # noqa: BLE001 - 数不出来不该让索引不可用
            self._ids_with_vec = set()

    @property
    def _vector_rows(self) -> int:
        return len(self._ids_with_vec)

    def has_vector(self, chunk_id: str) -> bool:
        """这一条到底有没有向量。按行判，而不是按索引级元数据猜。"""
        return chunk_id in self._ids_with_vec

    def vectors_missing(self) -> bool:
        """声明了向量空间，却没能覆盖全部行 —— 语义通道是不完整的。

        用**任意缺口**而不是"一条都没有"做判据：实测里这种损坏往往只差一点点
        （1070 条里 1061 条没向量，元数据照样写着 ``local:512``），
        按"零向量才算坏"去看，它会稳稳地漏过去。
        """
        if self._index_vector_key is None or not self._id_hash:
            return False
        return self._vector_rows < len(self._id_hash)

    def vectors_absent(self) -> bool:
        """更极端的一档：干脆一条向量都没有。"""
        return self._index_vector_key is not None and self._vector_rows <= 0

    def _vector_shortfall(self, vector_key: Optional[str], written: int) -> Optional[str]:
        """本该写向量却一条都没写进去 → 返回可读原因。

        原来是静默通过的：``embedder(texts)`` 返回空列表时，每条切片都被当成
        "没有向量"写下去，而 ``vectorKey`` 照记，调用方收到 ``ok=True``。
        结果是一份只有关键词通道的索引被当成完整的，直到有人去量原始分才发现。
        """
        if vector_key is None or written <= 0:
            return None
        if self._vector_rows > 0:
            return None
        return (
            f"向量化未产出任何向量：本次写入 {written} 条，但索引里带向量的行数为 0"
            f"（向量空间声明为 {vector_key}）。索引现在只有关键词通道 —— "
            "语义检索、原始分（ragTopRawScore）与「Top分/原分」都会缺失。"
            "常见原因：embedding 总开关被关掉、向量化通道报错、或模型名/Key 不可用。"
        )

    def vector_stats(self) -> dict:
        """向量覆盖情况，供 ``rag/status`` 与「能力体检」卡片显示。"""
        docs = len(self._id_hash)
        m = {
            "vectorRows": self._vector_rows,
            "vectorCoveragePct": round(self._vector_rows * 100.0 / docs, 1) if docs else 0.0,
            "vectorMissing": self.vectors_missing(),
            "vectorsAbsent": self.vectors_absent(),
        }
        if m["vectorMissing"]:
            # 说清"该怎么修"：这不是数据损坏，重算一次即可
            m["vectorWarning"] = (
                f"向量覆盖不全（{self._vector_rows}/{docs} 条）：语义通道只对一部分语料生效，"
                "缺失的那些只能靠关键词召回，原始分（ragTopRawScore）也出不来。"
                "开启 embedding 后重跑一次 POST /api/ai/rag/reindex 即可自动补算。"
            )
        return m

    # ---------------- 写入 ----------------

    def _write_chunk(self, owner: Optional[int], c: ChunkInput,
                     vec: Optional[np.ndarray], vector_key: Optional[str]) -> None:
        assert self._conn is not None
        cid = c.id
        company = GLOBAL_COMPANY if owner is None else str(owner)
        has_vec = vec is not None and len(vec) > 0
        self._conn.execute("DELETE FROM rag_chunk WHERE id=?", (cid,))
        self._conn.execute("DELETE FROM rag_fts WHERE id=?", (cid,))
        self._conn.execute(
            "INSERT INTO rag_chunk(id, company, type, schema, level, dept, title, body,"
            " hash, vkey, vbin, dim) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                company,
                c.source_type or "-",
                SCHEMA,
                _norm_level(c.level),
                _dept_of(c.dept_id),
                c.title or "",
                "" if c.text is None else c.text,
                _hash(c.title, c.text),
                # 逐行如实记账：没算出向量的那一行 **不能**写 vector_key。
                # 否则这一行看起来"属于 local:512 向量空间"，只是碰巧 vbin 空，
                # 事后无法区分"算法没产出"与"数据坏了"。
                vector_key if has_vec else None,
                _to_bytes(vec) if has_vec else None,
                int(len(vec)) if has_vec else 0,
            ),
        )
        self._conn.execute(
            "INSERT INTO rag_fts(id, tokens) VALUES(?,?)",
            (cid, _index_tokens(c.title, c.text)),
        )
        if has_vec:
            self._ids_with_vec.add(cid)
        else:
            self._ids_with_vec.discard(cid)

    def _delete_ids(self, ids: Iterable[str]) -> int:
        assert self._conn is not None
        n = 0
        for cid in ids:
            if cid is None or is_blank(cid):
                continue
            self._conn.execute("DELETE FROM rag_chunk WHERE id=?", (cid,))
            self._conn.execute("DELETE FROM rag_fts WHERE id=?", (cid,))
            self._id_hash.pop(cid, None)
            self._id_company.pop(cid, None)
            self._ids_with_vec.discard(cid)
            n += 1
        return n

    def _vector_key_equals(self, vector_key: Optional[str]) -> bool:
        """向量空间是否一致（索引级判定）。按行判定见 :meth:`_can_skip`。"""
        if vector_key is None or self._index_vector_key is None:
            return True
        return self._index_vector_key == vector_key

    def _can_skip(self, chunk_id: str, content_hash: str,
                  vector_key: Optional[str]) -> bool:
        """这一条能不能跳过重写（内容没变 + 向量已就位）。

        ‼️ **必须逐行看有没有向量**，不能只看索引级的 ``vectorKey``。
        事故复盘：索引元数据记着 ``local:512``，1070 条切片里 1061 条 ``vbin`` 是 NULL。
        只比键的话 ``local:512 == local:512`` 永远成立，这些行就永远被跳过 ——
        内容没变 → 不重写 → **永远修不好**，无论重启多少次、重建多少次。
        加上"这一行必须真有向量"之后，缺口会在下一次写入时被自动补上，
        而已有向量的行仍然走增量，不会白白重算。
        """
        if self._id_hash.get(chunk_id) != content_hash:
            return False
        if not self._vector_key_equals(vector_key):
            return False
        if vector_key is not None and not self.has_vector(chunk_id):
            return False
        return True

    def _rebuild_for_vector_change(self, vector_key: Optional[str]) -> bool:
        """向量空间变了（换 embedding 服务商 / 改维度）→ 全量重建。"""
        if vector_key is None or self._index_vector_key is None:
            return False
        if vector_key == self._index_vector_key:
            return False
        self._delete_all()
        self._id_hash.clear()
        self._id_company.clear()
        self._index_vector_key = None
        return True

    def sync(self, company_id: Optional[int], chunks: Optional[List[ChunkInput]],
             embedder: Optional[Embedder], vector_key: Optional[str]) -> SyncResult:
        """把某家企业的语料**增量对账**进索引。

        与 :meth:`upsert` 的区别：**本次没提到的会被删除**（"这一整批就是全部"语义）。
        """
        if not self.available():
            return SyncResult(0, 0, 0, 0, 0, False, False, "索引不可用")
        if company_id is None or chunks is None:
            return SyncResult(0, 0, 0, 0, 0, False, True, None)

        with self._lock:
            rebuilt = False
            try:
                rebuilt = self._rebuild_for_vector_change(vector_key)

                # 1) 删掉"索引里有、但这次不可见"的（文档被删 / 被降权）
                input_ids = {c.id for c in chunks if c is not None and c.id is not None}
                stale = [
                    cid for cid, owner in self._id_company.items()
                    if owner == company_id and cid not in input_ids
                ]
                removed = self._delete_ids(stale)

                # 2) 只对新 / 改动的 chunk 做向量化
                todo: List[ChunkInput] = []
                todo_texts: List[str] = []
                unchanged = 0
                for c in chunks:
                    if c is None or c.id is None:
                        continue
                    h = _hash(c.title, c.text)
                    if self._can_skip(c.id, h, vector_key):
                        unchanged += 1
                        continue
                    todo.append(c)
                    todo_texts.append((c.title or "") + "\n" + ("" if c.text is None else c.text))

                vecs: List[Optional[np.ndarray]] = []
                if embedder is not None and todo:
                    try:
                        vecs = list(embedder(todo_texts) or [])
                    except Exception:  # noqa: BLE001 - 向量化失败不该让同步整体失败
                        vecs = []

                added = updated = 0
                for i, c in enumerate(todo):
                    v = vecs[i] if i < len(vecs) else None
                    existed = c.id in self._id_hash
                    self._write_chunk(company_id, c, v, vector_key)
                    self._id_hash[c.id] = _hash(c.title, c.text)
                    self._id_company[c.id] = company_id
                    if existed:
                        updated += 1
                    else:
                        added += 1

                self._index_vector_key = vector_key
                self._conn.commit()
                self._refresh_vector_rows()
                vec_err = self._vector_shortfall(vector_key, added + updated)
                if vec_err:
                    return SyncResult(len(chunks), added, updated, removed, unchanged,
                                      rebuilt, False, vec_err)
                return SyncResult(len(chunks), added, updated, removed, unchanged,
                                  rebuilt, True, None)
            except Exception as e:  # noqa: BLE001
                return SyncResult(len(chunks), 0, 0, 0, 0, rebuilt, False, str(e))

    def upsert(self, chunks: Optional[List[ChunkInput]], embedder: Optional[Embedder],
               vector_key: Optional[str]) -> UpsertResult:
        """增量写入若干语料块（写入即索引的落点）。

        **不做"本次没提到就删除"** —— 一批事件只代表"这几条变了"，
        绝不能顺手把别的语料判成已删除。
        """
        if not self.available():
            return UpsertResult(0, 0, 0, 0, False, "索引不可用")
        if not chunks:
            return UpsertResult(0, 0, 0, 0, True, None)

        with self._lock:
            try:
                self._rebuild_for_vector_change(vector_key)

                todo: List[ChunkInput] = []
                todo_texts: List[str] = []
                skipped = 0
                for c in chunks:
                    if c is None or c.id is None:
                        continue
                    h = _hash(c.title, c.text)
                    if self._can_skip(c.id, h, vector_key):
                        skipped += 1
                        continue
                    todo.append(c)
                    todo_texts.append((c.title or "") + "\n" + ("" if c.text is None else c.text))

                vecs: List[Optional[np.ndarray]] = []
                if embedder is not None and todo:
                    try:
                        vecs = list(embedder(todo_texts) or [])
                    except Exception:  # noqa: BLE001
                        vecs = []

                added = updated = 0
                for i, c in enumerate(todo):
                    v = vecs[i] if i < len(vecs) else None
                    existed = c.id in self._id_hash
                    # 语料块自带企业优先（写入即索引时每条自带）
                    self._write_chunk(c.company_id, c, v, vector_key)
                    self._id_hash[c.id] = _hash(c.title, c.text)
                    self._id_company[c.id] = c.company_id
                    if existed:
                        updated += 1
                    else:
                        added += 1

                if vector_key is not None:
                    self._index_vector_key = vector_key
                self._conn.commit()
                self._refresh_vector_rows()
                vec_err = self._vector_shortfall(vector_key, added + updated)
                if vec_err:
                    return UpsertResult(len(chunks), added, updated, skipped, False, vec_err)
                return UpsertResult(len(chunks), added, updated, skipped, True, None)
            except Exception as e:  # noqa: BLE001
                return UpsertResult(len(chunks), 0, 0, 0, False, str(e))

    def remove(self, ids: Optional[Collection[str]]) -> int:
        """按 chunkId 精确删除（物理删除业务记录时用）。"""
        if not self.available() or not ids:
            return 0
        with self._lock:
            try:
                n = self._delete_ids(ids)
                if n > 0:
                    self._conn.commit()
                return n
            except Exception:  # noqa: BLE001
                return 0

    # ---------------- 查询 ----------------

    def keyword(self, company_id: Optional[int], terms: Optional[Sequence[str]], limit: int,
                max_level: Optional[int] = None, visible_depts: Optional[Set[int]] = None,
                source_types: Optional[Set[str]] = None) -> List[Cand]:
        """词法通道：BM25 倒排（OR 语义，命中任一 term 即召回）。"""
        if not self.available() or company_id is None or not terms:
            return []
        cleaned = [lower(t) for t in terms if t is not None and not is_blank(t)]
        if not cleaned:
            return []
        match = " OR ".join(_quote(t) for t in cleaned)

        where, args = self._acl_where(company_id, max_level, visible_depts, source_types)
        sql = (
            "SELECT rag_fts.id AS id, -bm25(rag_fts) AS score "
            "FROM rag_fts JOIN rag_chunk c ON c.id = rag_fts.id "
            f"WHERE rag_fts MATCH ?{where} "
            # 同分按 rowid（≈ Lucene 的内部 doc id，都是插入序），避免次序随机
            "ORDER BY score DESC, c.rowid ASC LIMIT ?"
        )
        try:
            cur = self._conn.execute(sql, [match, *args, max(1, limit)])
            return [Cand(r[0], float(r[1])) for r in cur.fetchall()]
        except Exception:  # noqa: BLE001 - 与 Java 的 try/catch 一致：查询失败给空列表
            return []

    def vector(self, company_id: Optional[int], qvec: Optional[np.ndarray], limit: int,
               max_level: Optional[int] = None, visible_depts: Optional[Set[int]] = None,
               source_types: Optional[Set[str]] = None) -> List[Cand]:
        """向量通道：精确余弦（Java 走 HNSW 近似，这里暴力点积，见模块头说明）。"""
        if not self.available() or company_id is None or qvec is None or len(qvec) == 0:
            return []
        q = np.asarray(qvec, dtype=np.float64)
        nq = float(np.linalg.norm(q))
        if nq <= 0:
            return []
        q = q / nq

        where, args = self._acl_where(company_id, max_level, visible_depts, source_types)
        sql = (
            "SELECT c.id, c.vbin FROM rag_chunk c "
            f"WHERE c.dim > 0 AND c.dim = ?{where} ORDER BY c.id ASC"
        )
        try:
            rows = self._conn.execute(sql, [int(len(qvec)), *args]).fetchall()
        except Exception:  # noqa: BLE001
            return []

        scored: List[Cand] = []
        for cid, blob in rows:
            if blob is None:
                continue
            v = _from_bytes(blob).astype(np.float64)
            nv = float(np.linalg.norm(v))
            if nv <= 0:
                continue
            scored.append(Cand(cid, float(np.dot(v, q) / nv)))
        # 大 = 好；同分按 id 升序，保证结果可复现（Java 的 HNSW 不保证同分次序）
        scored.sort(key=lambda c: (-c.score, c.id))
        return scored[: max(1, limit)]

    def read_vectors(self, ids: Optional[Collection[str]]) -> dict[str, np.ndarray]:
        """按 id 读回入库时的**精确向量**，供精排算真实余弦。"""
        out: dict[str, np.ndarray] = {}
        if not self.available() or not ids:
            return out
        want = {i for i in ids if i is not None}
        if not want:
            return out
        qmarks = ",".join("?" * len(want))
        try:
            cur = self._conn.execute(
                f"SELECT id, vbin FROM rag_chunk WHERE id IN ({qmarks})",
                list(want),
            )
            for cid, blob in cur.fetchall():
                if cid in want and blob is not None:
                    out[cid] = _from_bytes(blob)
        except Exception:  # noqa: BLE001
            pass
        return out

    def read_chunks(self, ids: Optional[Collection[str]]) -> dict[str, ChunkData]:
        """按 id 读回原文（title / body / 密级 / 部门 / 类型），供精排与摘要。"""
        out: dict[str, ChunkData] = {}
        if not self.available() or not ids:
            return out
        want = {i for i in ids if i is not None}
        if not want:
            return out
        qmarks = ",".join("?" * len(want))
        try:
            cur = self._conn.execute(
                "SELECT id, title, body, level, dept, company, type "
                f"FROM rag_chunk WHERE id IN ({qmarks})",
                list(want),
            )
            for cid, title, body, level, dept, company, ty in cur.fetchall():
                if cid in want:
                    out[cid] = ChunkData(cid, title, body, int(level or 1), dept, company, ty)
        except Exception:  # noqa: BLE001
            pass
        return out

    # ---------------- 内部 ----------------

    def _acl_where(self, company_id: int, max_level: Optional[int],
                   visible_depts: Optional[Set[int]],
                   source_types: Optional[Set[str]]) -> tuple[str, list]:
        """权限过滤：企业 + 密级上限 + 可见部门 + 来源类型。

        与 Java ``aclFilter`` 同口径：本企业语料 **OR** 全局语料，绝不跨企业。
        """
        where = " AND (c.company = ? OR c.company = ?)"
        args: list = [str(company_id), GLOBAL_COMPANY]
        if max_level is not None:
            where += " AND c.level <= ?"
            args.append(int(max_level))
        if visible_depts is not None:
            depts = {str(d) for d in visible_depts if d is not None}
            depts.add("-")
            where += f" AND c.dept IN ({','.join('?' * len(depts))})"
            args.extend(sorted(depts))
        if source_types:
            types = [t for t in source_types if t is not None and not is_blank(t)]
            if types:
                where += f" AND c.type IN ({','.join('?' * len(types))})"
                args.extend(sorted(types))
        return where, args
