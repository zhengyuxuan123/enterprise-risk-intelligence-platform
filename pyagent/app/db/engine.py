"""引擎与会话：一个进程一个 engine，查询返回 dict。

两点刻意的约束
--------------
1. **不建 ORM 实体**（理由见包 docstring）：查询只写列名，读回来是 dict。
2. **查询失败不外抛**：与 Java 侧各语料源 ``try/catch`` 的行为一致——
   某一类数据查不动，不能让整轮检索挂掉。需要"查不到就该报错"的场景由调用方判断。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable, List, Optional, Sequence

from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.engine import Row
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from . import tables

log = logging.getLogger(__name__)

_lock = threading.Lock()
_engine: Optional[Engine] = None
_factory: Optional[sessionmaker] = None


def _build() -> Engine:
    s = get_settings().db
    url = f"mysql+pymysql://{s.user}:{s.password}@{s.host}:{s.port}/{s.name}?charset=utf8mb4"
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=10,
        future=True,
    )


def engine() -> Engine:
    """进程内单例。测试里可用 ``reset_engine()`` 换掉。"""
    global _engine, _factory
    with _lock:
        if _engine is None:
            _engine = _build()
            _factory = sessionmaker(bind=_engine, future=True, expire_on_commit=False)
        return _engine


def reset_engine() -> None:
    """释放并清空单例（测试注入内存库时用）。"""
    global _engine, _factory
    with _lock:
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:  # noqa: BLE001 - 关闭失败不必影响调用方
                pass
        _engine = None
        _factory = None


def session() -> Session:
    engine()  # 确保 factory 已建
    assert _factory is not None
    return _factory()


def rows_to_dicts(rows: Iterable[Row]) -> List[dict]:
    """``Row`` → 普通 dict。这样下游只依赖键名，不依赖 SQLAlchemy 对象。"""
    return [dict(r._mapping) for r in rows]


class Database:
    """薄封装：查询返回 dict 列表；连接不可用时返回空列表而不是抛异常。

    为什么要"静默返回空"：agent 栈的所有调用点都在**增强**主流程
    （检索更准、证据更全），而不是主流程本身。索引挂了、某张表查不动，
    最坏情况应当是"这次少了一些证据"，而不是"这次分析整体失败"。
    """

    # -- 读 ----------------------------------------------------------

    def fetch_all(self, stmt, params: Optional[Sequence[Any]] = None) -> List[dict]:
        try:
            with session() as s:
                res = s.execute(stmt, params or {})
                return rows_to_dicts(res.fetchall())
        except SQLAlchemyError as e:
            log.warning("数据库查询失败（已返回空结果）: %s", e)
            return []
        except Exception as e:  # noqa: BLE001 - 驱动层异常也要兜住
            log.warning("数据库查询异常（已返回空结果）: %s", e)
            return []

    def fetch_one(self, stmt, params: Optional[Sequence[Any]] = None) -> Optional[dict]:
        r = self.fetch_all(stmt, params)
        return r[0] if r else None

    def scalar(self, stmt, params: Optional[Sequence[Any]] = None, default: Any = 0) -> Any:
        try:
            with session() as s:
                v = s.execute(stmt, params or {}).scalar()
                return default if v is None else v
        except Exception as e:  # noqa: BLE001
            log.warning("数据库标量查询失败（已返回默认值）: %s", e)
            return default

    def count(self, table, where=None) -> int:
        stmt = select(func.count()).select_from(table)
        if where is not None:
            stmt = stmt.where(where)
        return int(self.scalar(stmt, default=0))

    # -- 写（只用于 rag_chunk_state） -------------------------------

    def execute(self, stmt, params: Optional[Sequence[Any]] = None) -> int:
        """返回影响行数；失败返回 0。"""
        try:
            with session() as s:
                res = s.execute(stmt, params or {})
                s.commit()
                return int(res.rowcount or 0)
        except SQLAlchemyError as e:
            log.warning("数据库写入失败（已忽略）: %s", e)
            return 0
        except Exception as e:  # noqa: BLE001
            log.warning("数据库写入异常（已忽略）: %s", e)
            return 0

    def insert_id(self, stmt, params: Optional[Sequence[Any]] = None) -> Optional[int]:
        """插入并返回自增主键。

        别拿 :meth:`execute` 的返回值当 id —— 它返回的是**影响行数**，
        而自增表的插入永远只影响 1 行，于是"落库 id"会一直是 1。
        这个错最阴的地方是它**不报错**：写入真的成功了，
        只是后续所有按 id 的查询（血缘、留痕、详情）都指向另一条记录。
        """
        try:
            with session() as s:
                res = s.execute(stmt, params or {})
                s.commit()
                pk = res.inserted_primary_key
                if pk and pk[0] is not None:
                    return int(pk[0])
                lid = getattr(res, "lastrowid", 0) or 0
                return int(lid) if lid else None
        except SQLAlchemyError as e:
            log.warning("数据库写入失败（已忽略）: %s", e)
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("数据库写入异常（已忽略）: %s", e)
            return None

    def ping(self) -> bool:
        """连通性自检：能力体检卡片要用。"""
        try:
            with session() as s:
                s.execute(select(1))
            return True
        except Exception:  # noqa: BLE001
            return False


_db: Optional[Database] = None
_dbl = threading.Lock()


def get_db() -> Database:
    global _db
    with _dbl:
        if _db is None:
            _db = Database()
        return _db


__all__ = [
    "Database",
    "get_db",
    "engine",
    "reset_engine",
    "session",
    "rows_to_dicts",
    "tables",
    "select",
    "func",
]
