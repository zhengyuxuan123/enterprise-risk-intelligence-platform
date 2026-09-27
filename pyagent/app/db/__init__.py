"""数据库访问层（SQLAlchemy Core，只读业务表 + 读写 ``rag_chunk_state``）。

为什么用 Core 而不是 ORM
------------------------
Java 侧是 MyBatis-Plus 的实体映射，字段来自数据库列而非 Python 类声明。
若在这里再声明一套 ORM 实体，就出现了**第二份 schema 真相**——
业务表一加字段，Python 侧不会报错，只会静静地少读一列。

Core 只在查询里写列名，读回来的是 dict，缺列会直接 KeyError 而不是"悄悄为 None"。
所有用到的列都集中在 :mod:`app.db.tables` 里显式列出，加字段时改一处即可。
"""

from .engine import Database, get_db

__all__ = ["Database", "get_db"]
