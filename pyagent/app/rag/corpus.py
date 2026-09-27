"""语料块与渲染小工具。

字符串语义一律走 :mod:`app.rag.textnorm` 的 **Python 原生** 规则：

1. 空白是 **Unicode 集合** —— 全角空格 U+3000、NBSP U+00A0 都会被折叠/删除。
   （旧实现为了对齐 Java 只用 ASCII 空白集 `[ \\t\\n\\x0B\\f\\r]`，
   于是中文语料里的全角空格被原样留下，切片文本与「看起来」的不一样。）
2. 长度与截断按 **码点**，不是 UTF-16 码元。
   emoji / 扩展 B 区汉字在旧实现里一个算两个，截断位置会整体前移。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

#! ``trim`` **必须取别名**：本模块自己导出了一个两参数的 ``trim(s, max_len)``，
#! 直接 ``from .textnorm import trim`` 会被它遮蔽，函数体里那句 ``trim(...)``
#! 会变成递归调用（表现是 ``RecursionError``，而不是"参数个数不对"，很难一眼看出）。
from .textnorm import head, is_blank, length
from .textnorm import trim as trim_ws

# 空白折叠：Python 原生（Unicode 感知），全角空格也会被折成一个空格
_WS_RUN = re.compile(r"\s+")
_ELLIPSIS = "…"


class T:
    """来源类型常量 —— 与 Java ``CorpusChunk.T`` 同名同值。"""

    KNOWLEDGE = "knowledge"
    METRIC = "metric"
    EVENT = "event"
    COMPLAINT = "complaint"
    COMPETITOR = "competitor"
    COMPANY = "company"
    RULE = "rule"
    AGGREGATE = "aggregate"


@dataclass(frozen=True)
class CorpusChunk:
    """一个待入库的语料块（对应 Java 的 record ``CorpusChunk``）。"""

    chunk_id: str
    source_type: str
    source_id: str
    company_id: Optional[int]
    title: str
    text: str
    level: int
    dept_id: Optional[int] = None

    def hash(self) -> str:
        """内容指纹：SHA-256 前 8 字节 hex，与 ``RagIndexStore.hash()`` 同算法。"""
        payload = f"{self.title or ''}\x01{self.text or ''}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


# ------------------------------------------------------------------
# 渲染小工具（对应 source/Render）
# ------------------------------------------------------------------


def nz(s: Optional[str]) -> str:
    """空值兜底：切片里写「未填写」比写 null 更容易被检索，模型也不会误读。"""
    return "未填写" if is_blank(s) else trim_ws(s)


def blank(s: Optional[str]) -> str:
    """空值返回空串（用于「没有就不写这一段」的场景）。"""
    return "" if is_blank(s) else trim_ws(s)


def num(v) -> str:
    """``BigDecimal.stripTrailingZeros().toPlainString()`` 的等价物。

    Python 的 ``Decimal.normalize()`` 会把 100 变成 ``1E+2``，
    必须用 ``format(d, 'f')`` 还原成 ``100``，否则切片里出现科学计数法。
    """
    if v is None:
        return "—"
    try:
        return format(Decimal(str(v)).normalize(), "f")
    except (InvalidOperation, ValueError, ArithmeticError):
        return "—"


def date(d) -> str:
    """``LocalDate`` → ``yyyy-MM-dd``；None → 「未知日期」。"""
    if d is None:
        return "未知日期"
    try:
        return d.strftime("%Y-%m-%d")
    except (AttributeError, ValueError):
        return "未知日期"


def yes(v) -> bool:
    """``tinyint`` 的 1 判定。Java 是 ``v != null && v == 1``。"""
    if v is None:
        return False
    try:
        return int(v) == 1
    except (TypeError, ValueError):
        return False


def trim(s: Optional[str], max_len: int) -> str:
    """合并空白并截断：切片过长会稀释检索密度，也会挤占 prompt。"""
    if s is None:
        return ""
    t = trim_ws(_WS_RUN.sub(" ", s))
    if length(t) <= max_len:
        return t
    return head(t, max_len) + _ELLIPSIS


def is_chunkable(text: Optional[str]) -> bool:
    """空正文不入库：空切片只会污染索引，召回了也给不出证据。"""
    return not is_blank(text)
