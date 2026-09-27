"""文本与数值的规范化：**Python 原生语义**。

这里原来叫 ``jcompat``（Java 语义兼容层），存在的理由是"Java 与 Python 的
字符串操作看起来一样、语义其实不同"——分片起点差一个字符，后面所有分片都会跟着错位。

那个理由现在不成立了：**Java 侧已经下线**（代码只在 ``D:\\backup`` 里留档），
再为对齐 Java 保留一套非原生语义，代价是长期、隐蔽的：

* ``trim`` 不删全角空格 → 语料里以全角空格开头的段落，分片起点比"看起来"靠后一个字符；
* ``utf16_len`` / ``utf16_head`` 按 **UTF-16 码元** 计数 → emoji、扩展 B 区汉字
  一个字符算两个，分片长度在它们出现的地方整体偏长；
* ``java_round`` / ``fmt2`` 走 **HALF_UP** → 与 Python 其余部分的 ``round`` / ``f"{x:.2f}"``
  （银行家舍入）分叉，同一个分数在两处可能差 0.01。

三条都是"不会报错、只会悄悄偏一点"的类型 —— 正是最难排查的那一类。
所以这一层改成 Python 原生：**字符串按码点、空白按 Unicode、舍入按 Python 默认**。

.. WARNING::
    分片语义变了，**已入库的 chunk 是用旧规则切出来的**。
    改完之后必须重建索引（否则新旧分片混杂，同一段正文会有两种切法）。
"""

from __future__ import annotations


def trim(s: str | None) -> str:
    """``str.strip()``：删首尾所有 Unicode 空白（含全角空格 U+3000 与 NBSP U+00A0）。

    这是与旧 ``java_trim`` 唯一有实际差别的一条：Java 的 ``trim`` 只吃 ``<= U+0020``。
    """
    return "" if s is None else s.strip()


def is_blank(s: str | None) -> bool:
    """空、None、或全部由空白组成。与 :func:`trim` 同一套空白定义（不会各判一套）。"""
    return s is None or s.strip() == ""


def lower(v: str | None) -> str:
    return "" if v is None else v.lower()


def length(s: str | None) -> int:
    """字符串长度（**码点**）。emoji 算 1，不是 2。"""
    return 0 if s is None else len(s)


def head(s: str | None, n: int) -> str:
    """按码点取前 ``n`` 个字符。

    判断与切片**必须是同一个量纲**——历史上正是在这里出过一次错
    （用码元判断、用码点切片），所以把两件事合成一个函数，调用方不可能再写歪。
    """
    if s is None:
        return ""
    return s[:n] if n > 0 else ""


def round3(v: float) -> float:
    """保留三位小数（Python 原生 ``round``，银行家舍入）。"""
    return round(v, 3)


def fmt2(v: float) -> str:
    """两位小数的展示（Python 原生 ``:.2f``）。

    与 :func:`round3` 一样走 Python 默认舍入，不再为对齐 Java 走 HALF_UP。
    """
    return f"{v:.2f}"


__all__ = ["fmt2", "head", "is_blank", "length", "lower", "round3", "trim"]
