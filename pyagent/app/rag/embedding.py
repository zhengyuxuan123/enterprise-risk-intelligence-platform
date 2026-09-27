"""本地向量化（零成本），对应 Java ``LocalEmbedding``。

**不需要任何 API Key、不发一个网络包**：特征哈希（hashing trick）+ 亚线性词频 + L2 归一化。
每个 token 由 FNV-1a 哈希映射到固定维度的某一维，符号由第二个哈希决定，最后归一化成单位向量。

它不是什么（务必如实理解）：
它衡量的是**词面重叠**，不是真正的语义理解。问「资金链断裂」和文档里写「现金流紧张」
在字面上没有重叠，它给不出高分 —— 这类跨词面召回要靠 :mod:`query_rewrite` 的同义扩展，
或者改挂第三方 embedding。

选它当默认方案的理由：语义检索从此**永远可用** —— 没 Key、断网、第三方欠费熔断，
检索都不会退化成「裸关键词」，而且不会把任何密钥送到外部服务。

.. NOTE::
   FNV-1a 必须**按 Java 的 32 位有符号 int 溢出**来算：``h *= 0x01000193`` 在 Java 里
   是 int 乘法（自动截断），Python 整数不会溢出。写成裸乘法会得到一个巨大整数，
   取模后落到完全不同的桶位 —— 向量值全错，但不报错。
"""

from __future__ import annotations

import math

import numpy as np

from .textnorm import is_blank, lower

_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF


def _fnv1a(s: str) -> int:
    """FNV-1a 32 位，返回**无符号**位模式（调用方按需转有符号）。"""
    h = _FNV_OFFSET
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * _FNV_PRIME) & _MASK32
    return h


def _as_signed(u: int) -> int:
    return u - 0x100000000 if u >= 0x80000000 else u


def is_cjk(cp: int) -> bool:
    """``LocalEmbedding.isCjk(char)``：只认 CJK 基本区与扩展 A。

    注意这不是 ``\\p{IsHan}`` —— 扩展 B 区（含大量生僻字）**不算** CJK，
    会被当成"其他字符"丢弃。照抄，不要"顺手修正"。
    """
    return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)


def tokenize(text: str | None) -> list[str]:
    """分词：中文按字符 bigram（"经营风险" → 经营/营风/风险），英文数字按词。

    为什么中文用 bigram 而不是整句：中文没有空格，且本方案不做词典分词，
    bigram 是「不依赖分词器」时召回与精度的折中 —— 比单字抗噪声，比整句抗错序。
    """
    s = lower(text)
    out: list[str] = []
    ascii_buf: list[str] = []
    cjk_buf: list[str] = []

    def flush_cjk() -> None:
        if not cjk_buf:
            return
        seg = "".join(cjk_buf)
        cjk_buf.clear()
        if len(seg) == 1:
            out.append(seg)  # 单字（如"税"）也要能召回，否则短查询全空
        else:
            for i in range(len(seg) - 1):
                out.append(seg[i : i + 2])

    def flush_ascii() -> None:
        if not ascii_buf:
            return
        w = "".join(ascii_buf)
        ascii_buf.clear()
        out.append(w)
        if len(w) > 6:
            for i in range(len(w) - 3):
                out.append(w[i : i + 4])

    for ch in s:
        cp = ord(ch)
        if is_cjk(cp):
            flush_ascii()
            cjk_buf.append(ch)
        elif ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            flush_cjk()
            ascii_buf.append(ch)
        else:
            flush_ascii()
            flush_cjk()
    flush_ascii()
    flush_cjk()
    return out


class LocalEmbedding:
    """哈希向量化。同一段文本永远得到同一个向量（确定性）。"""

    def __init__(self, dim: int = 512) -> None:
        # 太小碰撞严重（语义失真），太大对短文本无收益；限制在合理区间。
        self.dim = 512 if dim <= 0 else max(64, min(dim, 4096))

    def dim(self) -> int:  # noqa: D102 - 与 Java 同名
        return self.dim

    def embed(self, text: str | None) -> np.ndarray | None:
        """文本 → 单位向量。空文本 / 无有效 token 时返回 ``None``。"""
        if text is None or is_blank(text):
            return None
        tf: dict[str, int] = {}
        for t in tokenize(text):
            tf[t] = tf.get(t, 0) + 1
        if not tf:
            return None

        # float32 与 Java 的 float[] 对齐：累加过程中的每一次舍入都要一致
        v = np.zeros(self.dim, dtype=np.float32)
        for tok, cnt in tf.items():
            h1 = _fnv1a(tok)
            h2 = _fnv1a(tok + "#s")  # 换一个种子位当符号，避免与桶位相关
            idx = (_as_signed(h1) & 0x7FFFFFFF) % self.dim
            sign = 1.0 if ((h2 >> 8) & 1) == 0 else -1.0
            w = 1.0 + math.log(cnt)  # 亚线性词频：抑制长文重复词的一家独大
            v[idx] = np.float32(v[idx] + np.float32(sign * w))

        norm = 0.0
        for x in v:
            norm += float(x) * float(x)
        if norm <= 0:
            return None
        inv = np.float32(1.0 / math.sqrt(norm))
        return (v * inv).astype(np.float32)


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """余弦相似度。两个向量都已归一化，点积即余弦；维度不一致时给 0。"""
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    s = 0.0
    for i in range(a.shape[0]):
        s += float(a[i]) * float(b[i])
    return s
