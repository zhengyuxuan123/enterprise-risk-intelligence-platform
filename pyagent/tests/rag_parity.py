"""检索层对账的**共享内核**。

``qa/_parity_rag.py``（命令行）与 ``tests/test_stage3a_rag.py``（pytest）都调这里，
避免"两份清单 / 两份比对逻辑各自漂移" —— 那是上一轮踩过的坑。

基准来自 jshell 跑 Java 真身的输出，重取方式见 ``qa/_java_ground_rag.jsh``。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent  # pyagent/

if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from app.rag import embedding as emb  # noqa: E402
from app.rag import text as ragtext  # noqa: E402
from app.rag.query_rewrite import LEXICON  # noqa: E402
from app.rag.query_rewrite import core as qr_core  # noqa: E402
from app.rag.query_rewrite import expanded as qr_expanded  # noqa: E402
from app.rag.query_rewrite import variants as qr_variants  # noqa: E402
from app.rag.relevance import SourceRelevanceFilter, WebSource  # noqa: E402
from app.rag.web_cleaner import clean as web_clean  # noqa: E402

CASES = HERE / "data" / "rag_cases.txt"
GROUND = HERE / "data" / "java_ground_rag.txt"

J = "#|#"


def _b(v: bool) -> str:
    """Java 的 boolean 字符串化是 ``true``/``false``，Python 是 ``True``/``False``。"""
    return "true" if v else "false"


def _bits_hex(vec) -> str:
    """按 IEEE-754 位模式比对，而不是按十进制 —— float32 的舍入差异一眼可见。"""
    return " ".join(format(int.from_bytes(v.tobytes(), "little"), "x") for v in vec)


def load_cases() -> list[list[str]]:
    out = []
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        out.append(line.split("\t", -1))
    return out


def load_ground() -> dict[tuple[str, int], str]:
    if not GROUND.exists():
        raise FileNotFoundError(
            f"基准文件不存在：{GROUND}\n"
            "先跑一次 jshell 取真（见 pyagent/README.md 的「重取基准」一节）。"
        )
    g: dict[tuple[str, int], str] = {}
    for line in GROUND.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t", -1)
        if len(parts) < 3:
            continue
        g[(parts[0], int(parts[1]))] = parts[2]
    return g


def py_result(fields: list[str]) -> str:
    kind = fields[0]
    if kind == "SPLIT":
        return J.join(ragtext.split(fields[3], int(fields[1]), int(fields[2])))
    if kind == "TOK":
        return J.join(ragtext.tokenize(fields[1]))
    if kind == "BIGRAM":
        return J.join(ragtext.bigrams(fields[1]))
    if kind == "OCC":
        return str(ragtext.occurrences(fields[1], fields[2]))
    if kind == "QR_VARIANTS":
        return J.join(qr_variants(fields[1]))
    if kind == "QR_CORE":
        return qr_core(fields[1])
    if kind == "QR_EXPANDED":
        return qr_expanded(fields[1])
    if kind == "LEXICON_ORDER":
        return J.join(LEXICON.keys())
    if kind == "CLEAN":
        c = web_clean(fields[1])
        return (
            ("null" if c.query is None else c.query)
            + J + _b(c.accepted) + J + c.reason + J + J.join(c.terms)
        )
    if kind == "EMBED":
        le = emb.LocalEmbedding(int(fields[1]))
        v = le.embed(fields[2])
        return f"{le.dim}{J}" + ("null" if v is None else _bits_hex(v))
    if kind == "REL":
        flt = SourceRelevanceFilter(emb.LocalEmbedding(512), 0.16, 0.18, 5)
        srcs = []
        k = 0
        for rec in fields[3].split("#@#"):
            if not rec:
                continue
            p = rec.split("#|#", -1)
            srcs.append(WebSource(k, p[0], p[1], p[2], p[3] if len(p) > 3 else ""))
            k += 1
        fr = flt.filter(fields[1], fields[2], srcs)
        s = J.join(x.url for x in fr.kept) + "##"
        for sc in fr.detail:
            s += f"{sc.score},{sc.coverage},{sc.cosine},{_b(sc.keep)},{sc.why};;"
        return s
    raise ValueError(f"未知用例类型：{kind}")


def run() -> tuple[int, list[str], int]:
    """返回 ``(通过数, 不一致明细, 缺基准数)``。"""
    cases = load_cases()
    ground = load_ground()
    ok = 0
    bad: list[str] = []
    missing = 0
    for i, f in enumerate(cases, start=1):
        exp = ground.get((f[0], i))
        if exp is None:
            missing += 1
            continue
        got = py_result(f)
        if exp == got:
            ok += 1
        else:
            bad.append(
                f"[{f[0]} #{i}] 输入={f[1:]!r}\n     期望 {exp!r}\n     实际 {got!r}"
            )
    return ok, bad, missing
