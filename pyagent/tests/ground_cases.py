# -*- coding: utf-8 -*-
"""
阶段 2B 对账的**共用输入定义**。

用例清单只在这里写一份：``qa/_parity_fallback.py``（人工跑对账）和
``tests/test_stage2b_fallback.py``（回归测试）都从这导入。
两边各写一份的话，改了一边另一边就悄悄测的是别的东西。

基准文件也只有一份：``tests/data/java_ground_fallback.txt``，
由 ``qa/_java_ground.jsh`` 用 jshell 跑 Java 真身生成（覆盖式重写，不是追加）。
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
GROUND_FILE = HERE / "data" / "java_ground_fallback.txt"

# ---- describe / isModelUnavailable 的输入（与 _java_ground.jsh 中的 cases 逐一对应）----
CASES: List[Tuple[int, Optional[str]]] = [
    (401, ""),
    (401, "invalid_api_key: bad key"),
    (401, "Arrearage"),
    (401, "overdue-payment"),
    (403, ""),
    (403, "permission denied"),
    (404, "ModelNotOpen"),
    (404, "not activated the model"),
    (404, "Model.NotExist"),
    (404, "model_not_found"),
    (404, ""),
    (404, "InvalidEndpointOrModel"),
    (429, ""),
    (429, "rate limit exceeded"),
    (500, "internal error"),
    (503, "upstream unavailable"),
    (400, "bad request"),
    (200, "Arrearage"),
    (200, "overdue-payment"),
    (400, "you do not have access to this model"),
    (418, ""),
    (0, ""),
    (429, "Arrearage and rate limit"),          # Arrearage 分支在 429 之前
    (404, '{"error":{"code":"ModelNotOpen","message":"x"}}'),
    (401, "invalid_api_key and Arrearage"),     # Arrearage 分支在前
    (0, None),
]

LONG_BODY = "abcdefghij" * 30      # 300 ASCII
CJK_BODY = "风险" * 120             # 240 个中文字符（BMP，1 码点 = 1 码元）
EMOJI_BODY = "\U0001f600" * 120    # 120 emoji = 240 个 UTF-16 码元（astral，1 码点 = 2 码元）
LONG_CASES: List[Tuple[int, str]] = [
    (400, LONG_BODY),
    (400, CJK_BODY),
    (400, EMOJI_BODY),
]

TRIM_IN: List[Optional[str]] = [
    None, "", "   ", "  a  b\nc  ",
    " non breaking ",
    "　全　角　",
    "  line1\t\tline2\r\nline3  ",
    LONG_BODY, CJK_BODY, EMOJI_BODY,
]
TRIM_MAX = [10, 10, 10, 200, 200, 200, 200, 200, 200, 200]

CAND_IN: List[Optional[str]] = [
    None, "", "   ", "a,b,c", "a, b ,c", "a\nb\r\nc", "a,,b,,a",
    " a , , b ", ",", "a,a,a", "ep-1,ep-2,ep-1",
]


def b64dec(s: str) -> str:
    return base64.b64decode(s).decode("utf-8")


def load_ground(path: Path = GROUND_FILE) -> Dict[str, Dict[int, Dict[str, str]]]:
    """读 jshell 产出的基准文件。

    返回 ``{kind: {idx: {field: value}}}``，``kind`` 为 case/long/trim/cand。
    """
    out: Dict[str, Dict[int, Dict[str, str]]] = {"case": {}, "long": {}, "trim": {}, "cand": {}}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        kind, idx, field = parts[0].lower(), parts[1], parts[2]
        val = parts[3] if len(parts) > 3 else ""
        try:
            i = int(idx)
        except ValueError:
            continue
        out.setdefault(kind, {}).setdefault(i, {})[field] = val
    return out


def diff_position(a: str, b: str) -> str:
    """首个不一致的位置：区分「量纲算错」和「文案抄错」。"""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return f"首差于第 {i} 个字符（期望 {a[i]!r} / 实际 {b[i]!r}）"
    return f"前缀相同，长度不同（期望 {len(a)} / 实际 {len(b)}）"
