"""Key / 配置读取的小工具 —— 移植 ``LlmClient`` 里的静态私有方法。

这些方法看着琐碎，但每一条都对应一次踩过的坑（见各自 docstring）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def not_blank(v: Optional[str]) -> bool:
    """Java: ``v != null && !v.isBlank()``。

    注意 ``isBlank()`` 与 Python 的 ``v.strip() == ""`` 语义一致（都按 Unicode 空白算），
    这里不做 ASCII 化 —— 与正则那处不同，这类空白判定两边本来就没有差异。
    """
    return v is not None and v.strip() != ""


@dataclass(frozen=True)
class KeySource:
    """Key 及其来源，便于把「用错 Key」这类问题一眼定位。"""

    name: str
    value: str


def resolve_source(explicit: Optional[str], openai: Optional[str]) -> Optional[KeySource]:
    """选 Key：显式配置的优先，其次环境变量。

    踩过的坑：机器上往往同时存在好几把来路不同的环境变量 Key（比如某个别的服务装好的
    ``OPENAI_API_KEY``）。只看变量名不看是谁装的，就会被拿去打完全无关的服务，
    表现是莫名其妙的 401 ``invalid_api_key``。

    所以这里**把来源记下来一起返回**，日志里打的是 "来源 app.ai.api-key / AI_API_KEY"，
    而不是只打一个脱敏后的 Key —— 后者没法区分"到底读到了哪一把"。
    """
    order = []
    if not_blank(explicit):
        order.append(KeySource("app.ai.api-key / AI_API_KEY", (explicit or "").strip()))
    if not_blank(openai):
        order.append(KeySource("OPENAI_API_KEY", (openai or "").strip()))
    for k in order:
        if not_blank(k.value):
            return k
    return None


def mask_key(k: Optional[str]) -> str:
    """脱敏后的 Key 摘要，日志/接口里只用它。

    Java 原文::

        "len=" + k.length() + " " + k.substring(0, Math.min(7, k.length()))
                + "***" + k.substring(Math.max(0, k.length() - 4))

    .. NOTE::
       长度与切片按**码点**（Python 原生）。对纯 ASCII 的 Key（真实场景都是）
       与历史上的 Java 版完全一致；若有人把中文塞进 Key，长度与前缀会和旧版不同 ——
       后果只是日志摘要的字数，不额外做转码。
    """
    if k is None or k.strip() == "":
        return "(未配置)"
    n = len(k)
    return f"len={n} {k[: min(7, n)]}***{k[max(0, n - 4) :]}"


def bool_or(raw: Optional[str], dflt: bool) -> bool:
    """宽松地把配置读成布尔值：空/null 用默认值，其余按宽松真值表。

    存在的理由：``@Value("${x:true}") boolean`` 在「环境变量已定义但为空」时不会用到默认值，
    而是拿到空串，随后抛 ``TypeMismatchException: Invalid boolean value []`` ——
    报错出现在 Bean 创建阶段，看不出是哪个变量干的，非常难查。这里直接把这个坑封掉。

    Java 原文走的是 ``Boolean.parseBoolean`` 的宽松分支：
    ``true`` / ``1`` / ``yes`` / ``on`` → 真，``false`` / ``0`` / ``no`` / ``off`` → 假，
    其余（含空串）→ 默认值。注意**未列出的字符串也回落到默认值**，而不是 false ——
    例如 ``boolOr("maybe", True)`` 是 ``True``。
    """
    if raw is None or raw.strip() == "":
        return dflt
    t = raw.strip()
    if t.lower() == "true" or t == "1" or t.lower() == "yes" or t.lower() == "on":
        return True
    if t.lower() == "false" or t == "0" or t.lower() == "no" or t.lower() == "off":
        return False
    return dflt


def strip_trailing_slash(b: Optional[str]) -> str:
    """去掉 base-url 末尾的斜杠。

    没有这一步，``base + "/models"`` 会拼成 ``.../v3//models`` —— 多数服务端能容忍，
    但方舟对部分路径会直接 404，而 404 又正好被 :func:`is_model_unavailable`
    判成"模型不可用"，表现为**莫名其妙地不停换模型**。所以这个一行的函数值得单列。
    """
    if b is None:
        return ""
    return b[:-1] if b.endswith("/") else b
