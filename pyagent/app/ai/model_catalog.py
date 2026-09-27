"""模型目录：**纯规则**部分 —— 与 ``LlmClient`` 逐行对齐，零网络、零 token。

移植来源：``service/ai/LlmClient.java`` 的
``isChatModel`` / ``modelScore`` / ``rankChatModels`` / ``isModelUnavailable`` /
``parseModelIds`` / ``describe`` / ``recommendedModels``。

这一层之所以值得单独切出来，是因为它**完全确定性**：
拿真实账号的 133 个模型 ID 喂进去，Java 与 Python 的排序结果必须逐字节相等。
不用发一次推理请求就能把差异钉出来 —— 这是整个阶段 2 里性价比最高的一块。
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Mapping, Optional, Sequence

# ---------------------------------------------------------------- 正则

#: ``LlmClient`` 里的 ``Pattern.compile("(\\d{6})")``。
#!
#! ``re.ASCII`` 是**必须**的：Java 的 ``\d`` 只认 ASCII 数字，Python 默认 Unicode 感知。
#! 模型 ID 目前都是 ASCII 所以看不出差别，但阶段 1 已经吃过一次"静默行为漂移"的亏
#! （见 pyagent/README.md 第 1 条），这里统一加标志，不给以后留雷。
MODEL_DATE = re.compile(r"(\d{6})", re.ASCII)


# ---------------------------------------------------------------- 模型筛选


def is_chat_model(model_id: Optional[str]) -> bool:
    """过滤掉不适合当「对话底座」的模型，避免自动降级时切到一个不会聊天的。

    实测本账号 ``GET /v3/models`` 返回 133 个模型，里面混着向量化、
    图像/视频/3D 生成、角色扮演、GUI 代理 —— 直接放进下拉框根本没法选，
    更要命的是降级可能切到 ``doubao-seedream-3-0-t2i-*``（文生图）这种压根不接受对话的。

    判定顺序与 Java 完全一致（先排除后判定，都是"只要命中一条就出局"）。
    """
    if model_id is None or model_id.strip() == "":
        return False
    s = model_id.lower()
    if (
        "embedding" in s
        or "rerank" in s
        or "tts" in s
        or "asr" in s
        or "speech" in s
        or "ocr" in s
        or "music" in s
    ):
        return False
    # 图像 / 视频 / 3D 生成类
    if "seedream" in s or "seededit" in s or "seedance" in s:
        return False
    if (
        s.startswith("wan2")
        or "-i2v" in s
        or "-t2v" in s
        or "-i2i" in s
        or "flf2v" in s
        or "video" in s
        or "image" in s
    ):
        return False
    if "3d" in s or "hyper3d" in s:
        return False
    # 角色扮演 / 预训练基座 / GUI 代理：都不是给「风险研判」用的
    if "character" in s or "pretrain" in s or "ui-tars" in s:
        return False
    return True


# ---------------------------------------------------------------- 打分排序


def model_score(model_id: str) -> int:
    """给候选模型打分，用来决定「自动挑一个」时的优先顺序。

    踩过的坑：方舟 ``GET /v3/models`` 返回的列表是**按上架时间从旧到新**排的，
    直接取第一个会挑到 2024 年的 ``doubao-lite-128k-240428`` 这种又老又弱的模型 ——
    「能跑」但结论质量明显下降，而且很难察觉。所以这里显式打分：
    能力代次 + 版本日期 - 不适合本场景的项（角色扮演 / 预训练基座 / 图形界面代理 /
    超小上下文 / 推理型太慢）。

    .. NOTE::
       版本日期取的是 **最后一组** 6 位数字（Java 里是 ``while (dm.find()) date = ...``，
       循环覆盖，留下的就是最后一个）。写成 ``findall()[0]`` 会取到第一组 ——
       对于 ``deepseek-v3-1-250821`` 这类只有一组的没差别，但
       ``xxx-2024-250821`` 就会取错，且错得毫无征兆。
    """
    s = model_id.lower()
    score = 0
    if s.startswith("deepseek-v3") or s.startswith("deepseek-v3-1") or s.startswith("deepseek-v3-2"):
        score += 30
    if "1-6" in s or "seed-1-6" in s:
        score += 28
    if "1-5-pro" in s or "1.5-pro" in s:
        score += 26
    if "pro" in s:
        score += 12
    if "lite" in s or "flash" in s or "mini" in s or "small" in s:
        score -= 18
    # 推理型（thinking / R1）在降级场景里更慢，容易吃满 3 分钟超时，降权
    if "thinking" in s or "-r1-" in s or s.endswith("-r1"):
        score -= 25
    if "character" in s or "pretrain" in s or "ui-tars" in s:
        score -= 40
    if "-4k" in s:
        score -= 20
    # 版本日期（id 里最后一组 6 位数字，YYMMDD）：数值越大越新，直接加进去做单调排序
    date = 0
    for m in MODEL_DATE.findall(s):
        date = int(m)
    return score + date


def rank_chat_models(ids: Sequence[str]) -> List[str]:
    """把「可用模型」里能对话的部分按优先级排好（新且强的在前）。

    .. NOTE::
       两边都必须是**稳定排序**：Java 用 ``List.sort``（TimSort，稳定），
       Python 用 ``sorted``（稳定）。分数相同的模型会保留**原始列表顺序**
       —— 而方舟的原始列表是按上架时间排的，所以"同分保持原序"本身就是有意义的语义。
       若把这里换成不稳定排序，同一批模型会跑出不同顺序，
       表现为"每次点刷新下拉框顺序都在变"，很难归因。
    """
    out = [x for x in ids if is_chat_model(x)]
    out.sort(key=model_score, reverse=True)
    return out


def recommended_models(ids: Sequence[str], limit: int) -> List[str]:
    """「推荐用它」的候选模型：能对话的按「新且强」排序后取前 ``limit`` 个。

    Java::

        if (!modelDiscovery) return List.of();          // 开关由调用方判断
        List<String> ranked = rankChatModels(listAvailableModels(refresh));
        return ranked.size() > limit ? new ArrayList<>(ranked.subList(0, limit)) : ranked;
    """
    ranked = rank_chat_models(ids)
    return ranked[:limit] if len(ranked) > limit else ranked


# ---------------------------------------------------------------- 错误分类
#
# .. IMPORTANT::
#    这两个函数**曾经在本文件里有一份实现**，并且已经和 :mod:`fallback` 的那份漂移了：
#    这里的 ``describe`` 按 Python 码点截断（``b[:200]``），而 Java 是 ``b.length() > 200``
#    （UTF-16 码元）—— 正文里出现 emoji 这类 astral 字符时两者截出不同的长度。
#
#    漂移是在拿 jshell 跑 Java 真身取基准（``qa/_java_ground.jsh``）之后才暴露的，
#    靠读代码看不出来。**同一份逻辑只允许有一处实现**，所以这里改成再导出。
#
#    .. NOTE::
#       ``fallback`` 那一份现在也改成按**码点**了（见 :mod:`app.rag.textnorm`），
#       所以"码点 vs 码元"这处差异已随之消失；留着这段只是说明「为什么不能有第二份实现」。

from .fallback import describe, is_model_unavailable  # noqa: E402,F401


# ---------------------------------------------------------------- 列表解析


def _java_string_value(v: Any) -> str:
    """对齐 Java 的 ``String.valueOf(Object)``。

    只为一处差异而存在：Java 的布尔值转字符串是 ``"true"`` / ``"false"``（小写），
    Python 的 ``str(True)`` 是 ``"True"``。模型列表里出现布尔 ID 的概率极低，
    但既然要对账，就不要留一个"理论上会不一致"的口子。
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def parse_model_ids(json_text: str) -> List[str]:
    """兼容两种返回结构：``{"data":[{"id":"x"}]}`` 与 ``{"data":["x"]}``。

    Java 原文能容错是因为它先 ``readValue(json, Map.class)`` 再判 ``data instanceof List``，
    任何结构不认识的情况都被 try/catch 吃掉、返回空列表 ——
    "发现不到模型"是可接受的降级，**不能**因为列表格式变了就让主链路崩掉。
    这里保持同样的宽容度。
    """
    ids: List[str] = []
    try:
        r = json.loads(json_text)
    except Exception:
        return ids
    if not isinstance(r, Mapping):
        return ids
    data = r.get("data")
    if isinstance(data, list):
        for o in data:
            if isinstance(o, Mapping):
                raw_id = o.get("id")
            else:
                raw_id = o
            if raw_id is None:
                continue
            t = _java_string_value(raw_id).strip()
            if t != "" and t not in ids:
                ids.append(t)
    return ids


def model_ids_from_response(payload: Mapping[str, Any]) -> List[str]:
    """从已解析的 dict 里取模型 ID。

    :func:`parse_model_ids` 走的是"原始 JSON 文本"这条路（与 Java 一致）；
    这个函数是给已经解析过响应的调用方用的便捷入口，语义完全相同。
    """
    return parse_model_ids(json.dumps(payload, ensure_ascii=False))


def chat_model_ids(ids: Sequence[str]) -> List[str]:
    """只留下能对话的（不分排序），给"看看排除了什么"这类排查用。"""
    return [x for x in ids if is_chat_model(x)]
