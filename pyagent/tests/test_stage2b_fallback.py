# -*- coding: utf-8 -*-
"""阶段 2B：模型降级层（冷却状态机 + 上游错误翻译）的回归测试。

分两组：

1. **对账组** —— 拿 ``tests/data/java_ground_fallback.txt`` 里 Java 真身的输出
   逐字节比对。这份基准是 jshell 跑出来的，不是手抄的期望值，所以它能证出
   「和 Java 一致」，而不只是「和我读到的源码一致」。
2. **状态机组** —— 注入假时钟测冷却/降级/提升的行为。这部分 Java 侧依赖
   ``System.currentTimeMillis()``，没法在两个进程间对拍，只能靠语义等价的测试守住。

全部不联网、不烧 token。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.fallback import (  # noqa: E402
    MODEL_DEAD_COOLDOWN_MS,
    MODEL_MAX_ATTEMPTS,
    ModelCooldown,
    ModelUnavailableException,
    describe,
    is_model_unavailable,
    parse_candidates,
    trim,
    upstream_error,
)

from ground_cases import (  # noqa: E402
    CASES,
    CAND_IN,
    GROUND_FILE,
    LONG_CASES,
    TRIM_IN,
    TRIM_MAX,
    b64dec,
    diff_position,
    load_ground,
)

GROUND = load_ground(GROUND_FILE)
_has_ground = bool(GROUND["case"])

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# ==================================================================
# 一、对账组：Python vs Java 真身
# ==================================================================

if _has_ground:

    def test_describe_matches_java_byte_for_byte():
        bad = []
        for i, (st, body) in enumerate(CASES):
            row = GROUND["case"].get(i, {})
            if "describe" not in row:
                continue
            exp, got = b64dec(row["describe"]), describe(st, body)
            if exp != got:
                bad.append(f"[{i}] status={st} body={body!r}｜{diff_position(exp, got)}")
        assert not bad, "describe 与 Java 不一致：\n" + "\n".join(bad)

    def test_is_model_unavailable_matches_java():
        bad = []
        for i, (st, body) in enumerate(CASES):
            row = GROUND["case"].get(i, {})
            if "isModelUnavailable" not in row:
                continue
            exp = row["isModelUnavailable"] == "true"
            got = is_model_unavailable(st, body)
            if exp != got:
                bad.append(f"[{i}] status={st} body={body!r} 期望 {exp} 实际 {got}")
        assert not bad, "isModelUnavailable 与 Java 不一致：\n" + "\n".join(bad)

    def test_describe_and_trim_still_match_java_where_java_was_native():
        """除「码元 vs 码点」「trim 空白集」这两条外，其余仍与 Java 基准一致。

        这两条**有意偏离**（见 :mod:`app.rag.textnorm` 的说明）：Java 已下线、
        基准无法重取，而它们恰恰是"不报错只悄悄偏"的那类规则，留着是纯负债。
        这里用白名单把这两条排掉，其余每条仍逐字节对账——
        **不是**把整组对账关掉，那会让"到底偏了哪些"变成盲区。
        """
        bad = []
        # 索引 4/5 是 NBSP / 全角空格，索引 9 是 emoji —— 见 ground_cases.TRIM_IN
        _TRIM_DEVIATIONS = {4, 5, 9}
        for i, s in enumerate(TRIM_IN):
            row = GROUND["trim"].get(i, {})
            if "out" not in row or i in _TRIM_DEVIATIONS:
                continue
            exp, got = b64dec(row["out"]), trim(s, TRIM_MAX[i])
            if exp != got:
                bad.append(f"[{i}] in={s!r}｜{diff_position(exp, got)}")
        assert not bad, "trim 与 Java 不一致（已排除有意偏离项）：\n" + "\n".join(bad)

        # describe 的长正文里只有 EMOJI_BODY 会分叉（码元 vs 码点）
        base = len(CASES)
        for j, (st, body) in enumerate(LONG_CASES):
            row = GROUND["long"].get(base + j, {})
            if "describe" not in row or "1f600" in body.encode("unicode_escape").decode():
                continue
            exp, got = b64dec(row["describe"]), describe(st, body)
            if exp != got:
                bad.append(f"[{j}]｜{diff_position(exp, got)}")
        assert not bad, "describe 截断与 Java 不一致（已排除 emoji）：\n" + "\n".join(bad)

    def test_parse_candidates_matches_java():
        bad = []
        for i, raw in enumerate(CAND_IN):
            row = GROUND["cand"].get(i, {})
            if "out" not in row:
                continue
            raw_out = b64dec(row["out"])
            exp = raw_out.split("\x01") if raw_out else []
            got = parse_candidates(raw)
            if exp != got:
                bad.append(f"[{i}] in={raw!r} 期望 {exp} 实际 {got}")
        assert not bad, "parseCandidates 与 Java 不一致：\n" + "\n".join(bad)

else:  # pragma: no cover
    def test_ground_truth_missing_is_visible():
        """基准文件丢了必须**显式失败**，而不是让这组测试悄悄全跳过。

         silently skipped 的对账等于没有对账 —— 这正是迁移期最容易自欺的地方。
        """
        pytest.fail(
            f"缺少 Java 基准 {GROUND_FILE}。先跑：\n"
            "  python qa/_javacp_prepare.py\n"
            "  jshell -q -R-Dground.out=<本文件绝对路径> "
            "--class-path '<classes>;<lib>/*' qa/_java_ground.jsh"
        )


# ==================================================================
# 二、语义组：不依赖基准文件的行为约束
# ==================================================================


def test_upstream_error_splits_retryable_from_not():
    """能靠换模型解决的走降级链，Key/账号类不做无谓重试。

    这条分界线一旦错了，表现是「Key 填错了，系统却挨个模型试一遍」，
    既慢又把「Key 错了」误导成「模型不对」。
    """
    assert isinstance(upstream_error(404, "ModelNotOpen"), ModelUnavailableException)
    assert isinstance(upstream_error(429, ""), ModelUnavailableException)
    assert isinstance(upstream_error(500, "boom"), ModelUnavailableException)
    # ---- 不换模型 ----
    assert not isinstance(upstream_error(401, "invalid_api_key"), ModelUnavailableException)
    assert not isinstance(upstream_error(401, ""), ModelUnavailableException)
    assert not isinstance(upstream_error(403, "forbidden"), ModelUnavailableException)
    assert not isinstance(upstream_error(429, "Arrearage"), ModelUnavailableException)
    # 关掉自动降级后，连 404 也不再换模型
    assert not isinstance(upstream_error(404, "ModelNotOpen", auto_fallback=False), ModelUnavailableException)


def test_trim_none_is_empty_string_not_exception():
    """「原因没给」是常态，不能抛异常把主链路打断。"""
    assert trim(None, 200) == ""
    assert trim("", 200) == ""


def test_parse_candidates_preserves_first_occurrence_order():
    assert parse_candidates("b,a,b") == ["b", "a"]
    assert parse_candidates("a\r\nb,a") == ["a", "b"]


def test_trim_strips_unicode_whitespace_not_just_ascii():
    """trim 删的是 **Unicode 空白**，不是 Java 那套「<= U+0020」。

    旧行为（对齐 ``String.trim()``）会留下全角空格 U+3000 与 NBSP U+00A0，
    于是「以全角空格开头的语料」分片起点比看起来靠后一个字符 —— 不报错，只是悄悄偏。
    Java 侧已下线，这里改为钉住 Python 原生语义。
    """
    # 全角空格是 Unicode 空白：首尾删掉、中间的被折叠成一个 ASCII 空格
    # （\s+ 不再带 re.ASCII，这是与旧实现最直观的一处差别）
    assert trim("\u3000全\u3000角\u3000", 200) == "全 角"
    assert trim("\u00a0nbsp\u00a0", 200) == "nbsp"        # NBSP 也算空白
    assert trim("  a  b\nc  ", 200) == "a b c"  # 空白折叠成一个空格


def test_describe_truncates_by_code_points_not_utf16_units():
    """截断按**码点**：一个 emoji 算一个字符，不是两个。

    以前按 UTF-16 码元（对齐 Java ``String.substring``），
    emoji 处会把截断位置整体前移一半 —— 这条正是"只看中文/ASCII 的话测试会一路绿灯"的地方。
    """
    s = "\U0001f600" * 300  # 300 个码点、600 个码元
    out = trim(s, 200)
    assert len(out) == 201, "截断到 200 个码点 + 一个省略号"
    assert out.endswith("…")
    assert out.count("\U0001f600") == 200


# ==================================================================
# 三、冷却状态机（假时钟）
# ==================================================================


class FakeClock:
    """毫秒时钟，可手动推进 —— 用来测 10 分钟冷却这种真实等不起的行为。"""

    def __init__(self, now: int = 1_000_000):
        self.now = now

    def __call__(self) -> int:
        return self.now

    def advance(self, ms: int) -> None:
        self.now += ms


def _cooldown(primary="m-main", candidates=("m-a", "m-b"), **kw) -> tuple:
    clock = kw.pop("clock", None) or FakeClock()
    cd = ModelCooldown(primary, candidates, clock=clock, **kw)
    return cd, clock


def test_cooling_model_is_ordered_last_but_not_dropped():
    """冷却的排到最后：不指望它，但也不至于完全不试。"""
    cd, clock = _cooldown(primary="m-main", candidates=("m-a",))
    cd.mark_dead("m-main", "未开通")
    # 冷却中的主模型仍在链里，只是排到了后面
    assert cd.model_chain() == ["m-a", "m-main"]


def test_chain_is_truncated_to_max_attempts():
    """撞太多只会把响应时间拖长，所以有硬上限 4 个。"""
    cd, _ = _cooldown(primary="m0", candidates=("m1", "m2", "m3", "m4", "m5"))
    assert len(cd.model_chain()) == MODEL_MAX_ATTEMPTS
    assert cd.model_chain()[0] == "m0"


def test_cooling_expires_after_cooldown_window():
    cd, clock = _cooldown()
    cd.mark_dead("m-a", "限流")
    assert cd.is_cooling("m-a")
    assert "m-a" in cd.unavailable()
    clock.advance(MODEL_DEAD_COOLDOWN_MS - 1)
    assert cd.is_cooling("m-a"), "差 1 毫秒也不该提前解除"
    clock.advance(1)
    assert not cd.is_cooling("m-a")
    assert cd.unavailable() == {}


def test_success_clears_cooldown_and_promotes():
    cd, clock = _cooldown(primary="m-main", candidates=("m-a",))
    cd.mark_dead("m-a", "限流")
    cd.note_success("m-a", "工具对话", reason="原模型不可用")
    assert cd.primary == "m-a", "用通的模型应被提升为当前模型"
    assert not cd.is_cooling("m-a")
    assert cd.last_switch is not None and "m-main → m-a" in cd.last_switch


def test_promote_false_keeps_primary_intact():
    """临时指定模型时**不能**把主模型悄悄换掉。

    否则后续所有分析都跟着降档，而没有任何报错 —— 表现出来就是
    「最近分析质量突然变差」，极难归因。
    """
    cd, _ = _cooldown(primary="m-main", candidates=("m-fast",))
    cd.note_success("m-fast", "快答", promote=False)
    assert cd.primary == "m-main"
    assert cd.last_switch is None


def test_with_fallback_switches_on_model_unavailable_only():
    """只有「换模型可能有用」的失败才换；Key 错这种直接抛出，不白试。"""
    cd, _ = _cooldown(primary="m-main", candidates=("m-a", "m-b"))
    tried = []

    def call(m):
        tried.append(m)
        if m == "m-main":
            raise ModelUnavailableException(404, "ModelNotOpen")
        return "ok:" + m

    assert cd.with_fallback("测试", call) == "ok:m-a"
    assert tried == ["m-main", "m-a"]
    assert cd.primary == "m-a"
    assert cd.is_cooling("m-main")


def test_with_fallback_does_not_retry_key_errors():
    """Key 错了就别换模型 —— 换了也是同一把坏 Key 再撞一次。"""
    cd, _ = _cooldown(primary="m-main", candidates=("m-a", "m-b"))
    tried = []

    def call(m):
        tried.append(m)
        raise RuntimeError(describe(401, "invalid_api_key"))

    with pytest.raises(RuntimeError):
        cd.with_fallback("测试", call)
    assert tried == ["m-main"], "401 不该触发换模型"


def test_with_fallback_second_round_uses_discovered_models():
    """第一轮全败后才拉账号可用列表补一轮 —— 正常情况不多花一次请求。"""
    discovered = ["d1", "d2", "d3"]
    calls = {"n": 0}

    def discover(force):
        calls["n"] += 1
        return discovered

    cd = ModelCooldown("m-main", ("m-a",), discover=discover, clock=FakeClock())
    tried = []

    def call(m):
        tried.append(m)
        raise ModelUnavailableException(404, "ModelNotOpen")

    with pytest.raises(RuntimeError) as ei:
        cd.with_fallback("测试", call)
    # 第一轮：主 + 候选（都没通）；第二轮：账号可用列表里还没试过的
    assert tried == ["m-main", "m-a", "d1", "d2", "d3"]
    assert calls["n"] == 1, "第二轮只拉一次，不该每个候选都拉一遍"
    assert "已尝试 5 个模型" in str(ei.value)


def test_second_round_prefers_newest_chat_model_over_upstream_order():
    """第二轮补试必须按「新且强」挑，**不能照上游返回的原始顺序**取前几个。

    真实背景（2026-09-22 线上实测）：方舟 ``GET /v3/models`` 按上架时间
    **从旧到新**返回，且列的是**平台全量**而不是本账号开通的 —— 134 个里
    91 个调用直接 404 NotFound、16 个 404 ModelNotOpen，真能调通的只有 14 个。
    改之前这里按原序取前 4 个，于是每次补试都是 2024 年的老模型，
    全部 404，对外表现成"自动降级从来不生效"，排查时极易误判为"上游没有可用模型"。
    """
    discovered = [
        "doubao-lite-128k-240428",       # 最老，排在前面
        "doubao-embedding-text-240715",  # 向量化：根本不该进降级候选
        "glm-5-3-flash-260828",          # 较新
        "doubao-seed-2-1-pro-260915",    # 最新最强
    ]
    cd = ModelCooldown("m-main", (), discover=lambda force: discovered, clock=FakeClock())
    tried = []

    def call(m):
        tried.append(m)
        raise ModelUnavailableException(404, "ModelNotOpen")

    with pytest.raises(RuntimeError):
        cd.with_fallback("测试", call)

    assert tried[0] == "m-main", "主模型永远先试"
    assert tried[1:] == [
        "doubao-seed-2-1-pro-260915",
        "glm-5-3-flash-260828",
        "doubao-lite-128k-240428",
    ], "第二轮必须按「新且强」排序"
    assert "doubao-embedding-text-240715" not in tried, \
        "embedding 模型不是对话底座，试它必然 404，纯浪费一次往返"


def test_with_fallback_raises_when_no_model_configured():
    cd = ModelCooldown("", (), discover=lambda force: [], clock=FakeClock())
    with pytest.raises(RuntimeError) as ei:
        cd.with_fallback("测试", lambda m: "x")
    assert "没有可用模型" in str(ei.value)
    assert "AI_MODEL" in str(ei.value), "失败信息里要给出可操作的处理办法"


def test_unavailable_reason_is_trimmed_to_200():
    """原因可能是一整段上游 JSON，冷却表里只留摘要。"""
    cd, _ = _cooldown()
    cd.mark_dead("m-a", "x" * 500)
    reason = cd.unavailable()["m-a"]
    assert reason.endswith("…")
    assert len(reason) <= 201


def test_add_candidate_is_idempotent():
    cd, _ = _cooldown(primary="m0", candidates=("m1",))
    assert cd.add_candidate("m2") is True
    assert cd.add_candidate("m2") is False
    assert cd.add_candidate("") is False
