"""阶段 2 对账测试：模型目录（零 token 的确定性部分）。

基准来源
--------
``tests/data/ark_models_baseline.json`` 是 **Java 版 ``/api/ai/models?refresh=true``
的实测快照**：账号真实可用的 133 个模型（按方舟返回顺序）+ Java 算出的推荐前 20。

也就是说这些测试比的是"同一批真实输入下，Python 的筛选/打分/排序结果
是否与 Java 逐项相等" —— 不是我自己编的期望值。

刷新基准：起 Java(8080) 后跑 ``python qa/_baseline_models.py``。
基准里的模型上下架变化会让 ``available`` 变化，但 ``recommended`` 的对账关系不变。
"""

from __future__ import annotations

import json
import os

import pytest

from app.ai import keys as K
from app.ai.model_catalog import (
    MODEL_DATE,
    describe,
    is_chat_model,
    is_model_unavailable,
    model_score,
    parse_model_ids,
    rank_chat_models,
    recommended_models,
)
from app.ai.discovery import StaticDiscovery

_BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ark_models_baseline.json")


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not os.path.exists(_BASELINE):
        pytest.skip("基准文件不存在：起 Java 后跑 qa/_baseline_models.py 生成")
    with open(_BASELINE, encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# 1. 与 Java 实测结果逐项对账（本文件最有价值的一组）
# =====================================================================


def test_recommended_top20_matches_java_baseline(baseline):
    """推荐前 20 必须与 Java 逐项相等 —— 筛选、打分、稳定排序三者同时正确才可能成立。"""
    got = recommended_models(baseline["available"], 20)
    assert got == baseline["recommended"], (
        "与 Java 版推荐列表不一致。\n"
        f"java={baseline['recommended']}\n py ={got}"
    )


def test_rank_prefix_equals_recommended(baseline):
    """全量排序的前 20 项 == 推荐列表（说明 limit 只是截断，没有二次重排）。"""
    ranked = rank_chat_models(baseline["available"])
    assert ranked[:20] == baseline["recommended"]


def test_dropped_count_matches_java(baseline):
    """133 个里 Java 剔掉了 47 个 → 留 86 个。这个数量级对上了，说明筛选口径没跑偏。"""
    available = baseline["available"]
    ranked = rank_chat_models(available)
    assert len(available) == 133
    assert len(ranked) == 86


def test_available_order_is_preserved_for_equal_scores(baseline):
    """同分模型必须保持方舟返回的原始顺序（两边都是稳定排序）。

    这条守的是"把稳定排序换成不稳定排序"这种改动 —— 它不会报错，
    只会让下拉框顺序每次刷新都变。构造法：把 available 逆序喂进去，
    同分组的相对顺序应当跟着反过来（即"跟随输入顺序"），而不是稳定地锁在原位。
    """
    available = baseline["available"]
    forward = rank_chat_models(available)
    backward = rank_chat_models(list(reversed(available)))

    def same_score_groups(seq):
        groups = {}
        for m in seq:
            groups.setdefault(model_score(m), []).append(m)
        return groups

    gf, gb = same_score_groups(forward), same_score_groups(backward)
    # 同分组里，正序输入应得到正序输出，逆序输入应得到逆序输出
    for score, members in gf.items():
        assert gb[score] == list(reversed(members)), f"同分组（score={score}）没有跟随输入顺序，排序不稳定"


# =====================================================================
# 2. isChatModel：真实被剔掉的 47 个各代表一类
# =====================================================================


@pytest.mark.parametrize(
    "mid",
    [
        "doubao-embedding-text-240515",          # 向量化
        "doubao-embedding-large-text-250515",    # 向量化
        "doubao-embedding-vision-251215",        # 向量化（vision 变体）
        "doubao-seedream-3-0-t2i-250415",        # 文生图
        "doubao-seededit-3-0-i2i-250628",        # 图生图
        "doubao-seedance-1-0-pro-250528",        # 视频生成
        "wan2-1-14b-i2v-250225",                 # 图生视频
        "hyper3d-gen2-260112",                   # 3D
        "hitem3d-2-0-251223",                    # 3D（含 "3d" 子串）
        "doubao-pro-32k-character-240528",       # 角色扮演
        "doubao-lite-4k-pretrain-character-240516",  # 预训练基座 + 角色扮演
        "doubao-1.5-ui-tars-250328",             # GUI 代理
    ],
)
def test_non_chat_models_are_rejected(mid):
    assert is_chat_model(mid) is False


@pytest.mark.parametrize(
    "mid",
    [
        "deepseek-v4-1-flash-260910",
        "doubao-seed-1-6-250615",
        "doubao-1-5-pro-32k-250115",
        "doubao-pro-32k-240828",
    ],
)
def test_chat_models_are_accepted(mid):
    assert is_chat_model(mid) is True


def test_chat_filter_edge_cases():
    """边界：空 / None / 以及几个容易误判的。"""
    assert is_chat_model(None) is False
    assert is_chat_model("") is False
    assert is_chat_model("   ") is False
    # 大小写不敏感：Java 先 toLowerCase 再判定
    assert is_chat_model("Doubao-Embedding-Text-240515") is False
    assert is_chat_model("SEEDREAM-3-0-T2I") is False


# =====================================================================
# 3. modelScore
# =====================================================================


def test_score_prefers_newer_generation():
    """1-6 代次应高于旧的 lite 模型 —— 这就是当初要打分的原因。"""
    assert model_score("doubao-seed-1-6-250615") > model_score("doubao-lite-128k-240428")


def test_score_includes_version_date_monotonically():
    """同代次下，版本日期越大分越高（id 尾部的 YYMMDD）。"""
    assert model_score("doubao-pro-32k-240828") > model_score("doubao-pro-32k-240515")


def test_score_uses_last_date_group_not_first():
    """版本日期取**最后一组** 6 位数字。

    构造一个前面带 6 位数字的 ID：``abc123456-250101``。
    取最后一组 → 250101（日期项大）；取第一组 → 123456（日期项小）。
    """
    last = model_score("abc123456-250101")
    first_style = model_score("abc250101-123456")
    assert last > first_style, "似乎取了第一组日期，Java 取的是最后一组"


def test_score_penalises_thinking_and_tiny_context():
    base = "doubao-seed-1-6-250615"
    assert model_score(base.replace("250615", "250615-thinking")) < model_score(base)
    assert model_score("doubao-seed-1-6-250615-4k") < model_score(base)


def test_model_date_regex_is_ascii():
    """``re.ASCII`` 守则：全角数字不该被当成日期。

    阶段 1 已经吃过一次"Java 的 \\d 只认 ASCII、Python 默认认 Unicode"的亏
    （见 pyagent/README.md 第 1 条）。这里对 ``model_score`` 用的同一个模式再守一遍。
    """
    assert MODEL_DATE.findall("doubao-２５０６１５") == []  # 全角
    assert MODEL_DATE.findall("doubao-250615") == ["250615"]


# =====================================================================
# 4. isModelUnavailable：换 / 不换模型的分界线
# =====================================================================


@pytest.mark.parametrize(
    "status,body,expected",
    [
        # Key / 账号的问题 → 不换（换模型只会拿同一把坏 Key 再撞一次）
        (401, '{"error":{"code":"invalid_api_key"}}', False),
        (403, "", False),
        (200, '{"code":"Arrearage"}', False),
        (400, "overdue-payment", False),
        # 模型 / 额度的问题 → 换
        (404, "", True),
        (429, "", True),
        (500, "", True),
        (503, "", True),
        (200, '{"code":"ModelNotOpen"}', True),
        (200, "Model.NotExist", True),
        (200, "model_not_found", True),
        (200, "InvalidEndpointOrModel", True),
        (200, "do not have access", True),
        (200, "not activated the model", True),
        # 普通业务错误 → 不换
        (400, "bad request", False),
    ],
)
def test_model_unavailable_matrix(status, body, expected):
    assert is_model_unavailable(status, body) is expected


def test_invalid_key_short_circuits_even_on_429():
    """判定顺序：Key/账号类**先于**状态码判断。

    Java 的 if 链是先看 401/403 与 invalid_api_key/Arrearage，再看 404/429/5xx。
    如果顺序写反，一句含 invalid_api_key 的 429 也会触发换模型 —— 把"Key 错了"
    误诊成"模型不对"，然后在候选链上白撞一圈。
    """
    assert is_model_unavailable(429, '{"error":"invalid_api_key"}') is False
    assert is_model_unavailable(404, "Arrearage") is False


# =====================================================================
# 5. describe：文案分支
# =====================================================================


def test_describe_detects_arrearage():
    s = describe(400, '{"error":{"code":"Arrearage"}}')
    assert "欠费" in s and "余额" in s


def test_describe_detects_bad_key():
    s = describe(401, '{"error":{"code":"invalid_api_key"}}')
    assert "API Key 无效" in s
    # 401 即使正文没提 invalid_api_key 也应命中
    assert "API Key 无效" in describe(401, "unauthorized")


def test_describe_detects_model_not_open_before_404():
    """``ModelNotOpen`` 必须优先于 ``status == 404`` 分支。

    方舟把「模型未开通」报成 404，若先判 404 就会走进"模型或接口不存在"那条，
    把一件"去控制台开一下就好"的事说成"接口不存在"。
    """
    s = describe(404, '{"error":{"code":"ModelNotOpen"}}')
    assert "尚未开通" in s
    assert "接口不存在" not in s.split("——")[0] or "很容易被误读" in s


def test_describe_404_generic():
    s = describe(404, '{"error":"nope"}')
    assert "模型或接口不存在" in s


def test_describe_429_and_fallback():
    assert "限流" in describe(429, "")
    assert "HTTP 400" in describe(400, "bad request")


def test_describe_truncates_long_body():
    """超长正文截断到 200，避免把整页 HTML 塞进接口响应。"""
    s = describe(400, "x" * 900)
    assert s.endswith("x" * 200)
    assert "x" * 201 not in s


# =====================================================================
# 6. parseModelIds：两种返回结构
# =====================================================================


def test_parse_model_ids_object_form():
    raw = '{"data":[{"id":"a","object":"model"},{"id":"b"},{"id":"a"}]}'
    assert parse_model_ids(raw) == ["a", "b"]  # 去重且保持顺序


def test_parse_model_ids_string_form():
    assert parse_model_ids('{"data":["x"," y ","","z"]}') == ["x", "y", "z"]


def test_parse_model_ids_tolerates_junk():
    """结构不认识就当没发现到 —— 不能因为列表格式变了就让主链路崩掉。"""
    assert parse_model_ids("not json at all") == []
    assert parse_model_ids('{"data":"a string"}') == []
    assert parse_model_ids('{"other":[{"id":"a"}]}') == []


def test_parse_model_ids_skips_null_ids():
    assert parse_model_ids('{"data":[{"id":null},{"id":"ok"}]}') == ["ok"]


def test_parse_model_ids_boolean_matches_java_string_value():
    """Java 的 ``String.valueOf(true)`` 是 ``"true"``，Python 的 ``str(True)`` 是 ``"True"``。"""
    assert parse_model_ids('{"data":[{"id":true}]}') == ["true"]


# =====================================================================
# 7. keys 工具
# =====================================================================


def test_mask_key_shape():
    assert K.mask_key(None) == "(未配置)"
    assert K.mask_key("") == "(未配置)"
    assert K.mask_key("   ") == "(未配置)"
    k = "ark-b5a7de73-55a8-4101-82b2-a220aec665c9-e0755"
    m = K.mask_key(k)
    assert m.startswith("len=46 ark-b5a***")
    assert m.endswith("0755")


def test_mask_key_short_key_does_not_raise():
    """短 Key 不能让 substring 越界（Java 用 min/max 夹住，Python 靠切片天然安全）。"""
    assert K.mask_key("abc") == "len=3 abc***abc"


def test_resolve_source_prefers_explicit():
    src = K.resolve_source("explicit-key", "openai-key")
    assert src is not None and src.name == "app.ai.api-key / AI_API_KEY" and src.value == "explicit-key"


def test_resolve_source_falls_back_to_openai():
    src = K.resolve_source("", "openai-key")
    assert src is not None and src.name == "OPENAI_API_KEY"


def test_resolve_source_returns_none_when_empty():
    assert K.resolve_source("", "") is None
    assert K.resolve_source(None, None) is None
    assert K.resolve_source("   ", "  ") is None


def test_resolve_source_trims():
    src = K.resolve_source("  k  ", "")
    assert src is not None and src.value == "k"


@pytest.mark.parametrize(
    "raw,dflt,expected",
    [
        (None, True, True),
        ("", True, True),
        ("  ", False, False),
        ("true", False, True),
        ("TRUE", False, True),
        ("1", False, True),
        ("yes", False, True),
        ("on", False, True),
        ("false", True, False),
        ("0", True, False),
        ("no", True, False),
        ("off", True, False),
        # 未列出的值回落到默认值，而不是 false —— 这是 Java 的行为，容易写错
        ("maybe", True, True),
        ("maybe", False, False),
    ],
)
def test_bool_or(raw, dflt, expected):
    assert K.bool_or(raw, dflt) is expected


def test_strip_trailing_slash():
    assert K.strip_trailing_slash(None) == ""
    assert K.strip_trailing_slash("") == ""
    assert K.strip_trailing_slash("https://a/v3/") == "https://a/v3"
    assert K.strip_trailing_slash("https://a/v3") == "https://a/v3"
    # 只去一个，不做 rstrip 式的批量删除（与 Java 的 endsWith + substring 一致）
    assert K.strip_trailing_slash("https://a/v3//") == "https://a/v3/"


# =====================================================================
# 8. discovery：缓存语义
# =====================================================================


def test_static_discovery_never_hits_network():
    d = StaticDiscovery(["a", "b"])
    assert d.is_configured is True
    assert d.list_available_models() == ["a", "b"]
    assert d.list_available_models(force=True) == ["a", "b"]
    assert d.cached_models == ["a", "b"]


def test_discovery_skips_request_without_key():
    """没 Key 就不发请求 —— 否则每次调用都白打一个必然 401 的请求。"""
    from app.ai.discovery import ModelDiscovery

    d = ModelDiscovery(base_url="https://example.invalid/api/v3", api_key="")
    assert d.is_configured is False
    assert d.list_available_models() == []
    assert d.last_issue is not None and "未配置" in d.last_issue


def test_discovery_negative_cache_prevents_repeat_calls():
    """失败也要做负缓存：第二次调用不应再打网络。

    用一个会抛异常的 transport 计数：第一次调用触发一次请求（失败），
    紧接着第二次调用必须**不再**触发。
    """
    import httpx

    from app.ai.discovery import ModelDiscovery

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("boom", request=request)

    d = ModelDiscovery(
        base_url="https://example.invalid/api/v3",
        api_key="k",
        transport=httpx.MockTransport(handler),
    )
    assert d.list_available_models() == []
    first = calls["n"]
    assert first == 1
    assert d.list_available_models() == []  # 触发负缓存分支
    assert calls["n"] == first, "负缓存没生效，第二次调用又打了一次网络"
    # force 应当绕过缓存
    d.list_available_models(force=True)
    assert calls["n"] == first + 1


def test_discovery_unwraps_nested_data_shape():
    """真实响应结构 ``{"data":[{"id":...}]}`` 能被解析出来。"""
    import httpx

    from app.ai.discovery import ModelDiscovery

    body = json.dumps({"data": [{"id": "m1"}, {"id": "m2"}, {"object": "model"}]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, request=request)

    d = ModelDiscovery(
        base_url="https://example.invalid/api/v3",
        api_key="k",
        transport=httpx.MockTransport(handler),
    )
    assert d.list_available_models() == ["m1", "m2"]
