"""让 ``import app.*`` 在仓库内无需安装即可工作，并守住「测试不得污染真实库」。

**真实事故（2026-09-22 定位）**：``test_stage5_orchestration`` 需要验证降级、路由、
缓存这些只有真跑一次 ``AgentService.analyze`` 才看得出来的行为，于是直接调用它 ——
而它内部会**落库**。结果每跑一次全量 pytest，真实 MySQL 的 ``ai_analysis`` 就多出
7 条名为「客户流失率为何上升」/「缓存命中测试问题」/「问题」的记录（历史累积上百条）。
这些记录会出现在用户的「历史分析」列表里，还会把首页健康卡的降级率一起拉低 ——
**测试的副作用跑到了用户的报表上**。

而且它不止污染历史列表：落库之后 ``_remember`` 还会抽一条长期记忆，
于是"【历史降级分析】问题：客户流失率为何上升"会留在用户的长期记忆里，
**在后续真实分析中被当成历史事实注入**。所以清理必须连派生记忆一起做。

测试**读**真实数据是刻意的（"企业不存在"这类路径就得拿真库验），
所以这里不去禁止连接，而是统一兜底收尾：每个用例跑完，把本次新增的分析记录
（连同留痕与派生记忆）删掉，并打日志点明是谁写的。
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ------------------------------------------------------------------
# 真实库"写入兜底"
# ------------------------------------------------------------------

_ENGINE = None          # 独立引擎：刻意**不**复用 app 的引擎单例
_DISABLED = False       # 连不上就把这道闸门关掉，别让环境问题变成用例失败


def pytest_configure(config):
    """注册自定义 marker。

    不建 pytest.ini 是为了不动既有的收集配置（testpaths / 告警策略等），
    在这里追加一行最省事。
    """
    config.addinivalue_line(
        "markers",
        "allows_real_client: 允许该用例构造真实模型客户端（默认禁止，见 _no_real_llm_client）")


def _real_engine():
    """按配置单独建一个引擎。

    不共用 ``app.db.engine`` 的单例：不少用例会用 ``reset_engine()`` 把单例换成内存库，
    共用单例就会互相打脸（那边换成 SQLite，这边以为在看真实库）。
    """
    global _ENGINE
    if _ENGINE is None:
        from sqlalchemy import create_engine

        from app.config import get_settings

        s = get_settings().db
        _ENGINE = create_engine(
            f"mysql+pymysql://{s.user}:{s.password}@{s.host}:{s.port}/{s.name}?charset=utf8mb4",
            pool_pre_ping=True, future=True)
    return _ENGINE


def _max_id(table: str) -> int:
    from sqlalchemy import text

    with _real_engine().connect() as c:
        return int(c.execute(text(f"SELECT MAX(id) FROM {table}")).scalar() or 0)


def _purge_created(after_id: int, after_mem: int):
    """删掉本次新增的分析记录（含留痕）与由它派生出的长期记忆。

    返回 ``(删掉的分析数, 删掉的记忆数)``。
    """
    from sqlalchemy import text

    with _real_engine().begin() as c:
        ids = [int(r[0]) for r in c.execute(
            text("SELECT id FROM ai_analysis WHERE id > :a"), {"a": after_id}).fetchall()]
        mem = int(c.execute(
            text("SELECT COUNT(*) FROM ai_memory WHERE id > :a"), {"a": after_mem}).scalar() or 0)
        if not ids and not mem:
            return 0, 0
        if ids:
            marks = ",".join(str(i) for i in ids)
            c.execute(text(f"DELETE FROM ai_analysis_trace WHERE analysis_id IN ({marks})"))
            c.execute(text(f"DELETE FROM ai_tool_log WHERE analysis_id IN ({marks})"))
            #: 只删"由这批分析派生"的记忆；``source_analysis_id IS NULL`` 兜住抽取异常路径
            c.execute(text(f"DELETE FROM ai_memory WHERE id > {int(after_mem)} "
                           f"AND (source_analysis_id IS NULL OR source_analysis_id IN ({marks}))"))
            c.execute(text(f"DELETE FROM ai_analysis WHERE id IN ({marks})"))
    return len(ids), mem


@pytest.fixture(autouse=True)
def _no_real_llm_client(request):
    """**禁止用例构造真实模型客户端**（默认对所有用例生效）。

    真实事故（2026-09-23）：把 pytest 强制切到 LangGraph 编排后，
    ``test_stage5_orchestration`` 里一批走 ``AgentService.analyze`` 全链路的用例
    开始向方舟**真实发请求**，服务端回：

        429 SetLimitExceeded · Your account [2132187327] has reached the set usage limit

    根因是这类用例只注入了 ``FakeLlm``（它**只对 legacy 分支生效**），而 LangGraph
    分支走自己的 ``ModelChain`` —— 没拿到 ``lg_model_factory`` 时它会老老实实去
    ``new`` 一个真的 ``ChatOpenAI``。于是两件事同时发生：用例悄悄烧额度，而且它
    以为在验证降级自愈、实际在验证网络。

    修法是注入适配器（见 ``tests/fake_langchain.py``），**这道闸门负责让忘记注入的
    用例当场失败**，而不是继续偷跑：

        pytest.Mark 标记 ``@pytest.mark.allows_real_client`` 可以显式放行。
    """
    if request.node.get_closest_marker("allows_real_client"):
        yield
        return

    try:
        from app.agent.langgraph_impl import ModelChain
    except Exception:  # noqa: BLE001 - 依赖缺失时不额外报错
        yield
        return

    original = ModelChain._new_client

    def guard(self, model: str) -> Any:  # noqa: ANN401
        raise AssertionError(
            f"用例试图构造真实模型客户端（model={model!r}）。测试必须是零 token 的："
            f"请注入 lg_model_factory（可用 tests/fake_langchain.py::adapt_factory），"
            f"若确有理由连真实客户端，给该用例加 @pytest.mark.allows_real_client。")

    ModelChain._new_client = guard  # type: ignore[method-assign]
    try:
        yield
    finally:
        ModelChain._new_client = original  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _no_history_pollution():
    """用例跑完把自己写进真实库的分析记录与派生记忆清掉（见模块头部说明）。"""
    global _DISABLED
    before, before_mem = -1, -1
    if not _DISABLED:
        try:
            before, before_mem = _max_id("ai_analysis"), _max_id("ai_memory")
        except Exception:  # noqa: BLE001 - 没有可用库（如纯离线环境）就跳过
            _DISABLED = True

    yield

    if _DISABLED or before < 0:
        return
    try:
        n, mem = _purge_created(before, before_mem)
        if n or mem:
            print(f"\n[conftest] 已清理用例写入真实库的分析记录 {n} 条、派生长期记忆 {mem} 条"
                  f" —— 用例应当自己收尾，或改用内存库夹具")
    except Exception as e:  # noqa: BLE001 - 清理失败不改判用例结果，但要吭声
        print(f"\n[conftest] 清理真实库记录失败（可能有残留）：{e}")
