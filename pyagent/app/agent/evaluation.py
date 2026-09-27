"""评估与门禁（对应 Java ``EvaluationService``）。

设计上坚持的几件事（都是从真实事故里长出来的，改动前请先读注释）：

* **六维加权**，不是单一通过率。安全用例挂一条 vs 检索质量普遍下降，
  在通过率上看起来差不多，但前者必须立刻止血。
* **缺维度不加权**（而不是记 0 分）。否则新增维度会让历史分数整体跳水，失去可比性。
* **安全维度必须满分**才可能通过门禁 —— 安全没有"差不多"。
* **0 条用例被执行 = 失败**，不是"0/0 全通过"。空集合天然满足"全部通过"，
  这是最容易被忽略的假绿灯。
* **先算分再落盘**。曾经是"先落盘、后算分"，落盘的那份 JSON 里 score/grade 全是空值，
  而界面历史读的恰恰是这份文件 —— 一份真实的 90 分报告在界面上显示成 0 分。
* **三种模式**：``fast``（零 token 确定性）/ ``live``（真调模型）/ ``full``。
  耗时只有在两次跑的是同一模式时才可比，跨模式不出这个数（宁可不出，也不出一个误导的数）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence

from ..config import get_settings
from ..db import tables as T
from ..db.engine import Database, get_db
from .agent_service import AgentService
from .eval_history import EvalHistoryStore
from .grounding import AnswerGroundingService
from .guardrail import GuardrailService
from .health import OperationalHealthService

log = logging.getLogger(__name__)


def round2(v: float) -> float:
    """Java ``Math.round(v * 100.0) / 100.0`` —— 注意是 floor(x+0.5)，不是取偶。"""
    import math
    return math.floor(v * 100.0 + 0.5) / 100.0


def bare_tool_name(trace_entry: Optional[str]) -> str:
    """``"get_metrics(5ms)"`` → ``"get_metrics"``。

    工具轨迹为了给人读，在工具名后面补了耗时（``(5ms)``）或来源（``(预取 370ms)``）。
    机器断言只认工具名，所以两边必须先归一 —— 否则断言永远不相等，
    而失败文案写的是「模型未调用期望工具 get_metrics，实际轨迹 ['get_metrics(5ms)']」，
    看上去自相矛盾，排查时极易被当成模型真的没有调用工具。
    """
    s = (trace_entry or "").strip()
    i = s.find("(")
    return (s if i < 0 else s[:i]).strip()


def _as_int(v: Any) -> Optional[int]:
    """留痕里的计数值转 int；缺失/不可解析一律 None（界面显示「—」而不是 0）。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


MODE_FAST = "fast"
MODE_LIVE = "live"
MODE_FULL = "full"

#: 「外部环境不可得」的特征串 —— 命中的 web 用例判 SKIP 而不是 FAIL。
#: 每一条都对应一个**使用者能自己动手解决**的动作，而不对应代码缺陷；
#: 把它们计为 FAIL 会让人去修一个并不存在的 bug。
_ENV_WEB_MARKERS = (
    ("ToolNotOpen", "方舟「联网内容插件」未开通（免费开通即可，不改代码）"),
    ("尚未开通", "方舟「联网内容插件」未开通（免费开通即可，不改代码）"),
    ("SetLimitExceeded", "方舟账号达到「安心体验模式」推理额度上限，模型服务已暂停"),
    ("安心体验模式", "方舟账号达到「安心体验模式」推理额度上限，模型服务已暂停"),
    ("Arrearage", "方舟账号欠费/余额不足"),
    ("invalid_api_key", "Key 无权调用 Responses API"),
    ("鉴权失败", "Key 无权调用 Responses API"),
    ("qcaptcha", "搜索引擎（360）跳验证码，免 Key 通道被挡"),
    ("访问异常页面", "搜索引擎返回访问异常页，免 Key 通道被挡"),
    ("全部与本次问题不相关", "免 Key 通道只返回无关网页，已被相关度质量门弃用"),
    ("引擎全部失败", "免 Key 直连引擎全部不可达"),
    ("未装配", "联网检索未装配"),
)


def _environmental_web_reason(reasons: List[str]) -> str:
    """从失败原因里认出"这是环境不可得，不是能力缺陷"。

    前置检查（``_web_environment_gate``）只能看清配置齐不齐，而「插件没开通」
    「账号到额」「搜索引擎只返回无关网页」这些**只有真跑一次才暴露**，
    所以定性放在跑完之后按真实原因来做。
    """
    joined = " | ".join(reasons or [])
    if not joined:
        return ""
    for marker, human in _ENV_WEB_MARKERS:
        if marker in joined:
            return human
    return ""
#: 六维权重。检索与引用最高 —— 这两个直接决定"结论有没有依据"。
WEIGHTS: Dict[str, float] = {
    "retrieval": 0.25, "citation": 0.20, "routing": 0.15,
    "structure": 0.15, "safety": 0.15, "cost": 0.10,
    # e2e（真实大模型）：唯一会真烧 token 的一类，也是唯一能回答
    # 「这套链路到底多久出得来、出来的是不是能用的东西」的一类。
    "e2e": 0.30,
}

#: camelCase → snake_case（用例文件是 camelCase，Python 字段是 snake_case）
_CAMEL = re.compile(r"(?<=[a-z0-9])([A-Z])")


DIM_CATEGORY: Dict[str, List[str]] = {
    "retrieval": ["rag"],
    "citation": ["citation", "repair"],
    "routing": ["route"],
    "structure": ["structure"],
    "safety": ["guardrail"],
    "e2e": ["e2e"],
}


# ------------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------------


@dataclass
class EvalCase:
    """一条评估用例。字段与 ``ai-eval-cases.json`` 一一对应。"""

    id: str = ""
    #: 纯注释：这条用例为什么存在、验的是什么。
    note: str = ""
    category: str = ""
    question: str = ""
    company_id: Optional[int] = None
    expect_tools: List[str] = field(default_factory=list)
    expect_keywords: List[str] = field(default_factory=list)
    min_rag_score: Optional[float] = None
    #: Top 归一分的上限（负样本）：问一个库里根本没有的话题时，召回分应当很低。
    max_rag_score: Optional[float] = None
    #: Top **原始分**的上限（负样本用这个判更可靠，因为归一分会被 Top1 拉到满分）。
    max_rag_raw_score: Optional[float] = None
    #: 负样本的相对判据：与一条已知相关的参照问题比，原始分不能高过它的多少倍。
    max_raw_ratio_vs_reference: Optional[float] = None
    reference_question: str = ""
    reference_company_id: Optional[int] = None
    #: Top 原始分的下限（绝对刻度），只用来兜住"向量通道整体失效"这类断崖式退化。
    min_rag_raw_score: Optional[float] = None
    top_k: Optional[int] = None
    expect_web: Optional[bool] = None
    min_web_results: Optional[int] = None
    expect_web_keywords: List[str] = field(default_factory=list)
    expect_blocked: Optional[bool] = None
    probe_tool: str = ""
    probe_args: Dict[str, Any] = field(default_factory=dict)

    # ---- memory 类（长期记忆；纯函数级，零 token，不落库）----
    #: 期望抽取出的记忆种类，如 ``["FACT", "EPISODE"]``
    memory_kinds: List[str] = field(default_factory=list)
    #: 记忆内容里**禁止**出现的串。最主要的用途是锁住"记忆不是证据"：
    #: 一旦注入文本里出现 ``[1]`` 或 ``来源ID=``，引用核对就会把历史数字验成本次证据。
    memory_forbid: List[str] = field(default_factory=list)
    #: 喂给抽取器的风险等级（HIGH 会额外产出一条「处置经过」）
    risk_level: str = ""

    # ---- citation 类 ----
    answer_fixture: str = ""
    fixture_web_sources: List[Dict[str, Any]] = field(default_factory=list)
    fixture_kb_sources: List[Dict[str, Any]] = field(default_factory=list)
    expect_ok: Optional[bool] = None
    expect_dangling_web: Optional[int] = None
    expect_dangling_kb: Optional[int] = None
    expect_min_web_cited: Optional[int] = None

    # ---- repair 类 ----
    expect_repaired_web: Optional[int] = None
    expect_repaired_kb: Optional[int] = None
    expect_reference_list: Optional[bool] = None
    expect_dangling_web_after: Optional[int] = None
    expect_all_sources_cited_after: Optional[bool] = None

    # ---- route 类 ----
    expect_agents: List[str] = field(default_factory=list)
    expect_tools_absent: List[str] = field(default_factory=list)

    # ---- structure 类 ----
    expect_sections: List[str] = field(default_factory=list)
    forbid_phrases: List[str] = field(default_factory=list)
    expect_contract_violation: Optional[bool] = None

    # ---- live（e2e）类 ----
    live: Optional[bool] = None
    live_depth: str = ""
    expect_answer_min_chars: Optional[int] = None
    max_duration_ms: Optional[int] = None
    expect_max_llm_calls: Optional[int] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvalCase":
        """宽容解析：未知字段忽略，但**字段名必须真的接上**。

        .. danger::
           用例文件（``ai-eval-cases.json``，与 Java 版共用同一份）是 **camelCase**，
           而这里的字段是 snake_case。不做转换的话所有断言会被**静默丢弃** ——
           44 条用例全部"通过"，实际上一条断言都没执行。这是最难发现的一种假绿灯：
           报告全绿、门禁 PASSED，而回归形同虚设。

        曾经用例文件里多写了一个没声明的扩展字段，Jackson 直接抛异常、整份用例一条没加载，
        而报告上显示的是「0/0 全通过」—— 同样是假绿灯，反方向。
        """
        known = set(EvalCase.__dataclass_fields__)  # type: ignore[attr-defined]
        kw: Dict[str, Any] = {}
        unknown: List[str] = []
        for k, v in (d or {}).items():
            key = _CAMEL.sub(r"_\1", k).lower()
            if key in known:
                kw[key] = v
            elif k in known:
                kw[k] = v
            else:
                unknown.append(k)
        if unknown:
            # 不抛异常（一个拼写错误不该让整轮回归空跑），但必须看得见
            log.warning("[Eval] 用例 %s 含未识别字段，已忽略: %s",
                        (d or {}).get("id") or "?", unknown)
        return EvalCase(**kw)


@dataclass
class CaseResult:
    id: str = ""
    category: str = ""
    question: str = ""
    company_id: Optional[int] = None
    passed: bool = False
    #: **跳过**（区别于失败）：用例本身没问题，是这条跑不起来的环境不具备。
    #:
    #: 2026-09-22 加这个字段正是因为 web 类用例：它们要真发一次联网检索，
    #: 而本机所有免 Key 通道都已不可用、方舟「联网内容插件」又没开通——
    #: 于是 5 条全 FAIL，报告看着像"联网能力崩了"，实际是"这次没法评"。
    #: FAIL 意味着**有缺陷**，会把人引去修一个并不存在的 bug；SKIP 说的是
    #: "这条在此刻无解"，把原因写在 :attr:`skip_reason` 里指条明路。
    skipped: bool = False
    skip_reason: str = ""
    #: 可空：只有真正跑了 RAG 的用例才有值。null 与 0 必须能区分开。
    rag_top_score: Optional[float] = None
    rag_top_raw_score: Optional[float] = None
    rag_hits: Optional[int] = None
    live: Optional[bool] = None
    web_hits: Optional[int] = None
    web_top_title: Optional[str] = None
    blocked: Optional[bool] = None
    probe_tool: Optional[str] = None
    probe_preview: Optional[str] = None
    probe_sources: Optional[int] = None
    citation_web_cited: Optional[int] = None
    citation_dangling: Optional[int] = None
    citation_summary: Optional[str] = None
    #: 长期记忆类用例实际抽出的记忆种类（解释"这条为什么通过/失败"）
    memory_kinds: str = ""
    repair_web: Optional[int] = None
    repair_kb: Optional[int] = None
    repair_reference: Optional[int] = None
    repair_dangling_after: Optional[int] = None
    route_agents: Optional[List[str]] = None
    route_tools: Optional[List[str]] = None
    #: 真实工具调用轨迹，**裸工具名**（``["get_metrics", ...]``）。
    #:
    #: 机器断言（``expectTools``）比对的就是这个字段，所以它必须是裸名。
    #: 展示用的带耗时版本另放 :attr:`trace_detail`——早先把两者合成一个字段，
    #: 轨迹里带上了 ``(5ms)`` 装饰，断言拿裸名去精确匹配于是**恒不相等**，
    #: 5 条端到端用例全部假失败（界面上一片「模型未调用期望工具」）。
    trace_tools: Optional[List[str]] = None
    #: 带耗时/预取的轨迹（``"get_metrics(5ms)"``），只给人看，不参与判定。
    #: Python 侧补充字段，Java 的 ``traceTools`` 只有裸名。
    trace_detail: Optional[List[str]] = None
    llm_calls: Optional[int] = None
    tool_calls: Optional[int] = None
    answer_chars: Optional[int] = None
    answer_preview: Optional[str] = None
    degrade_level: Optional[str] = None
    model: Optional[str] = None
    grounded: Optional[bool] = None
    single_round: Optional[bool] = None
    #: 本次真实生效的深度档位（quick / standard / deep）。
    #:
    #: 报告上「耗时」与「模型调用」两列只有配上档位才解释得通：
    #: quick 档本就该是单轮短稿，standard 是三轮到 1500 字，deep 更多。
    #: 早先 liveDepth 根本没传进 AgentBudget，报告里也就无从体现
    #: "这条为什么慢" —— 用户看到的是一堆无法解释的数字。
    depth: Optional[str] = None
    duration_ms: int = 0
    reasons: List[str] = field(default_factory=list)

    #: 字段名 → 对外 JSON 键。**必须与 Java 的 ``CaseResult`` 序列化结果逐字一致**：
    #: 前端就是按这些键读的，错一个字母就会静默变成空白单元格
    #: （``companyId`` 写成 ``company_id`` 的后果是「企业」整列消失、
    #: 「证据/模型调用/成稿/Top分/耗时」全部显示为「—」）。
    _JSON_KEYS: ClassVar[Dict[str, str]] = {
        "id": "id",
        "category": "category",
        "question": "question",
        "company_id": "companyId",
        "passed": "passed",
        "skipped": "skipped",
        "skip_reason": "skipReason",
        "rag_top_score": "ragTopScore",
        "rag_top_raw_score": "ragTopRawScore",
        "rag_hits": "ragHits",
        "live": "live",
        "web_hits": "webHits",
        "web_top_title": "webTopTitle",
        "blocked": "blocked",
        "probe_tool": "probeTool",
        "probe_preview": "probePreview",
        "probe_sources": "probeSources",
        "citation_web_cited": "citationWebCited",
        "citation_dangling": "citationDangling",
        "citation_summary": "citationSummary",
        "memory_kinds": "memoryKinds",
        "repair_web": "repairWeb",
        "repair_kb": "repairKb",
        "repair_reference": "repairReference",
        "repair_dangling_after": "repairDanglingAfter",
        "route_agents": "routeAgents",
        "route_tools": "routeTools",
        "trace_tools": "traceTools",
        "trace_detail": "traceDetail",
        "llm_calls": "llmCalls",
        "tool_calls": "toolCalls",
        "answer_chars": "answerChars",
        "answer_preview": "answerPreview",
        "degrade_level": "degradeLevel",
        "model": "model",
        "grounded": "grounded",
        "single_round": "singleRound",
        "depth": "depth",
        "duration_ms": "durationMs",
        "reasons": "reasons",
    }

    def to_dict(self) -> Dict[str, Any]:
        """按**驼峰键**输出。

        刻意不剔除 ``None``：契约要稳定可 diff，前端也靠「键在、值为 null」区分
        「这个指标不适用」（显示「—」）与「这条用例压根没这个字段」（同样是「—」，
        但排查时会误判成后端漏填）。空列表统一给 ``[]``，免得前端还要兜底
        ``row.reasons && row.reasons.length``。
        """
        out: Dict[str, Any] = {}
        for field_name, json_key in self._JSON_KEYS.items():
            v = getattr(self, field_name)
            if v is None and json_key == "reasons":
                v = []
            out[json_key] = v
        return out


@dataclass
class EvalReport:
    id: Optional[str] = None
    total: int = 0
    passed: int = 0
    #: 因环境不具备而未参评的条数（不计入 total/passRate，但要单独报出来）
    skipped: int = 0
    pass_rate: float = 0.0
    wall_clock_ms: int = 0
    model: str = ""
    generated_at: str = ""
    web_search_enabled: bool = False
    web_searches_run: int = 0
    parallel: bool = True
    wall_clock_hint_ms: int = 0
    total_case_ms: int = 0
    by_category: Dict[str, str] = field(default_factory=dict)
    comparison: Optional[Dict[str, Any]] = None
    score: float = 0.0
    grade: str = ""
    gate: str = ""
    gate_reason: str = ""
    dimensions: Dict[str, Any] = field(default_factory=dict)
    results: List[CaseResult] = field(default_factory=list)
    mode: str = MODE_FAST
    live_cases: int = 0
    live_total_ms: int = 0
    live_avg_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "total": self.total, "passed": self.passed,
            "skipped": self.skipped, "passRate": self.pass_rate, "wallClockMs": self.wall_clock_ms,
            "model": self.model, "generatedAt": self.generated_at,
            "webSearchEnabled": self.web_search_enabled,
            "webSearchesRun": self.web_searches_run, "parallel": self.parallel,
            "wallClockHintMs": self.wall_clock_hint_ms, "totalCaseMs": self.total_case_ms,
            "byCategory": self.by_category, "comparison": self.comparison,
            "score": self.score, "grade": self.grade, "gate": self.gate,
            "gateReason": self.gate_reason, "dimensions": self.dimensions,
            "results": [r.to_dict() for r in self.results],
            "mode": self.mode, "liveCases": self.live_cases,
            "liveTotalMs": self.live_total_ms, "liveAvgMs": self.live_avg_ms,
        }


# ------------------------------------------------------------------
# 服务
# ------------------------------------------------------------------


class EvaluationService:
    """跑一轮回归，给出六维评分与门禁判定。"""

    def __init__(
        self,
        rag=None,
        tools=None,
        web_search=None,
        guardrail: Optional[GuardrailService] = None,
        grounding: Optional[AnswerGroundingService] = None,
        history: Optional[EvalHistoryStore] = None,
        agent_router=None,
        health: Optional[OperationalHealthService] = None,
        agent_service: Optional[AgentService] = None,
        db: Optional[Database] = None,
        llm=None,
        cases_file: Optional[str] = None,
        gate_score: Optional[float] = None,
    ) -> None:
        s = get_settings()
        self.db = db or get_db()
        self.rag = rag
        self.tools = tools
        self.web_search = web_search
        self.guardrail = guardrail or GuardrailService()
        self.grounding = grounding or AnswerGroundingService()
        self.history = history if history is not None else EvalHistoryStore()
        self.agent_router = agent_router
        self.health = health or OperationalHealthService(self.db)
        self.agent_service = agent_service
        self.llm = llm
        self.cases_file = cases_file or s.eval.cases_file
        self.gate_score = float(gate_score if gate_score is not None else s.eval.gate_score)
        self._cases: Optional[List[EvalCase]] = None
        #: 用例加载失败的原因（None = 正常）。让「一条都没跑成」在报告里表现为故障，
        #: 而不是「0/0 全通过」。
        self.case_load_error: Optional[str] = None

    # -- 用例加载 ----------------------------------------------------

    def load_cases(self) -> List[EvalCase]:
        if self._cases is not None:
            return self._cases
        path = self.cases_file
        if not os.path.isabs(path):
            # 相对路径以 pyagent 包目录为基准，避免受调用方 cwd 影响
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.normpath(os.path.join(base, path))
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._cases = [EvalCase.from_dict(d) for d in (raw or [])]
            self.case_load_error = None
            log.info("[Eval] 已加载评估用例 %d 条（%s）", len(self._cases), path)
        except Exception as e:  # noqa: BLE001
            # 必须是 error 级：走到这里整轮回归等于空跑，而报告上的「0/0」
            # 看起来和全绿一模一样。
            log.error("[Eval] 加载评估用例失败，本轮回归将没有任何用例可跑: %s", e)
            self.case_load_error = str(e)
            self._cases = []
        return self._cases

    def case_count(self, mode: str = MODE_FAST) -> int:
        """该模式下**实际会执行**的用例数。

        不能拿文件里的总条数当进度条分母：fast 模式压根不跑那几条 live 用例，
        于是进度条永远差一截到不了 100%、报告里的用例数也会比真实执行的多。
        """
        m = (mode or MODE_FAST).strip().lower()
        want_live = m in (MODE_LIVE, MODE_FULL)
        want_fast = m in (MODE_FAST, MODE_FULL)
        n = 0
        for c in self.load_cases():
            if bool(c.live):
                n += 1 if want_live else 0
            elif want_fast:
                n += 1
        return n

    # -- 跑批 --------------------------------------------------------

    def evaluate(self, company_id: Optional[int] = None,
                 progress: Optional[Callable[[int, int], None]] = None,
                 mode: str = MODE_FAST) -> EvalReport:
        """跑一轮回归。默认只跑零 token 的确定性用例。"""
        t0 = time.time()
        m = (mode or MODE_FAST).strip().lower()
        want_live = m in (MODE_LIVE, MODE_FULL)
        want_fast = m in (MODE_FAST, MODE_FULL)

        fast_cases: List[EvalCase] = []
        live_cases: List[EvalCase] = []
        for c in self.load_cases():
            if bool(c.live):
                if want_live:
                    live_cases.append(c)
            elif want_fast:
                fast_cases.append(c)

        available = set()
        if self.tools is not None:
            available = {s.name for s in self.tools.specs()}

        total = len(fast_cases) + len(live_cases)
        state = {"done": 0, "web": 0}
        created: List[int] = []

        results: List[CaseResult] = []
        if fast_cases:
            results.extend(self._run_batch(fast_cases, 6, company_id, available, state,
                                           total, progress, created))
        if live_cases:
            # live 并发刻意压到 2：并发太高会撞限流，测出来的"耗时"就成了排队时间
            results.extend(self._run_batch(live_cases, 2, company_id, available, state,
                                           total, progress, created))

        # 保持报告顺序与用例文件一致，便于逐条对读
        order = [c.id for c in fast_cases] + [c.id for c in live_cases]
        by_id = {r.id: r for r in results}
        ordered = [by_id[i] for i in order if i in by_id]
        if len(ordered) == len(results):
            results = ordered

        self._cleanup_live_analyses(created)

        passed = sum(1 for r in results if r.passed)
        skipped_n = sum(1 for r in results if r.skipped)
        total_ms = sum(r.duration_ms for r in results)

        report = EvalReport()
        # ``total`` 是**参与评级**的条数：跳过的用例既不算过也不算不过，
        # 放进分母会让"环境不可得"看起来像"能力不行"（0/5 变成 89.8% 而不是 100%）。
        report.total = len(results) - skipped_n
        report.passed = passed
        report.skipped = skipped_n
        report.results = results
        report.generated_at = datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y").strip()
        report.web_search_enabled = bool(self.web_search is not None and self.web_search.is_enabled())
        report.web_searches_run = state["web"]
        report.parallel = True
        report.wall_clock_hint_ms = max((r.duration_ms for r in results), default=0)
        report.total_case_ms = total_ms
        bc: Dict[str, str] = {}
        for r in results:
            k = r.category or ""
            hit = sum(1 for x in results if (x.category or "") == k and x.passed)
            skip = sum(1 for x in results if (x.category or "") == k and x.skipped)
            allc = sum(1 for x in results if (x.category or "") == k)
            bc.setdefault(k, f"{hit}/{allc - skip}" + (f"（{skip} 条跳过）" if skip else ""))
        report.by_category = bc
        report.wall_clock_ms = int((time.time() - t0) * 1000)
        report.pass_rate = round2(passed * 100.0 / report.total) if report.total else 0.0
        report.model = self.describe_model()
        report.mode = m
        report.live_cases = len(live_cases)
        e2e_ms = [r.duration_ms for r in results if r.category == "e2e"]
        report.live_total_ms = sum(e2e_ms)
        report.live_avg_ms = int(round(sum(e2e_ms) / len(live_cases))) if live_cases else 0

        # 基线必须在落盘之前取：落盘会覆盖"最新一份"，取晚了就是拿自己跟自己比。
        baseline: Optional[Dict[str, Any]] = None
        if self.history is not None:
            try:
                baseline = self.history.latest_report()
            except Exception as e:  # noqa: BLE001
                log.warning("[Eval] 读取评估基线失败: %s", e)

        # ⚠️ 必须先算分再落盘
        self._score_and_gate(report)

        if self.history is not None:
            try:
                if baseline is not None:
                    report.comparison = self.compare(report, baseline)
                report.id = self.history.save(report)
            except Exception as e:  # noqa: BLE001
                log.warning("[Eval] 评估报告留档/比对失败: %s", e)
        self._persist_run(report)
        return report

    # -- 评分与门禁 --------------------------------------------------

    def _score_and_gate(self, report: EvalReport) -> None:
        dims: Dict[str, Any] = {}
        total_w = 0.0
        acc = 0.0
        for dim, w in WEIGHTS.items():
            s = self._dim_score(report, dim)
            one: Dict[str, Any] = {"weight": w}
            if s is None:
                # 缺维度不加权（而不是记 0 分），否则历史分数不可比
                one["score"] = None
                one["applicable"] = False
                one["note"] = "本次无该类用例，不参与加权"
            else:
                one["score"] = s
                one["applicable"] = True
                acc += s * w
                total_w += w
            dims[dim] = one
        report.dimensions = dims
        report.score = round2(acc / total_w) if total_w > 0 else 0.0
        report.grade = ("A" if report.score >= 90 else
                        "B" if report.score >= 80 else
                        "C" if report.score >= 70 else "D")

        verdict = (report.comparison or {}).get("verdict")
        regressed = verdict == "REGRESSED"
        safety = dims.get("safety") or {}
        safety_score = safety.get("score") if isinstance(safety.get("score"), (int, float)) else 100.0
        ok = report.score >= self.gate_score and not regressed and safety_score >= 100.0
        report.gate = "PASSED" if ok else "FAILED"
        if ok:
            report.gate_reason = (f"综合分 {report.score} ≥ 门禁 {self.gate_score}，"
                                  "无新增回归，安全维度满分")
        else:
            parts = []
            if report.score < self.gate_score:
                parts.append(f"综合分 {report.score} 低于门禁 {self.gate_score}；")
            if regressed:
                parts.append("存在新增回归用例；")
            if safety_score < 100.0:
                parts.append(f"安全维度未满分（{safety_score}）")
            report.gate_reason = "".join(parts).rstrip("；")

        # ⚠️ 空集合天然满足「全部通过」。这条是被真实事故逼出来的：
        # 用例文件里多写了一个未知字段，49 条全部加载失败，门禁照样 PASSED。
        if report.total == 0:
            report.gate = "FAILED"
            report.gate_reason = (
                "本轮 0 条用例被执行：用例文件未加载成功，或所选模式筛选后为空"
                + ("" if self.case_load_error is None else f"（加载错误：{self.case_load_error}）")
                + "。「0/0 全通过」不是通过，是没跑。"
            )

    def _dim_score(self, report: EvalReport, dim: str) -> Optional[float]:
        if dim == "cost":
            # cost 维度来自线上留痕，不是跑批 —— 这是六维里唯一反映真实表现的一维
            h = None if self.health is None else self.health.health(24)
            if not h or not h.get("available"):
                return None
            s = h.get("score")
            return round2(float(s)) if isinstance(s, (int, float)) else None
        cats = DIM_CATEGORY.get(dim)
        if not cats:
            return None
        rows = [r for r in (report.results or []) if (r.category or "") in cats]
        if not rows:
            return None
        return round2(sum(1 for r in rows if r.passed) * 100.0 / len(rows))

    def _persist_run(self, report: EvalReport) -> None:
        try:
            self.db.execute(T.ai_eval_run.insert().values(
                run_id=report.id or str(int(time.time() * 1000)),
                total=report.total, passed=report.passed, pass_rate=report.pass_rate,
                score=report.score, grade=report.grade, gate=report.gate,
                dimensions=json.dumps(report.dimensions or {}, ensure_ascii=False),
                model=report.model, wall_clock_ms=report.wall_clock_ms,
                created_at=datetime.now(),
            ))
        except Exception as e:  # noqa: BLE001
            log.warning("[Eval] 评估留档落库失败: %s", e)

    def describe_model(self) -> str:
        """一句话描述本次跑批的模型底座：换过底座的两次结果不可比。"""
        if self.llm is None:
            return "本地规则（未接入外部模型）"
        try:
            base = self.llm.get_base_url() or ""
            provider = ("火山方舟" if "volces" in base else
                        "硅基流动" if "siliconflow" in base else
                        "OpenAI" if "openai.com" in base else "自定义")
            return f"{provider} · {self.llm.get_model()}"
        except Exception:  # noqa: BLE001
            return "未知底座"

    # -- 基线对比 ----------------------------------------------------

    def compare(self, report: EvalReport, baseline: Dict[str, Any]) -> Dict[str, Any]:
        def num(v: Any) -> float:
            return float(v) if isinstance(v, (int, float)) else 0.0

        c: Dict[str, Any] = {}
        c["baselineId"] = baseline.get("id")
        c["baselineGeneratedAt"] = baseline.get("generatedAt")
        before_rate = num(baseline.get("passRate"))
        c["baselinePassRate"] = before_rate
        c["passRateDelta"] = round2(report.pass_rate - before_rate)

        # 口径校验：耗时只有两次跑的是同一套用例口径时才有可比性。
        # 基线可能是 fast-only、本轮却是 live，那种「+3040%」纯粹是把
        # 「不调模型」和「真调模型」放一起比。宁可不出这个数，也不出误导的数。
        base_mode = baseline.get("mode")
        base_mode = None if base_mode is None else str(base_mode)
        same = base_mode is not None and base_mode.lower() == report.mode
        c["baselineMode"] = base_mode if base_mode is not None else "(未知，早于 mode 字段)"
        c["currentMode"] = report.mode
        c["sameMode"] = same
        before_ms = int(num(baseline.get("wallClockMs")))
        c["baselineWallClockMs"] = before_ms
        if same:
            c["wallClockDeltaMs"] = report.wall_clock_ms - before_ms
            if before_ms > 0:
                c["wallClockDeltaPct"] = round2((report.wall_clock_ms - before_ms) * 100.0 / before_ms)
        else:
            c["wallClockComparable"] = False

        # 新增失败（回归）/ 已修复
        prev = {str(r.get("id")): bool(r.get("passed"))
                for r in (baseline.get("results") or []) if isinstance(r, dict)}
        now = {r.id: r.passed for r in report.results}
        regressed = sorted(i for i in now if i in prev and prev[i] and not now[i])
        fixed = sorted(i for i in now if i in prev and not prev[i] and now[i])
        c["regressedCases"] = regressed
        c["fixedCases"] = fixed
        c["verdict"] = "REGRESSED" if regressed else ("IMPROVED" if fixed else "SAME")
        return c

    # -- 批次执行 ----------------------------------------------------

    def _run_batch(self, cases: Sequence[EvalCase], workers: int, company_id: Optional[int],
                   available: set, state: Dict[str, int], total: int,
                   progress: Optional[Callable[[int, int], None]],
                   created: List[int]) -> List[CaseResult]:
        out: List[CaseResult] = []
        if not cases:
            return out
        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="eval-case") as ex:
            futures: List[Future] = []
            for c in cases:
                futures.append(ex.submit(self._one, c, company_id, available, state,
                                         total, progress, created))
            for i, f in enumerate(futures):
                try:
                    out.append(f.result(timeout=480))
                except Exception as e:  # noqa: BLE001
                    r = self._base(c, company_id)
                    r.passed = False
                    r.reasons = [f"用例超时或中断: {e}"]
                    out.append(r)
        return out

    def _one(self, c: EvalCase, company_id: Optional[int], available: set,
             state: Dict[str, int], total: int,
             progress: Optional[Callable[[int, int], None]],
             created: List[int]) -> CaseResult:
        try:
            return self.run_case(c, company_id, available, state, created)
        except Exception as e:  # noqa: BLE001
            r = self._base(c, company_id)
            r.passed = False
            r.reasons = [f"用例执行异常: {e}"]
            return r
        finally:
            state["done"] += 1
            if progress is not None:
                try:
                    progress(state["done"], total)
                except Exception:  # noqa: BLE001
                    pass

    def _cleanup_live_analyses(self, ids: Sequence[int]) -> None:
        """清掉 live 用例跑出来的分析记录。

        不清理的后果是实打实的：用户的「历史分析」列表会被几十条同名占满，
        首页健康卡的降级率也会被这些测试记录拉低。只删本次跑批自己产生的 id。
        """
        if not ids:
            return
        n = 0
        for aid in ids:
            if not aid:
                continue
            try:
                tr = T.ai_analysis_trace
                row = self.db.fetch_one(
                    tr.select().where(tr.c.analysis_id == aid).limit(1))
                trace_id = (row or {}).get("trace_id")
                self.db.execute(T.ai_tool_log.delete().where(T.ai_tool_log.c.analysis_id == aid))
                self.db.execute(T.ai_analysis_trace.delete().where(tr.c.analysis_id == aid))
                self.db.execute(T.ai_analysis.delete().where(T.ai_analysis.c.id == aid))
                n += 1
                if trace_id:
                    self.db.execute(T.ai_tool_log.delete().where(T.ai_tool_log.c.trace_id == trace_id))
            except Exception as e:  # noqa: BLE001
                log.warning("[Eval] 清理 live 用例产生的分析记录 #%s 失败: %s", aid, e)
        if n:
            log.info("[Eval] 已清理本次 live 跑批产生的 %d 条分析记录（不污染用户历史）", n)

    @staticmethod
    def _base(c: EvalCase, company_id: Optional[int]) -> CaseResult:
        r = CaseResult()
        r.id = c.id
        r.category = c.category or "rag"
        r.question = c.question
        r.company_id = c.company_id if c.company_id is not None else (company_id if company_id is not None else 1)
        #: 是不是真跑了大模型的用例。界面靠它区分「耗时 —（本来就不调模型）」
        #: 与「耗时 0.4s（真跑了一轮）」—— 原来两者都显示成 0s，看着像故障。
        r.live = bool(c.live)
        return r

    def _check_memory(self, c: EvalCase, r: CaseResult):
        """长期记忆的确定性检查（**零 token、不落库**）。

        只验两件最容易被"看起来做完了"骗过去的事：

        1. **该抽出来的东西抽得出来** —— 事实卡必须带数字与风险词（否则记忆会变成
           每次分析的副本），偏好必须只在出现"以后/记住/默认"这类长期措辞时才产生；
        2. **记忆不得伪装成证据** —— 注入文本里不许出现 ``[n]`` 或 ``来源ID=``。
           一旦记忆占了证据编号，引用核对就会把"记忆里的历史数字"验成本次证据，
           等于给"编造数字"开了一条合法通道。

        落库、冲突消解（旧口径被取代但仍可查）与容量淘汰由 pytest 的 SQLite 用例覆盖，
        这里不碰数据库：评估跑批必须是可重复的，不能因为跑一次回归就往用户库里塞记忆。
        """
        ok = True
        reasons: List[str] = []

        try:
            from .memory_extract import MemoryExtractor

            ex = MemoryExtractor()
            cands = ex.extract(
                company_id=r.company_id, user_id=1, question=c.question,
                answer_json={"conclusion_internal": c.answer_fixture or "",
                             "headline": "评测样本结论",
                             "risk_level": c.risk_level or ""},
                answer_text="", risk_level=c.risk_level or "",
                analysis_id=None, trace_id="eval", degrade_level="NONE",
                today="2026-01-01")
        except Exception as e:  # noqa: BLE001
            return False, [f"记忆抽取失败: {e}"]

        kinds = [x.kind for x in cands]
        r.memory_kinds = ",".join(kinds)
        # 哨兵 ``NONE`` = "这一问不该沉淀出任何记忆"（负样本用）。
        # 不能靠"给个空数组"表达 —— 空数组在断言里天然满足，等于没断言。
        want = list(c.memory_kinds)
        if "NONE" in want:
            want = [k for k in want if k != "NONE"]
            if kinds:
                ok = False
                reasons.append(f"本不该沉淀任何记忆，实际抽出：{kinds}")
        for k in want:
            if k not in kinds:
                ok = False
                reasons.append(f"未抽出期望的记忆种类 {k}（实际：{kinds or '无'}）")

        joined = " ".join(x.content for x in cands)
        for kw in c.expect_keywords:
            if kw not in joined:
                ok = False
                reasons.append(f"记忆内容缺少关键词: {kw}")
        for bad in c.memory_forbid:
            if bad in joined:
                ok = False
                reasons.append(f"记忆内容出现禁止串「{bad}」")

        # 注入层：内容没问题还不够，注入格式也必须保持"非证据"
        try:
            from .long_term_memory import LongTermMemory, MemoryHit

            hits = [MemoryHit(id=1, scope=x.scope, kind=x.kind, topic=x.topic,
                              content=x.content, importance=x.importance) for x in cands]
            ctx = LongTermMemory().build_context(hits, budget=600)
            for bad in ("[1]", "来源ID="):
                if ctx and bad in ctx:
                    ok = False
                    reasons.append(f"注入文本出现证据编号「{bad}」——记忆不是证据")
        except Exception as e:  # noqa: BLE001
            ok = False
            reasons.append(f"注入上下文构造失败: {e}")

        return ok, reasons

    # -- 单条确定性用例 ----------------------------------------------

    def _web_environment_gate(self) -> str:
        """判断"联网此刻能不能评"。返回非空字符串就应当跳过该类用例。

        注意区分两件事：
          * 能力缺陷（检索词清洗错、回引用错）→ 该 FAIL，让人去修；
          * 环境不可得（插件没开通、所有免 Key 通道拿不到相关结果）→ 该 SKIP，
            并给出可操作的那一步。
        """
        svc = self.web_search
        if svc is None:
            return "联网检索未装配"
        try:
            if not svc.is_enabled():
                return f"联网检索整体不可用：{svc.status_reason()}"
            info = svc.probe()
        except Exception as e:  # noqa: BLE001 - 前置检查自己出问题也按"不可评"处理
            return f"联网自检失败（按不可评处理）：{e}"
        if not self._web_zero_cost_channel_usable(info):
            detail = "；".join(str(h) for h in (info.get("hints") or [])) or "无可用通道"
            last = str(info.get("lastFailure") or "")
            return ("联网检索此刻无法给出可引用的来源（" + detail
                    + ("；上次失败：" + last if last else "")
                    + "）。付费通道未开通时请到方舟开通「联网内容插件」，"
                      "否则这类用例只能跳过 —— 这不是能力缺陷，不应计为失败。")
        return ""

    @staticmethod
    def _web_zero_cost_channel_usable(info: Dict[str, Any]) -> bool:
        """API 通道在配置层面是否可用。

        只能看到"配置齐不齐"，**看不到插件开没开通**——后者要真发一次请求才知道，
        所以真正的定性放在 :func:`_environmental_web_reason`（看真实失败原因）。
        """
        for ch in (info.get("channels") or []):
            if isinstance(ch, dict) and ch.get("name") == "api" and ch.get("note") == "可用":
                return True
        return False

    def run_case(self, c: EvalCase, company_id: Optional[int], available: set,
                 state: Dict[str, int], created: List[int]) -> CaseResult:
        if c.live:
            return self._run_live_case(c, company_id, created)
        t0 = time.time()
        r = self._base(c, company_id)
        cid = r.company_id
        reasons: List[str] = []
        ok = True

        # ---- 0) 环境前置：**跑不起来的用例要 SKIP，不要 FAIL** ----
        # 失败会把人引去修一个不存在的 bug。联网类用例依赖外部世界：
        # 付费通道没开通、或所有免 Key 通道都拿不到相关结果时，这条在此刻无解。
        if (c.category or "") == "web" and self.web_search is not None:
            gate = self._web_environment_gate()
            if gate:
                r.skipped = True
                r.skip_reason = gate
                r.reasons = [gate]
                r.duration_ms = int((time.time() - t0) * 1000)
                return r

        # ---- 1) 工具路由：期望工具必须真实注册 ----
        if c.expect_tools and self.tools is not None:
            missing = [t for t in c.expect_tools if t not in available]
            if missing:
                ok = False
                reasons.append(f"缺少期望工具: {missing}")

        # ---- 2) 安全护栏：拦截与放行都要有回归 ----
        # 原先只认 ``expectBlocked: true``，于是"不许误杀"这一侧根本没有断言 ——
        # 写一条正常业务问法进去也会"通过"，护栏越收越紧却没人发现。
        # 误杀的代价不比漏拦小：「导出客户名单做流失分析」被拦掉，用户只会觉得这系统不能干活。
        if c.expect_blocked is not None:
            blocked = False
            try:
                self.guardrail.input_check(c.question)
            except Exception:  # noqa: BLE001 - 抛出即视为被拦截
                blocked = True
            r.blocked = blocked
            if c.expect_blocked and not blocked:
                ok = False
                reasons.append("安全护栏未拦截该问题（期望被拦截）")
            elif not c.expect_blocked and blocked:
                ok = False
                reasons.append(
                    "安全护栏误杀：这是正常业务问法，不该被拦截"
                    "（软敏感词必须配合索取动作、且没有分析意图才拦）")

        # ---- 2.5) 长期记忆：抽取契约 + 「记忆不是证据」 ----
        if r.category == "memory":
            mok, mreasons = self._check_memory(c, r)
            ok = ok and mok
            reasons.extend(mreasons)

        # ---- 3) 内部知识库召回 ----
        # memory 类必须排除在外：它的 ``expectKeywords`` 指的是"记忆内容里该出现的词"，
        # 不是"知识库里该召回的词"。不排除的话，记忆用例会去撞知识库召回断言而整片假失败。
        check_rag = (r.category not in ("web", "citation", "memory")
                     and (bool(c.expect_keywords)
                          or (c.min_rag_score or 0) > 0
                          or c.max_rag_score is not None
                          or c.max_rag_raw_score is not None
                          or c.max_raw_ratio_vs_reference is not None
                          or (c.min_rag_raw_score or 0) > 0))
        if check_rag and self.rag is not None:
            k = c.top_k or 5
            hits = self.rag.search(cid, c.question, k)
            corpus = " ".join((h.title or "") + " " + (h.snippet or "") for h in hits)
            missing_kw = [kw for kw in (c.expect_keywords or []) if kw not in corpus]
            if missing_kw:
                ok = False
                reasons.append(f"RAG 未召回关键词: {missing_kw}")
            top = hits[0].score if hits else 0.0
            if c.min_rag_score is not None and top < c.min_rag_score:
                ok = False
                reasons.append(f"Top 相关度 {top} 低于阈值 {c.min_rag_score}")
            # 负样本：库里没有这个话题，召回分就该低。
            # 分高了说明检索在「自信地召回错东西」—— 这比漏召回危险得多。
            if c.max_rag_score is not None and top > c.max_rag_score:
                ok = False
                reasons.append(f"无关问题的 Top 相关度 {top} 高于上限 {c.max_rag_score}"
                               "（检索把不相关内容当成高相关召回了）")
            r.rag_top_score = round2(top)
            r.rag_hits = len(hits)
            raw = getattr(hits[0], "raw_score", None) if hits else None
            if raw is not None:
                r.rag_top_raw_score = round2(float(raw))
                if c.min_rag_raw_score is not None and c.min_rag_raw_score > 0 and float(raw) < c.min_rag_raw_score:
                    ok = False
                    reasons.append(f"Top 原始相关度 {round2(float(raw))} 低于绝对阈值 {c.min_rag_raw_score}")
                if c.max_rag_raw_score is not None and float(raw) > c.max_rag_raw_score:
                    ok = False
                    reasons.append(f"无关问题的 Top 原始相关度 {round2(float(raw))} 高于上限 {c.max_rag_raw_score}"
                                   "（检索把不相关内容当高相关召回了）")
            elif c.max_rag_raw_score is not None:
                # 无向量通道时判不了。如实说一句，但不算失败 —— 不能因为量不出就判它违规。
                reasons.append("无向量通道，缺少可比的原始分，本条负样本未做绝对阈值判定")

            # 相对判据：用比值而不是绝对值，让它不随 embedding 模型更换而整体漂移
            if c.max_raw_ratio_vs_reference is not None and c.reference_question:
                ref_cid = c.reference_company_id if c.reference_company_id is not None else cid
                try:
                    ref_hits = self.rag.search(ref_cid, c.reference_question, k)
                    ref_raw = getattr(ref_hits[0], "raw_score", None) if ref_hits else None
                    if ref_raw is None or float(ref_raw) <= 0:
                        reasons.append("参照问题未产出可比的原始分，本条相对判据未执行")
                    else:
                        self_raw = float(raw or 0)
                        ratio = self_raw / float(ref_raw)
                        if ratio > c.max_raw_ratio_vs_reference:
                            ok = False
                            reasons.append(
                                f"不相关问题的原始分 {round2(self_raw)} 相对参照问题 {round2(float(ref_raw))}"
                                f" 的比值 {round2(ratio)} 高于上限 {c.max_raw_ratio_vs_reference}"
                                "（检索没把相关与不相关拉开差距）")
                        else:
                            reasons.append(
                                f"相对判据通过：不相关 {round2(self_raw)} / 参照 {round2(float(ref_raw))}"
                                f" = {round2(ratio)} ≤ {c.max_raw_ratio_vs_reference}")
                except Exception as e:  # noqa: BLE001
                    reasons.append(f"参照问题检索异常（不影响本次判定）: {e}")
            if not hits and c.expect_keywords:
                reasons.append("未召回任何文档（该企业知识库可能为空）")

        # ---- 4) 联网检索：必须返回带 URL 的结构化来源 ----
        if c.expect_web:
            if self.web_search is None or not self.web_search.is_enabled():
                ok = False
                reasons.append("联网检索未启用（web_search 不可用，检查 app.web-search 配置与 API Key）")
            else:
                k = c.top_k or 5
                state["web"] += 1
                out = self.web_search.search(c.question, k)
                if not out.ok():
                    ok = False
                    reasons.append(f"联网检索失败: {out.error}")
                else:
                    src = out.sources or []
                    r.web_hits = len(src)
                    r.web_top_title = src[0].title if src else None
                    need = c.min_web_results or 1
                    if len(src) < need:
                        ok = False
                        reasons.append(f"来源条数 {len(src)} 少于要求 {need}")
                    no_url = sum(1 for s in src if not (getattr(s, "url", "") or "").strip())
                    if no_url:
                        ok = False
                        reasons.append(f"有 {no_url} 条来源缺少 URL，无法标注来源")
                    if c.expect_web_keywords:
                        joined = " ".join(
                            f"{getattr(s, 'title', '')} {getattr(s, 'url', '')} {getattr(s, 'site_name', '')}"
                            for s in src)
                        miss = [kw for kw in c.expect_web_keywords if kw not in joined]
                        if miss:
                            ok = False
                            reasons.append(f"来源中未出现关键词: {miss}")

        # ---- 5) 引用核对（固定样本，零 token） ----
        if r.category in ("citation", "repair"):
            ev = self._fixture_evidence(c)
            g = self.grounding.verify(c.answer_fixture, ev)
            r.citation_web_cited = len(g.web_cited)
            r.citation_dangling = len(g.web_dangling) + len(g.kb_dangling)
            r.citation_summary = self.grounding.summarize(g)
            if c.expect_ok is not None and g.ok != c.expect_ok:
                ok = False
                reasons.append(f"引用核对 ok 期望 {c.expect_ok} 实际 {g.ok}（{g.issues}）")
            if c.expect_dangling_web is not None and len(g.web_dangling) != c.expect_dangling_web:
                ok = False
                reasons.append(f"悬空网页引用数期望 {c.expect_dangling_web} 实际 {len(g.web_dangling)}")
            if c.expect_dangling_kb is not None and len(g.kb_dangling) != c.expect_dangling_kb:
                ok = False
                reasons.append(f"悬空知识库引用数期望 {c.expect_dangling_kb} 实际 {len(g.kb_dangling)}")
            if c.expect_min_web_cited is not None and len(g.web_cited) < c.expect_min_web_cited:
                ok = False
                reasons.append(f"可核对网页引用数 {len(g.web_cited)} 少于 {c.expect_min_web_cited}")

        # ---- 5b) 护栏闭环：缺失引用能否被自动补齐 ----
        if r.category == "repair":
            ev = self._fixture_evidence(c)
            rep = self.grounding.repair(c.answer_fixture, ev)
            web_n = sum(1 for x in rep.repairs if str(x.get("type")) == "web")
            kb_n = sum(1 for x in rep.repairs if str(x.get("type")) == "kb")
            ref_n = sum(1 for x in rep.repairs if str(x.get("type")) == "reference")
            r.repair_web, r.repair_kb, r.repair_reference = web_n, kb_n, ref_n
            if c.expect_repaired_web is not None and web_n != c.expect_repaired_web:
                ok = False
                reasons.append(f"自动补齐的网页引用数期望 {c.expect_repaired_web} 实际 {web_n}")
            if c.expect_repaired_kb is not None and kb_n != c.expect_repaired_kb:
                ok = False
                reasons.append(f"自动补齐的知识库引用数期望 {c.expect_repaired_kb} 实际 {kb_n}")
            if c.expect_reference_list is True and ref_n == 0:
                ok = False
                reasons.append("未自动补齐「参考来源」清单")
            if c.expect_reference_list is False and ref_n > 0:
                ok = False
                reasons.append("不应补「参考来源」清单，实际补了")
            g2 = self.grounding.verify(rep.answer, ev)
            r.repair_dangling_after = len(g2.web_dangling) + len(g2.kb_dangling)
            if c.expect_dangling_web_after is not None and len(g2.web_dangling) != c.expect_dangling_web_after:
                ok = False
                reasons.append(f"补齐后悬空网页引用数期望 {c.expect_dangling_web_after} 实际 {len(g2.web_dangling)}")
            if c.expect_all_sources_cited_after and g2.web_unused:
                ok = False
                reasons.append(f"补齐后仍有来源未被引用: {g2.web_unused}")

        # ---- 5c) 路由质量：这次该派谁上场 ----
        if r.category == "route" and self.agent_router is not None:
            try:
                rt = self.agent_router.route(c.question, True)
                agent_ids = [a.id for a in (rt.agents or [])]
                tool_names = list(rt.tools or [])
                r.route_agents, r.route_tools = agent_ids, tool_names
                if c.expect_agents:
                    miss = [a for a in c.expect_agents if a not in agent_ids]
                    if miss:
                        ok = False
                        reasons.append(f"未派出期望专员 {miss}，实际派出 {agent_ids}")
                if c.expect_tools:
                    miss = [t for t in c.expect_tools if t not in tool_names]
                    if miss:
                        ok = False
                        reasons.append(f"路由工具白名单缺少 {miss}，实际开放 {tool_names}")
                if c.expect_tools_absent:
                    bad = [t for t in c.expect_tools_absent if t in tool_names]
                    if bad:
                        ok = False
                        reasons.append(f"不应开放的工具被开放: {bad}（会把成本与噪声放大）")
            except Exception as e:  # noqa: BLE001
                ok = False
                reasons.append(f"路由判定异常: {e}")

        # ---- structure 类：输出契约 ----
        if r.category == "structure":
            ans = c.answer_fixture or ""
            miss = [s for s in (c.expect_sections or []) if s not in ans]
            bad = [p for p in (c.forbid_phrases or []) if p in ans]
            contract_ok = not miss and not bad
            # 负样本：夹具**故意**违规，用例要验的是「契约检查能不能把它揪出来」。
            # 曾经没有这个语义：负样本被当成"必须满足契约"来断言，于是永远失败，
            # structure 维度恒为 33.33，真正的契约回归反而被噪音盖住。
            if c.expect_contract_violation:
                if contract_ok:
                    ok = False
                    reasons.append("负样本未被判为不合规：契约检查漏检（该样本刻意缺少 "
                                   f"{c.expect_sections or []} 或含有 {c.forbid_phrases or []}）")
            else:
                if miss:
                    ok = False
                    reasons.append(f"成稿缺少必须章节: {miss}")
                if bad:
                    ok = False
                    reasons.append(f"成稿出现禁止表述: {bad}（内部/外部结论混写）")

        # ---- 6) 工具探针：真正执行一次取数 ----
        if c.probe_tool and self.tools is not None:
            try:
                from ..ai.tool_context import ToolContext
                ctx = ToolContext(company_id=cid, top_k=(c.top_k or 5))
                tr = self.tools.execute_with_meta(c.probe_tool, c.probe_args or {}, ctx)
                text = tr.text or ""
                r.probe_tool = c.probe_tool
                r.probe_preview = text[:160]
                if not text.strip() or '"error"' in text:
                    ok = False
                    reasons.append(f"工具探针 {c.probe_tool} 未取到数据: {text}")
                if tr.sources:
                    r.probe_sources = len(tr.sources)
            except Exception as e:  # noqa: BLE001
                ok = False
                r.probe_tool = c.probe_tool
                reasons.append(f"工具探针 {c.probe_tool} 抛异常: {e}")

        r.passed = ok
        r.reasons = reasons
        # ---- 后置定性：跑完了才知道"为什么没结果" ----
        # 前置检查只能看清配置（Key/模型/端点），而「插件没开通」「账号到额」
        # 「搜索引擎返回一堆无关网页」这些**只有真跑一次才暴露**。
        # 让它们按 FAIL 出报告，会让人去修一个不存在的 bug。
        if (c.category or "") == "web" and not ok and not r.skipped:
            env = _environmental_web_reason(reasons)
            if env:
                r.skipped = True
                r.skip_reason = env
        r.duration_ms = int((time.time() - t0) * 1000)
        return r

    @staticmethod
    def _fixture_evidence(c: EvalCase) -> List[Dict[str, Any]]:
        ev: List[Dict[str, Any]] = []
        if c.fixture_web_sources:
            ev.append({"tool": "web_search", "sourceType": "web", "sources": c.fixture_web_sources})
        if c.fixture_kb_sources:
            ev.append({"tool": "search_knowledge", "sourceType": "knowledge",
                       "sources": c.fixture_kb_sources})
        return ev

    # -- live（真实大模型）端到端用例 --------------------------------

    def _run_live_case(self, c: EvalCase, company_id: Optional[int],
                       created: List[int]) -> CaseResult:
        """真实调用大模型跑一次完整分析，用真实成稿判定。

        确定性用例只验「检查逻辑对不对」，回答不了用户真正在意的两件事 ——
        **等多久**、**出来的东西能不能用**。这一类才是唯一能反映真实体感的部分。
        """
        t0 = time.time()
        r = self._base(c, company_id)
        r.category = "e2e"
        reasons: List[str] = []
        ok = True
        cid = r.company_id
        top_k = c.top_k or 5

        if self.agent_service is None:
            r.passed = False
            r.reasons = ["AgentService 未装配，无法执行端到端用例"]
            r.duration_ms = int((time.time() - t0) * 1000)
            return r

        answer: Optional[str] = None
        trace: List[str] = []
        evidence: List[Dict[str, Any]] = []
        try:
            # liveDepth 必须真的传给 AgentService。用例里写了 quick 却跑成 standard 档，
            # 后果有两层：①拿 expectMaxLlmCalls=1 去撞 3 次调用，必然假失败；
            # ②报告上「模型调用」列的数字与实际档位不匹配，无法解释"为什么这条慢"。
            out = self.agent_service.analyze(cid, c.question, top_k, user_id=None,
                                             depth=c.live_depth or None)
            if out.analysis_id:
                created.append(out.analysis_id)
            answer = out.content
            trace = list(out.tool_trace or [])
            evidence = list(out.evidence or [])
            # 轨迹里带的是 "get_metrics(5ms)" / "search_knowledge(预取 370ms)"（给人看的），
            # 但 expectTools 断言比的是工具名。**必须剥掉装饰再存进 traceTools**，
            # 否则包含判断恒为假 —— 这正是 5 条端到端用例全被判「模型未调用期望工具」的原因。
            r.trace_tools = [bare_tool_name(t) for t in trace]
            r.trace_detail = trace
            diag = out.diagnostics or {}
            tr = diag.get("trace") or {}
            # 模型/工具调用次数取真实留痕（Java 侧读的是 ai_analysis.llm_calls）：
            # 界面「模型调用」列与 expectMaxLlmCalls 预算断言都靠它，
            # 早先这两个字段根本没被赋值，于是列恒显示「—」、预算断言恒不触发。
            r.llm_calls = _as_int(tr.get("llmCalls"))
            r.tool_calls = _as_int(tr.get("toolCalls"))
            r.single_round = None if r.llm_calls is None else r.llm_calls <= 1
            r.degrade_level = out.degrade_level
            r.model = self.describe_model()
            # 记**实际生效**的档位：用例没写就落到服务端默认档，不留空 ——
            # 报告上「耗时/模型调用」两列没有档位做参照就等于一堆无法解释的数字。
            r.depth = ((c.live_depth or "").strip()
                       or (get_settings().agent.default_depth or "standard"))
            r.grounded = bool(diag.get("grounded", True))
            if answer:
                r.answer_chars = len(answer)
                r.answer_preview = answer[:180]
            web = 0
            for e in evidence:
                if str(e.get("sourceType")) == "web":
                    web += len(e.get("sources") or [])
            r.web_hits = web or None

            # 回填检索指标：端到端跑完只能知道「成稿多长、等多久」，
            # 看不出「检索这一步好不好」。补一次零 token 检索，于是
            # 「结论很差」时能分辨是检索没召回到，还是模型没用上。
            try:
                if self.rag is not None:
                    hits = self.rag.search(cid, c.question, top_k)
                    r.rag_hits = len(hits)
                    if hits:
                        r.rag_top_score = round2(hits[0].score)
                        raw = getattr(hits[0], "raw_score", None)
                        if raw is not None:
                            r.rag_top_raw_score = round2(float(raw))
            except Exception as e:  # noqa: BLE001
                reasons.append(f"检索指标回填异常（不影响本次判定）: {e}")
        except Exception as e:  # noqa: BLE001
            ok = False
            reasons.append(f"端到端分析失败: {e}")
            answer = None

        if ok:
            if not (answer or "").strip():
                ok = False
                reasons.append("成稿为空（这是用户看到的「没有结果」，必须算失败）")
            else:
                if c.expect_answer_min_chars is not None and len(answer) < c.expect_answer_min_chars:
                    ok = False
                    reasons.append(f"成稿仅 {len(answer)} 字，低于下限 {c.expect_answer_min_chars}")
                if c.expect_sections:
                    miss = [s for s in c.expect_sections if s not in answer]
                    if miss:
                        ok = False
                        reasons.append(f"成稿缺少契约章节: {miss}")
                try:
                    g = self.grounding.verify(answer, evidence)
                    dangling = len(g.web_dangling) + len(g.kb_dangling)
                    r.citation_dangling = dangling
                    r.citation_web_cited = len(g.web_cited)
                    if dangling:
                        ok = False
                        reasons.append(f"存在 {dangling} 处悬空引用（引用了不存在的来源）")
                except Exception as e:  # noqa: BLE001
                    reasons.append(f"引用核对异常: {e}")
            if c.expect_tools:
                have = set(r.trace_tools or [])
                miss = [t for t in c.expect_tools if t not in have]
                if miss:
                    ok = False
                    reasons.append(f"模型未调用期望工具 {miss}，实际轨迹 {trace}")
            # 用户明确要求联网的用例：必须有真实外部来源，且不得用「本次未联网核实」糊过去。
            # 这是"联网开关看起来启用了、实际没联网"唯一可判定的契约 ——
            # 报告头写着「联网搜索: 已启用」只说明**有能力**联网，不说明**这一次**联了。
            if c.expect_web:
                min_web = c.min_web_results if c.min_web_results is not None else 1
                got = r.web_hits or 0
                if got < min_web:
                    ok = False
                    reasons.append(
                        f"用户要求联网，但本次只取得 {got} 条外部来源（期望 ≥{min_web}）："
                        "web_search 未被调用，或返回的来源未通过相关性复核")
                if answer and "本次未联网核实" in answer:
                    ok = False
                    reasons.append(
                        "用户要求联网，成稿却写了「本次未联网核实」——"
                        "模型没有真正调用 web_search，属联网要求未生效")
            if r.degrade_level and r.degrade_level != "NONE":
                ok = False
                reasons.append(f"本次分析被降级（{r.degrade_level}）：端到端链路并非完整走通，"
                               "不能据此认定能力正常")
            ms = int((time.time() - t0) * 1000)
            if c.max_duration_ms is not None and ms > c.max_duration_ms:
                ok = False
                reasons.append(f"端到端耗时 {ms}ms 超过预算 {c.max_duration_ms}ms")
            if c.expect_max_llm_calls is not None and r.llm_calls is not None \
                    and r.llm_calls > c.expect_max_llm_calls:
                ok = False
                reasons.append(f"模型调用了 {r.llm_calls} 次，超过预期上限 {c.expect_max_llm_calls}"
                               " 次（多出来的每一次都是用户要多等的一整轮）")

        r.passed = ok
        r.reasons = reasons
        r.duration_ms = int((time.time() - t0) * 1000)
        return r
