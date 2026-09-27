"""事件驱动的主动风险研判 —— 对应 Java ``ProactiveRiskService``。

**接在哪**（Java 侧的说明，Python 版保持一致）：
指标越阈值 → 规则引擎产出风险事件 → MQ 消费者 ``MetricRiskListener`` 跑完规则后，
若本次真的产生了 HIGH 级事件，就由这里再触发一次 Agent 研判。
之所以不新增队列：现有队列里传的是 metricId，且已被 ``MetricRiskListener`` 独占消费，
再加一个消费者监听同一队列会与它轮流分食消息，破坏原有链路。
接在下游既拿到完整上下文，又不动 MQ 拓扑。

**成本闸门（两道）**：

1. **授权闸门** —— 后台自动调用默认**关闭**（``APP_AI_AUTO_CONSUME=false``），
   未经使用者主动触发的调用一律不烧 token，因为免费/试用额度是一次性的；
2. **配额闸门** —— 每家企业每天最多研判 ``APP_PROACTIVE_DAILY_LIMIT`` 次（默认 5），
   计数落在 ``ai_proactive_alert`` 表。超了就直接返回、不再调模型——
   这是"事件驱动"最容易失控的地方：指标一抖就触发，一天能烧掉几百次调用。

**分派**：HIGH → ALERTED；MEDIUM → PENDING_APPROVAL；LOW/未知 → 仅归档（ARCHIVED）。
无论哪一档，动作都只是**提议**，照例要人点头才执行。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func, insert, select, update

from ..config import get_settings
from ..db import get_db
from ..db import tables as T

log = logging.getLogger(__name__)

LEVEL_PAT = re.compile(r"\"risk_level\"\s*:\s*\"([A-Za-z]+)\"")
HEADLINE_PAT = re.compile(r"\"headline\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"")


def _build_question(trigger_type: str, trigger_ref: Optional[str]) -> str:
    return ("【系统自动预警触发，非用户提问】触发源：" + trigger_type
            + "（" + (trigger_ref or "未知") + "）。\n"
            + "请只依据本企业的内部数据（经营指标、风险事件、客户投诉、知识库）做一次快速研判：\n"
            + "1. 不要调用联网检索工具；\n"
            + "2. 最多调用 2 次取数工具，优先看最近的风险事件与指标趋势；\n"
            + "3. 输出风险等级（HIGH / MEDIUM / LOW）、一句话核心判断、1-2 条最该做的处置动作；\n"
            + "4. 证据不足就直说「证据不足，建议人工核查」，不要臆测。")


def _parse_level(js: Optional[str]) -> Optional[str]:
    if not js:
        return None
    m = LEVEL_PAT.search(js)
    return m.group(1).upper() if m else None


def _parse_headline(js: Optional[str]) -> Optional[str]:
    if not js:
        return None
    m = HEADLINE_PAT.search(js)
    if not m:
        return None
    s = m.group(1)
    return s[:300] if len(s) > 300 else s


class ProactiveRiskService:
    """不用等人提问，业务事件自己把 Agent 唤起。"""

    def __init__(self, agent_service=None, approvals=None, llm=None, db=None,
                 enabled: Optional[bool] = None, daily_limit: Optional[int] = None,
                 fresh_minutes: Optional[int] = None, system_username: Optional[str] = None,
                 system_user_id: Optional[int] = None) -> None:
        self.agent_service = agent_service
        self.approvals = approvals
        self.llm = llm
        self.db = db
        s = get_settings()
        pr = getattr(s, "proactive", None)
        self.enabled = enabled if enabled is not None else bool(
            getattr(pr, "enabled", True))
        self.daily_limit = daily_limit if daily_limit is not None else int(
            getattr(pr, "daily_limit", 5) or 5)
        self.fresh_minutes = fresh_minutes if fresh_minutes is not None else int(
            getattr(pr, "fresh_minutes", 5) or 5)
        self.system_username = system_username or getattr(
            pr, "system_username", "admin") or "admin"
        self.system_user_id = system_user_id

    # ------------------------------------------------------------------

    def evaluate(self, company_id: Optional[int], trigger_type: str,
                 trigger_ref: Optional[str] = None) -> Dict[str, Any]:
        r: Dict[str, Any] = {"companyId": company_id, "triggerType": trigger_type}
        if not self.enabled:
            r["skipped"] = "主动预警未启用（APP_PROACTIVE_ENABLED=false）"
            return r
        if company_id is None:
            r["skipped"] = "缺少企业ID"
            return r

        # 授权闸门：后台自动调用默认关闭。免费/试用额度是一次性的，
        # 最容易被跑干的就是没人盯着的后台任务，所以这里必须挡一道。
        if self.llm is not None and not self.llm.is_auto_consume_allowed():
            r["skipped"] = ("后台自动研判未授权：未经使用者主动触发，系统不消耗模型额度"
                            "（APP_AI_AUTO_CONSUME=false）。要开启请在「AI Agent 智能分析」页把"
                            "【后台自动研判】开关打开，或设环境变量 APP_AI_AUTO_CONSUME=true 后重启后端；"
                            "页面上的分析不受影响，随时可用。")
            log.info("主动预警跳过：后台自动消耗未授权（company=%s，trigger=%s）",
                     company_id, trigger_type)
            return r

        used = self._count_today(company_id)
        if used >= self.daily_limit:
            r["skipped"] = ("该企业今日研判已达上限 %d 次（已完成 %d 次），本次跳过以控制成本"
                            % (self.daily_limit, used))
            log.info("主动预警跳过：company=%s 今日已 %d 次，达到上限 %d",
                     company_id, used, self.daily_limit)
            return r

        question = _build_question(trigger_type, trigger_ref)

        # ---- 配额占位：先落一条 RUNNING 记录，再去做研判 ----
        # 原来等分析完成后才落库，而每日配额是按这张表计数的：
        # 一旦落库失败，这次消耗就"没发生过"，同一个事件可以被反复触发、反复烧额度。
        # 占位写不进去说明计数不可靠，此时宁可不做（fail-closed），也不能无上限地跑。
        db = self._db()
        tbl = T.ai_proactive_alert
        alert_id = db.insert_id(insert(tbl).values(
            company_id=company_id, trigger_type=trigger_type, trigger_ref=trigger_ref,
            question=question, risk_level="PENDING", dispatched="RUNNING",
            created_at=datetime.now()))
        if not alert_id:
            r["skipped"] = "配额记录不可用，已跳过本次研判以避免无上限消耗"
            return r

        answer_json: Optional[str] = None
        err: Optional[str] = None
        try:
            answer_json = self._run_agent(company_id, question)
        except Exception as e:  # noqa: BLE001
            err = str(e)
        if err:
            r["error"] = err
            return r

        level = _parse_level(answer_json)
        headline = _parse_headline(answer_json)
        dispatched = ("ALERTED" if level == "HIGH"
                      else ("PENDING_APPROVAL" if level == "MEDIUM" else "ARCHIVED"))

        # 中高危才生成待审批动作；低危只归档，不去打扰人
        approval_id: Optional[int] = None
        if level in ("HIGH", "MEDIUM"):
            try:
                approval_id = self._submit_approval(
                    company_id, trigger_type, trigger_ref, headline, level)
            except Exception as e:  # noqa: BLE001
                log.warning("主动预警：生成待审批动作失败 %s", e)

        # 回填占位记录：无论研判成功与否，这一条都已计入当日配额
        try:
            db.execute(update(tbl).where(tbl.c.id == alert_id).values(
                risk_level=level or "UNKNOWN", headline=headline,
                answer_json=answer_json, dispatched=dispatched, approval_id=approval_id))
            r["alertId"] = alert_id
        except Exception as e:  # noqa: BLE001
            log.warning("主动预警记录回填失败（配额已计入）：%s", e)

        r.update(riskLevel=level, dispatched=dispatched, approvalId=approval_id,
                 todayUsed=used + 1, dailyLimit=self.daily_limit)
        return r

    def evaluate_by_metric(self, metric_id: Optional[int]) -> Dict[str, Any]:
        """MQ 链路入口：指标越阈值被规则引擎判为高风险后调用。

        只在新产生了 HIGH 事件时才真正触发研判，避免"指标抖动一下就烧一次模型"。
        """
        if not self.enabled or metric_id is None:
            return {"skipped": "未启用或缺少指标ID"}
        db = self._db()
        m = db.fetch_one(select(T.business_metric).where(T.business_metric.c.id == metric_id))
        if not m or m.get("company_id") is None:
            return {"skipped": "未找到指标 %s" % metric_id}
        since = datetime.now() - timedelta(minutes=max(1, self.fresh_minutes))
        fresh = db.fetch_all(
            select(T.risk_event).where(T.risk_event.c.company_id == m["company_id"])
            .where(T.risk_event.c.risk_level == "HIGH")
            .where(T.risk_event.c.created_at >= since)
            .order_by(desc(T.risk_event.c.created_at)).limit(3))
        if not fresh:
            return {"skipped": "本次未产生新的高危风险事件，不触发研判（避免无谓消耗）"}
        return self.evaluate(m["company_id"], "METRIC_THRESHOLD", "metric:%d" % metric_id)

    def recent(self, company_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        tbl = T.ai_proactive_alert
        n = max(1, min(100, limit))
        return self._db().fetch_all(
            select(tbl).where(tbl.c.company_id == company_id)
            .order_by(desc(tbl.c.created_at)).limit(n))

    # ------------------------------------------------------------------

    def _run_agent(self, company_id: int, question: str) -> str:
        """自动研判不需要流式输出，token sink 传 None 走同步路径。

        以系统账号身份执行：MQ 消费者线程没有请求上下文，
        而 Agent 全链路（权限校验、记忆、工具落日志）都依赖当前主体，所以这里临时装配一个。
        """
        from ..core.security import with_user

        uid = self._system_user_id()
        if uid is None:
            raise RuntimeError("未找到系统触发账号「%s」，无法执行自动研判"
                               "（请配置 APP_PROACTIVE_SYSTEM_USERNAME）" % self.system_username)
        # 自动研判不需要流式输出 → 走同步 analyze，且不开多智能体（省一轮）
        out = with_user(uid, lambda: self.agent_service.analyze(
            company_id, question, 3, None, False, uid))
        return _json_or_none(out.answer_json)

    def _submit_approval(self, company_id: int, trigger_type: str,
                         trigger_ref: Optional[str], headline: Optional[str],
                         level: str) -> Optional[int]:
        if self.approvals is None:
            return None
        return self.approvals(
            company_id, "CREATE_TICKET",
            {"title": "自动预警处置：" + trigger_type,
             "detail": headline or "", "riskLevel": level},
            "事件驱动自动研判（%s %s），风险等级 %s" % (trigger_type, trigger_ref, level),
            level, None, "proactive")

    def _count_today(self, company_id: int) -> int:
        try:
            tbl = T.ai_proactive_alert
            return int(self._db().scalar(
                select(func.count(tbl.c.id)).where(
                    tbl.c.company_id == company_id).where(
                    tbl.c.created_at >= datetime.now().replace(
                        hour=0, minute=0, second=0, microsecond=0)),
                default=0))
        except Exception as e:  # noqa: BLE001
            log.warning("主动预警配额计数失败（按 0 处理）：%s", e)
            return 0

    def _system_user_id(self) -> Optional[int]:
        if self.system_user_id is not None:
            return self.system_user_id
        try:
            row = self._db().fetch_one(
                select(T.sys_user).where(T.sys_user.c.username == self.system_username))
            if row:
                self.system_user_id = row.get("id")
                return self.system_user_id
        except Exception:  # noqa: BLE001
            pass
        return None

    def _db(self):
        if self.db is None:
            self.db = get_db()
        return self.db


def _json_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return None
