"""待审批动作（对应 Java ``ActionApprovalService``）。

**设计红线：AI 提议的动作一律只入审批队列，绝不直接落业务数据。**

写动作（建工单 / 改责任人 / 改事件状态）一旦让模型直接执行，
一次误判就会真实地改变业务状态，而且没有回滚路径。
所以工具只负责「提议」，由有权限的人在界面上确认后才生效。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, insert, select, update

from ..db import tables as T
from ..db.engine import Database, get_db

log = logging.getLogger(__name__)

ST_PENDING = "PENDING"
ST_APPROVED = "APPROVED"
ST_REJECTED = "REJECTED"


class ActionApprovalService:
    """审批队列的读写。"""

    def __init__(self, db: Optional[Database] = None) -> None:
        self.db = db or get_db()

    def submit(self, company_id: Optional[int], action_type: str, args: Dict[str, Any],
               reason: str, risk_level: Optional[str] = None,
               analysis_id: Optional[int] = None, source: str = "agent") -> int:
        """入队。返回记录 id（0 表示入库失败）。"""
        row = {
            "company_id": company_id,
            "analysis_id": analysis_id,
            "action_type": action_type,
            "action_args": json.dumps(args or {}, ensure_ascii=False),
            "reason": (reason or "")[:2000],
            "risk_level": (risk_level or "")[:20] or None,
            "status": ST_PENDING,
            "source": (source or "agent")[:20],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        # insert_id 而不是 execute：后者返回影响行数（恒为 1），
        # 拿它当 id 会让"审批 #1"永远指向同一条记录。
        rid = self.db.insert_id(insert(T.ai_action_approval).values(**row))
        log.info("[Approval] 入队 %s（企业 %s）", action_type, company_id)
        return int(rid or 0)

    def pending(self, company_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        stmt = select(T.ai_action_approval).where(T.ai_action_approval.c.status == ST_PENDING)
        if company_id is not None:
            stmt = stmt.where(T.ai_action_approval.c.company_id == company_id)
        return self.db.fetch_all(stmt.order_by(desc(T.ai_action_approval.c.id)).limit(limit))

    def decide(self, approval_id: int, approve: bool, approver_user_id: Optional[int] = None,
               comment: Optional[str] = None) -> bool:
        """批准 / 驳回。**只改状态**，真正的业务写入由执行方在确认后完成。"""
        now = datetime.now()
        n = self.db.execute(
            update(T.ai_action_approval)
            .where(T.ai_action_approval.c.id == approval_id)
            .values(status=ST_APPROVED if approve else ST_REJECTED,
                    approver_user_id=approver_user_id,
                    approve_comment=(comment or "")[:500] or None,
                    approved_at=now if approve else None,
                    updated_at=now))
        return bool(n)
