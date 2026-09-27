"""通知通道 —— 对应 Java ``NotificationService``。

把 Agent 提议的「通知责任人」真正落地。分两层：

1. **必达层**：每次通知都写操作日志（谁、什么时候、通知了谁、内容是什么），可审计、可追溯；
2. **投递层**：配置了 ``APP_NOTIFY_WEBHOOK_URL`` 就真实推送（企业微信/钉钉/自建网关均可），
   没配置则明确回执为"已登记待人工投递"，**不假装已送达**。

Java 原来的实现只在返回文案里写一句"系统内消息通道尚未接入，需人工完成实际通知"——
审批通过了却没有人真的被通知到，这在企业流程里等于动作没做成。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import request as urlrequest

from ..config import get_settings

log = logging.getLogger(__name__)


@dataclass
class Receipt:
    delivered: bool = False
    channel: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"delivered": self.delivered, "channel": self.channel, "detail": self.detail}


def _abbreviate(s: str) -> str:
    return s[:200] + "…" if len(s) > 200 else s


class NotificationService:
    def __init__(self, webhook_url: Optional[str] = None, timeout: Optional[int] = None) -> None:
        s = get_settings()
        notify = getattr(s, "notify", None)
        self.webhook_url = (webhook_url if webhook_url is not None
                            else (getattr(notify, "webhook_url", "") or ""))
        self._timeout = timeout if timeout is not None else int(
            getattr(notify, "timeout_seconds", 8) or 8)

    def timeout(self) -> int:
        return max(2, min(60, self._timeout))

    def notify_owner(self, event_id: Optional[int], assignee: Optional[int],
                     message: Optional[str]) -> Receipt:
        """通知风险事件责任人。"""
        from ..core.security import current_user_id

        body = message.strip() if message and message.strip() else "(无正文)"
        detail = ("风险事件#%s 责任人#%s 内容：%s"
                  % (event_id, "默认" if assignee is None else assignee, _abbreviate(body)))
        self._audit("AI_NOTIFY_OWNER", "risk_event", event_id, detail, current_user_id())

        url = (self.webhook_url or "").strip()
        if not url:
            return Receipt(False, "REGISTERED",
                           "已登记待人工投递（未配置 APP_NOTIFY_WEBHOOK_URL）：" + _abbreviate(body))
        try:
            from ..agent.trace import TraceContext

            payload = {"eventId": event_id, "assigneeUserId": assignee,
                       "message": body, "traceId": TraceContext.current()}
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urlrequest.Request(url, data=data, method="POST", headers={
                "Content-Type": "application/json; charset=utf-8"})
            with urlrequest.urlopen(req, timeout=self.timeout()) as resp:  # noqa: S310
                code = resp.getcode()
            ok = 200 <= code < 300
            return Receipt(ok, "WEBHOOK",
                           ("已投递（HTTP %d）" % code) if ok else ("投递失败（HTTP %d）" % code))
        except Exception as e:  # noqa: BLE001
            log.warning("[Notify] webhook 投递失败: %s", e)
            return Receipt(False, "WEBHOOK_FAILED", "投递失败：" + str(e))

    def _audit(self, action: str, resource_type: str, resource_id: Optional[int],
               detail: str, user_id: Optional[int]) -> None:
        """审计落库失败也要继续尝试投递，但不能假装审计成功了。"""
        try:
            from ..db import get_db
            from ..db import tables as T
            from sqlalchemy import insert

            get_db().execute(insert(T.operation_log).values(
                user_id=user_id, action=action, resource_type=resource_type,
                resource_id=resource_id, detail=detail[:2000], ip_address=""))
        except Exception as e:  # noqa: BLE001
            log.warning("[Notify] 审计落库失败: %s", e)
