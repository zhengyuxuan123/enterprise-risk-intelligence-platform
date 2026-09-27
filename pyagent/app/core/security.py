"""请求级安全上下文（对应 Java 的 ``CurrentUserService`` + ``SecurityContext``）。

Python 服务的定位是**内网服务**，由 Spring 按白名单转发 ``/api/ai/*``：
登录态与数据权限都在 Spring 侧判过了，这里只接收结果，不再重复鉴权。

两个必须记住的点：

1. **用户 id 为空会让落库直接失败**（``ai_analysis.user_id`` 是 NOT NULL）。
   而本服务没有自己的登录态，所以只能靠转发头。头也没有时退到配置默认，
   不能让"没人传"变成"整条分析记录写不进去"。
2. **线程池里跑的任务拿不到 contextvar**。工具在线程池并发执行，
   Java 版踩过的坑（``RagService → KnowledgeService`` 依赖 ThreadLocal 导致 NPE）
   在这里同样存在 —— 所以提供了 ``snapshot()/attach()`` 显式传递。
"""

from __future__ import annotations

import contextvars
from typing import Any, Dict, Optional

from ..config import get_settings

_USER_ID: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "pyagent_user_id", default=None)
_COMPANY_ID: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "pyagent_company_id", default=None)
#: 数据权限判定回调（由转发层注入；None = 不做越权检查，批处理/评估脚本）
_SCOPE: contextvars.ContextVar[Any] = contextvars.ContextVar("pyagent_scope", default=None)

HEADER_USER_ID = "X-User-Id"
HEADER_COMPANY_ID = "X-Company-Id"
HEADER_LEVEL = "X-Data-Level"
HEADER_DEPTS = "X-Visible-Depts"


class CurrentUser:
    """当前请求的用户与可视范围。"""

    @staticmethod
    def user_id() -> Optional[int]:
        v = _USER_ID.get()
        if v is not None:
            return v
        try:
            d = int(get_settings().default_user_id)
            return d if d > 0 else None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def company_id() -> Optional[int]:
        return _COMPANY_ID.get()

    @staticmethod
    def level() -> Optional[int]:
        return _LEVEL.get()

    @staticmethod
    def visible_depts() -> Optional[Any]:
        return _DEPTS.get()

    @staticmethod
    def scope() -> Any:
        return _SCOPE.get()

    # -- 设置（中间件用） --------------------------------------------

    @staticmethod
    def set_user_id(v: Optional[int]) -> None:
        _USER_ID.set(v)

    @staticmethod
    def set_company_id(v: Optional[int]) -> None:
        _COMPANY_ID.set(v)

    @staticmethod
    def set_level(v: Optional[int]) -> None:
        _LEVEL.set(v)

    @staticmethod
    def set_visible_depts(v: Optional[Any]) -> None:
        _DEPTS.set(v)

    @staticmethod
    def set_scope(fn: Any) -> None:
        _SCOPE.set(fn)

    # -- 跨线程传递 --------------------------------------------------

    @staticmethod
    def snapshot() -> Dict[str, Any]:
        """抓一份当前上下文，供线程池里的任务显式带上。"""
        return {"user_id": _USER_ID.get(), "company_id": _COMPANY_ID.get(),
                "level": _LEVEL.get(), "depts": _DEPTS.get(), "scope": _SCOPE.get()}

    @staticmethod
    def attach(snap: Optional[Dict[str, Any]]) -> None:
        """把 :meth:`snapshot` 抓到的上下文装回当前线程。"""
        if not snap:
            return
        _USER_ID.set(snap.get("user_id"))
        _COMPANY_ID.set(snap.get("company_id"))
        _LEVEL.set(snap.get("level"))
        _DEPTS.set(snap.get("depts"))
        _SCOPE.set(snap.get("scope"))

    @staticmethod
    def clear() -> None:
        _USER_ID.set(None)
        _COMPANY_ID.set(None)
        _LEVEL.set(None)
        _DEPTS.set(None)
        _SCOPE.set(None)


_LEVEL: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "pyagent_data_level", default=None)
_DEPTS: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "pyagent_visible_depts", default=None)


def current_user_id() -> Optional[int]:
    """模块级便捷入口（等价 ``CurrentUser.user_id()``）。"""
    return CurrentUser.user_id()


def with_user(user_id: Optional[int], fn):
    """以指定用户身份执行一段代码，用完还原。

    MQ 消费者 / 定时任务这类"没有请求"的线程拿不到上下文，
    而 Agent 全链路（权限校验、记忆、工具落日志）都依赖当前主体。
    """
    prev = _USER_ID.get()
    _USER_ID.set(user_id)
    try:
        return fn()
    finally:
        _USER_ID.set(prev)
