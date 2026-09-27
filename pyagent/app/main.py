"""FastAPI 入口。

与 Java 版的差异（有意为之，都为了迁移期更好用）：

1. **端口默认 8081**（``PY_AGENT_PORT``），Java 仍是 8080 —— 两个实现必须能同时跑，
   否则"双跑对照"就无从谈起。
2. 异常处理器**逐条对齐** Java 的 ``GlobalExceptionHandler``：
   ``BusinessError`` → HTTP 400 + ``ApiResponse.fail(code, msg)``；
   其它异常 → HTTP 500 + ``ApiResponse.fail(500, msg)``。
   注意业务码放在 body 里、HTTP 状态码固定 —— 前端读的是 body。
3. 未移植的端点**返回 501 而不是假装成功**。迁移期最危险的不是"缺功能"，
   而是"接口在、行为不对" —— 那会让双跑对照得出完全错误的结论。
   （2026-09-22 起已无未移植端点：最后一处 ``/api/ai/diagnose`` 已补齐。）
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .api.agent import router as agent_router
from .api.ai import router as ai_router
from .config import get_settings
from .core.errors import BusinessError
from .core.logging import get_logger, setup_logging

log = get_logger("pyagent.main")


def create_app() -> FastAPI:
    s = get_settings()
    setup_logging(s.log_level)

    app = FastAPI(
        title="企业经营风险智能分析平台 · Agent（Python 实现）",
        version=__version__,
        description=(
            "Agent 栈的 Python 实现，与 Java 版并存。"
            "HTTP 契约与 Java 版一致，由 Spring 按开关转发。"
        ),
    )

    # ---------------------------------------------------------------- 异常
    @app.exception_handler(BusinessError)
    async def _business_error(_: Request, exc: BusinessError) -> JSONResponse:
        # Java：ResponseEntity.badRequest()（HTTP 恒为 400，业务码在 body）
        return JSONResponse(
            status_code=400, content={"code": exc.code, "message": exc.message, "data": None}
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("[GlobalExceptionHandler] 未处理异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": str(exc) if str(exc) else "服务器异常",
                "data": None,
            },
        )

    # ---------------------------------------------------------------- 上下文
    @app.middleware("http")
    async def _context(request: Request, call_next):
        """把 Spring 转发头装进请求级上下文。

        登录态与数据权限都在 Spring 侧判过了，这里只接收结果；
        但 ``ai_analysis.user_id`` 是 NOT NULL，拿不到就必须有兜底，
        否则整条分析记录会因为一个空字段写不进去。
        """
        from .agent.trace import TraceContext
        from .core.security import CurrentUser

        CurrentUser.clear()
        TraceContext.clear()
        try:
            TraceContext.set(request.headers.get(TraceContext.HEADER))
            uid = request.headers.get("X-User-Id")
            if uid:
                CurrentUser.set_user_id(int(uid))
            lv = request.headers.get("X-Data-Level")
            if lv:
                CurrentUser.set_level(int(lv))
            depts = request.headers.get("X-Visible-Depts")
            if depts:
                CurrentUser.set_visible_depts({int(x) for x in depts.split(",") if x.strip()})
        except Exception:  # noqa: BLE001 - 头解析失败不该让请求 500
            pass
        try:
            response = await call_next(request)
        finally:
            CurrentUser.clear()
            TraceContext.clear()
        return response

    # ---------------------------------------------------------------- 路由
    app.include_router(ai_router)
    app.include_router(agent_router)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        """进程级探活（不含依赖）。给容器/守护进程用，与业务体检 ``/api/ai/health`` 分开。"""
        return {"status": "ok", "version": __version__}

    @app.on_event("startup")
    def _startup() -> None:
        log.info(
            "[PyAgent] 启动完成 端口=%s 配置：always-offer=%s 检索模式=%s embedding=%s",
            s.port,
            s.web_search.always_offer,
            s.web_search.mode,
            s.rag.embedding_mode,
        )
        # 索引可能处于"记录了向量空间、实际一条向量都没有"的状态 ——
        # 那是增量判定的盲区（内容 hash 没变就永远跳过），重启不会自愈，
        # 表现为语义检索静默失效、原始分恒 0。本地向量零成本，直接补算。
        try:
            from .api.agent import heal_rag_vectors_if_needed

            note = heal_rag_vectors_if_needed()
            if note:
                log.warning("[PyAgent] 向量自愈：%s", note)
        except Exception as e:  # noqa: BLE001 - 自愈失败绝不能挡住启动
            log.warning("[PyAgent] 向量自愈未执行：%s", e)

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    _s = get_settings()
    uvicorn.run("app.main:app", host=_s.host, port=_s.port, log_level=_s.log_level.lower())
