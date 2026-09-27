"""日志：与 Java 侧同样带 ``[组件]`` 前缀，方便两边日志并排比对。

Java 用的是 SLF4J：``log.info("[AgentRouter] 问题「{}」→ 专员：{}；开放工具：{}")``。
这里保留**完全相同的消息文案**，迁移期 grep 一条日志就能同时命中两个实现。

.. WARNING::
   **占位符要改成 ``%s``**：Python 的 ``logging`` 是 ``%`` 风格，照抄 SLF4J 的
   ``{}`` 不会在编译时报错，只会在**真正走到那行日志时**抛
   ``TypeError: not all arguments converted during string formatting``。
   也就是说这个错只在生产路径上炸，而且往往是在处理一个异常的过程中炸掉 ——
   把原始错误彻底盖住。文案照抄，占位符别照抄。
"""

from __future__ import annotations

import logging
import sys

_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-5s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # uvicorn 的 access 日志在双跑对照时会淹没有用信息
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """取一个 logger。名字用 ``pyagent.<模块>``，与 Java 的包路径一一对应。"""
    return logging.getLogger(name)
