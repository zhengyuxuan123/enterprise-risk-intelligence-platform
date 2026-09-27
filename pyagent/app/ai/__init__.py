"""模型层（阶段 2）。

对应 Java 侧的 ``service/ai/`` 包：

===============  ==========================================================
Java             Python
===============  ==========================================================
``LlmClient``    :mod:`pyagent.app.ai.model_catalog`（纯规则部分，零 token）
                 :mod:`pyagent.app.ai.discovery`（``GET /models`` 只读列举）
                 :mod:`pyagent.app.ai.llm_client`（阶段 2B：真实调用与工具循环）
``KeySource``    :mod:`pyagent.app.ai.keys`
===============  ==========================================================

**为什么先切出"纯规则"这一半**：``LlmClient`` 1488 行里，真正需要网络和 token 的
只是一部分；模型筛选（``isChatModel``）、打分排序（``modelScore`` / ``rankChatModels``）、
错误分类（``isModelUnavailable`` / ``describe``）都是**纯函数**，
可以拿 Java 版对真实账号 133 个模型的排序结果做逐字节对账 —— 不用花一分钱额度。
先把这部分钉死，后面接真实调用时才有干净的对照基准。
"""

from __future__ import annotations

__all__ = ["keys", "model_catalog", "discovery"]
