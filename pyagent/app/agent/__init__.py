"""Agent 编排层：与 Java 版 ``service/agent`` 包对齐。

七个阶段全部落地（进度自查见 ``GET /api/ai/migration``）：

* 零 token 的确定性层：``registry`` 分工表 / ``router`` 路由 / ``budget`` 预算 / ``guardrail`` 护栏
* 编排主链路：``agent_service``（工具循环、空正文自愈、引用核对、复核回环）
* 留痕与追溯：``trace`` / ``trace_query``
* 治理：``evaluation`` / ``eval_history`` / ``health``
* 运营：``proactive`` / ``notification`` / ``approvals`` / ``report_export``

本包不带 HTTP —— 路由在 ``app.api``，服务装配（进程内单例）集中在 ``app.api.agent``
的 ``_services`` 表里，不在这里做。
"""
