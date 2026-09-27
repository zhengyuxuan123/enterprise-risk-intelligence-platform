"""联网检索层（对应 ``service/web/*``）：MCP → 免 Key 直连 → 方舟 Responses API。

三条通道按「质量优先、成本兜底」排序，默认 ``auto`` 依次尝试：

1. **MCP**：按 Model Context Protocol 调外部检索服务（Brave / Tavily / Exa / 自建网关）。
   协议统一，不解析 HTML、不绑定厂商。需要先给出 command 或 url。
2. **免 Key 直连**（默认实际走这条）：直接抓搜索引擎结果页并解析。
   不需要 API Key、不通密钥外发；代价是要自己解析 HTML、还要剔除释义类结果。
3. **方舟 Responses API**：要一把 Key，且必须先开通「联网内容插件」，未开通返回 ``404 ToolNotOpen``。

永不抛异常：所有失败都收敛成 ``last_error``，让上层降级到下一条通道。
"""

from .direct_client import DirectWebSearchClient, Hit
from .mcp_client import McpWebSearchClient
from .search_service import SearchOutcome, WebSource, WebSearchService

__all__ = [
    "DirectWebSearchClient",
    "Hit",
    "McpWebSearchClient",
    "SearchOutcome",
    "WebSource",
    "WebSearchService",
]
