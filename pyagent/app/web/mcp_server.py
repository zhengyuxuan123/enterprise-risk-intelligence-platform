"""Built-in MCP stdio server for web search.

The main agent talks to this process only through MCP JSON-RPC. Keeping the
search worker out of the agent process gives us a real protocol boundary and
allows it to be replaced later by Brave, Tavily, Exa, or a remote MCP gateway
without changing Agent code.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

from .direct_client import DirectWebSearchClient, Hit


TOOL_NAME = "web_search"
# The parent MCP client owns the overall timeout. Retrying every public engine
# inside this child made one unavailable site consume that entire budget.
_client = DirectWebSearchClient(max_attempts=1)


def _search(query: str, count: int):
    """Run the local search adapter behind a real MCP protocol boundary.

    API fallback belongs to the outer WebSearchService. Doing it here as well
    made one request wait for the same slow provider twice before direct search.
    """
    return _client.search(query, count)


def _result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


def handle(request: Dict[str, Any]) -> Dict[str, Any] | None:
    """Handle one MCP request. Notifications intentionally return no reply."""
    method = str(request.get("method") or "")
    request_id = request.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return _result(request_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "risk-platform-web-search", "version": "1.0.0"},
        })
    if method == "tools/list":
        return _result(request_id, {"tools": [{
            "name": TOOL_NAME,
            "description": "Search public web pages and return citable URLs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        }]})
    if method != "tools/call":
        return _error(request_id, -32601, f"Method not found: {method}")

    params = request.get("params") or {}
    if str(params.get("name") or "") != TOOL_NAME:
        return _error(request_id, -32602, "Unknown tool")
    args = params.get("arguments") or {}
    query = str(args.get("query") or args.get("q") or "").strip()
    count = max(1, min(int(args.get("count") or args.get("topN") or 5), 10))
    if not query:
        return _error(request_id, -32602, "query is required")

    try:
        hits = _search(query, count)
        payload = [{"title": h.title, "url": h.url, "snippet": h.snippet}
                   for h in hits]
        return _result(request_id, {"content": [{
            "type": "text",
            "text": json.dumps(payload, ensure_ascii=False),
        }]})
    except Exception as exc:  # noqa: BLE001 - protocol errors must stay structured
        return _error(request_id, -32000, str(exc) or exc.__class__.__name__)


def main() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle(request)
        except Exception as exc:  # noqa: BLE001
            response = _error(None, -32700, str(exc) or "invalid request")
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
