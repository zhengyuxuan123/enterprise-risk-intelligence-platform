"""联网检索的 **MCP（Model Context Protocol）** 通道（对应 Java ``McpWebSearchClient``）。

为什么要有这一条通道
--------------------
前两条各有硬伤：直连抓取要自己解析 HTML 且常被科普页占据；
方舟 Responses API 质量最好但要 Key、要开通插件、额度受限时直接 429。

MCP 是第三条路：**接任何实现了 MCP 的检索服务**（Brave Search、Tavily、Exa、自建网关），
协议统一（``tools/list`` + ``tools/call``），不解析 HTML、不绑定厂商。

两种传输
--------
* ``stdio``：本地起子进程（如 ``npx -y @xxx/server-brave-search``），按行交换 JSON-RPC。
* ``http``：POST JSON-RPC 到 Streamable HTTP 端点，响应可能是纯 JSON 也可能是 SSE。

**永不抛异常**：所有失败收敛成 :attr:`last_error`，让上层降级到下一条通道，
而不是把一次分析打成 500。
"""

from __future__ import annotations

import json
import logging
import queue
import re
import subprocess
import threading
import time
from typing import Dict, List, Optional

from .direct_client import Hit, clean, domain_of

log = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s()（）\"'<>【】\[\]|]+")

#: 自动选工具时优先匹配的名字特征（越靠前越优先）。
TOOL_HINTS = ["web_search", "websearch", "search_web", "brave", "tavily", "exa",
              "search", "google", "bing", "duckduckgo", "query"]

#: 这些名字明显不是「检索」，即便含 search 也要跳过（避免挑中站内搜索/代码搜索）。
TOOL_EXCLUDE = ["fetch", "extract", "crawl", "read", "browser", "code", "repo", "file", "sql"]

DEFAULT_ARGS_TEMPLATE = '{"query":"{q}","count":{n}}'


class McpWebSearchClient:
    """MCP 检索通道。没配 command / url 时自动处于"未启用"，上层会跳过它。"""

    def __init__(self, cfg=None, http=None) -> None:
        from ..config import get_settings

        m = cfg or get_settings().web_search.mcp
        self.enabled = bool(m.enabled)
        self.command = (m.command or "").strip()
        self.url = (m.url or "").strip()
        self.transport = self._normalize_transport(m.transport, self.command, self.url)
        self.headers = self._parse_headers(m.headers)
        self.tool_name = (m.tool_name or "").strip()
        self.args_template = (m.args_template or "").strip() or DEFAULT_ARGS_TEMPLATE
        self.timeout_seconds = int(m.timeout_seconds or 60)

        self._http = http
        self._lock = threading.RLock()
        self._session: Optional[_StdioSession] = None
        self._resolved_tool: Optional[str] = None
        self._list_failed = False
        self._seq = 0
        self.last_error: Optional[str] = None
        self.last_tool: Optional[str] = None

        log.info("[WebSearch][MCP] enabled=%s transport=%s command=%s url=%s tool=%s",
                 self.enabled, self.transport, self.command or "-", self.url or "-",
                 self.tool_name or "(自动探测)")

    # -- 状态 --------------------------------------------------------

    def is_enabled(self) -> bool:
        return self.enabled and self.configured()

    def configured(self) -> bool:
        return bool(self.command or self.url)

    def get_transport(self) -> str:
        return self.transport

    def get_last_error(self) -> Optional[str]:
        return self.last_error

    def get_last_tool(self) -> Optional[str]:
        return self.last_tool

    def status_reason(self) -> Optional[str]:
        """不可用原因；``None`` 表示可用（只做**配置层面**的判断，不探活）。"""
        if not self.enabled:
            return "MCP 通道已关闭（app.web-search.mcp.enabled=false）"
        if not self.configured():
            return "MCP 未配置（需要 app.web-search.mcp.command 或 .url 之一）"
        return None

    def close(self) -> None:
        self._close_session()

    # -- 检索 --------------------------------------------------------

    def search(self, query: Optional[str], top_n: int) -> List[Hit]:
        if not query or not query.strip():
            self.last_error = "缺少检索词"
            return []
        n = max(1, min(top_n if top_n > 0 else 5, 10))
        try:
            with self._lock:  # stdio 子进程是独占资源：并发调用必须串行化，否则响应会串台
                tool = self._resolve_tool()
                if not tool:
                    return []
                result = self._call_tool(tool, self._build_args(query, n))
            hits = self._parse_hits(result, n)
            self.last_tool = tool
            if not hits:
                self.last_error = "MCP 工具已调用但未解析出可引用链接"
            else:
                self.last_error = None
            return hits
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e) or e.__class__.__name__
            log.warning("MCP 检索异常: %s", self.last_error)
            return []

    # -- 工具选择 ----------------------------------------------------

    def _resolve_tool(self) -> Optional[str]:
        """点了名就不做探测，省一次往返。"""
        if self.tool_name:
            return self.tool_name
        if self._resolved_tool:
            return self._resolved_tool
        if self._list_failed:
            raise RuntimeError("tools/list 曾失败，本次不再重试")
        try:
            res = self._request("tools/list", {})
        except Exception as e:  # noqa: BLE001
            self._list_failed = True
            raise RuntimeError(f"tools/list 失败: {e}") from e
        tools = (res or {}).get("tools") or []
        names = [str(t.get("name")) for t in tools if isinstance(t, dict) and t.get("name")]
        pick = self._pick_tool(names)
        if not pick:
            self._list_failed = True
            raise RuntimeError(f"MCP Server 未暴露检索类工具（共 {len(names)} 个：{names[:8]}）")
        self._resolved_tool = pick
        return pick

    @staticmethod
    def _pick_tool(names: List[str]) -> Optional[str]:
        for hint in TOOL_HINTS:
            for n in names:
                low = n.lower()
                if hint in low and not any(x in low for x in TOOL_EXCLUDE):
                    return n
        return None

    def _build_args(self, query: str, n: int) -> Dict:
        """各家 MCP Server 参数名不统一，所以走可配置模板：``{q}`` = 检索词、``{n}`` = 条数。"""
        tpl = self.args_template.replace("{q}", json.dumps(query)[1:-1]).replace("{n}", str(n))
        try:
            return json.loads(tpl)
        except json.JSONDecodeError:
            return {"query": query, "count": n}

    def _call_tool(self, tool: str, args: Dict) -> Dict:
        res = self._request("tools/call", {"name": tool, "arguments": args})
        return res or {}

    # -- 结果解析 ----------------------------------------------------

    def _parse_hits(self, result: Dict, n: int) -> List[Hit]:
        out: List[Hit] = []
        for c in (result.get("content") or []):
            if not isinstance(c, dict):
                continue
            text = c.get("text")
            if not isinstance(text, str):
                continue
            stripped = text.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                self._collect_from_json_array(stripped, out, n)
            if len(out) < n:
                self._collect_from_text(text, out, n)
        return out[:n]

    def _collect_from_json_array(self, text: str, out: List[Hit], n: int) -> None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        items = data if isinstance(data, list) else [data]
        for it in items:
            if len(out) >= n or not isinstance(it, dict):
                break
            url = _first_text(it, "url", "link", "href")
            if not url:
                continue
            title = _first_text(it, "title", "name") or domain_of(url)
            snip = _first_text(it, "snippet", "description", "content", "text")
            self._add(out, title, url, snip, n)

    def _collect_from_text(self, body: str, out: List[Hit], n: int) -> None:
        """MCP Server 大多把结果渲染成 Markdown 或纯文本，这里按「链接 + 上一行标题」来配。"""
        lines = body.splitlines()
        for i, line in enumerate(lines):
            if len(out) >= n:
                break
            for m in URL_RE.finditer(line):
                if len(out) >= n:
                    break
                url = m.group(0).rstrip(".,;。，；")
                title = _clean_title(lines[i - 1] if i > 0 else "", url) or domain_of(url)
                self._add(out, title, url, line.replace(url, " ").strip(), n)

    @staticmethod
    def _add(out: List[Hit], title: str, url: str, snip: str, n: int) -> None:
        if len(out) >= n or any(h.url == url for h in out):
            return
        out.append(Hit(title, url, snip or "", "mcp"))

    # -- JSON-RPC ----------------------------------------------------

    def _request(self, method: str, params: Dict) -> Dict:
        self._seq += 1
        payload = json.dumps({"jsonrpc": "2.0", "id": self._seq,
                              "method": method, "params": params}, ensure_ascii=False)
        raw = self._send_stdio(payload) if self.transport == "stdio" else self._send_http(payload)
        return self._extract(raw)

    @staticmethod
    def _extract(raw: str) -> Dict:
        """HTTP 传输的响应可能是纯 JSON，也可能是 SSE（``data: {...}``）。"""
        body = raw.strip()
        if body.startswith("data:"):
            body = body[5:].strip()
        try:
            root = json.loads(body)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                raise RuntimeError(f"MCP 返回非 JSON：{raw[:200]}")
            root = json.loads(m.group(0))
        if isinstance(root, list):
            root = next((x for x in root if isinstance(x, dict) and "result" in x), {})
        if "error" in root:
            err = root["error"]
            raise RuntimeError(f"MCP 错误 {err.get('code')}: {err.get('message')}")
        res = root.get("result")
        return res if isinstance(res, dict) else {}

    def _send_http(self, payload: str) -> str:
        import httpx

        url = self.url
        if not url:
            raise RuntimeError("MCP http 传输未配置 url")
        client = self._http or httpx.Client(timeout=self.timeout_seconds,
                                            follow_redirects=True)
        try:
            headers = {"Content-Type": "application/json", **self.headers}
            r = client.post(url, content=payload.encode("utf-8"), headers=headers)
            if r.status_code >= 400:
                raise RuntimeError(f"MCP HTTP {r.status_code}: {r.text[:200]}")
            return r.text
        finally:
            if self._http is None:
                client.close()

    def _send_stdio(self, payload: str) -> str:
        s = self._session or self._open_session()
        if s is None:
            raise RuntimeError("MCP stdio 子进程未启动")
        return s.roundtrip(payload, self.timeout_seconds)

    def _open_session(self):
        s = _StdioSession.open(self.command, self.timeout_seconds)
        self._session = s
        self._handshake(s)
        return s

    def _handshake(self, s) -> None:
        init = json.dumps({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05",
                       "capabilities": {},
                       "clientInfo": {"name": "risk-platform-pyagent", "version": "1.0"}},
        }, ensure_ascii=False)
        s.write(init)
        s.read_line(self.timeout_seconds)
        s.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def _close_session(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                pass
            self._session = None

    # -- 小工具 ------------------------------------------------------

    @staticmethod
    def _normalize_transport(raw: Optional[str], command: str, url: str) -> str:
        v = (raw or "").strip().lower()
        if v in ("stdio", "http"):
            return v
        # 没显式给就按"有什么用什么"推断：给了 command 就是本地子进程
        return "stdio" if command else ("http" if url else "stdio")

    @staticmethod
    def _parse_headers(raw: Optional[str]) -> Dict[str, str]:
        """``K=V`` 用换行或分号分隔。"""
        out: Dict[str, str] = {}
        if not raw or not raw.strip():
            return out
        for part in re.split(r"[\n;]+", raw):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k:
                out[k] = v
        return out


class _StdioSession:
    """本地 MCP 子进程：按行交换 JSON-RPC。"""

    def __init__(self, proc):
        self.proc = proc
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        """Move blocking pipe reads off the request thread so timeout is real on Windows."""
        try:
            assert self.proc.stdout is not None
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    break
                self._lines.put(line)
        finally:
            self._lines.put(None)

    @staticmethod
    def open(command: str, timeout: float):
        if not command:
            raise RuntimeError("MCP stdio 未配置 command")
        import os
        import shlex
        import sys

        args = shlex.split(command)
        # Use the same interpreter as the running agent. On Windows, `python`
        # on PATH may point at another environment without this project's deps.
        if args and args[0].lower() in ("python", "python.exe", "python3", "python3.exe"):
            args[0] = sys.executable
        env = os.environ.copy()
        # Pipes do not inherit a UTF-8 console on Windows. Without these flags,
        # Chinese JSON-RPC requests are decoded as GBK in the child and search
        # for mojibake; responses can also raise UnicodeDecodeError in the parent.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                bufsize=1, env=env,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"MCP 命令不存在：{command}") from e
        return _StdioSession(proc)

    def write(self, payload: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(payload + "\n")
        self.proc.stdin.flush()

    def read_line(self, timeout: float) -> str:
        try:
            line = self._lines.get(timeout=max(0.01, float(timeout)))
        except queue.Empty as e:
            raise TimeoutError(f"MCP stdio 响应超时（{timeout:g} 秒）") from e
        if line is None:
            raise RuntimeError("MCP 子进程已退出")
        return line

    def roundtrip(self, payload: str, timeout: float) -> str:
        """发一行、读到第一条含 id 的响应为止（中间的 notification 要跳过）。"""
        self.write(payload)
        deadline = time.monotonic() + max(0.01, float(timeout))
        for _ in range(200):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"MCP stdio 响应超时（{timeout:g} 秒）")
            line = self.read_line(remaining).strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                return line
        raise RuntimeError("MCP stdio 未收到有效响应")

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass


def _first_text(node: Dict, *keys: str) -> str:
    for k in keys:
        v = node.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _clean_title(line: str, url: str) -> str:
    t = re.sub(r"^[-*#>\[\]\d.\s]+", "", line)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    return clean(t)[:120]
