"""评估报告的本地档案库（对应 Java ``EvalHistoryStore``）。

「单次体检」和「回归」的区别在于有没有历史：只有把每次跑出来的结果按序号留档，
才能回答"这次是不是比上次差了、差在哪几条、是不是变慢了"。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import get_settings

log = logging.getLogger(__name__)

_SAFE = re.compile(r"[^0-9A-Za-z\-]")

#: 旧版落盘报告的用例字段名（snake_case）→ 现在对外用的驼峰名。
#:
#: 留档文件是**只追加、不重写**的历史证据，不能为了改键名去改旧文件；
#: 所以在读取时做一次只读归一。否则「历史趋势」里点开旧报告，
#: 企业/证据/模型调用/成稿/Top分/耗时 会整片空白 —— 看起来像报告坏了，
#: 其实只是那一版把字段名写成了下划线。
_LEGACY_CASE_KEYS = {
    "company_id": "companyId", "rag_top_score": "ragTopScore",
    "rag_top_raw_score": "ragTopRawScore", "rag_hits": "ragHits",
    "web_hits": "webHits", "web_top_title": "webTopTitle",
    "probe_tool": "probeTool", "probe_preview": "probePreview",
    "probe_sources": "probeSources", "citation_web_cited": "citationWebCited",
    "citation_dangling": "citationDangling", "citation_summary": "citationSummary",
    "repair_web": "repairWeb", "repair_kb": "repairKb",
    "repair_reference": "repairReference", "repair_dangling_after": "repairDanglingAfter",
    "route_agents": "routeAgents", "route_tools": "routeTools",
    "trace_tools": "traceTools", "llm_calls": "llmCalls", "tool_calls": "toolCalls",
    "answer_chars": "answerChars", "answer_preview": "answerPreview",
    "degrade_level": "degradeLevel", "single_round": "singleRound",
    "duration_ms": "durationMs",
}


class EvalHistoryStore:
    """每份报告一个 JSON 文件 + 按文件名索引。``max_history`` 之外的自动归档。"""

    def __init__(self, report_dir: Optional[str] = None, max_history: Optional[int] = None) -> None:
        s = get_settings().eval
        self.report_dir = os.path.abspath(report_dir or s.report_dir)
        self.max_history = int(max_history if max_history is not None else s.max_history)
        self._lock = threading.RLock()

    # -- 目录 --------------------------------------------------------

    def _dir(self) -> str:
        os.makedirs(self.report_dir, exist_ok=True)
        return self.report_dir

    def _path(self, rid: str) -> str:
        return os.path.join(self._dir(), f"report-{_SAFE.sub('', rid)}.json")

    # -- 写入 --------------------------------------------------------

    def save(self, report: Any) -> str:
        """保存一份报告并返回它的 id。

        .. note::
           ``id`` 必须**先写进对象再序列化**，否则落盘的 JSON 里 id 是 null，
           下一轮拿它当基线时就报不出"对比的是哪一份"。
        """
        with self._lock:
            base = datetime.now().strftime("%Y%m%d-%H%M%S")
            rid = base
            p = self._path(rid)
            # 同一秒连跑两次会撞 id，后一份会覆盖前一份、基线也就丢了
            n = 2
            while os.path.exists(p):
                rid = f"{base}-{n}"
                p = self._path(rid)
                n += 1
            if isinstance(report, dict):
                report["id"] = rid
            elif hasattr(report, "id"):
                report.id = rid
            try:
                with open(p, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(self._dumpable(report), f, ensure_ascii=False, indent=2)
                self._prune()
                log.info("[Eval] 评估报告已留档: %s", p)
            except Exception as e:  # noqa: BLE001
                log.warning("[Eval] 评估报告写入失败: %s", e)
            return rid

    @staticmethod
    def _dumpable(report: Any) -> Any:
        if isinstance(report, dict):
            return report
        to_dict = getattr(report, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        return getattr(report, "__dict__", {})

    # -- 读取 --------------------------------------------------------

    def history(self) -> List[Dict[str, Any]]:
        """历史摘要（最新在前）。"""
        with self._lock:
            d = self._dir()
            names = [n for n in os.listdir(d) if n.startswith("report-") and n.endswith(".json")]
            names.sort(reverse=True)
            out: List[Dict[str, Any]] = []
            for n in names:
                m = self._read(os.path.join(d, n))
                if m is None or m.get("total") is None:
                    continue
                rid = n[len("report-"):-len(".json")]
                out.append({
                    "id": rid,
                    "generatedAt": str(m.get("generatedAt") or ""),
                    "total": int(m.get("total") or 0),
                    "passed": int(m.get("passed") or 0),
                    "passRate": float(m.get("passRate") or 0),
                    "wallClockMs": int(m.get("wallClockMs") or 0),
                    "model": str(m.get("model") or ""),
                })
            return out

    def latest_report(self) -> Optional[Dict[str, Any]]:
        h = self.history()
        return self.report(h[0]["id"]) if h else None

    def report(self, rid: Optional[str]) -> Optional[Dict[str, Any]]:
        if not rid:
            return None
        p = self._path(rid)
        if not os.path.isfile(p):
            return None
        m = self._read(p)
        if m is None:
            return None
        # 旧版本落盘的报告里 id 是 null（只在文件名里）。按文件名回填，
        # 历史档案立刻可用，不必等跑出新报告才自愈。
        if m.get("id") is None:
            m["id"] = rid
        _normalize_case_keys(m)
        return m

    def delete(self, rid: str) -> bool:
        with self._lock:
            try:
                os.remove(self._path(rid))
                return True
            except OSError:
                return False

    # -- 内部 --------------------------------------------------------

    @staticmethod
    def _read(p: str) -> Optional[Dict[str, Any]]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001 - 坏文件不该让整份历史读不出来
            return None

    def _prune(self) -> None:
        d = self._dir()
        names = sorted([n for n in os.listdir(d) if n.startswith("report-")], reverse=True)
        for n in names[self.max_history:]:
            try:
                os.remove(os.path.join(d, n))
            except OSError:
                pass


def _normalize_case_keys(report: Dict[str, Any]) -> None:
    """把旧报告里的 snake_case 用例字段就地改名（只改内存里的这一份）。"""
    rows = report.get("results")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        for old, new in _LEGACY_CASE_KEYS.items():
            if old in row and new not in row:
                row[new] = row.pop(old)
        # 旧报告里 traceTools 混着 "(5ms)" / "(预取 370ms)" 装饰，断言压根比不过；
        # 读的时候顺手剥一次，老报告的「期望工具」也能看出到底调没调。
        if isinstance(row.get("traceDetail"), list) and not row.get("traceTools"):
            row["traceTools"] = [_bare(t) for t in row["traceDetail"]]
        elif isinstance(row.get("traceTools"), list):
            row["traceTools"] = [_bare(t) for t in row["traceTools"]]


def _bare(entry: Any) -> str:
    s = str(entry or "").strip()
    i = s.find("(")
    return (s if i < 0 else s[:i]).strip()
