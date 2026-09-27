"""向量化通道（对应 Java ``EmbeddingClient``）：本地零成本向量 / 第三方 embedding 二选一。

``mode`` 取值与 Java 完全一致：

* ``local``（默认）：FNV-1a 特征哈希 + L2 归一化，**不联网、不计费**，维度由配置给。
* ``remote``：调 OpenAI 兼容的 ``/embeddings``，带 TTL 缓存。
* ``auto``：配了第三方就用第三方，否则退回本地。

为什么要有缓存
--------------
同一次分析里，查询向量化和语料向量化会反复请求同一批文本。
第三方 embedding 按 token 计费，没有缓存的话一次分析能刷出几百次等价调用。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import get_settings
from .embedding import LocalEmbedding

log = logging.getLogger(__name__)


class EmbeddingClient:
    """统一入口。未配置第三方时自动退化为本地向量，不会抛异常。"""

    def __init__(self, local: Optional[LocalEmbedding] = None, http=None) -> None:
        s = get_settings()
        r = s.rag
        self.enabled = bool(r.embedding_enabled)
        self._mode = (r.embedding_mode or "local").strip().lower()
        self.local = local or LocalEmbedding(int(r.local_embedding_dim))
        self.base_url = (r.embedding_base_url or "").rstrip("/")
        self.api_key = r.embedding_api_key or s.ai.api_key
        self.model = r.embedding_model or s.ai.embedding_model
        self.dimensions = int(r.embedding_dimensions or 0)
        self.timeout = float(r.embedding_timeout_seconds)
        self._http = http

        self._cache_ttl = max(0, int(r.embedding_cache_ttl_minutes)) * 60.0
        self._cache: Dict[str, Tuple[float, np.ndarray]] = {}
        self._lock = threading.RLock()
        self._remote_dim: Optional[int] = None

    # -- 状态 --------------------------------------------------------

    @property
    def mode(self) -> str:
        """实际生效的模式：``local`` / ``remote``。

        ``auto`` 在这里被消解掉——下游只关心"到底走的哪条路"，
        因为写入时是否算向量就取决于它（本地恒算、第三方留到巡检）。
        """
        if self._mode == "remote" or self._mode == "auto":
            if self._mode == "remote" or self._remote_ready():
                return "remote"
        return "local"

    def is_enabled(self) -> bool:
        return bool(self.enabled)

    @property
    def raw_mode(self) -> str:
        """配置里写的模式原值（``local`` / ``remote`` / ``auto``），供诊断回显。

        与 :attr:`mode` 的区别：``mode`` 是**实际生效**的那条路（``auto`` 已被消解），
        诊断卡片要同时说清"配的是什么"和"实际走的是哪条"。
        """
        return self._mode

    def is_semantic(self) -> bool:
        """是否真的走向量语义：本地哈希向量不算语义检索（对应 Java ``isSemantic()``）。"""
        if not self.is_enabled():
            return False
        return self.mode == "remote"

    def status_note(self) -> str:
        """供诊断展示：当前实际在哪条路上（对应 Java ``statusNote()``）。"""
        if not self.enabled:
            return "已关闭"
        if self.mode == "local":
            return f"ok（本地哈希向量 {self.local.dim} 维 · 零成本 · 无需 Key）"
        if not self._remote_ready():
            return "未配置可用的第三方 Key / 模型"
        return "ok（第三方 embedding）"

    def dim(self) -> int:
        return self._remote_dim or self.local.dim if self.mode == "remote" else self.local.dim

    def _remote_ready(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    # -- 单条 --------------------------------------------------------

    def embed_one(self, text: Optional[str]):
        """单条向量化。失败返回 None——调用方会退化成"本次不走向量通道"。"""
        if not self.enabled or text is None:
            return None
        if self.mode == "local":
            return self.local.embed(text)
        got = self._remote([text])
        return got[0] if got else None

    # -- 批量（带缓存） ----------------------------------------------

    def embed_cached(self, texts: Optional[Sequence[str]]) -> List[np.ndarray]:
        """批量向量化，命中缓存的直接回。

        返回长度**恒等于入参长度**（某条失败用零向量占位，这样"位置"始终对齐）；
        唯一的例外是"没活可干"——``enabled=False`` 或入参为空时返回 ``[]``。

        ‼️ 这个"例外"是个陷阱：调用方若把 ``[]`` 当成"逐条结果"，就会静默地
        把每一条都当作"没有向量"，同时对外宣称向量化成功。所以
        **写入路径必须先判 :meth:`is_enabled` 再来调**（见
        ``RagIndexingService._embedder_for_write``），不要靠返回值反推开关状态。
        """
        src = list(texts or [])
        if not self.enabled or not src:
            return []
        dim = self.dim()
        out: List[Optional[np.ndarray]] = [None] * len(src)
        miss_idx: List[int] = []
        miss_text: List[str] = []

        with self._lock:
            now = time.time()
            for i, t in enumerate(src):
                hit = self._cache.get(t or "")
                if hit is not None and (self._cache_ttl <= 0 or now - hit[0] < self._cache_ttl):
                    out[i] = hit[1]
                else:
                    miss_idx.append(i)
                    miss_text.append(t or "")

        if miss_text:
            got = self._remote(miss_text) if self.mode == "remote" else [
                self.local.embed(t) for t in miss_text
            ]
            with self._lock:
                now = time.time()
                for k, i in enumerate(miss_idx):
                    v = got[k] if k < len(got) else None
                    if v is None:
                        v = np.zeros(dim, dtype=np.float32)
                    out[i] = v
                    self._cache[src[i] or ""] = (now, v)
        return [v if v is not None else np.zeros(dim, dtype=np.float32) for v in out]

    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # -- 远端 --------------------------------------------------------

    def _remote(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """OpenAI 兼容的 ``/embeddings``。逐条失败不影响其余条目。"""
        if not self._remote_ready():
            return [None] * len(texts)
        try:
            import httpx
        except ImportError:  # pragma: no cover
            log.warning("httpx 不可用，第三方向量通道本次停用")
            return [None] * len(texts)

        payload: Dict[str, object] = {"model": self.model, "input": texts}
        if self.dimensions > 0:
            payload["dimensions"] = self.dimensions
        try:
            client = self._http or httpx.Client(timeout=self.timeout)
            resp = client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
            )
            if resp.status_code >= 400:
                log.warning("embedding 调用失败 %s: %s", resp.status_code, resp.text[:200])
                return [None] * len(texts)
            data = resp.json().get("data") or []
            by_index = {int(d.get("index", i)): d.get("embedding")
                        for i, d in enumerate(data)}
            out: List[Optional[np.ndarray]] = []
            for i in range(len(texts)):
                vec = by_index.get(i)
                if not vec:
                    out.append(None)
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                n = float(np.linalg.norm(arr))
                out.append(arr / n if n > 0 else arr)
                self._remote_dim = int(arr.shape[0])
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("embedding 调用异常: %s", e)
            return [None] * len(texts)
