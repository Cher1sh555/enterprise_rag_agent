"""Embedding 模型封装。"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from src.config import Settings
from src.utils.exceptions import EmbeddingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    """Embedding 模型接口：批量编码 + 单条查询编码。"""

    dim: int

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...

class HashEmbedder:
    """字符 bigram -> 固定维度哈希桶 -> L2 归一化。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._encode(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(text)

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for gram in _bigrams(text):
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vector[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class STEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError("缺少 sentence-transformers", {"model": model_name}) from exc
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts))

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        try:
            vectors = self._model.encode(
                texts, batch_size=self._batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError("向量化失败", {"count": len(texts)}) from exc
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise EmbeddingError("输出维度异常", {"expected": self.dim, "got": vectors.shape})
        return vectors


def build_embedder(settings: Settings, force_hash: bool = False) -> Embedder:
    name = settings.models.embedding_model
    if force_hash or name.startswith("hash"):
        return HashEmbedder()
    return STEmbedder(name)


def _bigrams(text: str) -> list[str]:
    text = "".join(ch for ch in text if not ch.isspace())
    return [text[i:i + 2] for i in range(len(text) - 1)] or [text]
