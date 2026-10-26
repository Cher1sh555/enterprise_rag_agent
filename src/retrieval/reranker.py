"""Rerank 精排：把"召回了 20 条"收敛成"哪 5 条真的能答"。

两种后端，接口一致（`score_many(query, docs) -> ndarray`）：

- `LexicalReranker`：query 词覆盖率的 IDF 加权，输出 0~1。零下载、确定性、
  可解释，用来把评测/阈值定标的流程跑通。**它不是 cross-encoder**，对同义改写无效。
- `FlagReranker`：真的 cross-encoder（BAAI/bge-reranker-v2-m3），要做
  `(query, doc)` 成对前向，首次使用会下载几 GB 权重。

为什么阈值必须作用在 rerank 分上：RRF 分是 1/(60+rank)，量级 0.016~0.033，
既没有上下界也没有概率含义，拿它和 0.5 比等于随机。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from src.schema import ScoredChunk
from src.utils.text import tokenize


class LexicalReranker:
    """query 词覆盖率重排（离线兜底，非 cross-encoder）。"""

    name = "lexical"

    def __init__(self, idf: Mapping[str, float], default_idf: float = 1.0) -> None:
        self._idf = dict(idf)
        self._default_idf = default_idf

    @classmethod
    def from_tokens(cls, token_lists: Sequence[Sequence[str]]) -> "LexicalReranker":
        """用语料的 token 列表估 IDF（平滑版），未登录词按最稀有处理。"""
        n = len(token_lists)
        df: Counter[str] = Counter()
        for tokens in token_lists:
            df.update(set(tokens))
        idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
        return cls(idf, default_idf=math.log(n + 1) + 1.0)

    def score_many(self, query: str, docs: Sequence[str]) -> np.ndarray:
        query_terms = set(tokenize(query))
        if not query_terms or not docs:
            return np.zeros(len(docs), dtype=np.float32)

        total = sum(self._idf.get(term, self._default_idf) for term in query_terms)
        if total <= 0:
            return np.zeros(len(docs), dtype=np.float32)

        scores = []
        for doc in docs:
            covered = sum(
                self._idf.get(term, self._default_idf)
                for term in query_terms & set(tokenize(doc))
            )
            scores.append(min(1.0, covered / total))
        return np.asarray(scores, dtype=np.float32)


class FlagReranker:
    """真 cross-encoder（FlagEmbedding）。惰性导入：不装也能跑离线评测。"""

    name = "flag"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        use_fp16: bool = False,
    ) -> None:
        from FlagEmbedding import FlagReranker as _FlagReranker  # 惰性：重依赖

        self._model = _FlagReranker(model_name, use_fp16=use_fp16, device=device)
        self._batch_size = batch_size
        self._max_length = max_length

    def score_many(self, query: str, docs: Sequence[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float32)
        pairs = [[query, doc] for doc in docs]
        scores = self._model.compute_score(
            pairs,
            normalize=True,             # 不归一化出来是 logits，阈值会失效
            max_length=self._max_length,
            batch_size=self._batch_size,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def rerank(
    query: str,
    candidates: Sequence[ScoredChunk],
    reranker,
    top_n: int,
) -> list[ScoredChunk]:
    """按 rerank 分重排并截断；保留 chunk 身份与 sources。"""
    if not candidates or top_n <= 0:
        return []

    scores = reranker.score_many(query, [item.chunk.text for item in candidates])
    rescored = [
        replace(item, rerank_score=float(score))
        for item, score in zip(candidates, scores)
    ]
    rescored.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
    return rescored[:top_n]


def build_reranker(settings, bundle, backend: str = "lexical"):
    """按 backend 造 reranker；backend='none' 返回 None（保持 v0 行为）。"""
    if backend in (None, "none"):
        return None
    if backend == "lexical":
        return LexicalReranker.from_tokens(bundle.tokens)
    if backend == "flag":
        return FlagReranker(settings.models.rerank_model)
    raise ValueError(f"未知的 reranker backend: {backend}")
