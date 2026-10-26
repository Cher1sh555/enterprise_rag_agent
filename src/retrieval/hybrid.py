"""双路召回（向量 + BM25）与分数融合。"""

from __future__ import annotations

import numpy as np

from src.schema import Chunk, ScoredChunk
from src.utils.text import tokenize

def vector_search(bundle, qvec, k, allowed_ids=None) -> list[tuple[str, float]]:
    # 注意用 is not None 判断：空集合表示"一条都看不到"，和 None（不过滤）语义完全不同
    filtered = allowed_ids is not None
    over = min(k * 4, bundle.faiss.ntotal) if filtered else k
    scores, idxs = bundle.faiss.search(np.ascontiguousarray(qvec[None, :], dtype=np.float32), max(over, 1))
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        chunk = bundle.chunks[idx]
        if filtered and chunk.chunk_id not in allowed_ids:
            continue
        out.append((chunk.chunk_id, float(score)))
        if len(out) >= k:
            break
    return out


def bm25_search(bundle, query, k, allowed_ids=None) -> list[tuple[str, float]]:
    filtered = allowed_ids is not None
    scores = bundle.bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][: k * 4 if filtered else k]
    out = []
    for idx in order:
        if scores[idx] <= 0:
            continue
        chunk = bundle.chunks[int(idx)]
        if filtered and chunk.chunk_id not in allowed_ids:
            continue
        out.append((chunk.chunk_id, float(scores[idx])))
        if len(out) >= k:
            break
    return out


def rrf_fuse(ranked_lists: list[list[tuple[str, float]]],
             chunks: dict[str, Chunk], k: int = 60) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    names = ("vector", "bm25")
    for name, ranked in zip(names, ranked_lists):
        for rank, (chunk_id, _score) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(chunk_id, []).append(name)
    return [
        ScoredChunk(chunk=chunks[cid], score=score, sources=tuple(sources[cid]))
        for cid, score in sorted(scores.items(), key=lambda kv: -kv[1])
    ]