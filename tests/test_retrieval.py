"""retrieval 契约：RRF 融合语义 + 拒答判定。

这里钉两条最容易被改坏的语义：
1. RRF 的 k 决定"单路第一"与"两路共识"谁更占优（k 小 → 单路通吃；k 大 → 共识优先）；
2. 判定层在 v0 只对"空候选"拒答，分数阈值留给 reranker 定标后的 M3。
"""

from __future__ import annotations

from types import SimpleNamespace

import faiss
import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from src.retrieval.hybrid import bm25_search, rrf_fuse, vector_search
from src.retrieval.threshold import decide
from src.schema import Chunk, ScoredChunk
from src.utils.text import tokenize


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        source="data/raw_docs/研发/制度.md",
        page_no=1,
        seq=1,
        kind="text",
        text=text,
        char_len=len(text),
        department="研发",
        sensitivity="机密",
        ocr_used=False,
        low_confidence=False,
    )


def _by_id(*chunks: Chunk) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


# ---------------------------------------------------------------- threshold


def test_empty_candidates_refused(settings):
    decision = decide([], settings)

    assert decision.answered is False
    assert decision.reason == "no_result"
    assert decision.top1 == 0.0


def test_single_candidate_margin_equals_top1(settings):
    scored = [ScoredChunk(chunk=_chunk("d#0001", "正文"), score=0.42, sources=("bm25",))]

    decision = decide(scored, settings)

    assert decision.answered is True
    assert decision.top1 == 0.42
    assert decision.margin == 0.42


# ------------------------------------------------------------ threshold（M3）


def test_below_threshold_is_refused(settings):
    """有 rerank 分且低于阈值 -> low_score，而不是硬答。"""
    threshold = settings.retrieval.relevance_threshold
    scored = [
        ScoredChunk(
            chunk=_chunk("d#0001", "正文"),
            score=0.03,
            sources=("vector",),
            rerank_score=threshold - 0.01,
        )
    ]

    decision = decide(scored, settings)

    assert decision.answered is False
    assert decision.reason == "low_score"
    assert decision.top1_rerank == pytest.approx(threshold - 0.01)


def test_threshold_boundary_is_answered(settings):
    """恰好等于阈值算通过（>= 语义，写死在测试里）。"""
    threshold = settings.retrieval.relevance_threshold
    scored = [
        ScoredChunk(
            chunk=_chunk("d#0001", "正文"),
            score=0.03,
            sources=("vector",),
            rerank_score=threshold,
        )
    ]

    decision = decide(scored, settings)

    assert decision.answered is True
    assert decision.reason == "ok"


# ---------------------------------------------------------------------- RRF


def test_rrf_prefers_document_found_by_both_lists():
    a, b, c = _chunk("a", "甲"), _chunk("b", "乙"), _chunk("c", "丙")

    fused = rrf_fuse(
        [
            [("a", 0.9), ("b", 0.5)],
            [("b", 0.8)],
        ],
        _by_id(a, b, c),
    )

    assert [item.chunk.chunk_id for item in fused] == ["b", "a"]
    assert fused[0].sources == ("vector", "bm25")


def test_rrf_small_k_lets_single_list_win():
    """k=0 时"某路第一"通吃；k 很大时"两路都命中"反超。"""
    a, b, c, d = (_chunk(i, i) for i in ("a", "b", "c", "d"))
    lists = [
        [("a", 1.0), ("b", 0.9), ("c", 0.8)],
        [("d", 1.0), ("c", 0.9), ("b", 0.8)],
    ]
    chunks = _by_id(a, b, c, d)

    small = rrf_fuse(lists, chunks, k=0)
    large = rrf_fuse(lists, chunks, k=100)

    assert small[0].chunk.chunk_id in {"a", "d"}
    assert large[0].chunk.chunk_id in {"b", "c"}


def test_rrf_empty_lists_returns_empty():
    assert rrf_fuse([[], []], {}) == []


# ------------------------------------------------------------------- search


@pytest.fixture()
def bundle():
    texts = ["差旅住宿标准 600 元", "年假 5 天", "报销 30 天"]
    chunks = [_chunk(f"d#{i:04d}", text) for i, text in enumerate(texts, start=1)]
    index = faiss.IndexFlatIP(3)
    index.add(np.eye(3, dtype=np.float32))
    return SimpleNamespace(
        faiss=index,
        chunks=chunks,
        bm25=BM25Okapi([tokenize(t) for t in texts]),
    )


def test_vector_search_respects_allowed_ids(bundle):
    hits = vector_search(
        bundle, np.eye(3, dtype=np.float32)[0], k=3, allowed_ids={"d#0002"}
    )

    assert [chunk_id for chunk_id, _score in hits] == ["d#0002"]


def test_bm25_search_finds_exact_terms(bundle):
    hits = bm25_search(bundle, "差旅住宿标准", k=1)

    assert hits and hits[0][0] == "d#0001"
