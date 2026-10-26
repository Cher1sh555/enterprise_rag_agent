"""reranker 契约：分数 0~1、按分排序、保住 chunk 身份、空输入不炸。"""

from __future__ import annotations

import numpy as np

from src.retrieval.reranker import LexicalReranker, rerank
from src.schema import Chunk, ScoredChunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d",
        source="a.md",
        page_no=1,
        seq=1,
        kind="text",
        text=text,
        char_len=len(text),
        department="研发",
        sensitivity="普通",
        ocr_used=False,
        low_confidence=False,
    )


def _scored(chunk_id: str, text: str, rrf: float) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(chunk_id, text), score=rrf, sources=("vector",))


def _reranker() -> LexicalReranker:
    corpus = [
        ["差旅", "住宿", "标准", "600", "元"],
        ["年假", "天数", "工龄"],
        ["报销", "发票", "30", "天"],
    ]
    return LexicalReranker.from_tokens(corpus)


def test_scores_are_unit_interval():
    scores = _reranker().score_many("差旅住宿标准", ["差旅住宿标准 600 元", "年假 5 天"])

    assert scores.shape == (2,)
    assert scores.dtype == np.float32
    assert all(0.0 <= float(s) <= 1.0 for s in scores)


def test_exact_term_scores_higher():
    scores = _reranker().score_many("差旅住宿标准", ["差旅住宿标准 600 元", "年假 5 天"])

    assert float(scores[0]) > float(scores[1])


def test_rerank_sorts_and_truncates():
    candidates = [
        _scored("c1", "年假 5 天", 0.03),
        _scored("c2", "差旅住宿标准 600 元", 0.02),
        _scored("c3", "报销 30 天", 0.01),
    ]
    out = rerank("差旅住宿标准", candidates, _reranker(), top_n=2)

    assert len(out) == 2
    assert out[0].chunk.chunk_id == "c2"
    assert out[0].rerank_score >= out[1].rerank_score


def test_rerank_preserves_identity_and_rrf():
    candidates = [_scored("c1", "差旅住宿标准", 0.033)]
    out = rerank("差旅住宿标准", candidates, _reranker(), top_n=5)

    assert out[0].chunk.chunk_id == "c1"
    assert out[0].score == 0.033
    assert out[0].sources == ("vector",)
    assert out[0].rerank_score is not None


def test_rerank_empty_candidates():
    assert rerank("任意", [], _reranker(), top_n=5) == []


def test_rerank_zero_top_n():
    candidates = [_scored("c1", "差旅住宿标准", 0.03)]

    assert rerank("差旅住宿标准", candidates, _reranker(), top_n=0) == []
