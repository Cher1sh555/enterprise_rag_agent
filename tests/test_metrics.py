"""评测指标的单测：指标算错了，后面所有参数实验都是假的。"""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    QuestionResult,
    correct_refuse_rate,
    false_refuse_rate,
    leak_rate,
    match_rank,
    mrr,
    normalize,
    p95,
    recall_at_k,
    summarize,
)
from src.schema import Chunk


def _chunk(source: str, text: str) -> Chunk:
    return Chunk(
        chunk_id="d#0001",
        doc_id="d",
        source=source,
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


def _result(qid, qtype, answered, rank=None, leaked=False, rerank=None, latency=1.0):
    return QuestionResult(
        qid=qid,
        qtype=qtype,
        answered=answered,
        reason="ok" if answered else "low_score",
        first_hit_rank=rank,
        rerank_top1=rerank,
        latency_ms=latency,
        leaked=leaked,
    )


def test_normalize_strips_all_whitespace():
    assert normalize("每晚 不超过\n600 元") == "每晚不超过600元"


def test_match_rank_matches_source_and_contains():
    chunks = [
        _chunk("a.md", "无关内容"),
        _chunk("b.md", "一线城市每晚不超过 600 元"),
    ]
    expect = [{"source": "b.md", "contains": "600 元"}]

    assert match_rank(chunks, expect) == 2


def test_match_rank_ignores_chunk_from_other_source():
    chunks = [_chunk("a.md", "每晚不超过 600 元")]
    expect = [{"source": "b.md", "contains": "600 元"}]

    assert match_rank(chunks, expect) is None


def test_match_rank_empty_expect_returns_none():
    assert match_rank([_chunk("a.md", "x")], []) is None


def test_recall_and_mrr():
    results = [
        _result("q1", "normal", True, rank=1),
        _result("q2", "normal", True, rank=3),
        _result("q3", "normal", False, rank=None),
    ]

    assert recall_at_k(results, 5) == pytest.approx(2 / 3)
    assert mrr(results) == pytest.approx((1 + 1 / 3) / 3)


def test_refuse_rates_are_two_directions():
    results = [
        _result("q1", "normal", True),
        _result("q2", "normal", False),      # 该答却拒
        _result("q3", "boundary", False),    # 该拒就拒
        _result("q4", "boundary", True),     # 不该答却答
    ]

    assert false_refuse_rate(results) == 0.5
    assert correct_refuse_rate(results) == 0.5


def test_leak_rate_only_counts_denied():
    results = [
        _result("q1", "denied", True, leaked=True),
        _result("q2", "denied", False, leaked=False),
        _result("q3", "normal", True, leaked=True),   # normal 不计入
    ]

    assert leak_rate(results) == 0.5


def test_p95_uses_nearest_rank():
    values = [float(i) for i in range(1, 21)]      # 1..20

    assert p95(values) == 19.0
    assert p95([]) == 0.0


def test_summarize_shape():
    results = [
        _result("q1", "normal", True, rank=1, rerank=0.8),
        _result("q2", "boundary", False),
        _result("q3", "denied", True, leaked=True),
    ]
    summary = summarize(results)

    assert summary["count"] == 3
    assert summary["counts_by_type"] == {"normal": 1, "boundary": 1, "denied": 1}
    assert summary["recall_at_5"] == 1.0
    assert summary["leak_rate"] == 1.0
