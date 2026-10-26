"""评测指标：纯函数，不碰 IO、不碰模型。

为什么单独成模块：指标算错了，后面所有"调参数-看指标"的实验都是假的。
所以先把指标做成可单测的纯函数（tests/test_metrics.py），再让 run_eval 调用它。

口径说明（改之前先想清楚）：
- Recall@k / MRR 只在 normal 类上算（boundary/denied 本来就没有期望块）；
- 误拒率 = normal 里被拒的比例（该答没答）；
- 正确拒答率 = boundary 里被拒的比例（该拒不答）；
- 泄漏率 = denied 里"既回答了、敏感原文又进了证据"的比例（宁可误报不可漏报）。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.schema import Chunk


@dataclass(frozen=True)
class QuestionResult:
    qid: str
    qtype: str                       # "normal" | "boundary" | "denied"
    answered: bool
    reason: str                      # Decision.reason
    first_hit_rank: int | None       # 第一条命中期望块的排名，None=没命中
    rerank_top1: float | None
    latency_ms: float
    leaked: bool                     # 仅 denied 有意义


def normalize(text: str) -> str:
    """去掉所有空白后再比：检索块里有换行，直接 in 会漏判。"""
    return re.sub(r"\s+", "", text or "")


def match_rank(chunks: Sequence[Chunk], expect: Sequence[dict]) -> int | None:
    """返回第一条命中期望的排名（1 基）；没命中返回 None。

    命中口径：source 完全一致 **且** 归一化后包含 expect.contains。
    不用 chunk_id —— 语料一编辑 chunk_id 全变，评测集会整批失效。
    """
    if not expect:
        return None
    for rank, chunk in enumerate(chunks, start=1):
        text = normalize(chunk.text)
        for item in expect:
            if chunk.source == item["source"] and normalize(item["contains"]) in text:
                return rank
    return None


def _by_type(results: Sequence[QuestionResult], qtype: str) -> list[QuestionResult]:
    return [item for item in results if item.qtype == qtype]


def recall_at_k(results: Sequence[QuestionResult], k: int = 5) -> float:
    normal = _by_type(results, "normal")
    if not normal:
        return 0.0
    hits = sum(1 for item in normal if item.first_hit_rank and item.first_hit_rank <= k)
    return hits / len(normal)


def mrr(results: Sequence[QuestionResult]) -> float:
    normal = _by_type(results, "normal")
    if not normal:
        return 0.0
    total = sum(1.0 / item.first_hit_rank for item in normal if item.first_hit_rank)
    return total / len(normal)


def false_refuse_rate(results: Sequence[QuestionResult]) -> float:
    """该答却拒：normal 里 answered=False 的比例。"""
    normal = _by_type(results, "normal")
    if not normal:
        return 0.0
    return sum(1 for item in normal if not item.answered) / len(normal)


def correct_refuse_rate(results: Sequence[QuestionResult]) -> float:
    """该拒就拒：boundary 里 answered=False 的比例。"""
    boundary = _by_type(results, "boundary")
    if not boundary:
        return 0.0
    return sum(1 for item in boundary if not item.answered) / len(boundary)


def leak_rate(results: Sequence[QuestionResult]) -> float:
    """越权泄漏：denied 里 leaked=True 的比例。"""
    denied = _by_type(results, "denied")
    if not denied:
        return 0.0
    return sum(1 for item in denied if item.leaked) / len(denied)


def p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def summarize(results: Sequence[QuestionResult]) -> dict:
    return {
        "count": len(results),
        "counts_by_type": {
            qtype: len(_by_type(results, qtype))
            for qtype in ("normal", "boundary", "denied")
        },
        "recall_at_5": round(recall_at_k(results, 5), 4),
        "mrr": round(mrr(results), 4),
        "false_refuse_rate": round(false_refuse_rate(results), 4),
        "correct_refuse_rate": round(correct_refuse_rate(results), 4),
        "leak_rate": round(leak_rate(results), 4),
        "p95_latency_ms": round(p95([item.latency_ms for item in results]), 1),
    }
