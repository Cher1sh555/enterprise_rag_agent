"""相关性阈值判定（低于阈值拒答）。

三个分支，顺序不能换：
1. 没有候选            -> no_result（检索层什么都没找到）
2. 有 rerank 分且低于阈值 -> low_score（找到了但不够相关，宁可拒答）
3. 其它               -> ok

为什么阈值作用在 rerank 分而不是 RRF 分：RRF 是 1/(60+rank)，量级 0.02，
没有概率含义；只有 reranker 的 0~1 分数才能和 relevance_threshold 比。
没有 reranker 时（v0 行为）只判空，不做分数判断。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings
from src.schema import ScoredChunk


@dataclass(frozen=True)
class Decision:
    answered: bool
    reason: str                    # "ok" | "no_result" | "low_score"
    top1: float                    # RRF 融合分（非相关度）
    margin: float
    top1_rerank: float | None = None   # reranker 的 0~1 分；无 reranker 时为 None


def decide(scored: list[ScoredChunk], settings: Settings) -> Decision:
    """空候选拒答；有 rerank 分时再按阈值拒答。"""
    if not scored:
        return Decision(answered=False, reason="no_result", top1=0.0, margin=0.0)

    top1 = scored[0].score
    margin = top1 - scored[1].score if len(scored) > 1 else top1
    top1_rerank = scored[0].rerank_score

    if top1_rerank is None:
        return Decision(answered=True, reason="ok", top1=top1, margin=margin)

    if top1_rerank < settings.retrieval.relevance_threshold:
        return Decision(
            answered=False,
            reason="low_score",
            top1=top1,
            margin=margin,
            top1_rerank=top1_rerank,
        )
    return Decision(
        answered=True, reason="ok", top1=top1, margin=margin, top1_rerank=top1_rerank
    )
