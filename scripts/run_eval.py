"""跑评测集，输出指标报表。

只做检索 + 判定（不调 LLM），所以可以离线、可重复、秒级。
"答案里是否真的泄漏"要等 M4 接上生成才测得准，这里用"敏感原文进了证据集"
作为较保守的替代口径（宁可误报，不可漏报）。

用法：
    python scripts/run_eval.py --reranker lexical --out data/evaluation/report.json
    python scripts/run_eval.py --reranker none --threshold 0.0   # 关掉拒答看上限
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import get_settings
from src.evaluation.metrics import QuestionResult, match_rank, normalize, summarize
from src.ingest.embedder import build_embedder
from src.ingest.indexer import load_index
from src.retrieval.hybrid import bm25_search, rrf_fuse, vector_search
from src.retrieval.reranker import build_reranker, rerank
from src.retrieval.threshold import decide
from src.schema import User
from src.security.permission import allowed_ids
from src.utils.exceptions import RAGError


def load_questions(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="跑评测集出指标")
    parser.add_argument("--questions", default="data/evaluation/questions.jsonl")
    parser.add_argument("--top-k", type=int, default=None, help="融合后进入重排的候选数")
    parser.add_argument("--reranker", choices=("lexical", "flag", "none"), default="lexical")
    parser.add_argument("--threshold", type=float, default=None, help="临时覆盖相关性阈值")
    parser.add_argument("--out", default=None, help="明细 JSON 落盘路径")
    parser.add_argument("--dump-scores", default=None, help="导每条 rerank_top1，供阈值定标")
    parser.add_argument(
        "--no-permission",
        action="store_true",
        help="关闭权限过滤（对比用：会看到泄漏率飙升）",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.threshold is not None:
        settings = settings.model_copy(
            update={
                "retrieval": settings.retrieval.model_copy(
                    update={"relevance_threshold": args.threshold}
                )
            }
        )

    try:
        questions = load_questions(Path(args.questions))
        bundle = load_index(settings.paths.index)
    except RAGError as exc:
        print(f"[错误] {exc}")
        return 1
    if not questions:
        print("[错误] 评测集为空")
        return 1

    embedder = build_embedder(
        settings, force_hash=bundle.meta.get("embedding_model") == "hash"
    )
    reranker = build_reranker(settings, bundle, args.reranker)
    k = args.top_k or settings.retrieval.top_k

    results: list[QuestionResult] = []
    details: list[dict] = []
    score_rows: list[dict] = []
    by_id = {chunk.chunk_id: chunk for chunk in bundle.chunks}

    for question in questions:
        started = time.perf_counter()

        person = question.get("user") or {}
        user = User(
            user_id=question["id"],
            department=person.get("department", settings.ingest.unknown_department),
            level=person.get("level", settings.default_sensitivity),
        )
        visible = (
            None
            if args.no_permission
            else allowed_ids(bundle.chunks, user, settings)
        )

        fused = rrf_fuse(
            [
                vector_search(bundle, embedder.encode_query(question["question"]), k, visible),
                bm25_search(bundle, question["question"], k, visible),
            ],
            by_id,
        )
        if reranker is None:
            evidence = fused[: settings.retrieval.rerank_top_k]
        else:
            evidence = rerank(
                question["question"], fused[:k], reranker, settings.retrieval.rerank_top_k
            )

        decision = decide(evidence, settings)
        rank = match_rank([item.chunk for item in evidence], question.get("expect", []))

        denied = question["type"] == "denied"
        leaked = bool(
            denied
            and decision.answered
            and any(
                normalize(marker) in normalize(item.chunk.text)
                for item in evidence
                for marker in question.get("must_not_contain", [])
            )
        )

        latency_ms = (time.perf_counter() - started) * 1000
        results.append(
            QuestionResult(
                qid=question["id"],
                qtype=question["type"],
                answered=decision.answered,
                reason=decision.reason,
                first_hit_rank=rank,
                rerank_top1=decision.top1_rerank,
                latency_ms=latency_ms,
                leaked=leaked,
            )
        )
        details.append(
            {
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "answered": decision.answered,
                "reason": decision.reason,
                "first_hit_rank": rank,
                "rerank_top1": None
                if decision.top1_rerank is None
                else round(decision.top1_rerank, 4),
                "evidence": [item.chunk.chunk_id for item in evidence],
                "visible_chunks": None if visible is None else len(visible),
                "latency_ms": round(latency_ms, 2),
            }
        )
        score_rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "rerank_top1": decision.top1_rerank,
                "answered": decision.answered,
                "reason": decision.reason,
            }
        )

    summary = summarize(results)
    reranker_name = "none" if reranker is None else getattr(reranker, "name", "?")

    print(
        f"样本 {summary['count']} "
        f"(normal {summary['counts_by_type']['normal']} / "
        f"boundary {summary['counts_by_type']['boundary']} / "
        f"denied {summary['counts_by_type']['denied']})"
        f"   索引 {bundle.faiss.ntotal} 块 dim={bundle.faiss.d}"
        f"   模型 {bundle.meta.get('embedding_model')}"
        f"   reranker {reranker_name}"
        f"   threshold {settings.retrieval.relevance_threshold}"
        f"   permission {'off' if args.no_permission else 'on'}"
    )
    print()
    print(f"Recall@5        {summary['recall_at_5']:.3f}      MRR        {summary['mrr']:.3f}")
    print(
        f"误拒率          {summary['false_refuse_rate']:.3f}      正确拒答率   "
        f"{summary['correct_refuse_rate']:.3f}"
    )
    print(
        f"泄漏率          {summary['leak_rate']:.3f}      P95 延迟    "
        f"{summary['p95_latency_ms']:.1f}ms"
    )

    if args.out:
        payload = {
            "meta": {
                "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "questions": str(args.questions),
                "index_meta": bundle.meta,
                "config": {
                    "top_k": k,
                    "rerank_top_k": settings.retrieval.rerank_top_k,
                    "threshold": settings.retrieval.relevance_threshold,
                    "reranker": reranker_name,
                    "permission": not args.no_permission,
                },
            },
            "summary": summary,
            "details": details,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细 -> {out}")

    if args.dump_scores:
        out = Path(args.dump_scores)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in score_rows),
            encoding="utf-8",
        )
        print(f"分数 -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
