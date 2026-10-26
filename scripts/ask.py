"""问一句，答一句：权限过滤 -> 检索 -> 重排 -> (可选)拒答 -> LLM -> 带引用的答案。

三条硬约束：
1. 证据集合只有一份（`evidence`）：打印、判定、喂给模型都用它；
2. 权限过滤发生在**检索之前**（allowed_ids），不是拿到结果再删；
3. 每次提问都写一条审计，审计失败不影响回答。
"""

from __future__ import annotations

import argparse
import sys
import time

from src.config import get_settings
from src.ingest.embedder import build_embedder
from src.ingest.indexer import load_index
from src.llm.client import build_messages, chat
from src.retrieval.hybrid import bm25_search, rrf_fuse, vector_search
from src.retrieval.reranker import build_reranker, rerank
from src.retrieval.threshold import decide
from src.schema import User
from src.security.audit import write_event
from src.security.permission import allowed_ids
from src.utils.exceptions import RAGError


def parse_user(raw: str) -> User:
    """把 `部门:密级` 解析成 User；格式不对就给可读错误。"""
    department, _, level = raw.partition(":")
    if not department or not level:
        raise SystemExit(f"--user 格式应为 部门:密级，例如 研发:机密；收到 {raw!r}")
    return User(user_id=f"cli-{department}-{level}", department=department, level=level)


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库问答")
    parser.add_argument("question")
    parser.add_argument("--no-llm", action="store_true", help="只打印召回结果，不调模型")
    parser.add_argument("--top-k", type=int, default=None, help="融合后进入重排的候选数")
    parser.add_argument(
        "--user", default="研发:机密", help="提问者身份，格式 部门:密级（默认 研发:机密）"
    )
    parser.add_argument(
        "--reranker",
        choices=("lexical", "flag", "none"),
        default="lexical",
        help="lexical=词覆盖率（离线）；flag=cross-encoder（需下载模型）；none=不重排",
    )
    args = parser.parse_args()

    settings = get_settings()
    user = parse_user(args.user)

    try:
        bundle = load_index(settings.paths.index)
    except RAGError as exc:
        print(f"[错误] {exc}")
        return 1

    embedder = build_embedder(
        settings, force_hash=bundle.meta.get("embedding_model") == "hash"
    )
    reranker = build_reranker(settings, bundle, args.reranker)

    visible = allowed_ids(bundle.chunks, user, settings)
    print(
        f"[权限] {user.department}/{user.level} 可见 {len(visible)}/{len(bundle.chunks)} 块"
    )

    started = time.perf_counter()
    k = args.top_k or settings.retrieval.top_k
    by_id = {chunk.chunk_id: chunk for chunk in bundle.chunks}
    fused = rrf_fuse(
        [
            vector_search(bundle, embedder.encode_query(args.question), k, visible),
            bm25_search(bundle, args.question, k, visible),
        ],
        by_id,
    )

    if reranker is None:
        evidence = fused[: settings.retrieval.rerank_top_k]
    else:
        evidence = rerank(
            args.question, fused[:k], reranker, settings.retrieval.rerank_top_k
        )

    decision = decide(evidence, settings)
    latency_ms = (time.perf_counter() - started) * 1000

    top1_rerank = "—" if decision.top1_rerank is None else f"{decision.top1_rerank:.3f}"
    print(
        f"[检索] 候选 {len(fused)}，证据 {len(evidence)}，"
        f"rerank_top1={top1_rerank}，判定={decision.reason}"
    )
    for rank, item in enumerate(evidence, start=1):
        rerank_part = "—" if item.rerank_score is None else f"{item.rerank_score:.3f}"
        print(
            f"  [{rank}] rrf={item.score:.4f} rerank={rerank_part} "
            f"{'+'.join(item.sources)}  {item.chunk.source} p.{item.chunk.page_no}"
        )
        print(f"      {item.chunk.text[:60]}...")

    answer = ""
    if not args.no_llm and decision.answered and evidence:
        try:
            answer = chat(build_messages(args.question, [item.chunk for item in evidence]))
        except RAGError as exc:
            print(f"\n[错误] {exc}")
            _audit(settings, user, args.question, evidence, decision, latency_ms, "")
            return 2

    if not args.no_llm:
        if decision.answered and answer:
            print(f"\n[回答] {answer}")
        else:
            print("\n[回答] 知识库中未找到相关依据。")

    _audit(settings, user, args.question, evidence, decision, latency_ms, answer)
    return 0


def _audit(settings, user, question, evidence, decision, latency_ms, answer) -> None:
    levels = settings.security.security_levels
    sensitivity = max(
        (item.chunk.sensitivity for item in evidence),
        key=lambda name: levels.index(name) if name in levels else -1,
        default=None,
    )
    write_event(
        {
            "user_id": user.user_id,
            "department": user.department,
            "level": user.level,
            "question": question,
            "hit_ids": [item.chunk.chunk_id for item in evidence],
            "sensitivity_max": sensitivity,
            "answered": decision.answered,
            "reason": decision.reason,
            "rerank_top1": decision.top1_rerank,
            "latency_ms": round(latency_ms, 2),
            "answer_chars": len(answer),
        },
        settings,
    )


if __name__ == "__main__":
    sys.exit(main())
