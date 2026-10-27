"""各节点逻辑实现。

铁律：节点只做"读 state → 调用已有函数 → 写 state"，**不重新实现检索/生成**。
这样 `run_eval` 的基准和 `--graph` 的行为不会悄悄分叉。

用到的已有能力（M3/M5 做完的）：
    allowed_ids（权限） / vector_search + bm25_search + rrf_fuse（召回）
    rerank（重排） / decide（阈值判定） / build_messages + chat_with_fallback（生成）
"""

from __future__ import annotations

import re
from dataclasses import fields
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage

from src.agent.tools import run_tool
from src.config import Settings, get_settings
from src.ingest.embedder import build_embedder
from src.ingest.indexer import load_index
from src.llm.client import build_messages, chat_with_fallback
from src.retrieval.hybrid import bm25_search, rrf_fuse, vector_search
from src.retrieval.reranker import build_reranker, rerank
from src.retrieval.threshold import decide
from src.schema import Chunk, ScoredChunk, User
from src.security.permission import allowed_ids
from src.utils.exceptions import RAGError
from src.utils.logger import get_logger

logger = get_logger(__name__)

REFUSE_TEXT = {
    "no_result": "知识库中未找到相关依据。",
    "low_score": "知识库中未找到足够相关的依据，建议补充更具体的提问。",
    "forbidden": "你没有访问相关资料的权限。",
    "llm_error": "服务暂时不可用，请稍后重试。",
    "citation_out_of_scope": "回答引用了无权访问的资料，已拦截。",
}

_KNOWLEDGE_HINTS = ("制度", "规定", "标准", "流程", "政策", "报销", "年假", "差旅", "第", "条")
_TOOL_HINTS = ("计算", "算一下", "帮我算", "等于多少")
_EXPR_RE = re.compile(r"[-\d+*/().\s]{3,}")
_CITE_RE = re.compile(r"\[(\d{1,2})\]")


# --------------------------------------------------------------------- 资源


@lru_cache(maxsize=1)
def _resources() -> tuple[Settings, Any, Any, Any]:
    """索引 / embedder / reranker 只加载一次（建索引后记得 reset_resources）。"""
    settings = get_settings()
    bundle = load_index(settings.paths.index)
    embedder = build_embedder(
        settings, force_hash=bundle.meta.get("embedding_model") == "hash"
    )
    reranker = build_reranker(settings, bundle, "lexical")
    return settings, bundle, embedder, reranker


def reset_resources() -> None:
    """测试 / 重建索引后调用。"""
    _resources.cache_clear()


# ------------------------------------------------------------------- 路由


def route_intent(state: dict) -> dict:
    """规则优先，默认落 knowledge（最坏多检索一次，不会答错）。"""
    question = state["question"].strip()

    if any(hint in question for hint in _KNOWLEDGE_HINTS):
        return {"intent": "knowledge"}
    if any(hint in question for hint in _TOOL_HINTS) or _pure_expression(question):
        return {"intent": "tool"}
    return {"intent": "knowledge"}


def after_intent(state: dict) -> str:
    return "tool" if state.get("intent") == "tool" else "retrieve"


def _pure_expression(text: str) -> bool:
    return bool(re.fullmatch(r"[-\d+*/().\s]{3,}", text))


# ------------------------------------------------------------------- 检索


def retrieve(state: dict) -> dict:
    settings, bundle, embedder, reranker = _resources()
    person = state.get("user") or {}
    user = User(
        user_id=person.get("user_id", "unknown"),
        department=person.get("department", settings.ingest.unknown_department),
        level=person.get("level", settings.default_sensitivity),
    )

    visible = allowed_ids(bundle.chunks, user, settings)
    if not visible:
        return {"visible_ids": [], "evidence": [], "refused": True, "reason": "forbidden"}

    k = settings.retrieval.top_k
    by_id = {chunk.chunk_id: chunk for chunk in bundle.chunks}
    fused = rrf_fuse(
        [
            vector_search(bundle, embedder.encode_query(state["question"]), k, visible),
            bm25_search(bundle, state["question"], k, visible),
        ],
        by_id,
    )

    if reranker is None:
        evidence = fused[: settings.retrieval.rerank_top_k]
    else:
        evidence = rerank(
            state["question"], fused[:k], reranker, settings.retrieval.rerank_top_k
        )

    decision = decide(evidence, settings)
    return {
        "visible_ids": sorted(visible),
        "evidence": [_serialize(item) for item in evidence],
        "refused": not decision.answered,
        "reason": decision.reason,
    }


def after_retrieve(state: dict) -> str:
    return "refuse" if state.get("refused") else "generate"


# ------------------------------------------------------------------- 生成


def generate_node(state: dict) -> dict:
    evidence = state.get("evidence") or []
    if not evidence:
        return {"refused": True, "reason": "no_result", "answer": REFUSE_TEXT["no_result"]}

    chunks = [_to_chunk(item) for item in evidence]
    messages = build_messages(state["question"], chunks, history=_history(state))

    try:
        answer, degraded = chat_with_fallback(messages)
    except RAGError as exc:
        logger.warning("生成失败: %s", exc)
        return {
            "refused": True,
            "reason": "llm_error",
            "answer": REFUSE_TEXT["llm_error"],
            "citations": [],
            "error": str(exc),
        }

    citations, invalid = parse_citations(answer, evidence)
    visible = state.get("visible_ids")
    if visible is not None:
        allowed = set(visible)
        out_of_scope = [cid for cid in citations if cid not in allowed]
    else:
        out_of_scope = []

    if invalid or out_of_scope:
        logger.warning("引用越界，拦截：invalid=%s out_of_scope=%s", invalid, out_of_scope)
        return {
            "refused": True,
            "reason": "citation_out_of_scope",
            "answer": REFUSE_TEXT["citation_out_of_scope"],
            "citations": [],
            "degraded": degraded,
            "messages": [AIMessage(content=REFUSE_TEXT["citation_out_of_scope"])],
        }

    return {
        "answer": answer,
        "citations": citations,
        "refused": False,
        "reason": "ok",
        "degraded": degraded,
        "messages": [AIMessage(content=answer)],
    }


def parse_citations(answer: str, evidence: list[dict]) -> tuple[list[str], list[str]]:
    """把答案里的 [n] 映射成 chunk_id。返回 (合法引用, 越界编号)。"""
    citations: list[str] = []
    invalid: list[str] = []
    for raw in _CITE_RE.findall(answer or ""):
        index = int(raw) - 1
        if 0 <= index < len(evidence):
            chunk_id = evidence[index]["chunk_id"]
            if chunk_id not in citations:
                citations.append(chunk_id)
        else:
            invalid.append(raw)
    return citations, invalid


def _history(state: dict) -> list[dict]:
    """最近两轮对话（排除当前问题）。"""
    messages = state.get("messages") or []
    prior = list(messages)[:-1][-4:]
    history: list[dict] = []
    for message in prior:
        role = "user" if getattr(message, "type", "") == "human" else "assistant"
        content = getattr(message, "content", "")
        if content:
            history.append({"role": role, "content": str(content)[:400]})
    return history


# --------------------------------------------------------------- 拒答/工具


def refuse_node(state: dict) -> dict:
    reason = state.get("reason") or "no_result"
    text = REFUSE_TEXT.get(reason, REFUSE_TEXT["no_result"])
    return {
        "answer": text,
        "refused": True,
        "citations": [],
        "messages": [AIMessage(content=text)],
    }


def tool_node(state: dict) -> dict:
    expression = _extract_expression(state["question"])
    result = run_tool("calculator", expression)
    return {
        "answer": result,
        "citations": [],
        "refused": result.startswith("[工具错误]"),
        "reason": "ok" if not result.startswith("[工具错误]") else "tool_error",
        "messages": [AIMessage(content=result)],
    }


def _extract_expression(text: str) -> str:
    matches = _EXPR_RE.findall(text)
    return max(matches, key=len).strip() if matches else text


# ------------------------------------------------------------------ 序列化

_CHUNK_FIELDS = tuple(field.name for field in fields(Chunk))


def _serialize(item: ScoredChunk) -> dict:
    payload = {name: getattr(item.chunk, name) for name in _CHUNK_FIELDS}
    payload.update(
        {"rrf": item.score, "rerank": item.rerank_score, "sources": list(item.sources)}
    )
    return payload


def _to_chunk(payload: dict) -> Chunk:
    return Chunk(**{name: payload[name] for name in _CHUNK_FIELDS})
