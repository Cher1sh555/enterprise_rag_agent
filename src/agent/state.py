"""LangGraph 状态定义。

所有字段都必须可序列化（要进 checkpointer）：
- `user` 用 dict，不用 frozen dataclass；
- `evidence` 存 dict，需要 `Chunk` 时按字段还原；
- `messages` 用 add_messages reducer，多轮才能累加而不是覆盖。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class RAGState(TypedDict, total=False):
    # ---- 输入 ----
    question: str
    user: dict[str, str]            # {"user_id","department","level"}
    trace_id: str

    # ---- 路由 ----
    intent: str                     # "knowledge" | "tool" | "smalltalk"

    # ---- 检索 / 判定 ----
    visible_ids: list[str] | None   # None = 不过滤（仅对照实验）；[] = 一个都看不到
    evidence: list[dict[str, Any]]  # Chunk 字段 + rrf/rerank/sources
    refused: bool
    reason: str                     # ok | no_result | low_score | forbidden | citation_out_of_scope | llm_error

    # ---- 生成 ----
    answer: str
    citations: list[str]            # chunk_id 列表，顺序与答案里的 [n] 对应
    degraded: bool

    # ---- 多轮 ----
    messages: Annotated[list, add_messages]

    # ---- 观测 ----
    latency_ms: float
    error: str
