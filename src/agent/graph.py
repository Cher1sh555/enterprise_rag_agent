"""状态图：节点与条件边（意图识别 / 检索 / 生成 / 拒答 / 工具）。

图只是编排层：真正的逻辑在 nodes.py，而 nodes 又只调用 M3/M5 已经验证过的函数。
对外唯一入口是 `run()`——脚本、UI、测试都走它，避免三处各写一套调用顺序。

多轮说明：`thread_id` 决定会话隔离。**同一个 thread_id 会被不同用户共用历史**，
所以 UI 必须按 `(用户, 会话)` 生成 id；CLI 默认每次随机（= 不复用历史），
只有显式传 `--thread` 才是多轮。
"""

from __future__ import annotations

import time
import uuid
from functools import lru_cache

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    after_intent,
    after_retrieve,
    generate_node,
    refuse_node,
    retrieve,
    route_intent,
    tool_node,
)
from src.agent.state import RAGState
from src.config import get_settings
from src.schema import User
from src.security.audit import write_event
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_graph(checkpointer=None):
    """显式传 checkpointer=None 可得到无状态图（测试用）。"""
    graph = StateGraph(RAGState)

    graph.add_node("route_intent", route_intent)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("tool", tool_node)

    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges(
        "route_intent",
        after_intent,
        {"retrieve": "retrieve", "tool": "tool"},
    )
    graph.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {"generate": "generate", "refuse": "refuse"},
    )
    graph.add_edge("tool", END)
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)

    return graph.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def _compiled():
    return build_graph(MemorySaver())


def reset_graph() -> None:
    """测试 / 换索引后调用。"""
    _compiled.cache_clear()


def run(
    question: str,
    user: User,
    thread_id: str | None = None,
) -> dict:
    """跑一次完整链路，返回最终 state（含 answer / citations / reason）。"""
    started = time.perf_counter()
    conversation = thread_id or uuid.uuid4().hex[:12]

    initial: dict = {
        "question": question,
        "user": {
            "user_id": user.user_id,
            "department": user.department,
            "level": user.level,
        },
        "trace_id": conversation,
        "messages": [HumanMessage(content=question)],
    }
    state = _compiled().invoke(
        initial, config={"configurable": {"thread_id": conversation}}
    )
    state["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    _audit(state, user)
    return state


def _audit(state: dict, user: User) -> None:
    settings = get_settings()
    levels = settings.security.security_levels
    evidence = state.get("evidence") or []
    sensitivity = max(
        (item["sensitivity"] for item in evidence),
        key=lambda name: levels.index(name) if name in levels else -1,
        default=None,
    )
    write_event(
        {
            "trace_id": state.get("trace_id"),
            "user_id": user.user_id,
            "department": user.department,
            "level": user.level,
            "question": state.get("question"),
            "intent": state.get("intent"),
            "hit_ids": [item["chunk_id"] for item in evidence],
            "citations": state.get("citations") or [],
            "sensitivity_max": sensitivity,
            "answered": not state.get("refused", False),
            "reason": state.get("reason"),
            "degraded": state.get("degraded", False),
            "latency_ms": state.get("latency_ms"),
            "answer_chars": len(state.get("answer") or ""),
        },
        settings,
    )
