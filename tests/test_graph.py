"""图契约：知识 / 拒答 / 工具 / 越权引用 / 权限五条路径。

这些用例都不加载真模型：`nodes.chat_with_fallback` 被替换成固定回复。
关键断言有两条：
- **拒答路径不能调用模型**（省钱、也不给幻觉机会）；
- **越权引用必须被拦截**（前置过滤挡不住生成侧的幻觉引用）。
"""

from __future__ import annotations

import pytest

import src.agent.graph as graph_module
import src.agent.nodes as nodes
from src.agent.graph import reset_graph, run
from src.agent.nodes import reset_resources
from src.schema import User


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """每个用例：审计只进内存、图和资源冷启动，避免用例间串状态。"""
    events: list[dict] = []
    monkeypatch.setattr(
        graph_module, "write_event", lambda event, settings: events.append(event) or True
    )
    reset_resources()
    reset_graph()
    yield events
    reset_resources()
    reset_graph()


def _stub_reply(monkeypatch, text: str, sink: list | None = None) -> None:
    def fake(messages, **kwargs):
        if sink is not None:
            sink.append(messages)
        return text, False

    monkeypatch.setattr(nodes, "chat_with_fallback", fake)


def test_knowledge_path_returns_grounded_citations(monkeypatch, isolated):
    _stub_reply(monkeypatch, "一线城市每晚不超过 600 元 [1]")

    state = run("一线城市差旅住宿标准是多少？", User("u1", "研发", "机密"))

    evidence_ids = {item["chunk_id"] for item in state["evidence"]}
    assert state["intent"] == "knowledge"
    assert state["refused"] is False
    assert state["citations"]
    assert set(state["citations"]) <= evidence_ids
    assert isolated and isolated[-1]["answered"] is True
    assert isolated[-1]["citations"] == state["citations"]


def test_refuse_path_never_calls_llm(monkeypatch):
    calls: list = []
    _stub_reply(monkeypatch, "这行不该被调用", sink=calls)

    state = run("公司有通勤班车吗？", User("u1", "研发", "机密"))

    assert state["refused"] is True
    assert state["reason"] in {"low_score", "no_result"}
    assert calls == []


def test_tool_path_bypasses_retrieval(monkeypatch):
    calls: list = []
    _stub_reply(monkeypatch, "不该被调用", sink=calls)

    state = run("计算 (600*2+400)*1.1", User("u1", "研发", "机密"))

    assert state["intent"] == "tool"
    assert state["answer"] == "1760"
    assert (state.get("evidence") or []) == []      # 工具路径不检索，state 里就没有这个键
    assert calls == []


def test_hallucinated_citation_is_blocked(monkeypatch):
    _stub_reply(monkeypatch, "薪酬结构是 12000 元 [9]")   # 只有 5 条证据，[9] 是幻觉

    state = run("一线城市差旅住宿标准是多少？", User("u1", "研发", "机密"))

    assert state["refused"] is True
    assert state["reason"] == "citation_out_of_scope"
    assert state["citations"] == []


def test_normal_user_never_sees_secret_evidence(monkeypatch):
    _stub_reply(monkeypatch, "x [1]")

    state = run("研发部 P5 的月薪带宽是多少？", User("u2", "研发", "普通"))

    sensitivities = {item["sensitivity"] for item in state.get("evidence") or []}
    assert "机密" not in sensitivities
    assert state["refused"] is True


def test_route_intent_keeps_knowledge_questions_out_of_tool(monkeypatch):
    """含"多少/标准"的制度类问题不能被误判成计算题。"""
    _stub_reply(monkeypatch, "每晚不超过 600 元 [1]")

    state = run("差旅住宿标准是多少？", User("u1", "研发", "机密"))

    assert state["intent"] == "knowledge"
