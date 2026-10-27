"""Streamlit 页面：首页 / 问答 / 文档 / 审计。

结构：
    顶栏（旋转标志 + 索引状态）
    ├─ 首页：hero(粒子画布) + 标志轮播 + 指标 + 能力卡片 + 快捷入口
    ├─ 问答：身份可见块数 + 提问表单 + 对话式历史 + 证据高亮
    ├─ 文档：来源聚合表（🔒 标记当前身份读不到）
    └─ 审计：最近 200 条 + 身份过滤 + 延迟条形图
    底部固定导航（首页/问答/文档/审计）

设计原则：UI 是 Agent 的薄壳。一问一答只调用 `src.agent.graph.run()`，
不自己拼检索/权限逻辑；没有 LLM key 时证据仍然展示，用来验证检索质量。
"""

from __future__ import annotations

import os
import uuid

import streamlit as st

from src.agent.graph import reset_graph, run as run_graph
from src.agent.nodes import reset_resources
from src.config import get_settings
from src.ingest.indexer import load_index
from src.schema import User
from src.security.audit import read_events
from src.security.permission import allowed_ids
from src.ui import components as ui
from src.ui import hero, theme
from src.utils.exceptions import RAGError

NAV = [("home", "首页"), ("ask", "问答"), ("docs", "文档"), ("audit", "审计")]

TECH = [
    "LangGraph",
    "FAISS · IndexFlatIP",
    "BM25 · jieba",
    "bge-m3 / hash",
    "RRF 融合",
    "权限前置过滤",
    "引用后置校验",
    "审计 JSONL",
]

CAPABILITIES = [
    ("01", "混合检索", "向量 + BM25 双路召回，RRF 融合，cross-encoder 重排"),
    ("02", "权限隔离", "密级 + 部门双维前置过滤，越权片段进不了候选集"),
    ("03", "引用溯源", "答案逐条标注来源，引用编号经后置校验，幻觉引用被拦截"),
    ("04", "Agent 编排", "意图路由 · 工具调用 · 拒答分支，五节点状态图"),
]


# --------------------------------------------------------------------- 资源


@st.cache_resource(show_spinner=False)
def _bundle():
    return load_index(get_settings().paths.index)


def _settings():
    return get_settings()


def _init_session() -> None:
    st.session_state.setdefault("section", "home")
    st.session_state.setdefault("thread_id", uuid.uuid4().hex[:12])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("department", _settings().security.departments[0])
    st.session_state.setdefault("level", _settings().security.security_levels[-1])
    st.session_state.setdefault("motion", True)


def _go(section: str) -> None:
    st.session_state["section"] = section
    st.rerun()


# --------------------------------------------------------------------- 侧栏


def _sidebar() -> User:
    settings = _settings()
    st.sidebar.markdown("### 身份与操作")
    st.sidebar.caption("切换身份会立即改变可见知识块范围。")

    department = st.sidebar.selectbox(
        "部门",
        settings.security.departments,
        index=settings.security.departments.index(st.session_state["department"]),
    )
    level = st.sidebar.selectbox(
        "密级权限",
        settings.security.security_levels,
        index=settings.security.security_levels.index(st.session_state["level"]),
    )
    st.session_state["department"] = department
    st.session_state["level"] = level
    user = User(user_id=f"ui-{department}-{level}", department=department, level=level)

    st.sidebar.divider()
    st.sidebar.caption(f"会话 `{st.session_state['thread_id']}`")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("新会话", use_container_width=True):
        st.session_state["thread_id"] = uuid.uuid4().hex[:12]
        st.session_state["history"] = []
        st.rerun()
    if col2.button("重载索引", use_container_width=True):
        _bundle.clear()
        reset_resources()
        reset_graph()
        st.rerun()

    if not os.getenv("LLM_API_KEY", "").strip():
        st.sidebar.warning("未配置 LLM_API_KEY：只展示检索证据，不生成答案。")

    st.sidebar.divider()
    st.sidebar.toggle(
        "界面动效",
        key="motion",
        help="关闭后停止旋转 Logo、光斑漂移与入场动画（系统开启『减少动态效果』时也以此开关为准）。",
    )
    st.sidebar.caption("问答 · 文档 · 审计，用底部导航切换。")
    return user


# --------------------------------------------------------------------- 首页


def _page_home(bundle, user: User) -> None:
    hero.render(
        title_html="企业内部知识，<em>一句话抵达</em>",
        subtitle="多格式文档解析 · 混合检索 · 权限隔离 · 引用可溯源。"
        "答案只来自你被授权看到的那部分资料。",
        badges=["混合检索", "密级隔离", "引用校验", "离线可跑"],
        force_motion=st.session_state.get("motion", True),
    )

    ui.marquee(TECH)

    visible = allowed_ids(bundle.chunks, user, _settings())
    sources = {chunk.source for chunk in bundle.chunks}
    secret = sum(1 for chunk in bundle.chunks if chunk.sensitivity == "机密")
    ui.stat_strip(
        [
            ("知识文档", str(len(sources))),
            ("知识块", str(len(bundle.chunks))),
            ("机密块", str(secret)),
            ("当前身份可见", f"{len(visible)}/{len(bundle.chunks)}"),
        ]
    )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    ui.card_grid(CAPABILITIES)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.caption("快速进入")
    col1, col2, col3 = st.columns(3)
    if col1.button("开始提问 →", use_container_width=True, type="primary"):
        _go("ask")
    if col2.button("查看知识库 →", use_container_width=True):
        _go("docs")
    if col3.button("查看审计 →", use_container_width=True):
        _go("audit")

    history = st.session_state.get("history") or []
    if history:
        st.divider()
        st.caption(f"本会话最近 {min(len(history), 3)} 个问题")
        for turn in history[-3:]:
            st.markdown(f"- {turn['question']}")


# --------------------------------------------------------------------- 问答


def _page_ask(user: User, bundle) -> None:
    visible = allowed_ids(bundle.chunks, user, _settings())
    ui.section_header(
        "知识问答",
        f"当前身份 {user.department}/{user.level} · 可见 {len(visible)}/{len(bundle.chunks)} 个知识块",
    )

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input(
            "问题",
            placeholder="例如：一线城市差旅住宿标准是多少？",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("提问", type="primary")

    if submitted and question.strip():
        with st.spinner("检索 · 重排 · 生成…"):
            try:
                state = run_graph(
                    question.strip(), user, thread_id=st.session_state["thread_id"]
                )
            except RAGError as exc:
                st.error(f"执行失败：{exc}")
                return
        st.session_state["history"].append({"question": question.strip(), "state": state})

    history = st.session_state["history"]
    if not history:
        st.info("试试问：一线城市差旅住宿标准是多少？ / 计算 (600*2+400)*1.1")
        return

    for turn in reversed(history):
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            state = turn["state"]
            ui.render_status(state)
            ui.render_answer(state)
            evidence = state.get("evidence") or []
            if evidence:
                with st.expander(f"证据（{len(evidence)} 条）", expanded=False):
                    ui.render_evidence(
                        evidence, state.get("citations") or [], turn["question"]
                    )


# --------------------------------------------------------------------- 文档


def _page_docs(user: User, bundle) -> None:
    visible = allowed_ids(bundle.chunks, user, _settings())
    ui.section_header(
        "知识库",
        f"按来源聚合 · 当前身份 {user.department}/{user.level} · 🔒 表示读不到",
    )
    ui.render_chunk_table(bundle.chunks, visible_ids=visible)

    with st.expander("索引元信息"):
        st.json(bundle.meta)


# --------------------------------------------------------------------- 审计


def _page_audit() -> None:
    ui.section_header("审计日志", "每次提问的身份、命中块、最高密级与判定都留痕")
    events = read_events(_settings(), limit=200)
    if not events:
        st.info("还没有审计记录，先去问一个问题。")
        return

    users = sorted({f"{e.get('department')}/{e.get('level')}" for e in events})
    chosen = st.multiselect("按身份过滤", users, default=users)
    filtered = [e for e in events if f"{e.get('department')}/{e.get('level')}" in chosen]
    if not filtered:
        st.info("没有匹配的记录。")
        return

    st.dataframe(
        ui.audit_rows(filtered),
        use_container_width=True,
        hide_index=True,
        column_config={
            "延迟(ms)": st.column_config.NumberColumn(format="%.0f"),
            "命中块": st.column_config.NumberColumn(format="%d", width="small"),
        },
    )

    latencies = [float(event.get("latency_ms") or 0) for event in filtered]
    if latencies:
        st.caption("延迟分布（ms）")
        st.bar_chart({"延迟": latencies}, height=150)

    with st.expander("原始 JSONL（最近 20 条）"):
        st.json(filtered[-20:])


# ------------------------------------------------------------------- 底部导航


def _bottom_nav() -> None:
    with st.container(key="bottomnav"):
        cols = st.columns(len(NAV))
        for col, (key, label) in zip(cols, NAV):
            active = st.session_state["section"] == key
            if col.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                _go(key)


# --------------------------------------------------------------------- 入口


def main() -> None:
    st.set_page_config(
        page_title="企业知识库 · RAG Studio", page_icon="◍", layout="wide"
    )
    theme.inject(st.session_state.get("motion", True))
    _init_session()

    try:
        bundle = _bundle()
    except RAGError as exc:
        st.error(f"索引不可用：{exc}")
        st.code("python scripts/build_index.py --hash-embed", language="powershell")
        st.stop()

    user = _sidebar()
    ui.topbar(len(bundle.chunks), f"{user.department}/{user.level}")

    section = st.session_state["section"]
    if section == "ask":
        _page_ask(user, bundle)
    elif section == "docs":
        _page_docs(user, bundle)
    elif section == "audit":
        _page_audit()
    else:
        _page_home(bundle, user)

    _bottom_nav()


if __name__ == "__main__":
    main()
