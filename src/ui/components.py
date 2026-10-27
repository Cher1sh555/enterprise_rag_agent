"""UI 组件：把 state 渲染成人能看的东西。

只做展示，不做业务判断——权限、拒答、引用校验全部在 Agent/检索层完成。
UI 如果自己再判断一次，就会出现"界面说能看、实际查不到"的分裂。

这一版新增：
- 顶栏（旋转标志 + 索引状态）
- 标志轮播（纯 CSS marquee）
- 指标条 / 能力卡片 / 分区标题
- 证据摘要 + 命中高亮（snippet）
- 引用芯片（把 chunk_id 换成人能读的"部门/文件名 p.x"）
"""

from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st

from src.utils.text import tokenize

REASON_PILL = {
    "ok": ("已作答", "ok"),
    "no_result": ("未检索到依据", "muted"),
    "low_score": ("相关性不足 · 已拒答", "warn"),
    "forbidden": ("无访问权限", "danger"),
    "citation_out_of_scope": ("引用越界 · 已拦截", "danger"),
    "llm_error": ("模型不可用", "danger"),
    "tool_error": ("工具执行失败", "warn"),
}


# ------------------------------------------------------------------ 布局块


def topbar(chunk_count: int, user_label: str) -> None:
    st.markdown(
        f"""
        <div class="topbar fade-up">
          <div class="logo">
            <span class="ring"></span>
            <span class="wm">企业知识库<em>RAG Studio</em></span>
          </div>
          <div class="status">
            <span class="dot"></span>索引 {chunk_count} 块 · 当前身份 {html.escape(user_label)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def marquee(items: list[str]) -> None:
    """标志轮播：内容复制一份实现无缝循环，hover 暂停。"""
    loop = "".join(f'<span class="mq"><i></i>{html.escape(item)}</span>' for item in items)
    st.markdown(
        f'<div class="marquee"><div class="marquee-track">{loop}{loop}</div></div>',
        unsafe_allow_html=True,
    )


def stat_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="stat"><b>{html.escape(value)}</b><span>{html.escape(key)}</span></div>'
        for key, value in items
    )
    st.markdown(f'<div class="stats fade-up d2">{cells}</div>', unsafe_allow_html=True)


def card_grid(cards: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f'<div class="card fade-up d{min(index + 1, 4)}">'
        f'<div class="idx">{idx}</div><h3>{html.escape(title)}</h3>'
        f"<p>{html.escape(desc)}</p></div>"
        for index, (idx, title, desc) in enumerate(cards)
    )
    st.markdown(f'<div class="grid">{cells}</div>', unsafe_allow_html=True)


def section_header(title: str, caption: str = "") -> None:
    st.markdown(
        f'<div class="fade-up"><h2 style="margin:6px 0 2px;font-size:22px;">{html.escape(title)}</h2>'
        f'<p style="margin:0;color:#64748b;font-size:13px;">{html.escape(caption)}</p></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ 文本处理


def short_source(path: str | None) -> str:
    """data/raw_docs/财务/普通/报销细则.md -> 财务/报销细则"""
    if not path:
        return "?"
    parts = path.replace("\\", "/").split("/")
    name = parts[-1].removesuffix(".md")
    dept = parts[-3] if len(parts) >= 3 else ""
    return f"{dept}/{name}" if dept else name


def snippet(text: str, query: str, window: int = 220) -> str:
    """截取命中附近的一段并高亮 query 词；不命中就取开头。"""
    tokens = [token for token in tokenize(query) if len(token) >= 2][:8]
    if not tokens:
        return html.escape(text[:window]) + ("…" if len(text) > window else "")

    pattern = re.compile("|".join(re.escape(token) for token in tokens))
    hit = pattern.search(text)
    start = max(0, hit.start() - window // 3) if hit else 0
    fragment = text[start : start + window]

    safe = html.escape(fragment)
    alternation = "|".join(
        re.escape(html.escape(token)) for token in sorted(tokens, key=len, reverse=True)
    )
    safe = re.sub(f"({alternation})", r"<mark>\1</mark>", safe)

    prefix = "…" if start > 0 else ""
    suffix = "…" if start + window < len(text) else ""
    return f"{prefix}{safe}{suffix}"


# ------------------------------------------------------------------ 回答区


def render_status(state: dict) -> None:
    reason = state.get("reason") or "unknown"
    label, cls = REASON_PILL.get(reason, (reason, "muted"))
    evidence = state.get("evidence") or []
    top1 = evidence[0].get("rerank") if evidence else None
    latency = state.get("latency_ms")

    metrics = [
        ("意图", state.get("intent") or "—"),
        ("延迟", "—" if latency is None else f"{latency:.0f} ms"),
        ("证据", str(len(evidence))),
        ("rerank", "—" if top1 is None else f"{top1:.3f}"),
    ]
    cells = "".join(
        f'<span class="metric"><span class="k">{key}</span>'
        f'<span class="v">{html.escape(value)}</span></span>'
        for key, value in metrics
    )
    degraded = '<span class="pill pill-warn">已降级</span>' if state.get("degraded") else ""
    st.markdown(
        f'<div class="metric-row"><span class="pill pill-{cls}">{label}</span>'
        f"{cells}{degraded}</div>",
        unsafe_allow_html=True,
    )


def render_answer(state: dict) -> None:
    st.markdown(state.get("answer") or "（无输出）")

    citations = state.get("citations") or []
    if not citations:
        return

    evidence = {item["chunk_id"]: item for item in state.get("evidence") or []}
    chips = []
    for rank, chunk_id in enumerate(citations, start=1):
        item = evidence.get(chunk_id, {})
        chips.append(
            f'<span class="chip">{rank} · {html.escape(short_source(item.get("source")))}'
            f' p.{item.get("page_no", "?")}</span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)


def render_evidence(evidence: list[dict], citations: list[str], query: str = "") -> None:
    if not evidence:
        st.caption("没有检索到证据。")
        return

    cited = set(citations or [])
    for index, item in enumerate(evidence, start=1):
        rerank = item.get("rerank")
        score = "—" if rerank is None else f"{rerank:.3f}"
        secret = " secret" if item.get("sensitivity") == "机密" else ""
        badge = " · ✅ 已引用" if item["chunk_id"] in cited else ""
        st.markdown(
            f'<div class="evidence{secret}">'
            f'<div class="meta">[{index}] {html.escape(short_source(item.get("source")))}'
            f" · p.{item.get('page_no', '?')} · rerank {score}"
            f" · {html.escape(str(item.get('department', '?')))}/"
            f"{html.escape(str(item.get('sensitivity', '?')))}"
            f"{badge}</div>"
            f'<div class="body">{snippet(item.get("text", ""), query)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------ 数据表


def render_chunk_table(chunks: list[Any], visible_ids: set[str] | None = None) -> None:
    rows: dict[str, dict] = {}
    for chunk in chunks:
        row = rows.setdefault(
            chunk.source,
            {
                "可读": "✅",
                "来源": short_source(chunk.source),
                "部门": chunk.department,
                "密级": chunk.sensitivity,
                "块数": 0,
                "字数": 0,
            },
        )
        row["块数"] += 1
        row["字数"] += chunk.char_len
        if visible_ids is not None and chunk.chunk_id not in visible_ids:
            row["可读"] = "🔒"

    if not rows:
        st.info("索引里还没有任何 chunk。")
        return

    data = list(rows.values())
    max_chars = max(row["字数"] for row in data) or 1
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "可读": st.column_config.TextColumn(width="small"),
            "字数": st.column_config.ProgressColumn(
                "字数", min_value=0, max_value=max_chars, format="%d"
            ),
            "块数": st.column_config.NumberColumn(format="%d"),
            "密级": st.column_config.TextColumn(width="small"),
        },
    )


def audit_rows(events: list[dict]) -> list[dict]:
    return [
        {
            "时间": event.get("ts"),
            "身份": f"{event.get('department')}/{event.get('level')}",
            "问题": event.get("question"),
            "意图": event.get("intent"),
            "判定": event.get("reason"),
            "命中块": len(event.get("hit_ids") or []),
            "最高密级": event.get("sensitivity_max"),
            "延迟(ms)": event.get("latency_ms"),
        }
        for event in reversed(events)
    ]
