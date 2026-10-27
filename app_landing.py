"""Ocean 风格落地首页：企业 RAG 问答 Agent。

直接运行：
    streamlit run app_landing.py

设计目标（Ocean / 深蓝低饱和 / 商务静谧）：
- 深海蓝 + 浅青 + 米白三色为主，标题衬线、正文无衬线；
- Hero 全屏海景，导航与文字浮在图上；
- 下方为 RAG 问答工作区，左侧对话、右侧文档入库与检索参数。

代码结构（自上而下）：
    1) 常量与配色        —— 换背景图 / 换色只改这里
    2) 全局 CSS          —— 关闭默认外壳、字体、导航、Hero、组件皮肤
    3) Hero HTML         —— 纯 HTML + CSS，不用前端框架
    4) RAG 接入层        —— 三个接入点：检索 / 参数 / 入库（含降级演示）
    5) 页面区块          —— 对话区、文档区、参数区
    6) main()

接入你现有仓库的方式已经写好（src.agent.graph / src.ingest.pipeline），
即使这些模块不可用，页面也能以"演示模式"完整跑起来。
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# 让 `import src.xxx` 在任意工作目录下都成立（streamlit run 只会把脚本目录加进 sys.path）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# 1) 常量：替换背景图 / 字体 / 品牌名，只改这一段
# ===========================================================================

# ★ 背景图位置：换成你自己的海景日出图即可（建议 2400×1600 以上、深蓝低饱和）
HERO_IMAGE = (
    "https://images.unsplash.com/photo-1505142468610-359e7d316be0"
    "?auto=format&fit=crop&w=2400&q=80"
)
# 备选（同样是免费可商用的 Unsplash 海景）：
#   https://images.unsplash.com/photo-1500375592092-40eb2168fd21?auto=format&fit=crop&w=2400&q=80
#   https://images.unsplash.com/photo-1518837695005-2083093ee35b?auto=format&fit=crop&w=2400&q=80

BRAND = "Enterprise RAG Agent"
FONT_HREF = (
    "https://fonts.googleapis.com/css2?"
    "family=Playfair+Display:wght@400;500&family=Inter:wght@300;400;500&display=swap"
)

st.set_page_config(
    page_title="Enterprise RAG Agent",
    page_icon="🌊",
    layout="wide",                       # 宽屏布局
    initial_sidebar_state="collapsed",   # 关闭默认侧边栏
)


# ===========================================================================
# 2) 全局 CSS：关闭默认外壳 + Ocean 设计系统
# ===========================================================================

CSS = """
<style>
@import url('__FONT__');

:root{
  --sea-900:#061A2B;
  --sea-800:#0A2136;
  --sea-700:#0E3A5C;
  --sea-500:#1B5578;
  --cyan-300:#9CC3CE;
  --cyan-100:#DDEBEF;
  --paper:#F7F3EC;
  --paper-2:#F1EBE0;
  --ink:#0A2136;
  --muted:#5B7385;
  --line:rgba(10,33,54,.10);
  --serif:"Playfair Display","Cormorant Garamond",Georgia,"Songti SC","Noto Serif SC",serif;
  --sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
}

/* ---------- 关闭 Streamlit 默认外壳 ---------- */
header[data-testid="stHeader"]{ display:none !important; height:0 !important; }
[data-testid="stSidebar"]{ display:none !important; }
[data-testid="collapsedControl"]{ display:none !important; }
#MainMenu, footer{ visibility:hidden; }
html{ scroll-behavior:smooth; }
html, body{ overflow-x:hidden; }
/* 正文容器：正常宽度，Hero 用负边距"冲"出去做全屏 */
.block-container{ max-width:1120px !important; margin:0 auto !important; padding:0 24px !important; }
html, body, .stApp,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
[data-testid="stExpander"] summary,
h1, h2, h3, h4, h5, h6, p, li, label, summary,
input, textarea, button, select{ font-family:var(--sans); color:var(--ink); }

/* 图标字体必须单独救回来：通配选择器会把 Material 图标渲染成连字名（keyboard_arrow_right） */
[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
span[class*="material-symbols"],
span[class*="material-icons"],
.material-symbols-rounded, .material-symbols-outlined, .material-icons{
  font-family:"Material Symbols Rounded","Material Symbols Outlined","Material Icons" !important;
  font-feature-settings:"liga" !important;
  letter-spacing:normal !important;
}
.stApp{ background:var(--paper); }

/* ---------- 2.1 Hero：全屏海景 + 浮层导航 ---------- */
.hero{
  position:relative; min-height:100svh; color:var(--paper);
  background-size:cover; background-position:center; background-repeat:no-repeat;
  display:flex; flex-direction:column; overflow:hidden;
  /* 全屏出血：抵消 .block-container 的 24px 内边距 */
  margin-left:calc(50% - 50vw); margin-right:calc(50% - 50vw);
}
.hero-veil{
  position:absolute; inset:0; pointer-events:none;
  background:
    linear-gradient(180deg, rgba(6,26,43,.58) 0%, rgba(6,26,43,.18) 34%, rgba(6,26,43,.62) 100%),
    radial-gradient(120% 90% at 18% 8%, rgba(27,85,120,.30), transparent 62%);
}

/* 导航：叠加在大图上方 */
.nav{
  position:absolute; top:0; left:0; right:0; z-index:20;
  display:flex; align-items:center; justify-content:space-between;
  padding:26px 44px; gap:24px;
}
.brand{ display:flex; align-items:center; gap:11px; text-decoration:none; color:var(--paper); }
.brand svg{ width:30px; height:30px; opacity:.92; }
.brand-txt{ font-size:14px; letter-spacing:.02em; font-weight:500; }
.nav-links{ display:flex; gap:34px; }
.nav-links a{
  font-size:13px; color:rgba(247,243,236,.78); text-decoration:none;
  letter-spacing:.02em; padding-bottom:3px; border-bottom:1px solid transparent;
  transition:color .2s ease, border-color .2s ease;
}
.nav-links a:hover{ color:var(--paper); border-bottom-color:rgba(247,243,236,.55); }
.nav-cta{
  font-size:12.5px; color:var(--paper); text-decoration:none;
  border:1px solid rgba(247,243,236,.42); border-radius:999px;
  padding:9px 20px; letter-spacing:.06em; transition:all .22s ease;
}
.nav-cta:hover{ background:rgba(247,243,236,.12); border-color:rgba(247,243,236,.72); }

/* Hero 内容 */
.hero-inner{
  position:relative; z-index:10; flex:1;
  display:flex; flex-direction:column; justify-content:space-between;
  padding:150px 44px 40px;
}
.kicker{
  margin:0 0 22px; font-size:11px; letter-spacing:.32em; text-transform:uppercase;
  color:rgba(247,243,236,.68); font-weight:500;
}
h1.hero-title{
  margin:0; font-family:var(--serif); font-weight:400;
  font-size:clamp(40px, 7vw, 92px); line-height:1.03; letter-spacing:-.02em;
  max-width:16em; text-shadow:0 2px 40px rgba(6,26,43,.35);
}
.lede{
  margin:26px 0 0; font-size:15.5px; line-height:1.75; font-weight:300;
  color:rgba(247,243,236,.82); max-width:34em;
}
.cta-row{ display:flex; flex-wrap:wrap; gap:14px; margin-top:38px; }

/* 按钮 */
.btn{
  display:inline-flex; align-items:center; gap:10px;
  padding:14px 26px; border-radius:999px; text-decoration:none;
  font-size:13.5px; font-weight:500; letter-spacing:.02em;
  transition:transform .2s ease, background .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.btn-solid{ background:var(--sea-700); color:var(--paper); box-shadow:0 14px 34px rgba(6,26,43,.38); }
.btn-solid:hover{ background:var(--sea-500); transform:translateY(-1px); }
.btn-ghost{ border:1px solid rgba(247,243,236,.45); color:var(--paper); }
.btn-ghost:hover{ background:rgba(247,243,236,.10); border-color:rgba(247,243,236,.75); }
.play{
  width:26px; height:26px; border-radius:50%;
  border:1px solid rgba(247,243,236,.6);
  display:grid; place-items:center; font-size:8px;
}

/* Hero 底部：左标识 / 右滚动提示 */
.hero-foot{
  display:flex; align-items:flex-end; justify-content:space-between;
  gap:16px; margin-top:48px;
}
.foot-label{ font-size:11px; letter-spacing:.18em; color:rgba(247,243,236,.62); }
.scroll{
  display:inline-flex; align-items:center; gap:10px; text-decoration:none;
  font-size:10.5px; letter-spacing:.26em; color:rgba(247,243,236,.72);
}
.scroll .arrow{ display:inline-block; animation:bob 2.6s ease-in-out infinite; }
@keyframes bob{ 0%,100%{ transform:translateY(0); opacity:.65 } 50%{ transform:translateY(5px); opacity:1 } }

/* ---------- 2.2 下方纸张区（米白） ---------- */
.sheet{ background:var(--paper); }
.sec{
  max-width:1120px; margin:0 auto; padding:0 24px;
}
.sec-head{ padding:76px 0 26px; }
.sec-label{
  font-size:10.5px; letter-spacing:.3em; text-transform:uppercase;
  color:var(--sea-500); margin:0 0 14px;
}
.sec-title{
  font-family:var(--serif); font-weight:400; letter-spacing:-.01em;
  font-size:clamp(28px,3.4vw,40px); line-height:1.15; margin:0; color:var(--ink);
}
.sec-sub{ margin:14px 0 0; font-size:14px; line-height:1.7; color:var(--muted); max-width:44em; }

/* Streamlit 区块容器（key 会生成 st-key-xxx 类） */
.sec{ padding:0; }

/* ---------- 2.3 对话区皮肤 ---------- */
[data-testid="stChatMessage"]{
  background:#fff; border:1px solid var(--line); border-radius:16px;
  padding:10px 16px; box-shadow:0 1px 2px rgba(10,33,54,.03);
}
[data-testid="stChatMessage"] p{ font-size:14px; line-height:1.75; color:#16324a; }

/* 输入框：胶囊感 */
.stTextInput input{
  border-radius:999px !important; border:1px solid var(--line) !important;
  background:#fff !important; padding:13px 20px !important; font-size:14px !important;
  color:var(--ink) !important;
}
.stTextInput input::placeholder{ color:#8ba0ae; }
.stTextInput input:focus{ border-color:var(--sea-500) !important; box-shadow:0 0 0 3px rgba(27,85,120,.10) !important; }
.stFormSubmitButton button{
  border-radius:999px !important; background:var(--sea-700) !important; color:#F7F3EC !important;
  border:1px solid var(--sea-700) !important; padding:12px 22px !important; font-weight:500 !important;
  transition:background .2s ease, transform .2s ease;
}
.stFormSubmitButton button:hover{ background:var(--sea-500) !important; border-color:var(--sea-500) !important; transform:translateY(-1px); }

/* 上传 / 折叠面板 / 滑块的容器皮肤 */
[data-testid="stFileUploaderDropzone"]{
  border:1px dashed rgba(10,33,54,.22); border-radius:16px; background:rgba(255,255,255,.62);
}
[data-testid="stExpander"]{
  border:1px solid var(--line); border-radius:14px; background:#fff;
  box-shadow:0 1px 2px rgba(10,33,54,.03);
}
[data-testid="stExpander"] summary{ font-size:13px; color:var(--ink); }

/* 引用芯片 */
.cite-chip{
  display:inline-block; margin:4px 6px 0 0; padding:3px 11px; border-radius:999px;
  font-size:11.5px; color:var(--sea-700); background:var(--cyan-100);
  border:1px solid rgba(27,85,120,.16); letter-spacing:.01em;
}
.meta-line{ font-size:11.5px; color:var(--muted); letter-spacing:.02em; margin-top:8px; }
.rule{ height:1px; background:var(--line); margin:0; border:0; }
.foot-note{
  max-width:1120px; margin:0 auto; padding:34px 24px 60px;
  font-size:11px; letter-spacing:.14em; color:#8ba0ae; text-transform:uppercase;
}

/* ---------- 2.4 响应式：移动端 ---------- */
@media (max-width: 900px){
  .nav{ padding:20px 22px; }
  .nav-links{ display:none; }                 /* 小屏收起中间导航 */
  .hero-inner{ padding:120px 22px 32px; }
  .cta-row .btn{ width:100%; justify-content:center; }
  .hero-foot{ flex-direction:column; align-items:flex-start; gap:10px; }
  .sec-head{ padding:56px 0 18px; }
}
@media (prefers-reduced-motion: reduce){
  .scroll .arrow{ animation:none; }
  html{ scroll-behavior:auto; }
}
</style>
"""

st.markdown(CSS.replace("__FONT__", FONT_HREF), unsafe_allow_html=True)


# ===========================================================================
# 3) Hero：纯 HTML（导航 + 主视觉），不使用任何前端框架
#    说明：Hero 内的按钮是锚点链接，点击平滑滚动到下方对应区块；
#          如果你希望它们触发 Streamlit 逻辑，把 <a> 换成 st.button（见文末注释）。
# ===========================================================================

HERO_HTML = """
<section class="hero" style="background-image:url('__IMG__')">
  <div class="hero-veil"></div>

  <!-- 1. 悬浮导航 -->
  <header class="nav">
    <a class="brand" href="#top">
      <!-- 简约波浪 logo（内联 SVG，换成你的 logo 只需替换这段 path） -->
      <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M2 20c4-4 8-4 12 0s8 4 12 0" stroke="#DDEBEF" stroke-width="1.6"
              stroke-linecap="round" opacity=".95"/>
        <path d="M2 25c4-4 8-4 12 0s8 4 12 0" stroke="#9CC3CE" stroke-width="1.6"
              stroke-linecap="round" opacity=".7"/>
        <path d="M2 15c4-4 8-4 12 0s8 4 12 0" stroke="#DDEBEF" stroke-width="1.6"
              stroke-linecap="round" opacity=".45"/>
      </svg>
      <span class="brand-txt">__BRAND__</span>
    </a>

    <nav class="nav-links">
      <a href="#docs">知识库管理</a>
      <a href="#chat">对话历史</a>
      <a href="#config">系统配置</a>
    </nav>

    <a class="nav-cta" href="#chat">快速问答</a>
  </header>

  <!-- 2. 主视觉 -->
  <div class="hero-inner">
    <div class="hero-top">
      <p class="kicker">Knowledge flows like the ocean</p>
      <h1 class="hero-title">Closer to your<br/>enterprise knowledge</h1>
      <p class="lede">
        A moment of stillness. Reliable answers from your internal documents.
      </p>

      <div class="cta-row">
        <a class="btn btn-solid" href="#chat">Start Chat · 开始问答</a>
        <a class="btn btn-ghost" href="#docs">
          <span class="play">▶</span> Manage Docs · 文档库
        </a>
      </div>
    </div>

    <div class="hero-foot">
      <span class="foot-label">Enterprise Knowledge · RAG Agent</span>
      <a class="scroll" href="#chat">Scroll to chat <span class="arrow">↓</span></a>
    </div>
  </div>
</section>
"""

st.markdown(
    HERO_HTML.replace("__IMG__", HERO_IMAGE).replace("__BRAND__", BRAND),
    unsafe_allow_html=True,
)

# 锚点：#chat / #docs / #config 的落点
st.markdown('<div id="chat"></div>', unsafe_allow_html=True)


# ===========================================================================
# 4) RAG 接入层（三个接入点，均有降级实现，保证页面永远能跑）
# ===========================================================================


def _load_agent():
    """接入点 A：把问题交给现有的 LangGraph Agent。

    返回 (run_graph, User) 或 (None, None)。
    换成你自己的模型服务时，只要保持 run_graph(question, user, thread_id) 这个签名。
    """
    try:
        from src.agent.graph import run as run_graph
        from src.schema import User

        return run_graph, User
    except Exception:  # noqa: BLE001 - 缺依赖/缺索引时走演示模式
        return None, None


def _apply_runtime_params(top_k: int, threshold: float, use_threshold: bool) -> None:
    """接入点 B：把界面上的检索参数写回运行时配置。

    注意：get_settings() 是 lru_cache 单例，这里直接改内存字段，属于"演示级"做法；
    生产环境应该给 graph.run 增加显式参数，避免隐式全局状态。
    """
    try:
        from src.config import get_settings

        settings = get_settings()
        settings.retrieval.top_k = int(top_k)
        settings.retrieval.relevance_threshold = float(threshold if use_threshold else 0.0)
    except Exception:  # noqa: BLE001
        pass


def ask_agent(
    question: str,
    *,
    top_k: int,
    threshold: float,
    use_threshold: bool,
    department: str,
    level: str,
    thread_id: str,
) -> dict:
    """统一问答出口：优先真实链路，失败则返回演示答案（并标注 demo）。"""
    _apply_runtime_params(top_k, threshold, use_threshold)

    run_graph, User = _load_agent()
    if run_graph is None:
        return {
            "answer": (
                "（演示模式）还没有接入检索索引。请先运行 "
                "`python scripts/build_index.py --hash-embed`，"
                "或把 `_load_agent()` 换成你自己的模型服务。\n\n"
                "关于你问的「" + question + "」：这里会返回带引用的答案。"
            ),
            "citations": [],
            "evidence": [],
            "reason": "demo",
            "latency_ms": 0,
        }

    try:
        state = run_graph(question, User("web", department, level), thread_id=thread_id)
    except Exception as exc:  # noqa: BLE001 - 任何后端异常都要变成可读提示
        return {
            "answer": f"回答失败：{exc}",
            "citations": [],
            "evidence": [],
            "reason": "error",
            "latency_ms": 0,
        }

    return {
        "answer": state.get("answer") or "（无输出）",
        "citations": state.get("citations") or [],
        "evidence": state.get("evidence") or [],
        "reason": state.get("reason") or "ok",
        "latency_ms": state.get("latency_ms") or 0,
    }


def save_uploads(files, department: str, level: str) -> list[str]:
    """接入点 C：把上传文件落到 data/raw_docs/{部门}/{密级}/ 下（目录即标签）。"""
    try:
        from src.config import get_settings

        base = Path(get_settings().paths.raw_docs) / department / level
    except Exception:  # noqa: BLE001
        base = PROJECT_ROOT / "data" / "raw_docs" / department / level

    base.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        target = base / file.name
        target.write_bytes(file.getbuffer())
        saved.append(str(target.relative_to(PROJECT_ROOT)))
    return saved


def rebuild_index() -> tuple[bool, str]:
    """上传后重建索引；失败也只返回提示，不抛异常到界面。"""
    try:
        from src.ingest.pipeline import run_ingest

        report = run_ingest(force_hash=True)
        return True, (
            f"索引重建完成：{report.docs} 篇文档 / {report.chunks} 个知识块"
            f"（跳过 {report.docs_skipped}）"
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"索引重建失败：{exc}"


def _short_source(path: str | None) -> str:
    if not path:
        return "?"
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else path


# ===========================================================================
# 5) 页面区块
# ===========================================================================

# 会话状态：消息列表 + 多轮 thread id
st.session_state.setdefault("messages", [])
st.session_state.setdefault("thread_id", uuid.uuid4().hex[:12])

# ---- 区块标题 ----
st.markdown(
    """
    <div class="sec-head sec">
      <p class="sec-label">Conversation</p>
      <h2 class="sec-title">Ask the ocean of documents</h2>
      <p class="sec-sub">
        问题只会命中你有权访问的资料；回答附带引用来源，没有依据时会明确拒答。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2.05, 1], gap="large")

# ---------------------------------------------------------------- 左：对话
with left:
    with st.container(key="chat_section"):
        if not st.session_state.messages:
            st.markdown(
                '<p class="meta-line">试试问：一线城市差旅住宿标准是多少？</p>',
                unsafe_allow_html=True,
            )

        for message in st.session_state.messages:
            with st.chat_message(
                "user" if message["role"] == "user" else "assistant",
                avatar="🧭" if message["role"] == "user" else "🌊",
            ):
                st.markdown(message["content"])
                if message.get("citations"):
                    chips = "".join(
                        f'<span class="cite-chip">{i} · {_short_source(src)}</span>'
                        for i, src in enumerate(message["citations"], start=1)
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                if message.get("meta"):
                    st.markdown(
                        f'<p class="meta-line">{message["meta"]}</p>',
                        unsafe_allow_html=True,
                    )

        # 输入区：胶囊输入框 + 提交按钮（同一行）
        with st.form("composer", clear_on_submit=True):
            col_input, col_send = st.columns([5, 1])
            question = col_input.text_input(
                "问题",
                placeholder="向知识库提问…",
                label_visibility="collapsed",
            )
            sent = col_send.form_submit_button("发送", use_container_width=True)

        if sent and question.strip():
            text = question.strip()
            st.session_state.messages.append({"role": "user", "content": text})

            with st.spinner("检索 · 重排 · 生成…"):
                result = ask_agent(
                    text,
                    top_k=st.session_state.get("top_k", 20),
                    threshold=st.session_state.get("threshold", 0.45),
                    use_threshold=st.session_state.get("use_threshold", True),
                    department=st.session_state.get("department", "研发"),
                    level=st.session_state.get("level", "机密"),
                    thread_id=st.session_state["thread_id"],
                )

            citations = [
                item.get("source", "")
                for item in result.get("evidence", [])
                if item.get("chunk_id") in set(result.get("citations") or [])
            ]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": citations,
                    "meta": f"判定 {result['reason']} · 用时 {result['latency_ms']:.0f} ms",
                }
            )
            st.rerun()

        if st.session_state.messages:
            if st.button("清空对话", key="clear_chat"):
                st.session_state.messages = []
                st.session_state.thread_id = uuid.uuid4().hex[:12]
                st.rerun()

# ------------------------------------------------- 右：文档入库 + 参数配置
with right:
    st.markdown('<div id="docs"></div>', unsafe_allow_html=True)
    with st.container(key="docs_section"):
        st.markdown(
            '<p class="sec-label" style="margin-bottom:8px">Knowledge base</p>',
            unsafe_allow_html=True,
        )

        department = st.selectbox(
            "文档所属部门",
            ["研发", "人力", "财务", "公共"],
            index=0,
            help="目录即标签：文件会保存到 data/raw_docs/{部门}/{密级}/",
        )
        level = st.selectbox("文档密级", ["普通", "机密"], index=0)
        st.session_state["department"] = department
        st.session_state["level"] = level

        uploaded = st.file_uploader(
            "上传文档（PDF / Markdown / TXT / DOCX）",
            type=["pdf", "md", "markdown", "txt", "docx"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("入库并重建索引", use_container_width=True):
            saved = save_uploads(uploaded, department, level)
            ok, message = rebuild_index()
            st.success(f"已保存 {len(saved)} 个文件：" + "、".join(saved))
            (st.success if ok else st.warning)(message)

        st.markdown("<hr class='rule'/>", unsafe_allow_html=True)

        st.markdown('<div id="config"></div>', unsafe_allow_html=True)
        with st.expander("检索参数 / 系统配置", expanded=False):
            top_k = st.slider("top_k · 召回条数", 5, 30, 20, 1)
            use_threshold = st.toggle("启用相似度阈值", value=True)
            threshold = st.slider(
                "threshold · 相关性阈值",
                0.0, 1.0, 0.45, 0.01,
                disabled=not use_threshold,
            )
            st.session_state["top_k"] = top_k
            st.session_state["use_threshold"] = use_threshold
            st.session_state["threshold"] = threshold
            st.caption("阈值越高越保守：拒答更多，幻觉更少。")

# ---- 页脚 ----
st.markdown(
    f'<div class="foot-note">{BRAND} · ocean edition</div>',
    unsafe_allow_html=True,
)


# ===========================================================================
# 6) 备注：如果你想让 Hero 里的按钮直接触发 Streamlit 逻辑（而不是锚点滚动），
#    把 HERO_HTML 里的 <a class="btn ..."> 换成占位 div，然后用 st.button 覆盖：
#
#        st.markdown('<div class="cta-row"><div class="btn btn-solid">…</div></div>')
#        if st.button("Start Chat", key="hero_start"):
#            st.session_state.messages = []
#
#    但请优先用锚点：landing page 的按钮语义就是"滚动到功能区"，
#    这样也不必为了按钮位置去调 Streamlit 的流式布局。
# ===========================================================================
