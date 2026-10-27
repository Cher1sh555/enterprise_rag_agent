"""视觉系统：极简白 + 无衬线 + 空间分层 + 克制的科技动效。

设计规则（改之前先读）：
1. **一个强调色**（--accent），其余颜色只表达语义（成功/警告/危险）；
2. **三档字号**：标题 26 / 正文 15 / 元信息 12.5，不再多；
3. **两层空间**：背景网格 + 浮动态光斑（固定层）→ 内容卡片（抬升层）；
4. **动效只做三件事**：标志旋转、光斑漂移、内容入场；其余一律不加；
5. 尊重 `prefers-reduced-motion`：动效可以关。
"""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
:root{
  --accent:#1d4ed8; --accent-soft:#eff6ff; --accent-line:#dbeafe;
  --ink:#0f172a; --body:#1f2937; --muted:#64748b; --faint:#94a3b8;
  --line:#e6e8ee; --surface:#ffffff; --surface-2:#fafbfd;
  --ok:#15803d; --ok-bg:#f0fdf4; --ok-line:#bbf7d0;
  --warn:#b45309; --warn-bg:#fffbeb; --warn-line:#fde68a;
  --danger:#b91c1c; --danger-bg:#fef2f2; --danger-line:#fecaca;
  --radius:12px; --shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.05);
}

/* ---------- 基础排版：极简白、无衬线、紧字距 ---------- */
/* 字体：不要用 [class*="st-"] 这类通配属性选择器——
   Streamlit 的每个元素都带 st-emotion-cache-xxx 类，通配会连"图标字体"一起覆盖，
   结果 Material 图标被渲染成它的连字名（keyboard_arrow_right），压在标题上像重影。*/
body, .stApp,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
[data-testid="stChatMessage"], [data-testid="stExpander"] summary,
[data-testid="stDataFrame"], [data-testid="stMetricLabel"],
h1, h2, h3, h4, h5, h6, p, li, label, summary, th, td,
input, textarea, button, select {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Hiragino Sans GB", "Microsoft YaHei", system-ui, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

/* 把图标字体救回来：只匹配 material 系类名，不用通配 */
[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
span[class*="material-symbols"],
span[class*="material-icons"],
.material-symbols-rounded, .material-symbols-outlined, .material-icons {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined",
               "Material Icons" !important;
  font-feature-settings: "liga" !important;
  -webkit-font-feature-settings: "liga" !important;
  letter-spacing: normal !important;
}

.stApp { background:#fff; color:var(--ink); }
header[data-testid="stHeader"] { background:transparent; height:0; }
#MainMenu, footer { visibility:hidden; }
.block-container { max-width:1120px; padding-top:1.4rem; padding-bottom:140px; }
h1,h2,h3,h4 { letter-spacing:-.02em; color:var(--ink); font-weight:680; }
p, li { color:var(--body); }

/* ---------- 背景层：网格 + 漂移光斑（固定，不随滚动） ---------- */
.stApp::before{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    linear-gradient(to right, rgba(15,23,42,.045) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(15,23,42,.045) 1px, transparent 1px);
  background-size:56px 56px, 56px 56px;
  mask-image: radial-gradient(circle at 50% 12%, #000 0%, transparent 72%);
  animation: gridDrift 44s linear infinite;
}
@keyframes gridDrift{ from{background-position:0 0,0 0} to{background-position:56px 56px,56px 56px} }

.bg-orbs{ position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
.bg-orbs i{ position:absolute; border-radius:50%; filter:blur(64px); opacity:.8; }
.bg-orbs .o1{ width:560px; height:560px; left:-150px; top:-180px;
  background:radial-gradient(circle,#bfdbfe 0%,transparent 70%); animation:float1 26s ease-in-out infinite; }
.bg-orbs .o2{ width:500px; height:500px; right:-140px; top:80px;
  background:radial-gradient(circle,#c7d2fe 0%,transparent 70%); animation:float2 32s ease-in-out infinite; }
.bg-orbs .o3{ width:420px; height:420px; left:38%; bottom:-160px;
  background:radial-gradient(circle,#bae6fd 0%,transparent 70%); animation:float3 38s ease-in-out infinite; }
@keyframes float1{ 0%,100%{transform:translate3d(0,0,0)} 50%{transform:translate3d(60px,40px,0)} }
@keyframes float2{ 0%,100%{transform:translate3d(0,0,0)} 50%{transform:translate3d(-50px,50px,0)} }
@keyframes float3{ 0%,100%{transform:translate3d(0,0,0)} 50%{transform:translate3d(-40px,-30px,0)} }

/* 内容抬到背景之上：stMain 是 1.66 的主容器，.main / stAppViewContainer 是旧版兜底 */
[data-testid="stMain"], section.main,
[data-testid="stAppViewContainer"] > .main{ position:relative; z-index:2; }

/* ---------- 顶栏：旋转标志 + 状态 ---------- */
.topbar{ display:flex; align-items:center; justify-content:space-between;
  padding:10px 0 18px; margin-bottom:6px; border-bottom:1px solid var(--line); }
.logo{ display:flex; align-items:center; gap:11px; }
.ring{ width:30px; height:30px; border-radius:50%; flex:0 0 auto;
  background:conic-gradient(from 0deg, var(--accent), #93c5fd, var(--accent));
  animation:spin 9s linear infinite; position:relative;
  box-shadow:0 0 0 4px rgba(29,78,216,.06); }
.ring::after{ content:""; position:absolute; inset:5px; border-radius:50%; background:#fff; }
.ring::before{ content:""; position:absolute; inset:-7px; border-radius:50%;
  background:radial-gradient(circle, rgba(29,78,216,.22), transparent 68%);
  animation:pulse 3.6s ease-in-out infinite; z-index:-1; }
@keyframes pulse{ 0%,100%{opacity:.45; transform:scale(.92)} 50%{opacity:.9; transform:scale(1.06)} }
@keyframes spin{ to{ transform:rotate(360deg); } }
.wm{ font-size:15.5px; font-weight:700; letter-spacing:-.01em; }
.wm em{ font-style:normal; color:var(--accent); margin-left:6px; font-weight:700; }
.topbar .status{ font-size:12.5px; color:var(--muted); display:flex; align-items:center; gap:8px; }
.dot{ width:7px; height:7px; border-radius:50%; background:var(--ok);
  box-shadow:0 0 0 3px rgba(21,128,61,.12); }

/* ---------- 卡片：抬升 + 悬停 ---------- */
.grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:18px 0 6px; }
@media (max-width:900px){ .grid{ grid-template-columns:repeat(2,1fr); } }
.card{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
  padding:16px 16px 14px; box-shadow:var(--shadow);
  transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
.card:hover{ transform:translateY(-3px); border-color:var(--accent-line);
  box-shadow:0 2px 4px rgba(15,23,42,.05), 0 16px 34px rgba(29,78,216,.10); }
.card .idx{ font-size:11.5px; font-weight:700; color:var(--accent); letter-spacing:.08em; }
.card h3{ margin:8px 0 6px; font-size:15.5px; }
.card p{ margin:0; font-size:13px; color:var(--muted); line-height:1.6; }

/* ---------- 指标条 ---------- */
.stats{ display:flex; flex-wrap:wrap; gap:10px; margin:16px 0 4px; }
.stat{ flex:1 1 150px; background:var(--surface-2); border:1px solid var(--line);
  border-radius:10px; padding:12px 14px; }
.stat b{ display:block; font-size:22px; font-weight:700; color:var(--ink);
  font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
.stat span{ font-size:12px; color:var(--muted); }

/* ---------- 标志轮播（marquee） ---------- */
.marquee{ position:relative; overflow:hidden; margin:22px 0 4px;
  mask-image:linear-gradient(90deg,transparent,#000 8%,#000 92%,transparent); }
.marquee-track{ display:flex; gap:34px; width:max-content;
  animation:scrollX 26s linear infinite; }
.marquee:hover .marquee-track{ animation-play-state:paused; }
.mq{ font-size:13px; color:var(--faint); letter-spacing:.02em; white-space:nowrap;
  display:flex; align-items:center; gap:8px; }
.mq i{ width:5px; height:5px; border-radius:50%; background:var(--accent-line); }
@keyframes scrollX{ from{transform:translateX(0)} to{transform:translateX(-50%)} }

/* ---------- 状态药丸 / 指标 / 芯片 ---------- */
.pill{ display:inline-flex; align-items:center; gap:6px; padding:3px 11px;
  border-radius:999px; font-size:12.5px; font-weight:640; border:1px solid transparent; }
.pill-ok{ color:var(--ok); background:var(--ok-bg); border-color:var(--ok-line); }
.pill-warn{ color:var(--warn); background:var(--warn-bg); border-color:var(--warn-line); }
.pill-danger{ color:var(--danger); background:var(--danger-bg); border-color:var(--danger-line); }
.pill-muted{ color:#475569; background:#f4f5f7; border-color:var(--line); }

.metric-row{ display:flex; gap:22px; flex-wrap:wrap; align-items:center; margin:10px 0 2px; }
.metric{ font-variant-numeric:tabular-nums; }
.metric .k{ font-size:11.5px; color:var(--faint); letter-spacing:.03em; display:block; }
.metric .v{ font-size:17px; font-weight:660; color:var(--ink); }

.chip{ display:inline-block; margin:3px 6px 3px 0; padding:2px 10px; border-radius:999px;
  font-size:12px; color:var(--accent); background:var(--accent-soft);
  border:1px solid var(--accent-line); font-variant-numeric:tabular-nums; }

/* ---------- 证据卡片 ---------- */
.evidence{ border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:10px; padding:11px 14px; margin-bottom:10px; background:#fff;
  transition:box-shadow .18s ease, transform .18s ease; }
.evidence:hover{ transform:translateY(-1px); box-shadow:var(--shadow); }
.evidence.secret{ border-left-color:var(--danger); }
.evidence .meta{ font-size:12px; color:var(--muted); margin-bottom:6px;
  font-variant-numeric:tabular-nums; }
.evidence .body{ font-size:14px; line-height:1.75; color:var(--body); }
mark{ background:#fef08a; padding:0 2px; border-radius:3px; }

/* ---------- 对话气泡 ---------- */
[data-testid="stChatMessage"]{ background:#fff; border:1px solid var(--line);
  border-radius:var(--radius); box-shadow:var(--shadow); padding:6px 12px; }

/* ---------- 侧栏 ---------- */
section[data-testid="stSidebar"]{ background:#fbfbfd; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container{ padding-top:1.2rem; }

/* ---------- 按钮 ---------- */
.stButton > button{ border-radius:10px; border:1px solid var(--line); font-weight:600;
  transition:transform .12s ease, box-shadow .18s ease, border-color .18s ease; }
.stButton > button:hover{ transform:translateY(-1px); border-color:var(--accent-line);
  box-shadow:0 8px 20px rgba(29,78,216,.12); }
.stButton > button[kind="primary"]{ background:var(--accent); border-color:var(--accent); }

/* ---------- 底部导航（固定层，随侧栏移动） ----------
   问题：底栏用 left:50% 是"视口中心"，不是"内容区中心"；
        侧栏展开后内容区整体右移，底栏没跟着走，左半边就被压在侧栏下面。
   解决：用 :has() 检测侧栏展开，把 left 改成 calc(50% + 半个侧栏宽)，
        并同步收窄宽度，左右各留 24px 安全间距。
   --sidebar-half 默认 200px（官方默认侧栏 336px 的一半是 168px，
   这里故意取大一点：宁可稍微偏右，也绝不被遮住；把侧栏拖宽时也还安全）。 */
:root{ --sidebar-half: 200px; }

.st-key-bottomnav{
  position:fixed; left:50%; transform:translateX(-50%);
  bottom:16px; width:min(620px, calc(100vw - 48px)); z-index:1000;
  background:rgba(255,255,255,.88); backdrop-filter:blur(14px);
  border:1px solid var(--line); border-radius:999px; padding:7px 9px;
  box-shadow:0 10px 34px rgba(15,23,42,.10);
  transition:left .18s ease, width .18s ease;
}

/* 侧栏展开：中心右移，宽度同时收窄，左右都留出安全间距。
   Streamlit 1.66 把展开状态写在「收起按钮」上
   （data-testid="stSidebarCollapseButton" + aria-expanded + aria-controls="stSidebar"），
   所以这里以该按钮为准；再补一条旧版 sidebar 元素上的 aria-expanded 作为兜底。 */
body:has([data-testid="stSidebarCollapseButton"][aria-expanded="true"]) .st-key-bottomnav,
body:has([data-testid="stSidebar"][aria-expanded="true"]) .st-key-bottomnav{
  left: calc(50% + var(--sidebar-half));
  width: min(620px, calc(100vw - var(--sidebar-half) * 2 - 48px));
}

/* 窄屏：侧栏变成抽屉覆盖，底栏保持视口居中 */
@media (max-width: 900px){
  body:has([data-testid="stSidebarCollapseButton"][aria-expanded="true"]) .st-key-bottomnav,
  body:has([data-testid="stSidebar"][aria-expanded="true"]) .st-key-bottomnav{
    left: 50%;
    width: min(620px, calc(100vw - 32px));
  }
}

.st-key-bottomnav [data-testid="stHorizontalBlock"]{ gap:6px; }
.st-key-bottomnav .stButton > button{ width:100%; border-radius:999px; border:1px solid transparent;
  background:transparent; color:var(--muted); box-shadow:none; font-weight:600; }
.st-key-bottomnav .stButton > button:hover{ background:#f4f6fb; color:var(--ink);
  transform:none; box-shadow:none; }
.st-key-bottomnav .stButton > button[kind="primary"]{ background:var(--accent); color:#fff; }

/* ---------- 折叠面板：显式 flex，图标与标题各占一格，杜绝叠加 ---------- */
[data-testid="stExpander"] summary{
  display:flex !important; align-items:center; gap:10px;
  padding:10px 14px !important; line-height:1.5; list-style:none;
}
[data-testid="stExpander"] summary::-webkit-details-marker{ display:none; }
[data-testid="stExpander"] summary p{ margin:0 !important; }
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"]{
  flex:0 0 auto; display:inline-flex; align-items:center; justify-content:center;
  width:16px; height:16px; overflow:hidden;
}

/* ---------- 入场动画：只给 hero 与卡片 ---------- */
.fade-up{ animation:fadeUp .6s cubic-bezier(.22,.61,.36,1) both; }
.fade-up.d1{ animation-delay:.06s } .fade-up.d2{ animation-delay:.12s }
.fade-up.d3{ animation-delay:.18s } .fade-up.d4{ animation-delay:.24s }
@keyframes fadeUp{ from{opacity:0; transform:translateY(10px)} to{opacity:1; transform:none} }

@media (prefers-reduced-motion: reduce){
  .ring,.bg-orbs i,.marquee-track,.stApp::before{ animation:none !important; }
  .fade-up{ animation:none !important; }
}
</style>
"""

ORBS = (
    '<div class="bg-orbs"><i class="o1"></i><i class="o2"></i><i class="o3"></i></div>'
)

# 用户显式勾选"界面动效"时，覆盖系统级 prefers-reduced-motion（自己的选择优先于系统默认）
FORCE_MOTION = """
<style>
@media (prefers-reduced-motion: reduce){
  .ring{ animation:spin 9s linear infinite !important; }
  .ring::before{ animation:pulse 3.6s ease-in-out infinite !important; }
  .bg-orbs i{ animation:float1 26s ease-in-out infinite !important; }
  .marquee-track{ animation:scrollX 26s linear infinite !important; }
  .stApp::before{ animation:gridDrift 44s linear infinite !important; }
  .fade-up{ animation:fadeUp .5s cubic-bezier(.22,.61,.36,1) both !important; }
}
</style>
"""

# 用户关闭动效：无论系统设置如何，全部停掉
NO_MOTION = """
<style>
.ring, .ring::before, .bg-orbs i, .marquee-track, .stApp::before, .fade-up{
  animation:none !important;
}
</style>
"""


def inject(motion: bool = True) -> None:
    """注入样式与背景层。

    Args:
        motion: 是否允许动效。True 时即使系统开了"减少动态效果"也照常播放
                （用户在页面里显式选择优先）；False 时全部静止。
    """
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(ORBS, unsafe_allow_html=True)
    st.markdown(FORCE_MOTION if motion else NO_MOTION, unsafe_allow_html=True)

