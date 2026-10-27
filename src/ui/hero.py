"""首页 Hero：粒子 / 连线画布 + 标题层（文字背后有交互动效）。

为什么用 iframe 组件而不是 st.markdown：
Streamlit 会剥离 markdown 里的 <script>，canvas 必须在独立文档里跑。
代价是动效只覆盖 hero 区域 —— 而这恰好符合"文字背后有动效"的需求，
也不会因为全屏 canvas 拖慢问答页。

交互：鼠标在画布上移动时，粒子被轻微推开，连线随距离淡出。
无障碍：`prefers-reduced-motion` 下只画一帧静态图。
"""

from __future__ import annotations

import streamlit.components.v1 as components

_TEMPLATE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<style>
  *{ box-sizing:border-box; }
  html,body{ margin:0; height:100%; overflow:hidden; background:#fff;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif; }
  #stage{ position:relative; width:100%; height:100%; }
  canvas{ position:absolute; inset:0; width:100%; height:100%; display:block; }
  .veil{ position:absolute; inset:0; pointer-events:none;
    background:radial-gradient(120% 90% at 12% 0%, rgba(219,234,254,.55) 0%, rgba(255,255,255,0) 60%); }
  .copy{ position:absolute; inset:0; display:flex; flex-direction:column; justify-content:center;
    padding:0 6%; pointer-events:none; }
  .kicker{ font-size:11.5px; letter-spacing:.18em; color:#1d4ed8; font-weight:700;
    margin-bottom:16px; animation:rise .7s cubic-bezier(.22,.61,.36,1) both; }
  h1{ margin:0; font-size:38px; line-height:1.18; letter-spacing:-.025em; color:#0f172a;
    font-weight:720; max-width:15em; animation:rise .7s .06s cubic-bezier(.22,.61,.36,1) both; }
  h1 em{ font-style:normal; color:#1d4ed8; }
  p.sub{ margin:14px 0 0; font-size:14.5px; line-height:1.7; color:#475569; max-width:34em;
    animation:rise .7s .12s cubic-bezier(.22,.61,.36,1) both; }
  .badges{ margin-top:22px; display:flex; gap:8px; flex-wrap:wrap;
    animation:rise .7s .18s cubic-bezier(.22,.61,.36,1) both; }
  .badge{ font-size:12px; color:#334155; background:rgba(255,255,255,.82);
    border:1px solid #e6e8ee; border-radius:999px; padding:5px 12px;
    box-shadow:0 1px 2px rgba(15,23,42,.04); backdrop-filter:blur(6px); }
  @keyframes rise{ from{opacity:0; transform:translateY(12px)} to{opacity:1; transform:none} }
  @media (prefers-reduced-motion: reduce){ .kicker,h1,p.sub,.badges{ animation:none; } }
</style></head>
<body>
<div id="stage">
  <canvas id="cv"></canvas>
  <div class="veil"></div>
  <div class="copy">
    <div class="kicker">ENTERPRISE KNOWLEDGE · RAG STUDIO</div>
    <h1>__TITLE__</h1>
    <p class="sub">__SUB__</p>
    <div class="badges">__BADGES__</div>
  </div>
</div>
<script>
(function(){
  const cv = document.getElementById('cv');
  const ctx = cv.getContext('2d');
  const reduce = __REDUCE__;
  let w = 0, h = 0, dpr = 1, pts = [];
  const mouse = { x: -9999, y: -9999 };
  const COUNT = 58, LINK = 118, PUSH = 92;

  function resize(){
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = cv.clientWidth; h = cv.clientHeight;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
  }
  function seed(){
    pts = [];
    for (let i = 0; i < COUNT; i++) {
      pts.push({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - .5) * .28, vy: (Math.random() - .5) * .28,
        r: Math.random() * 1.2 + .9
      });
    }
  }
  function frame(){
    ctx.clearRect(0, 0, w, h);

    for (const p of pts) {
      if (!reduce) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
        const dx = p.x - mouse.x, dy = p.y - mouse.y;
        const d = Math.hypot(dx, dy);
        if (d < PUSH && d > .1) { p.x += dx / d * .7; p.y += dy / d * .7; }
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(29,78,216,.38)';
      ctx.fill();
    }

    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const a = pts[i], b = pts[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < LINK) {
          const alpha = (0.16 * (1 - d / LINK)).toFixed(3);
          ctx.strokeStyle = 'rgba(29,78,216,' + alpha + ')';
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      }
    }
    if (!reduce) requestAnimationFrame(frame);
  }

  window.addEventListener('mousemove', function(e){
    const r = cv.getBoundingClientRect();
    mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top;
  });
  window.addEventListener('mouseleave', function(){ mouse.x = -9999; mouse.y = -9999; });
  window.addEventListener('resize', resize);

  resize();
  frame();
})();
</script>
</body></html>
"""


REDUCED_MOTION_EXPR = "window.matchMedia('(prefers-reduced-motion: reduce)').matches"


def render(
    title_html: str,
    subtitle: str,
    badges: list[str],
    height: int = 384,
    force_motion: bool = False,
) -> None:
    """渲染英雄区。

    Args:
        force_motion: True 时忽略系统"减少动态效果"设置，强制播放粒子动画。
                      由页面上的"界面动效"开关控制（用户显式选择优先于系统默认）。
    """
    badges_html = "".join(f'<span class="badge">{badge}</span>' for badge in badges)
    html = (
        _TEMPLATE.replace("__TITLE__", title_html)
        .replace("__SUB__", subtitle)
        .replace("__BADGES__", badges_html)
        .replace("__REDUCE__", "false" if force_motion else REDUCED_MOTION_EXPR)
    )
    components.html(html, height=height, scrolling=False)
