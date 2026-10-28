# 企业知识库 RAG 智能问答 Agent

面向企业内部文档的问答系统：**混合检索 → 权限隔离 → 重排定标 → 引用可溯源**。
带 30 条评测集与可复现的指标，而不是只跑通一条 demo。

> **实测（30 条评测集，14 个知识块）**
> Recall@5 **1.000** · MRR **1.000** · 误拒率 **0.200** · 正确拒答率 **1.000** · 越权泄漏率 **0.000** · P95 **10.5 ms**
> 关闭权限过滤做对照，泄漏率升到 **0.800** —— 这条差值就是权限模块存在的理由。

---

## 一、核心能力

| 能力 | 做法 | 可验证证据 |
|---|---|---|
| 混合检索 | 向量（FAISS）+ BM25（jieba）双路召回，RRF 融合 | `tests/test_retrieval.py` 9 条 |
| 权限隔离 | 密级 + 部门**检索前**过滤；目录即标签 `{部门}/{密级}/` | 泄漏率 0.000；`tests/test_permission.py` 9 条 |
| 引用溯源 | 答案逐条标注 `[n]`，映射到"部门/文件 p.x"；**引用后置校验**拦截幻觉引用 | `reason=citation_out_of_scope`；`tests/test_graph.py` 6 条 |
| 拒答机制 | 无依据 / 相关性不足 / 无权限 三种拒答，话术不同、审计可查 | 正确拒答率 1.000 |
| Agent 编排 | LangGraph 五节点：意图路由 / 检索 / 工具 / 生成 / 拒答 | `src/agent/graph.py` |
| LLM 工程 | 令牌桶限流 → LRU 缓存 → 超时重试 → 降级（`degraded` 可见） | `tests/test_llm.py` 11 条 |
| 评测闭环 | 30 条样本 + 6 个指标 + 阈值定标流程 | `data/evaluation/questions.jsonl` |

---

## 二、架构

```
                      ┌──────────────────────────────┐
   documents ───────▶ │  ingest（离线）              │
   data/raw_docs/     │  loader → cleaner → splitter │
   {部门}/{密级}/      │        → embedder → indexer  │
                      └──────────────┬───────────────┘
                                     ▼
                          data/index/  4 个文件
                    chunks.jsonl · tokens.jsonl
                    vectors.npy  · meta.json

   question ──▶ ┌───────────────┐   allowed_ids（权限前置过滤）
                │   retrieve    │◀──────────────┐
                │ vector+BM25   │               │
                │   → RRF       │        ┌──────┴──────┐
                │   → rerank    │        │  permission │
                │   → threshold │        └──────┬──────┘
                └───────┬───────┘               │
                        ▼                       │
              ┌──────────────────┐              │
              │ generate / refuse│──▶ 引用校验 ─┘
              └────────┬─────────┘
                       ▼
                  answer + citations + audit
```

| 层 | 模块 | 职责 |
|---|---|---|
| ① 检索 | `src/ingest/`、`src/retrieval/` | 解析、切分、双路召回、RRF、重排、阈值 |
| ② 编排 | `src/agent/` | 状态图、意图路由、工具、拒答分支 |
| ③ LLM | `src/llm/` | 缓存、限流、重试、降级、prompt 组装 |
| ④ 安全 | `src/security/` | 打标、权限、审计 |
| ⑤ 工程 | `configs/`、`scripts/`、`tests/`、`src/ui/` | 配置、CLI、106 个测试、界面 |
| ⑥ 评测 | `src/evaluation/`、`data/evaluation/` | 指标计算与定标 |

---

## 三、快速开始

```bash
# 1) 环境（Python ≥ 3.10）
python -m venv .venv && .\.venv\Scripts\activate      # Windows
pip install -r requirements.txt                       # 最小运行集，不含 torch（约 300MB）
# 需要真实向量/真 cross-encoder 时再装：pip install torch sentence-transformers FlagEmbedding
# 想以包形式安装（可 import src.*）：pip install -e . --no-deps

# 2) 配置
copy .env.example .env        # 填 LLM_API_KEY；不填也能跑检索与评测

# 3) 建索引（--hash-embed = 完全离线，不下载任何模型）
python scripts/build_index.py --hash-embed
# → {"docs": 9, "docs_skipped": 1, "chunks": 14, "labels_from_default": 0}

# 4) 提问
python scripts/ask.py "一线城市差旅住宿标准是多少？" --user 研发:机密
python scripts/ask.py "计算 (600*2+400)*1.1" --graph        # 走 Agent 图 + 工具
python scripts/ask.py "财务部薪酬带宽多少？" --user 研发:普通  # 越权 → 拒答

# 5) 评测
python scripts/run_eval.py --reranker lexical
```

---

## 四、两个界面入口

```bash
streamlit run src/main.py        # 主应用：首页 / 问答 / 文档 / 审计（底部导航）
streamlit run app_landing.py     # 落地页：Ocean 风格全屏 Hero + 对话区
```

- `src/main.py`：身份切换（部门 × 密级）、对话、文档可见性一览、审计日志。
- `app_landing.py`：对外展示用，粒子动效 Hero + 胶囊输入框 + 参数面板，同样接真实链路。

---

## 五、命令速查

| 目的 | 命令 |
|---|---|
| 建/重建索引 | `python scripts/build_index.py [--hash-embed]` |
| 单次问答（直线） | `python scripts/ask.py "问题" --user 部门:密级` |
| 单次问答（走图） | `python scripts/ask.py "问题" --graph [--thread t1]` |
| 只看召回 | `python scripts/ask.py "问题" --no-llm` |
| 评测 | `python scripts/run_eval.py [--reranker lexical\|flag\|none] [--threshold 0.45]` |
| 阈值定标取分数 | `python scripts/run_eval.py --dump-scores scores.jsonl` |
| 测试 | `python -m pytest -q -m "not slow"` |

---

## 六、评测与阈值定标

**样本**：`data/evaluation/questions.jsonl`，30 条 = normal 10 / boundary 10 / denied 10。

判据不用 `chunk_id`，而用 `source + contains`：

```json
{"id": "q001", "type": "normal", "question": "一线城市差旅住宿标准是多少？",
 "expect": [{"source": "data/raw_docs/研发/普通/差旅制度.md", "contains": "每晚不超过 600 元"}],
 "user": {"department": "研发", "level": "机密"}}
```

> 为什么不用 `chunk_id`：`doc_id = sha1(相对路径)`，语料一挪目录 id 全变，30 条评测集会整批失效。

**指标**：Recall@5 / MRR / 误拒率 / 正确拒答率 / 泄漏率 / P95 延迟（定义见 `src/evaluation/metrics.py`）。

**定标方法**（`relevance_threshold` 现在的 0.45 就是这么来的）：

| 分组 | rerank 分分布 | 结论 |
|---|---|---|
| normal | 0.201 – 0.808（中位 0.554） | 有 2 条被疑问词压低 |
| **boundary** | 0.111 – **0.443** | 阈值 ≥ 0.45 即可 100% 拒答 |

阈值 0.50 → 误拒 0.40；0.45 → 误拒 0.20 且正确拒答仍为 1.00。**换 reranker 必须重新定标。**

---

## 七、权限模型

两级目录约定：`data/raw_docs/{部门}/{密级}/xxx.md`。任一层不在白名单 → **fail-closed**（未知部门 + 最严格密级）。

| 文档 \ 身份 | 研发/机密 | 研发/普通 | 财务/机密 |
|---|---|---|---|
| 研发 / 普通 | ✅ | ✅ | ✅ |
| 研发 / 机密 | ✅ | ❌ | ❌ |
| 财务 / 机密 | ❌ | ❌ | ✅ |
| 公共 / 机密 | ✅ | ❌ | ✅ |

规则：**最低密级全员可读；更高密级仅同部门或"公共"部门可读。**

---

## 八、关键设计取舍

1. **前置过滤而非后置**：越权片段在进入候选集之前就消失，模型没有机会说出来；再叠加"引用后置校验"防自身逻辑出错。
2. **RRF 而非分数加权**：余弦 0~1、BM25 无上界，归一化又依赖当次候选集；RRF 只看排名，免疫量纲。`k=60` 决定"单路第一"与"两路共识"谁占优（有测试钉死）。
3. **打标 fail-closed**：标不上按最严处理，避免"一个 bug = 泄漏一份机密"。
4. **索引存 `.npy` 而不是 `faiss.index`**：FAISS 的 C++ 文件 IO 在 Windows 上走窄字符 `fopen`，路径含中文（`C:\Users\…`）直接 `No such file or directory`；numpy 用宽字符 API，加载时重建索引只要几毫秒。
5. **图只编排、不重写**：节点只调用检索/生成函数，所以 `run_eval` 的基准与 Agent 图不会悄悄分叉。
6. **LexicalReranker 作为离线兜底**：零下载、确定性、可解释，用来把"评测 → 定标"流程完整跑通；它**不是 cross-encoder**，对同义改写无效（`--reranker flag` 可切真模型）。

---

## 九、已知边界

| 项 | 现状 | 原因 |
|---|---|---|
| PDF / 扫描件 / OCR | 未启用 | `paddlepaddle` 无 Python 3.14 wheel（实测），loader 里保留 v1 实现 |
| 真 cross-encoder 重排 | 代码就绪，未做基线对比 | 需下载 ~2GB 权重 |
| 语料规模 | 9 篇 / 14 块 | **块数 < `top_k=20`，Recall@5 偏乐观，不能当真实水平写进简历** |
| 增量索引 | 全量重建 | 14 块重建 0.4s |
| 身份认证 | 命令行参数 / 界面下拉 | 真实系统应接 SSO / JWT |
| 审计 | 本地 JSONL，无轮转/签名 | 生产要只追加存储 + 哈希链 |
| 多轮 | 最近 2 轮拼接 | 未做摘要 |

---

## 十、踩坑记录

1. **FAISS + 中文路径**：`faiss.write_index` 在含中文的用户目录下静默失败，改为 `vectors.npy` 存向量。
2. **Streamlit 通配 CSS 杀死图标字体**：`[class*="st-"]` 会匹配所有 emotion 类名，把 Material 图标渲染成 `keyboard_arrow_right` 字面量，压在标题上像重影。改为点名 `data-testid`。
3. **Windows 下 pytest 临时目录**：系统 Temp 不可写时 `tempfile` 会回退到 cwd，污染工作区；`tests/conftest.py` 显式指定 `.pytest-tmp/` 并加入 gitignore。

---

## 十一、目录结构

```
configs/            settings.yaml（唯一配置真相）· prompts.yaml
data/
  raw_docs/         语料，目录即标签 {部门}/{密级}/
  evaluation/       questions.jsonl（评测资产，入库）
  index/ processed/ 可重建产物（gitignore）
scripts/            build_index · ask · run_eval · export_audit
src/
  schema.py         Page/RawDoc/Block/Chunk/DocMeta/User/ScoredChunk
  config.py         pydantic 校验 + 路径绝对化
  ingest/           loader · cleaner · splitter · embedder · indexer · pipeline
  retrieval/        hybrid（RRF）· reranker · threshold
  agent/            state · nodes · graph · tools（AST 安全计算器）
  llm/              client（重试/降级）· cache（LRU）· ratelimit（令牌桶）
  security/         metadata · permission · audit
  evaluation/       metrics（纯函数）
  ui/               theme · hero · components · app（Streamlit）
tests/              106 个测试函数（含 1 条 slow）
```

---

## 十二、开发与测试

```bash
python -m pytest -q -m "not slow"     # 单元测试（不加载真模型）
python -m pytest -q                    # 含 slow（会下载模型）
```

CI：`.github/workflows/ci.yml` 在 push / PR 时跑同样的命令（只装轻依赖，不装 torch）。

> 测试为什么快：单测用 `HashEmbedder`（字符 bigram 哈希）和假 LLM，只有 `@pytest.mark.slow` 才碰真模型。
