# 企业知识库 RAG 智能问答 Agent

面向企业内部文档的智能问答系统：多格式文档解析 → 清洗 → 分块 → 混合检索 → 权限过滤 → 重排 → LLM 生成，带权限隔离与幻觉抑制。

## 技术栈

Python · LangChain · LangGraph · FAISS · BM25 · Streamlit · Docker

## 核心能力

- 多类型文档解析：PDF / Word / 扫描件(OCR)，清洗页眉页脚冗余文本
- 混合检索：向量检索 + BM25 关键词检索，搭配 Rerank 重排
- 权限隔离：向量块绑定部门 / 密级标签，检索阶段自动过滤无权限片段
- 答案溯源：回答附带原文段落来源，方便核验
- 幻觉防护：检索相关性低于阈值时拒绝编造
- Agent 工具能力：自动判断知识库检索或调用自定义工具
- 基础审计日志：记录提问、检索片段、用户 ID

## 目录结构

```
enterprise-rag-agent/
├── configs/          # 全局配置（settings.yaml / prompts.yaml）
├── data/
│   ├── raw_docs/     # 原始文档
│   ├── processed/    # 清洗后中间文本
│   ├── index/        # FAISS / BM25 索引落盘
│   └── evaluation/   # 评测集
├── src/
│   ├── ingest/       # 离线索引管线
│   ├── retrieval/    # 双路召回 / 重排 / 阈值
│   ├── security/     # 元数据 / 权限 / 审计
│   ├── agent/        # LangGraph 状态图
│   ├── llm/          # 大模型调用工程化
│   ├── ui/           # Streamlit 前端
│   └── utils/        # 工具
├── scripts/          # 建索引 / 评测 / 审计导出
├── tests/            # 单元测试
└── logs/             # 运行与审计日志
```

## 快速开始（待补充）

1. 安装依赖：`pip install -r requirements.txt`
2. 复制 `.env.example` 为 `.env` 并填写 API Key
3. 配置 `configs/settings.yaml`
4. 构建索引：`python scripts/build_index.py`
5. 启动：`streamlit run src/main.py`
