# data 目录说明

- `raw_docs/`    原始文档：PDF / Word / 扫描件 / 带表格文档（建议 10~20 份模拟企业文档）
- `processed/`   清洗后的中间文本（可复查，不入库）
- `index/`       FAISS / BM25 索引落盘（不入库）
- `evaluation/`  评测集：questions.jsonl + 标注答案（正常 / 边界 / 越权三类样本）
