# 应用镜像：Streamlit 主应用（默认离线模式，不装 torch）
#
# 构建： docker build -t enterprise-rag-agent .
# 运行： docker run --rm -p 8501:8501 --env-file .env -v "%cd%/data:/app/data" enterprise-rag-agent
#
# 说明：镜像里不包含 torch / sentence-transformers / FlagEmbedding，
#       因此请用 `--hash-embed` 建索引（零下载）；需要真实向量时再自行安装这三者。

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# faiss 运行时依赖 libgomp；curl 用于健康检查
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# 可重建产物目录（生产建议用卷挂载覆盖）
RUN mkdir -p data/index data/processed logs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
