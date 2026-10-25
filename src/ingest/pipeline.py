"""编排整条 ingest 流程，供 scripts/build_index.py 调用。

错误策略：单文档失败 fail-soft（记 failures 继续），模型/索引失败 fail-fast。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import Settings, get_settings
from src.ingest.cleaner import clean_document
from src.ingest.embedder import build_embedder
from src.ingest.indexer import build_index
from src.ingest.loader import iter_documents, load_document
from src.ingest.splitter import split_document
from src.schema import Chunk
from src.security.metadata import resolve_doc_meta
from src.utils.exceptions import IndexBuildError, RAGError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestReport:
    docs: int = 0
    docs_skipped: int = 0
    chunks: int = 0
    labels_from_default: int = 0
    elapsed_s: float = 0.0
    failures: list[dict] = field(default_factory=list)


def run_ingest(settings: Settings | None = None, force_hash: bool = False) -> IngestReport:
    settings = settings or get_settings()
    started = time.perf_counter()
    report = IngestReport()
    chunks: list[Chunk] = []

    for path in iter_documents(settings.paths.raw_docs, settings.ingest.supported_extensions):
        try:
            raw = load_document(path, settings)
            meta = resolve_doc_meta(raw.rel_path, settings)
            clean = clean_document(raw, settings)
            doc_chunks = split_document(clean, meta, settings)
        except RAGError as exc:      # 单文档失败不阻断整批
            report.docs_skipped += 1
            report.failures.append({"path": str(path), "error": type(exc).__name__,
                                    "detail": exc.detail})
            logger.warning("跳过 %s: %s", path.name, exc)
            continue

        report.docs += 1
        if meta.labels_source == "default":
            report.labels_from_default += len(doc_chunks)
        chunks.extend(doc_chunks)
        _dump_processed(settings, meta.doc_id, clean)
        logger.info("%s -> %d 块（%s/%s）", path.name, len(doc_chunks),
                    meta.department, meta.sensitivity)

    if not chunks:
        raise IndexBuildError("没有产生任何 chunk", {"docs": report.docs, "skipped": report.docs_skipped})

    embedder = build_embedder(settings, force_hash=force_hash)
    vectors = embedder.encode_documents([c.text for c in chunks])
    build_index(chunks, vectors, settings.paths.index, {
        "version": 1,
        "embedding_model": settings.models.embedding_model if not force_hash else "hash",
        "dim": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "chunk_size": settings.chunking.chunk_size,
        "chunk_overlap": settings.chunking.chunk_overlap,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    report.chunks = len(chunks)
    report.elapsed_s = round(time.perf_counter() - started, 2)
    return report


def _dump_processed(settings: Settings, doc_id: str, clean) -> None:
    out = Path(settings.paths.processed) / f"{doc_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(asdict(b), ensure_ascii=False) + "\n" for b in clean.blocks),
        encoding="utf-8",
    )