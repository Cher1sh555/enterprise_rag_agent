"""索引构建与加载：chunks.jsonl / tokens.jsonl / faiss.index / meta.json。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.schema import Chunk
from src.utils.exceptions import IndexBuildError, IndexNotFoundError
from src.utils.logger import get_logger
from src.utils.text import tokenize

logger = get_logger(__name__)

_FILES = ("chunks.jsonl", "tokens.jsonl", "faiss.index", "meta.json")


@dataclass
class IndexBundle:
    faiss: object
    chunks: list[Chunk]
    tokens: list[list[str]]
    bm25: BM25Okapi
    meta: dict


def build_index(
    chunks: list[Chunk], vectors: np.ndarray, out_dir: Path, meta: dict
) -> None:
    tokens = [tokenize(chunk.text) for chunk in chunks]
    _assert_aligned(chunks, vectors, tokens)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    faiss.write_index(index, str(out / "faiss.index"))

    (out / "chunks.jsonl").write_text(
        "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in chunks),
        encoding="utf-8",
    )
    (out / "tokens.jsonl").write_text(
        "".join(
            json.dumps({"chunk_id": c.chunk_id, "tokens": t}, ensure_ascii=False) + "\n"
            for c, t in zip(chunks, tokens)
        ),
        encoding="utf-8",
    )
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(out_dir: Path) -> IndexBundle:
    out = Path(out_dir)
    missing = [name for name in _FILES if not (out / name).is_file()]
    if missing:
        raise IndexNotFoundError(
            "索引文件缺失", {"path": str(out), "missing": missing, "hint": "先跑 scripts/build_index.py"}
        )

    chunks = [
        Chunk(**json.loads(line))
        for line in (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    tokens = [
        json.loads(line)["tokens"]
        for line in (out / "tokens.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    index = faiss.read_index(str(out / "faiss.index"))
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))

    _assert_aligned(chunks, np.zeros((len(chunks), 1)), tokens)
    if index.ntotal != len(chunks):
        raise IndexBuildError(
            "索引与 chunk 数量不一致", {"ntotal": index.ntotal, "chunks": len(chunks)}
        )
    logger.info("索引加载完成：%d 块，dim=%d，模型=%s",
                len(chunks), index.d, meta.get("embedding_model"))
    return IndexBundle(faiss=index, chunks=chunks, tokens=tokens,
                       bm25=BM25Okapi(tokens), meta=meta)


def _assert_aligned(chunks, vectors, tokens) -> None:
    if not (len(chunks) == len(vectors) == len(tokens)):
        raise IndexBuildError(
            "chunks / vectors / tokens 数量不一致",
            {"chunks": len(chunks), "vectors": len(vectors), "tokens": len(tokens)},
        )