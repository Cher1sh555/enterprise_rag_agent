"""索引构建与加载：chunks.jsonl / tokens.jsonl / vectors.npy / meta.json。

为什么向量存 .npy 而不是 faiss.index：
    faiss 的 C++ 文件 IO 在 Windows 上走窄字符 fopen，路径里带中文
    （例如 C:\\Users\\秦家鑫\\...）会直接 "could not open ... No such file or directory"。
    numpy 的 np.save/np.load 用宽字符 API，不受影响；加载时用向量重建
    IndexFlatIP 只要几毫秒，还顺带摆脱了 faiss 文件格式的版本耦合。
"""

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

_FILES = ("chunks.jsonl", "tokens.jsonl", "vectors.npy", "meta.json")


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

    np.save(out / "vectors.npy", np.ascontiguousarray(vectors, dtype=np.float32))

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
    vectors = np.load(out / "vectors.npy").astype(np.float32)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))

    _assert_aligned(chunks, vectors, tokens)
    expected_dim = int(meta.get("dim", vectors.shape[1]))
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise IndexBuildError(
            "向量维度与 meta.json 不一致",
            {"shape": list(vectors.shape), "dim": expected_dim},
        )

    index = faiss.IndexFlatIP(expected_dim)
    index.add(np.ascontiguousarray(vectors))

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