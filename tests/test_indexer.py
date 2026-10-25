"""indexer 契约：四文件落盘 + 对齐不变量 + 可加载可检索。

落盘约定（chunks.jsonl / tokens.jsonl / faiss.index / meta.json），
不变量：len(chunks) == len(tokens) == faiss.ntotal。
这一条断了，检索会把 A 的分数安到 B 头上，而且不报错。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ingest.embedder import HashEmbedder
from src.ingest.indexer import build_index, load_index
from src.schema import Chunk
from src.utils.exceptions import IndexBuildError, IndexNotFoundError

META = {"version": 1, "embedding_model": "hash", "dim": 512, "chunk_count": 2}


def _chunk(text: str, seq: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc1#{seq:04d}",
        doc_id="doc1",
        source="data/raw_docs/研发/制度.md",
        page_no=1,
        seq=seq,
        kind="text",
        text=text,
        char_len=len(text),
        department="研发",
        sensitivity="机密",
        ocr_used=False,
        low_confidence=False,
    )


@pytest.fixture()
def chunks() -> list[Chunk]:
    return [
        _chunk("差旅住宿标准 一线城市 600 元", 1),
        _chunk("年假 5 天 满一年", 2),
    ]


@pytest.fixture()
def vectors(chunks) -> np.ndarray:
    return HashEmbedder(dim=512).encode_documents([c.text for c in chunks])


def test_roundtrip(tmp_path, chunks, vectors):
    build_index(chunks, vectors, tmp_path, META)

    for name in ("chunks.jsonl", "tokens.jsonl", "faiss.index", "meta.json"):
        assert (tmp_path / name).is_file(), f"缺少落盘文件 {name}"

    bundle = load_index(tmp_path)
    assert bundle.faiss.ntotal == len(chunks) == len(bundle.tokens)
    assert [c.chunk_id for c in bundle.chunks] == [c.chunk_id for c in chunks]
    assert bundle.meta["embedding_model"] == "hash"


def test_search_finds_itself(tmp_path, chunks, vectors):
    build_index(chunks, vectors, tmp_path, META)
    bundle = load_index(tmp_path)

    scores, ids = bundle.faiss.search(
        np.ascontiguousarray(vectors[0][None, :], dtype=np.float32), 1
    )
    assert int(ids[0][0]) == 0
    assert float(scores[0][0]) > 0.99


def test_alignment_mismatch_raises(tmp_path, chunks, vectors):
    with pytest.raises(IndexBuildError) as ei:
        build_index(chunks, vectors[:1], tmp_path, META)
    assert "vectors" in ei.value.detail


def test_missing_files_raise(tmp_path):
    with pytest.raises(IndexNotFoundError) as ei:
        load_index(tmp_path)
    assert ei.value.detail["missing"]


def test_tokens_are_real_words(tmp_path, chunks, vectors):
    build_index(chunks, vectors, tmp_path, META)
    tokens = load_index(tmp_path).tokens

    assert all(tokens), "中文块不应切出空 token 列表"
    assert not any(t.strip() == "" for t in tokens[0])


def test_rebuild_is_idempotent(tmp_path, chunks, vectors):
    build_index(chunks, vectors, tmp_path, META)
    first = [c.chunk_id for c in load_index(tmp_path).chunks]

    build_index(chunks, vectors, tmp_path, META)
    second = [c.chunk_id for c in load_index(tmp_path).chunks]

    assert first == second
