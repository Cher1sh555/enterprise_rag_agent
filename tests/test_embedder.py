"""embedder 契约：一个接口两种实现，输出必须已 L2 归一化。

归一化不是"顺手做的"——FAISS 用 IndexFlatIP，只有归一化后内积才等于余弦。
这条约束放在 embedder 里，indexer 就不必关心索引类型。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ingest.embedder import HashEmbedder, STEmbedder, build_embedder


def test_shape_dtype_and_norm():
    vecs = HashEmbedder(dim=512).encode_documents(["差旅住宿标准", "年假政策"])

    assert vecs.shape == (2, 512)
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-5)


def test_deterministic():
    embedder = HashEmbedder(dim=512)
    np.testing.assert_array_equal(
        embedder.encode_query("差旅标准"), embedder.encode_query("差旅标准")
    )


def test_similar_text_scores_higher():
    embedder = HashEmbedder(dim=512)
    base = embedder.encode_query("差旅住宿标准")
    near = embedder.encode_query("差旅住宿")
    far = embedder.encode_query("年假政策")

    assert float(base @ near) > float(base @ far)


def test_empty_input_returns_empty_matrix():
    vecs = HashEmbedder(dim=64).encode_documents([])
    assert vecs.shape == (0, 64)
    assert vecs.dtype == np.float32


def test_query_is_one_dimensional():
    assert HashEmbedder(dim=64).encode_query("文本").shape == (64,)


def test_build_embedder_force_hash(settings):
    assert isinstance(build_embedder(settings, force_hash=True), HashEmbedder)


def test_build_embedder_by_model_name(settings):
    """模型名以 hash 开头即走 HashEmbedder（测试/离线路径）。"""
    cfg = settings.model_copy(
        update={"models": settings.models.model_copy(update={"embedding_model": "hash-512"})}
    )
    assert isinstance(build_embedder(cfg), HashEmbedder)


@pytest.mark.slow
def test_st_embedder_real_model():
    """真模型冒烟：需要下载约 95MB 权重，默认不跑（-m "not slow"）。"""
    embedder = STEmbedder("BAAI/bge-small-zh-v1.5")
    vecs = embedder.encode_documents(["差旅住宿标准", "年假政策"])

    assert vecs.shape == (2, embedder.dim)
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-4)

    query = embedder.encode_query("住宿标准")
    assert float(query @ vecs[0]) > float(query @ vecs[1])
