"""splitter 契约：CleanDoc + DocMeta -> list[Chunk]。

测试即约束，这里钉死两个设计取舍：
1. 段落贪心打包到 <= chunk_size，**打包阶段不重复内容**；
2. 只有单段超长时才按 chunk_overlap 滑动切，相邻窗口共享 overlap 个字符。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ChunkingConfig
from src.ingest.splitter import split_document
from src.schema import Block, CleanDoc, DocMeta, Page, RawDoc

META = DocMeta(doc_id="abc123", department="研发", sensitivity="机密", labels_source="path")


@pytest.fixture()
def small(settings):
    """100/20 的小窗口：让断言不依赖 settings.yaml 里的 512/64。"""
    return settings.model_copy(
        update={"chunking": ChunkingConfig(chunk_size=100, chunk_overlap=20)}
    )


def _clean(texts: list[str], rel_path: str = "data/raw_docs/研发/制度.md") -> CleanDoc:
    raw = RawDoc(
        path=Path(rel_path),
        rel_path=rel_path,
        ext="md",
        pages=(Page(page_no=1, text="\n\n".join(texts)),),
    )
    return CleanDoc(raw=raw, blocks=tuple(Block(kind="text", page_no=1, text=t) for t in texts))


def _distinct(n: int, base: int = 0x4E00) -> str:
    """生成 n 个互不相同的汉字。

    用重复字符（如 "甲"*500）测重叠窗口是自欺欺人——任何两块都"相等"。
    """
    return "".join(chr(base + i) for i in range(n))


def test_short_doc_single_chunk(small):
    chunks = split_document(_clean(["第一段", "第二段", "第三段"]), META, small)
    assert len(chunks) == 1
    assert "第一段" in chunks[0].text
    assert "第三段" in chunks[0].text


def test_every_chunk_within_size(small):
    chunks = split_document(_clean(["甲" * 90, "乙" * 90, "丙" * 90]), META, small)
    assert len(chunks) == 3
    assert all(c.char_len <= 100 for c in chunks)
    assert all(c.char_len == len(c.text) for c in chunks)


def test_long_paragraph_shares_overlap_window(small):
    text = _distinct(500)
    chunks = split_document(_clean([text]), META, small)

    # 步长 = size - overlap = 80 -> 0,80,160,240,320,400 共 6 块
    assert len(chunks) == 6
    assert all(c.char_len <= 100 for c in chunks)
    assert chunks[0].text[-20:] == chunks[1].text[:20]
    assert chunks[0].text != chunks[1].text


def test_no_overlap_when_packing_short_paragraphs(small):
    first, second = _distinct(40), _distinct(40, base=0x5B00)
    chunks = split_document(_clean([first, second]), META, small)

    assert len(chunks) == 1
    assert chunks[0].text.count(first) == 1
    assert chunks[0].text.count(second) == 1


def test_chunk_ids_stable_and_formatted(small):
    doc = _clean(["第一段", "第二段"])
    first = split_document(doc, META, small)
    second = split_document(doc, META, small)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id == "abc123#0001"
    assert [c.seq for c in first] == list(range(1, len(first) + 1))


def test_small_tail_is_merged_into_previous_chunk(small):
    """尾块 < size*0.3 时并入前一块，避免碎块污染召回。"""
    body, tail = "甲" * 90, "乙" * 20
    chunks = split_document(_clean([body, tail]), META, small)

    assert len(chunks) == 1
    assert body in chunks[0].text and tail in chunks[0].text
    assert 100 < chunks[0].char_len <= 115      # 允许上浮到 size*1.15


def test_tail_merge_respects_cap(small):
    """合并后会超过 size*1.15 则不合并，保留两块。"""
    body, tail = "甲" * 100, "乙" * 20
    chunks = split_document(_clean([body, tail]), META, small)

    assert len(chunks) == 2
    assert chunks[-1].char_len == 20


def test_metadata_inherited(small):
    chunk = split_document(_clean(["正文"]), META, small)[0]

    assert (chunk.doc_id, chunk.department, chunk.sensitivity) == ("abc123", "研发", "机密")
    assert chunk.source == "data/raw_docs/研发/制度.md"
    assert chunk.kind == "text"


def test_ocr_flags_propagate(settings):
    raw = RawDoc(
        path=Path("x.md"),
        rel_path="x.md",
        ext="md",
        pages=(Page(page_no=1, text="正文", ocr_used=True, ocr_confidence=0.3, low_confidence=True),),
    )
    doc = CleanDoc(raw=raw, blocks=(Block(kind="text", page_no=1, text="正文"),))

    chunk = split_document(doc, META, settings)[0]
    assert chunk.ocr_used is True
    assert chunk.low_confidence is True
