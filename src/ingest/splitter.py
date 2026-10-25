"""递归字符分块 + 重叠窗口 + 表格独立策略。"""
from __future__ import annotations

from collections.abc import Sequence

from src.config import Settings
from src.schema import Block, Chunk, CleanDoc, DocMeta

def split_document(doc: CleanDoc, meta: DocMeta, settings: Settings) -> list[Chunk]:
    size = settings.chunking.chunk_size
    overlap = settings.chunking.chunk_overlap
    packed = _pack(doc.blocks, size, overlap)
    ocr_used = any(page.ocr_used for page in doc.raw.pages)
    low_conf = any(page.low_confidence for page in doc.raw.pages)

    return [
        Chunk(
            chunk_id=f"{meta.doc_id}#{seq:04d}",
            doc_id=meta.doc_id,
            source=doc.raw.rel_path,
            page_no=page_no,
            seq=seq,
            kind="text",
            text=text,
            char_len=len(text),
            department=meta.department,
            sensitivity=meta.sensitivity,
            ocr_used=ocr_used,
            low_confidence=low_conf,
        )
        for seq, (text, page_no) in enumerate(packed, start=1)
    ]

def _pack(blocks: Sequence[Block], size: int, overlap: int) -> list[tuple[str, int]]:
    """把段落装进 <= size 的块；返回 [(块文本, 起始页)]。"""
    packed: list[tuple[str, int]] = []
    buf, buf_page = "", 0
    for block in blocks:
        for piece in _split_long(block.text, size, overlap):
            if buf and len(buf) + 1 + len(piece) > size:
                packed.append((buf, buf_page))
                buf, buf_page = piece, block.page_no
            elif buf:
                buf = f"{buf}\n{piece}"
            else:
                buf, buf_page = piece, block.page_no
    if buf.strip():
        packed.append((buf, buf_page))
    return packed


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """单段超长时按字符窗口滑动切，窗口之间保留 overlap。"""
    if len(text) <= size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        pieces.append(text[start:end])
        if end >= len(text):
            break
        nxt = end - overlap
        start = nxt if nxt > start else end     # 防死循环（overlap>=size 时）
    return pieces