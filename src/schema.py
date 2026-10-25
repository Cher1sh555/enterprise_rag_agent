from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal  #类型注解工具：**限定变量只能是给定的几个固定字面量之中的一个**

Kind = Literal["text", "table"]  #`Literal["a","b"]`：**只能是这几个固定字符串**

@dataclass(frozen=True)
class Page:
    page_no: int                 # 1-based；docx/txt/md 视为 1
    text: str
    ocr_used: bool = False
    ocr_confidence: float | None = None   # 该页 OCR 的页均置信度
    low_confidence: bool = False          # < ocr_min_confidence 时置 True（标记不丢弃）

@dataclass(frozen=True)
class RawDoc:
    path: Path
    rel_path: str                # 相对项目根，用于 source 与 doc_id
    ext: str                     # 文件后缀 pdf / docx / md
    pages: tuple[Page, ...]

@dataclass(frozen=True)
class Block:
    kind: Kind
    page_no: int
    text: str                    # text: 段落文本；table: markdown 渲染（含表头）
    header: tuple[str, ...] | None = None          # 仅 table
    rows: tuple[tuple[str, ...], ...] | None = None  # 仅 table，切分时按行分组要用
    table_id: str | None = None

@dataclass(frozen=True)
class CleanDoc:
    raw: RawDoc
    blocks: tuple[Block, ...]

@dataclass(frozen=True)
class Chunk:
    chunk_id: str                # f"{doc_id}#{seq:04d}"，稳定，不用 uuid
    doc_id: str                  # sha1(rel_path)[:12]
    source: str                  # rel_path，回答里展示的"出处"
    page_no: int
    seq: int                     # 当前文档内分片序号，0001、0002...，用来拼接chunk_id
    kind: Kind
    text: str
    char_len: int
    department: str              # ④ 用
    sensitivity: str             # ④ 用
    ocr_used: bool               # ④ 用（识别质量提示）
    low_confidence: bool

@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    department: str
    sensitivity: str
    labels_source: str          # "path" | "default"

@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    sources: tuple[str, ...]    # ("vector",) / ("bm25",) / ("vector","bm25")

