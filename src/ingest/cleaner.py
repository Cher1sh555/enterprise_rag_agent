"""文档清洗：换行归一 + 页码行剔除 + 按空行切段。

v0 范围：只处理纯文本段落。页眉页脚的跨页频率统计、表格结构化留到 v1——
单页的 md/txt/docx 没有"页眉跨页重复"这个概念，现在做等于给自己加假需求。
"""

from __future__ import annotations

import re

from src.config import Settings
from src.schema import Block, CleanDoc, RawDoc
from src.utils.exceptions import CleanError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BLANK_LINE = re.compile(r"\n\s*\n")
_PAGE_NO_LINE = re.compile(r"^\s*[-—–]?\s*\d{1,4}\s*[-—–]?\s*$")


def clean_document(doc: RawDoc, settings: Settings) -> CleanDoc:
    """把每页文本切成段落块；清洗后一块不剩 -> CleanError。

    settings 目前未使用：保留参数是为了和 load/split 的调用约定一致，
    v1 的表格策略与页眉统计会从这里读配置。
    """
    blocks: list[Block] = []
    for page in doc.pages:
        for paragraph in _paragraphs(page.text):
            blocks.append(Block(kind="text", page_no=page.page_no, text=paragraph))
    if not blocks:
        raise CleanError("清洗后没有任何文本", {"path": doc.rel_path})
    return CleanDoc(raw=doc, blocks=tuple(blocks))


def _paragraphs(text: str) -> list[str]:
    """先按空行切段，再在段内删页码行（顺序不能反：先删空行就没得切了）。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    for chunk in _BLANK_LINE.split(text):
        lines = [line.strip() for line in chunk.split("\n")]
        lines = [line for line in lines if line and not _PAGE_NO_LINE.match(line)]
        if lines:
            paragraphs.append("\n".join(lines))
    return paragraphs
