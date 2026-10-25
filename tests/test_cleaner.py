"""cleaner 契约：RawDoc -> CleanDoc。段落粒度，页码剔除，换行归一。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.cleaner import clean_document
from src.schema import Page, RawDoc
from src.utils.exceptions import CleanError


def _raw(pages: list[str], rel_path: str = "data/raw_docs/研发/制度.md") -> RawDoc:
    return RawDoc(
        path=Path(rel_path),
        rel_path=rel_path,
        ext="md",
        pages=tuple(Page(page_no=i, text=t) for i, t in enumerate(pages, start=1)),
    )


def test_splits_paragraphs_on_blank_lines(settings):
    doc = clean_document(_raw(["段一\n\n段二\n\n\n段三"]), settings)
    assert [b.text for b in doc.blocks] == ["段一", "段二", "段三"]
    assert all(b.kind == "text" for b in doc.blocks)


def test_page_number_line_removed(settings):
    doc = clean_document(_raw(["正文一\n- 3 -\n正文二"]), settings)
    assert "- 3 -" not in doc.blocks[0].text
    assert "正文一" in doc.blocks[0].text
    assert "正文二" in doc.blocks[0].text


def test_crlf_normalized(settings):
    doc = clean_document(_raw(["第一行\r\n第二行"]), settings)
    assert "\r" not in doc.blocks[0].text


def test_page_no_preserved(settings):
    doc = clean_document(_raw(["第一页正文", "第二页正文"]), settings)
    assert [b.page_no for b in doc.blocks] == [1, 2]


@pytest.mark.parametrize("pages", [[""], ["   "], ["\n\n\n"], ["- 1 -\n\n- 2 -"]])
def test_empty_after_clean_raises(settings, pages):
    with pytest.raises(CleanError):
        clean_document(_raw(pages), settings)