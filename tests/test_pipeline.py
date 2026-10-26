"""pipeline 契约：fail-soft、报表自洽、打标覆盖统计。

两个关键错误策略（写在这里就是文档）：
- 单文档失败 fail-soft：记进 report.failures，其余文档照常入库；
- 配置/模型/索引失败 fail-fast：直接抛，不产出半成品索引。
"""

from __future__ import annotations

import pytest

from src.ingest.indexer import load_index
from src.ingest.pipeline import run_ingest
from src.utils.exceptions import IndexBuildError


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_empty_raw_docs_raises(settings):
    """一份文档都没有时必须失败，而不是产出一个空索引。"""
    with pytest.raises(IndexBuildError):
        run_ingest(settings)


def test_bad_doc_is_skipped_not_fatal(settings):
    _write(
        settings.paths.raw_docs / "研发" / "差旅.md",
        "差旅住宿标准\n\n一线城市每晚不超过 600 元",
    )
    _write(settings.paths.raw_docs / "研发" / "年假.md", "年假\n\n满一年 5 天")
    (settings.paths.raw_docs / "空文件.md").write_bytes(b"")

    report = run_ingest(settings)

    assert report.docs == 2
    assert report.docs_skipped == 1
    assert [f["error"] for f in report.failures] == ["LoadError"]


def test_report_matches_index(settings):
    _write(
        settings.paths.raw_docs / "研发" / "差旅.md",
        "差旅住宿标准\n\n一线城市每晚不超过 600 元",
    )

    report = run_ingest(settings)
    bundle = load_index(settings.paths.index)

    assert report.chunks == len(bundle.chunks) > 0
    assert bundle.faiss.ntotal == report.chunks


def test_processed_jsonl_written(settings):
    _write(settings.paths.raw_docs / "研发" / "差旅.md", "正文一\n\n正文二")

    run_ingest(settings)

    assert len(list(settings.paths.processed.glob("*.jsonl"))) == 1


def test_labels_from_default_counted(settings):
    """放在 raw_docs 根目录 -> 认不出部门 -> 走 fail-closed 兜底并计数。"""
    _write(settings.paths.raw_docs / "制度.md", "没有部门目录的文档")

    report = run_ingest(settings)

    assert report.chunks > 0
    assert report.labels_from_default == report.chunks


def test_labels_inherited_from_two_level_path(settings):
    """两级目录 研发/普通/ -> 部门与密级都来自路径，不计入兜底。"""
    _write(settings.paths.raw_docs / "研发" / "普通" / "制度.md", "正文")

    report = run_ingest(settings)
    bundle = load_index(settings.paths.index)

    assert report.labels_from_default == 0
    assert {(c.department, c.sensitivity) for c in bundle.chunks} == {("研发", "普通")}


def test_incomplete_path_falls_back_to_default(settings):
    """只有部门层、没有密级层 -> 整体 fail-closed（未知 + 最严格密级）。"""
    _write(settings.paths.raw_docs / "研发" / "制度.md", "正文")

    report = run_ingest(settings)
    bundle = load_index(settings.paths.index)

    assert report.labels_from_default == report.chunks > 0
    assert {(c.department, c.sensitivity) for c in bundle.chunks} == {("未知", "机密")}
