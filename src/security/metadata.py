from __future__ import annotations

import hashlib
from pathlib import Path

from src.config import Settings
from src.schema import DocMeta
from src.utils.logger import get_logger, project_root

logger = get_logger(__name__)


def doc_id_for(rel_path: str) -> str:
    """根据相对路径生成文档 ID（稳定：重建索引后引用不失效）。"""
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]


def resolve_doc_meta(rel_path: str, settings: Settings) -> DocMeta:
    labels = labels_from_path(rel_path, settings)
    department, sensitivity = labels or (
        settings.ingest.unknown_department,
        settings.default_sensitivity,
    )
    return DocMeta(
        doc_id=doc_id_for(rel_path),
        department=department,
        sensitivity=sensitivity,
        labels_source="path" if labels else "default",
    )


def labels_from_path(rel_path: str, settings: Settings) -> tuple[str, str] | None:
    """从 {部门}/{密级}/file 解析标签；任一层不合法返回 None。"""
    relative = _relative_to_raw(rel_path, settings)
    if relative is None:
        return None

    parts = relative.parts
    if len(parts) < 3:                      # 至少 部门/密级/文件
        return None

    department, sensitivity = parts[0], parts[1]
    if (
        department in settings.security.departments
        and sensitivity in settings.security.security_levels
    ):
        return department, sensitivity
    return None


def _relative_to_raw(rel_path: str, settings: Settings) -> Path | None:
    """把 rel_path 还原成"相对 raw_docs 的路径"。

    兼容两种输入：
    - 项目根相对（生产：data/raw_docs/研发/普通/制度.md）；
    - 绝对路径（raw_docs 挂在项目根之外，例如测试的 tmp_path、容器卷挂载）。
    两者都还原不了就返回 None，由调用方走 fail-closed 兜底。
    """
    raw = Path(settings.paths.raw_docs).resolve()
    candidate = Path(rel_path)
    possibilities = (
        [candidate] if candidate.is_absolute() else [project_root() / candidate, candidate]
    )
    for item in possibilities:
        try:
            return item.resolve().relative_to(raw)
        except (ValueError, OSError):
            continue
    return None
