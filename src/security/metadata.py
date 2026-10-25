"""部门 / 密级标签规范与校验。

v0 打标只做路径约定：raw_docs/{部门}/xxx.md -> department = 部门。
标不上就退回 unknown_department + 最严格密级（fail-closed），
并把来源记成 labels_source="default"，方便 build_index 统计打标覆盖率。
"""

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
    department = _department_from_path(rel_path, settings)
    return DocMeta(
        doc_id=doc_id_for(rel_path),
        department=department,
        sensitivity=settings.default_sensitivity,
        labels_source="path" if department != settings.ingest.unknown_department else "default",
    )


def _department_from_path(rel_path: str, settings: Settings) -> str:
    """raw_docs 下第一层目录名若命中部门白名单，就采信。"""
    relative = _relative_to_raw(rel_path, settings)
    if relative is None:
        return settings.ingest.unknown_department

    parts = relative.parts
    if len(parts) > 1 and parts[0] in settings.security.departments:
        return parts[0]
    return settings.ingest.unknown_department


def _relative_to_raw(rel_path: str, settings: Settings) -> Path | None:
    """把 rel_path 还原成"相对 raw_docs 的路径"。

    兼容两种输入：
    - 项目根相对（生产：data/raw_docs/研发/制度.md）；
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
