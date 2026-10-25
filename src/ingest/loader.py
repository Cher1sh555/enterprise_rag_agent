from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from src.config import Settings
from src.schema import Page, RawDoc
from src.utils.exceptions import LoadError, UnsupportedFormatError
from src.utils.logger import get_logger, project_root

logger = get_logger(__name__)

_SKIP_DIRS = {"__pycache__", ".git", ".idea", ".vscode", ".venv", "venv", ".pytest_cache"}
_LOCK_PREFIX = "~$"            # Office 打开文档时的临时锁文件
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")   # gb18030 兜住老 Windows 文档

def iter_documents(raw_dir: Path, extensions: Sequence[str]) -> Iterator[Path]:
    """按路径排序产出待索引文件；排序保证构建结果可复现。"""
    root = Path(raw_dir)
    if not root.is_dir():
        raise LoadError("原始文档目录不存在", {"path": str(root)})
    allowed = {ext.lower() for ext in extensions}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith(_LOCK_PREFIX):
            continue
        if allowed and path.suffix.lower() not in allowed:
            continue
        if _SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        yield path

def load_document(path: Path, settings: Settings) -> RawDoc:
    """加载单个文档，返回 RawDoc。"""
    file_path = Path(path).resolve()#转为真实路径
    suffix = check_file(file_path,settings)

    if suffix == ".docx":
        text,table_count = _read_docx(file_path)
        if table_count :
            logger.warning(
                "%s 含 %d 个表格，v0 不索引表格内容", file_path.name, table_count
            )

    else:
        text = read_text(file_path)

    if not text.strip():
        raise LoadError("未提取到任何文本", {"path": str(file_path), "ext": suffix})

    return RawDoc(
        path=file_path,
        rel_path=_rel_path(file_path),
        ext=suffix.lstrip("."),
        pages=(Page(page_no=1, text=text),),
    )


# ------------------------------------------------------------------
# 内部
# ------------------------------------------------------------------

def check_file(path: Path, settings: Settings) -> str:
    if not path.exists():
        raise LoadError("文件不存在", {"path": str(path)})
    if not path.is_file():
        raise LoadError("不是文件", {"path": str(path)})
    if path.stat().st_size == 0:
        raise LoadError("文件为空", {"path": str(path)})
    suffix = path.suffix.lower()
    allowed = {ext.lower() for ext in settings.ingest.supported_extensions}
    if suffix not in allowed:
        raise UnsupportedFormatError(
            "不支持的格式",
            {"path": str(path), "ext": suffix, "supported": sorted(allowed)},
        )
    return suffix

def read_text(path: Path) -> str:
    """读取文本文件，尝试多种编码。"""
    raw = path.read_bytes()
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding) #一次性读取文件全部内容，返回 `bytes` 字节对象
        except UnicodeDecodeError:
            continue
    logger.warning("编码无法确定，按 utf-8 强解，可能出现乱码: %s", path.name)
    return raw.decode("utf-8", errors="replace")

def _read_docx(path: Path) -> tuple[str, int]:
    from docx import Document
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    parts: list[str] = []
    table_count = 0
    for block in doc.element.body:
        if block.tag.endswith("}p"):
            text = Paragraph(block, doc).text.strip()   # 用 python-docx 的 Paragraph 取文本
            if text:
                parts.append(text)
        elif block.tag.endswith("}tbl"):
            table_count += 1
    return "\n".join(parts), table_count

def _rel_path(path: Path) -> str:
    """返回相对于项目根的路径，便于索引存储和展示。"""
    try:
        return path.relative_to(project_root()).as_posix()   # 统一正斜杠，跨平台
    except ValueError:
        logger.warning("文件不在项目根下，rel_path 退化为文件名: %s", path.name)
        # path 不在项目根下，直接返回绝对路径
        return str(path)















