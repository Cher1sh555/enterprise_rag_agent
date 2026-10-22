"""多格式文档加载：PDF / Word / 扫描件(OCR) 区分。

要点：
1. paddleocr / pymupdf 必须在函数体内导入 —— tests/test_smoke.py 会用
   pkgutil.walk_packages 导入 src 下每个模块，顶层导入未安装的库会让冒烟测试全红。
2. _needs_ocr 是页级函数；文档级只做统计（见 _load_pdf 末尾的日志）。
3. max_pages 在循环上界截断，且在 OCR 判定之前，不渲染多余页。
4. .docx 只取 paragraphs；表格单元格里的段落不在 paragraphs 中，
   留给 cleaner 通过 Document.tables 处理，两边不重复计入。
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np

from src.config import IngestConfig, Settings
from src.schema import Page, RawDoc
from src.utils.exceptions import LoadError, OCRError, UnsupportedFormatError
from src.utils.logger import get_logger, project_root

logger = get_logger(__name__)

_HIDDEN_DIRS = {"__pycache__", ".git", ".idea", ".vscode", ".venv", "venv", ".pytest_cache"}
_OFFICE_LOCK_PREFIX = "~$"
_WHITESPACE_RE = re.compile(r"\s+")
# 不可靠字符：替换符 + Unicode 私用区（CID 字体缺 ToUnicode 映射的典型产物）
_UNRELIABLE_RE = re.compile(r"[\ufffd\uE000-\uF8FF]")
# 可靠率下限：低于它说明文本层是乱码，需要走 OCR 而不是直接使用
_MIN_RELIABLE_RATIO = 0.7


def iter_documents(raw_dir: Path, ext: Sequence[str]) -> Iterator[Path]:
    """遍历待索引文件。

    跳过：Office 临时锁文件（~$）、隐藏/环境目录、非白名单后缀。
    """
    root = Path(raw_dir)
    suffixes = {suffix.lower() for suffix in ext}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(_OFFICE_LOCK_PREFIX):
            continue
        if any(part in _HIDDEN_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in suffixes:
            yield path


def load_documents(path: Path, settings: Settings) -> RawDoc:
    """加载单个文件，返回 RawDoc（doc_id 由后续 split 阶段生成）。"""
    file_path = Path(path).resolve()
    _check_file(file_path, settings)

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        pages = _load_pdf(file_path, settings.ingest)
    elif suffix == ".docx":
        pages = _load_docx(file_path, settings.ingest)
    else:
        pages = _load_text(file_path, settings.ingest)

    if not any(page.text.strip() for page in pages):
        raise LoadError("未提取到任何文本", {"path": str(file_path), "ext": suffix})

    return RawDoc(
        path=file_path,
        rel_path=_rel_path(file_path),
        ext=suffix.lstrip("."),
        pages=pages,
    )


def _load_pdf(path: Path, settings: IngestConfig) -> tuple[Page, ...]:
    fitz = _import_pymupdf()
    doc = _open_pdf(fitz, path)
    try:
        total = doc.page_count
        # 要点 3：上界在这里截断，后面的 OCR 判定与渲染都只发生在 limit 之内
        limit = total if settings.max_pages is None else min(total, settings.max_pages)
        if limit < total:
            logger.warning("文档共 %d 页，按 max_pages=%d 截断: %s", total, limit, path.name)

        texts: list[str] = []
        for index in range(limit):
            try:
                texts.append(doc.load_page(index).get_text("text") or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("第 %d 页文本层提取失败: %s", index + 1, exc)
                texts.append("")
    finally:
        doc.close()

    # 页级判定：只 OCR 真正需要的页（0 基索引，与 fitz.load_page 对齐）
    ocr_pages = [index for index, text in enumerate(texts) if _needs_ocr(text, settings)]
    ocr_results: dict[int, tuple[str, float]] = {}
    if ocr_pages:
        ocr_results = _ocr_pages(path, ocr_pages, settings)

    pages: list[Page] = []
    for index, text in enumerate(texts):
        used_ocr = False
        confidence: float | None = None
        if index in ocr_results:
            ocr_text, confidence = ocr_results[index]
            # OCR 没比文本层多出内容时保留文本层，避免噪声覆盖水印/页码
            if _visible_chars(ocr_text) > _visible_chars(text):
                text = ocr_text
                used_ocr = True
        pages.append(
            Page(
                page_no=index + 1,
                text=text.strip(),
                ocr_used=used_ocr,
                ocr_confidence=confidence if used_ocr else None,
                low_confidence=bool(
                    used_ocr
                    and confidence is not None
                    and confidence < settings.ocr_min_confidence
                ),
            )
        )

    # 要点 2：文档级只做统计
    logger.info(
        "PDF %s：共 %d 页（原始 %d 页），OCR 命中 %d 页，平均文本密度 %.2f",
        path.name,
        len(pages),
        total,
        sum(1 for page in pages if page.ocr_used),
        sum(_text_density(page.text) for page in pages) / max(len(pages), 1),
    )
    return tuple(pages)


def _load_text(path: Path, settings: IngestConfig) -> tuple[Page, ...]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
        logger.warning("编码无法确定，按 utf-8 强解，可能乱码: %s", path)

    return (Page(page_no=1, text=text.strip()),)


def _load_docx(path: Path, settings: IngestConfig) -> tuple[Page, ...]:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise LoadError("DOCX 解析需要 python-docx", {"path": str(path)}) from exc

    try:
        document = DocxDocument(str(path))
    except Exception as exc:  # noqa: BLE001
        raise LoadError(
            "DOCX 打开失败（旧版 .doc 请先另存为 .docx）", {"path": str(path)}
        ) from exc

    # 要点 4：只取 body 直属段落。表格单元格里的段落在 w:tbl 下，
    # 不会出现在 document.paragraphs 里，因此与 cleaner 取 tables 不会重复。
    text = "\n".join(
        paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()
    )
    return (Page(page_no=1, text=text),)


def _text_density(text: str) -> float:
    """文本密度 = 可见字符数 / 全部字符数（0~1）。

    文档级统计指标，不参与 OCR 判定 —— 判定走 _needs_ocr 的页级阈值。
    """
    if not text:
        return 0.0
    return _visible_chars(text) / len(text)


def _reliable_ratio(text: str) -> float:
    """可靠率 = 1 - 不可靠字符占比。

    扫描件常带一层"看不见"的文本层：字符数不少，但全是 U+FFFD 或私用区字符。
    这类页字数阈值判不出来，靠可靠率兜住。
    """
    visible = _visible_chars(text)
    if visible == 0:
        return 0.0
    return 1.0 - len(_UNRELIABLE_RE.findall(text)) / visible


def _needs_ocr(text: str, settings: IngestConfig) -> bool:
    """页级判定：这一页是否需要 OCR。"""
    if _visible_chars(text) < settings.scan_threshold_chars_per_page:
        return True
    return _reliable_ratio(text) < _MIN_RELIABLE_RATIO


def _ocr_pages(
    path: Path, pages: Sequence[int], settings: IngestConfig
) -> dict[int, tuple[str, float]]:
    """对指定页做 OCR。

    Args:
        pages: 0 基页码（与 fitz.load_page 一致）。

    Returns:
        {0 基页码: (识别文本, 页均置信度)}，页号基准与入参一致。
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRError(
            "OCR 需要 paddleocr",
            {"path": str(path), "hint": "pip install paddleocr paddlepaddle"},
        ) from exc

    engine = _build_paddle_engine(PaddleOCR, settings)
    fitz = _import_pymupdf()
    doc = _open_pdf(fitz, path)
    results: dict[int, tuple[str, float]] = {}
    try:
        for index in pages:
            try:
                pixmap = doc.load_page(index).get_pixmap(
                    dpi=settings.render_dpi, alpha=False
                )
            except Exception as exc:  # noqa: BLE001
                raise OCRError(
                    "页面渲染失败", {"path": str(path), "page": index + 1}
                ) from exc
            text, confidence = _recognize(engine, _pixmap_to_bgr(pixmap))
            results[index] = (text, confidence)
    finally:
        doc.close()
    return results


# ------------------------------------------------------------------
# 内部工具
# ------------------------------------------------------------------

def _visible_chars(text: str) -> int:
    """可见字符数（忽略空白），OCR 判定与统计的统一口径。"""
    return len(_WHITESPACE_RE.sub("", text or ""))


def _check_file(path: Path, settings: Settings) -> None:
    if not path.exists():
        raise LoadError("文件不存在", {"path": str(path)})
    if not path.is_file():
        raise LoadError("传入的不是文件", {"path": str(path)})
    if path.name.startswith(_OFFICE_LOCK_PREFIX):
        raise UnsupportedFormatError("Office 临时锁文件不应参与索引", {"path": str(path)})
    if path.stat().st_size == 0:
        raise LoadError("空文件（0 字节）", {"path": str(path)})
    supported = {suffix.lower() for suffix in settings.ingest.supported_extensions}
    if path.suffix.lower() not in supported:
        raise UnsupportedFormatError(
            "不支持的格式", {"path": str(path), "supported": sorted(supported)}
        )


def _rel_path(path: Path) -> str:
    """相对项目根的路径，用于 source 展示与 doc_id 生成。"""
    try:
        return path.relative_to(project_root()).as_posix()
    except ValueError:
        logger.warning("文件不在项目根下，rel_path 退化为文件名: %s", path)
        return path.name


def _import_pymupdf():
    """兼容新包名 pymupdf 与旧导入名 fitz。"""
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except ImportError:
        pass
    try:
        import fitz  # type: ignore

        return fitz
    except ImportError as exc:
        raise LoadError(
            "PDF 解析需要 PyMuPDF",
            {"missing": "pymupdf", "hint": "pip install pymupdf"},
        ) from exc


def _open_pdf(fitz, path: Path):
    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        raise LoadError("PDF 打开失败，文件可能损坏", {"path": str(path)}) from exc
    # PyMuPDF 的解锁方法是 authenticate()；decrypt() 属于 pypdf/PyPDF2
    if doc.needs_pass and not doc.authenticate(""):
        doc.close()
        raise LoadError(
            "PDF 已加密，需要密码", {"path": str(path), "reason": "encrypted"}
        )
    if doc.page_count == 0:
        doc.close()
        raise LoadError("PDF 不含任何页面", {"path": str(path)})
    return doc


def _build_paddle_engine(factory, settings: IngestConfig):
    """构造 PaddleOCR，兼容 2.x / 3.x 的构造参数差异。"""
    params = inspect.signature(factory.__init__).parameters
    kwargs: dict[str, object] = {"lang": settings.ocr_lang}
    if "use_textline_orientation" in params:  # 3.x
        kwargs["use_textline_orientation"] = True
    elif "use_angle_cls" in params:  # 2.x
        kwargs["use_angle_cls"] = True
    if "use_gpu" in params:  # 2.x
        kwargs["use_gpu"] = settings.ocr_use_gpu
    if "device" in params and settings.ocr_use_gpu:  # 3.x
        kwargs["device"] = "gpu"
    if "show_log" in params:
        kwargs["show_log"] = False

    try:
        return factory(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise OCRError("PaddleOCR 初始化失败", {"error": repr(exc)}) from exc


def _pixmap_to_bgr(pixmap) -> np.ndarray:
    buffer = np.frombuffer(pixmap.samples, dtype=np.uint8)
    image = buffer.reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n >= 3:
        return image[:, :, :3][:, :, ::-1].copy()  # RGB -> BGR
    return image[:, :, 0].copy()


def _recognize(engine, image: np.ndarray) -> tuple[str, float]:
    try:
        if hasattr(engine, "predict"):  # 3.x
            raw = engine.predict(image)
        else:  # 2.x
            raw = engine.ocr(image, cls=True)
    except Exception as exc:  # noqa: BLE001
        raise OCRError("OCR 识别失败", {"error": repr(exc)}) from exc
    return _parse_ocr_result(raw)


def _parse_ocr_result(raw) -> tuple[str, float]:
    """解析 OCR 结果，兼容 2.x 与 3.x 两种返回结构。

    2.x: [[[box, (text, score)], ...]]
    3.x: [OCRResult]，其中 OCRResult 含 rec_texts / rec_scores
    """
    texts: list[str] = []
    scores: list[float] = []

    for item in raw if isinstance(raw, (list, tuple)) else [raw]:
        mapping = _as_mapping(item)
        if mapping is not None and "rec_texts" in mapping:
            for text, score in zip(
                mapping.get("rec_texts") or [], mapping.get("rec_scores") or []
            ):
                if str(text).strip():
                    texts.append(str(text).strip())
                    scores.append(float(score))
            continue

        nested = item if isinstance(item, (list, tuple)) else []
        for entry in nested:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            payload = entry[1]
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                text, score = payload
                if str(text).strip():
                    texts.append(str(text).strip())
                    scores.append(float(score))

    if not texts:
        return "", 0.0
    return "\n".join(texts), sum(scores) / len(scores)


def _as_mapping(obj) -> Mapping[str, object] | None:
    """把 PaddleOCR 3.x 的 OCRResult / dict / json 统一成 Mapping。"""
    data = None
    if isinstance(obj, Mapping):
        data = obj
    else:
        for attr in ("json", "to_dict"):
            candidate = getattr(obj, attr, None)
            if candidate is None:
                continue
            try:
                data = candidate() if callable(candidate) else candidate
            except Exception:  # noqa: BLE001
                continue
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    data = None
            if isinstance(data, Mapping):
                break

    if not isinstance(data, Mapping):
        return None
    nested = data.get("res") if "res" in data else None
    return nested if isinstance(nested, Mapping) else data
