"""阶段 0 冒烟测试。

验证"地基"是否可用，而不是验证业务逻辑：
1. src 下所有模块都能 import（防止语法错、循环导入）
2. configs/*.yaml 能被 YAML 解析且关键段落齐全（防止缩进错误）
3. 日志能落盘，且不重复挂 handler
4. 异常体系层级正确
5. data 下的关键目录存在（防止建索引时才发现目录没了）
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import yaml

import src


def test_all_src_modules_import() -> None:
    """src 下每个模块都能被导入。"""
    failures: list[str] = []
    for module_info in pkgutil.walk_packages(src.__path__, prefix="src."):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:  # noqa: BLE001 - 冒烟测试要收集全部失败项
            failures.append(f"{module_info.name}: {type(exc).__name__}: {exc}")
    assert not failures, "以下模块导入失败:\n" + "\n".join(failures)


def test_settings_yaml_is_valid(project_root: Path) -> None:
    """settings.yaml 能解析，且必需段落都存在。"""
    path = project_root / "configs" / "settings.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "settings.yaml 顶层必须是映射"

    for section in ("app", "paths", "models", "chunking", "retrieval", "security", "audit"):
        assert section in data, f"settings.yaml 缺少段落: {section}"

    # 关键数值字段存在且类型正确
    assert isinstance(data["chunking"]["chunk_size"], int)
    assert isinstance(data["chunking"]["chunk_overlap"], int)
    assert isinstance(data["retrieval"]["top_k"], int)
    assert data["retrieval"]["rerank_top_k"] <= data["retrieval"]["top_k"], (
        "rerank_top_k 不应大于 top_k，否则截断逻辑会出错"
    )


def test_prompts_yaml_is_valid(project_root: Path) -> None:
    """prompts.yaml 能解析（当前只有注释，解析结果为 None）。"""
    path = project_root / "configs" / "prompts.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data is None or isinstance(data, dict)


def test_logger_writes_file(isolated_log_dir: Path) -> None:
    """日志能写入 app.log，且重复 setup 不会导致重复输出。"""
    from src.utils.logger import get_logger, setup_logging

    logger = setup_logging(level="INFO", log_dir=isolated_log_dir, force=True)
    get_logger("smoke.test").info("冒烟测试日志")
    for handler in logger.handlers:
        handler.flush()

    log_file = isolated_log_dir / "app.log"
    assert log_file.exists(), "日志文件未创建"
    content = log_file.read_text(encoding="utf-8")
    assert "冒烟测试日志" in content
    # 只出现一次 -> 说明没有重复挂 handler
    assert content.count("冒烟测试日志") == 1


def test_logger_setup_is_idempotent(isolated_log_dir: Path) -> None:
    """不带 force 的重复调用不应叠加 handler。"""
    from src.utils.logger import setup_logging

    setup_logging(level="INFO", log_dir=isolated_log_dir, force=True)
    first = len(setup_logging().handlers)
    second = len(setup_logging().handlers)
    assert first == second


def test_exception_hierarchy() -> None:
    """异常都继承 RAGError，且能携带结构化 detail。"""
    from src.utils import exceptions as exc

    for cls in (
        exc.ConfigError,
        exc.LoadError,
        exc.UnsupportedFormatError,
        exc.CleanError,
        exc.SplitError,
        exc.EmbeddingError,
        exc.IndexBuildError,
        exc.IndexNotFoundError,
        exc.RetrievalError,
        exc.PermissionDeniedError,
        exc.LLMCallError,
        exc.LLMTimeoutError,
        exc.LLMRateLimitError,
        exc.AuditError,
    ):
        assert issubclass(cls, exc.RAGError), f"{cls.__name__} 未继承 RAGError"

    assert issubclass(exc.UnsupportedFormatError, exc.LoadError)
    assert issubclass(exc.LLMTimeoutError, exc.LLMCallError)
    assert issubclass(exc.LLMRateLimitError, exc.LLMCallError)

    err = exc.IndexNotFoundError("索引不存在", {"path": "data/index"})
    assert "索引不存在" in str(err)
    assert "data/index" in str(err)


def test_data_directories_exist(project_root: Path) -> None:
    """data 下的关键目录存在（空目录靠 .gitkeep 保活）。"""
    for name in ("raw_docs", "evaluation"):
        assert (project_root / "data" / name).is_dir(), f"缺少目录 data/{name}"
