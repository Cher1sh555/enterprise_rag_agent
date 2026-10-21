"""日志封装：控制台 + 按天轮转文件，双通道。

阶段性约定：
- 阶段 0 先只读环境变量（LOG_LEVEL / LOG_DIR），默认写到 <项目根>/logs/app.log。
- 阶段 2 再改成从 Settings 注入，避免这里反向依赖 src.config 造成循环导入
  （src.config 自己也要打日志，直接互相 import 会炸）。

用法：
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# 所有项目日志都挂在这个命名空间下，方便统一开关（logging.getLogger("rag").setLevel(...)）
ROOT_LOGGER_NAME = "rag"

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 模块级状态：保证多次 import / 多次调用不会重复挂 handler（重复挂 = 每行日志打印 N 遍）
_configured = False


def project_root() -> Path:
    """返回项目根目录。

    src/utils/logger.py -> parents[0]=utils, [1]=src, [2]=项目根。
    """
    return Path(__file__).resolve().parents[2]


def _resolve_log_dir(log_dir: Path | str | None) -> Path:
    """把 LOG_DIR 解析成绝对路径。

    相对路径一律相对"项目根"解析，而不是当前工作目录 ——
    否则从 scripts/ 下运行和从项目根运行会写到两个不同的地方。
    """
    if log_dir is None:
        raw = os.getenv("LOG_DIR", "").strip()
        path = Path(raw) if raw else Path("logs")
    else:
        path = Path(log_dir)

    if not path.is_absolute():
        path = project_root() / path
    return path


def setup_logging(
    level: str | int | None = None,
    log_dir: Path | str | None = None,
    force: bool = False,
) -> logging.Logger:
    """初始化日志系统并返回根 logger。

    Args:
        level: 日志级别，默认取环境变量 LOG_LEVEL，再默认 INFO。
        log_dir: 日志目录，默认 <项目根>/logs。
        force: 清空已有 handler 重新配置（测试里替换目录时用）。
    """
    global _configured

    logger = logging.getLogger(ROOT_LOGGER_NAME)

    if _configured and not force:
        return logger

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    level_value = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level_value, str):
        level_value = level_value.strip().upper() or "INFO"
    logger.setLevel(level_value)
    # 不向上冒泡：避免 root logger 上的默认 handler 再打印一遍
    logger.propagate = False

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)

    # ---- 控制台：走 stderr，让 stdout 留给脚本的统计输出（build_index 要打印指标）----
    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ---- 文件：按天切分，保留 14 天 ----
    try:
        directory = _resolve_log_dir(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            filename=directory / "app.log",
            when="midnight",
            backupCount=14,
            encoding="utf-8",
            delay=True,  # 有第一条日志才创建文件，避免空日志文件
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # 日志写不了不能拖垮主流程：降级为仅控制台，并明确告警
        logger.warning("日志文件初始化失败，已降级为仅控制台输出: %s", exc)

    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取带项目统一配置的 logger。

    Args:
        name: 通常传 __name__，会变成 rag.src.xxx.yyy 的层级命名。
    """
    setup_logging()
    if not name:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


__all__ = ["ROOT_LOGGER_NAME", "get_logger", "setup_logging", "project_root"]
