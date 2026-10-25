"""pytest 全局夹具与路径设置。

作用：即使没有执行 `pip install -e .`，测试里也能 `import src.xxx`。
（pip install -e . 是第一道保险，这里是第二道，双保险不嫌多。）
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# 固定测试临时目录
#
# 背景：某些环境（本机就是）下 Python 进程写不了系统 Temp，
# tempfile 会回退到"当前工作目录"，导致 pytest 在项目根建
# `pytest-of-<用户>` 目录，污染工作区。
# 这里显式指定到项目内的 .pytest-tmp/，行为可预测、可 gitignore。
# 必须在任何 fixture 创建临时目录之前执行，所以放在模块顶层。
# ------------------------------------------------------------------
_PYTEST_TMP = PROJECT_ROOT / ".pytest-tmp"
_PYTEST_TMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_PYTEST_TMP)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录，供需要读 configs/ 或 data/ 的用例使用。"""
    return PROJECT_ROOT


@pytest.fixture()
def isolated_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把日志目录临时指到 tmp_path，避免测试污染真实 logs/。"""
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    return tmp_path


# tests/conftest.py 追加（放在文件末尾）
from src.config import PathConfig, Settings, load_settings


@pytest.fixture(scope="session")
def base_settings() -> Settings:
    """真实配置只读一次。

    为什么不手写一份 Settings(...) 全量构造：它有 27 个字段，你以后每加一个
    字段就得同步改 fixture，改漏了测试全红。用真实 yaml 才是单一真相源，
    yaml 的形状由 test_smoke 单独守（那条契约测试就该干这个）。
    """
    return load_settings()


@pytest.fixture()
def settings(base_settings: Settings, tmp_path: Path) -> Settings:
    """把四条路径全部重定向到 tmp_path，并把 embedding 固定成 hash。"""
    paths = PathConfig(
        raw_docs=tmp_path / "raw_docs",
        processed=tmp_path / "processed",
        index=tmp_path / "index",
        evaluation=tmp_path / "evaluation",
    )
    for path in (paths.raw_docs, paths.processed, paths.index, paths.evaluation):
        path.mkdir(parents=True, exist_ok=True)      # iter_documents 要求目录存在

    return base_settings.model_copy(
        update={
            "paths": paths,
            "models": base_settings.models.model_copy(update={"embedding_model": "hash"}),
        }
    )
