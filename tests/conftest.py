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
