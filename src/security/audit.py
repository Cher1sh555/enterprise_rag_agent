"""审计日志：提问、命中片段、用户 ID 留痕。

约定（exceptions.py 里 `AuditError` 的注释也是这个意思）：
**审计失败只上报告警，绝不中断主问答链路。** 日志写不进去是运维问题，
不该让用户问不了问题；但必须留 warning，否则就是静默丢证据。

格式：JSONL，一行一条，追加写。字段由调用方给，这里只补时间戳。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import Settings
from src.utils.logger import get_logger, project_root

logger = get_logger(__name__)

AUDIT_FILENAME = "audit.jsonl"


def audit_path(settings: Settings) -> Path:
    directory = Path(settings.audit.log_dir)
    if not directory.is_absolute():
        directory = project_root() / directory
    return directory / AUDIT_FILENAME


def write_event(event: Mapping[str, Any], settings: Settings) -> bool:
    """追加一条审计事件；返回是否写成功，任何异常都降级为告警。"""
    if not settings.audit.enabled:
        return False

    payload = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **dict(event)}
    try:
        path = audit_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("审计写入失败（不影响主链路）: %s", exc)
        return False


def read_events(settings: Settings, limit: int = 50) -> list[dict]:
    """读最近 limit 条，供 UI / 排查用；文件不存在返回空列表。"""
    path = audit_path(settings)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
