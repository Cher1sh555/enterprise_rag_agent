"""审计契约：JSONL 追加、可关闭、写失败绝不抛。"""

from __future__ import annotations

import json

from src.security.audit import audit_path, read_events, write_event


def _with_log_dir(settings, path):
    return settings.model_copy(
        update={"audit": settings.audit.model_copy(update={"log_dir": path})}
    )


def test_write_event_appends_jsonl(tmp_path, settings):
    scoped = _with_log_dir(settings, tmp_path)

    assert write_event({"question": "差旅标准", "hit_ids": ["c1"]}, scoped) is True

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["question"] == "差旅标准"
    assert payload["ts"]                       # 自动补时间戳


def test_write_event_appends_not_overwrites(tmp_path, settings):
    scoped = _with_log_dir(settings, tmp_path)

    write_event({"n": 1}, scoped)
    write_event({"n": 2}, scoped)

    events = read_events(scoped, limit=10)
    assert [event["n"] for event in events] == [1, 2]


def test_disabled_writes_nothing(tmp_path, settings):
    scoped = _with_log_dir(settings, tmp_path).model_copy(
        update={"audit": settings.audit.model_copy(update={"log_dir": tmp_path, "enabled": False})}
    )

    assert write_event({"n": 1}, scoped) is False
    assert not (tmp_path / "audit.jsonl").exists()


def test_write_failure_is_swallowed(tmp_path, settings):
    """把 log_dir 指到一个同名文件上 -> mkdir 失败，但必须返回 False 而不是抛。"""
    blocker = tmp_path / "blocked"
    blocker.write_text("x", encoding="utf-8")

    assert write_event({"n": 1}, _with_log_dir(settings, blocker)) is False


def test_audit_path_resolves_relative_to_project_root(settings):
    assert audit_path(settings).name == "audit.jsonl"
