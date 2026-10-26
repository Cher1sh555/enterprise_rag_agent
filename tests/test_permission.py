"""权限契约：密级是硬门槛，部门是范围。

规则（改之前先想清楚，这几条决定了会不会越权）：
1. 密级：chunk 等级 > 用户等级 -> 不可读；
2. 最低密级（普通）-> 全员可读；
3. 更高密级（机密）-> 仅同部门或"公共"部门可读；
4. 未知密级 -> 不可读（fail-closed）；
5. 空白名单 != 不过滤（这是最容易写出越权的地方）。
"""

from __future__ import annotations

from types import SimpleNamespace

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.hybrid import bm25_search, vector_search
from src.schema import Chunk, User
from src.security.permission import allowed_ids, can_read, level_order
from src.utils.text import tokenize


def _chunk(chunk_id: str, department: str, sensitivity: str, text: str = "正文") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d",
        source="a.md",
        page_no=1,
        seq=1,
        kind="text",
        text=text,
        char_len=len(text),
        department=department,
        sensitivity=sensitivity,
        ocr_used=False,
        low_confidence=False,
    )


def _user(department: str, level: str) -> User:
    return User(user_id="u1", department=department, level=level)


def test_level_order_follows_config(settings):
    order = level_order(settings)

    assert order[settings.security.security_levels[0]] < order[
        settings.security.security_levels[-1]
    ]


def test_lowest_level_readable_by_everyone(settings):
    """HR 的年假政策如果只有 HR 能查，知识库就没意义了。"""
    chunk = _chunk("c1", "人力", settings.security.security_levels[0])

    assert can_read(_user("研发", settings.security.security_levels[-1]), chunk, settings)
    assert can_read(_user("财务", settings.security.security_levels[0]), chunk, settings)


def test_secret_readable_within_department(settings):
    chunk = _chunk("c1", "研发", settings.security.security_levels[-1])

    assert can_read(_user("研发", settings.security.security_levels[-1]), chunk, settings)


def test_secret_blocked_across_departments(settings):
    chunk = _chunk("c1", "财务", settings.security.security_levels[-1])

    assert not can_read(_user("研发", settings.security.security_levels[-1]), chunk, settings)


def test_secret_in_public_department_is_readable(settings):
    """公共部门的机密文档 = 全员可读的机密（由公司层面发布）。"""
    chunk = _chunk("c1", "公共", settings.security.security_levels[-1])

    assert can_read(_user("研发", settings.security.security_levels[-1]), chunk, settings)


def test_insufficient_level_blocked(settings):
    chunk = _chunk("c1", "研发", settings.security.security_levels[-1])

    assert not can_read(_user("研发", settings.security.security_levels[0]), chunk, settings)


def test_unknown_level_is_fail_closed(settings):
    chunk = _chunk("c1", "研发", "绝密")          # 不在 security_levels 里

    assert not can_read(_user("研发", settings.security.security_levels[-1]), chunk, settings)


def test_allowed_ids_matches_can_read(settings):
    chunks = [
        _chunk("c1", "研发", settings.security.security_levels[0]),
        _chunk("c2", "研发", settings.security.security_levels[-1]),
        _chunk("c3", "财务", settings.security.security_levels[-1]),
    ]
    user = _user("研发", settings.security.security_levels[-1])

    visible = allowed_ids(chunks, user, settings)

    assert visible == {"c1", "c2"}


def test_empty_whitelist_does_not_leak():
    """空白名单必须"什么都查不到"，绝不能退化成"不过滤"。"""
    texts = ["差旅住宿标准", "年假 5 天"]
    chunks = [
        _chunk("c1", "研发", "普通", texts[0]),
        _chunk("c2", "人力", "普通", texts[1]),
    ]
    index = faiss.IndexFlatIP(3)
    index.add(np.eye(2, 3, dtype=np.float32))
    bundle = SimpleNamespace(
        faiss=index, chunks=chunks, bm25=BM25Okapi([tokenize(t) for t in texts])
    )

    assert vector_search(bundle, np.eye(3, dtype=np.float32)[0], 5, set()) == []
    assert bm25_search(bundle, "差旅住宿标准", 5, set()) == []
