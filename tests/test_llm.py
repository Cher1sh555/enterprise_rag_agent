"""LLM 工程契约：缓存 key、LRU 淘汰、令牌桶、降级。"""

from __future__ import annotations

import pytest

import src.llm.client as client_module
from src.llm.cache import ResponseCache, make_key
from src.llm.ratelimit import TokenBucket
from src.utils.exceptions import LLMRateLimitError

MESSAGES = [{"role": "user", "content": "差旅标准是多少"}]


# ------------------------------------------------------------------ 缓存


def test_cache_key_includes_model_and_temperature():
    assert make_key("a", MESSAGES, 0.2) != make_key("b", MESSAGES, 0.2)
    assert make_key("a", MESSAGES, 0.2) != make_key("a", MESSAGES, 0.7)


def test_cache_key_includes_messages():
    other = [{"role": "user", "content": "年假几天"}]

    assert make_key("a", MESSAGES, 0.2) != make_key("a", other, 0.2)


def test_cache_lru_evicts_oldest():
    cache = ResponseCache(maxsize=2)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k3", "v3")

    assert cache.get("k1") is None
    assert len(cache) == 2


def test_cache_get_refreshes_recency():
    cache = ResponseCache(maxsize=2)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    assert cache.get("k1") == "v1"      # k1 变成最近使用

    cache.put("k3", "v3")

    assert cache.get("k1") == "v1"
    assert cache.get("k2") is None


def test_cache_hit_rate():
    cache = ResponseCache(maxsize=4)
    cache.put("k", "v")
    cache.get("k")
    cache.get("missing")

    assert cache.hits == 1 and cache.misses == 1
    assert cache.hit_rate == 0.5


# ---------------------------------------------------------------- 令牌桶


def test_bucket_burst_then_exhausted():
    now = [0.0]
    bucket = TokenBucket(rate=1.0, capacity=2, clock=lambda: now[0])

    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False


def test_bucket_refills_over_time():
    now = [0.0]
    bucket = TokenBucket(rate=1.0, capacity=2, clock=lambda: now[0])
    bucket.try_acquire()
    bucket.try_acquire()

    now[0] = 1.0                        # 过 1 秒回 1 个令牌

    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False


def test_bucket_acquire_raises_after_timeout():
    bucket = TokenBucket(rate=0.001, capacity=1)
    bucket.try_acquire()

    with pytest.raises(LLMRateLimitError):
        bucket.acquire(timeout=0.05)


# ------------------------------------------------------------------ 降级


def test_fallback_on_rate_limit(monkeypatch):
    def fake_chat(messages, **kwargs):
        if kwargs.get("model") == "primary":
            raise LLMRateLimitError("限流", {"model": "primary"})
        return "降级答案"

    monkeypatch.setattr(client_module, "chat", fake_chat)

    text, degraded = client_module.chat_with_fallback(
        MESSAGES, primary="primary", fallback="backup"
    )

    assert (text, degraded) == ("降级答案", True)


def test_no_fallback_reraises(monkeypatch):
    def fake_chat(messages, **kwargs):
        raise LLMRateLimitError("限流", {"model": "primary"})

    monkeypatch.setattr(client_module, "chat", fake_chat)
    monkeypatch.setattr(client_module, "resolve_fallback_model", lambda: None)

    with pytest.raises(LLMRateLimitError):
        client_module.chat_with_fallback(MESSAGES, primary="primary")


def test_build_messages_places_history_between_system_and_question():
    from src.schema import Chunk

    chunk = Chunk(
        chunk_id="c1", doc_id="d", source="a.md", page_no=1, seq=1, kind="text",
        text="每晚不超过 600 元", char_len=10, department="研发", sensitivity="普通",
        ocr_used=False, low_confidence=False,
    )
    messages = client_module.build_messages(
        "住宿标准？", [chunk], history=[{"role": "user", "content": "上一轮问题"}]
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "上一轮问题"
    assert "<doc id=\"1\">" in messages[-1]["content"]
