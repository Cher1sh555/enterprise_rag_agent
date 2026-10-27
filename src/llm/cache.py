"""LLM 响应缓存（LRU）。

key 必须包含 **model + temperature + messages**：
只按 messages 做 key 的话，你调了采样参数还会命中旧答案——
这种"缓存正确但结果过期"的 bug 极难查，因为看上去一切都对。
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

DEFAULT_MAXSIZE = 256


def make_key(model: str, messages: list[dict], temperature: float) -> str:
    payload = json.dumps(
        {"model": model, "temperature": temperature, "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """极简 LRU：命中率统计 + 容量上限。"""

    make_key = staticmethod(make_key)

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize 必须为正")
        self._maxsize = maxsize
        self._data: OrderedDict[str, str] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        if key not in self._data:
            self.misses += 1
            return None
        self.hits += 1
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: str) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
