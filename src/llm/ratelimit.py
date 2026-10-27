"""令牌桶限流。

两个方法，用途严格区分：

- `try_acquire()`：不等待，立刻返回 True/False —— **测试用它**；
- `acquire()`：拿不到就短暂轮询，超时抛 `LLMRateLimitError` —— **生产用它**。

时钟可注入（默认 `time.monotonic`）：测试传假时钟，既能验证"补充后恢复"，
又不用 `sleep`，也不会因为机器慢而 flaky。

注意：`acquire()` 内部用真实 `time.sleep` 推进等待，只有在真实时钟下才有意义；
测试请只用 `try_acquire()`，否则假时钟永远不前进。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.utils.exceptions import LLMRateLimitError

_POLL_INTERVAL = 0.01


class TokenBucket:
    def __init__(
        self,
        rate: float,
        capacity: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate 与 capacity 必须为正")
        self._rate = float(rate)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._clock = clock
        self._last = clock()

    @property
    def tokens(self) -> float:
        return self._tokens

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def acquire(self, tokens: float = 1.0, timeout: float = 5.0) -> None:
        deadline = self._clock() + timeout
        while True:
            if self.try_acquire(tokens):
                return
            if self._clock() >= deadline:
                raise LLMRateLimitError(
                    "触发本地限流",
                    {"rate": self._rate, "capacity": self._capacity, "tokens": round(self._tokens, 3)},
                )
            time.sleep(_POLL_INTERVAL)
