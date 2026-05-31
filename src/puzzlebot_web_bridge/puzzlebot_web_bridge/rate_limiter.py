"""Simple token-bucket-style rate limiter based on wall-clock time."""

import time


class RateLimiter:
    """Allow at most `max_hz` calls per second. Thread-safe via GIL."""

    def __init__(self, max_hz: float):
        self._min_interval = (1.0 / max_hz) if max_hz > 0 else 0.0
        self._last_sent = 0.0

    def should_send(self) -> bool:
        if self._min_interval == 0.0:
            return True
        now = time.monotonic()
        if now - self._last_sent >= self._min_interval:
            self._last_sent = now
            return True
        return False
