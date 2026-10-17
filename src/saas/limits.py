"""Request-level resource limits.

``BodySizeLimit`` caps every request body before the multipart parser
spools it to disk -- a size check inside the route would run only after
the whole upload had been written to a temp file. ``RateLimiter`` is a
per-process sliding window; with more than one web replica it moves to
Redis (P1).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException

MULTIPART_OVERHEAD = 1 << 20


class BodySizeLimit:
    def __init__(self, app, max_bytes) -> None:
        # max_bytes: int, or a callable so tests can change the settings
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.max_bytes() if callable(self.max_bytes) else self.max_bytes
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            return await _too_large(send, limit)
        seen = 0
        started = False

        async def counted_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counted_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await _too_large(send, limit)


class _BodyTooLarge(Exception):
    pass


async def _too_large(send, limit: int) -> None:
    body = f'{{"detail":"request body is larger than {limit} bytes"}}'.encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class RateLimiter:
    def __init__(self, limit: int, window_s: float) -> None:
        self.limit, self.window_s = limit, window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                raise HTTPException(status_code=429, detail="too many requests; slow down",
                                    headers={"Retry-After": str(int(self.window_s))})
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
