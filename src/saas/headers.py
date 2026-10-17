"""Browser security headers for every response.

The page is one static file with a single inline ``<script>`` and no
third-party assets, so scripts are pinned to that script's hash: an
injected ``<script>`` or ``onerror=`` attribute doesn't run even if a
customer value slips past ``esc()``. Inline ``style`` attributes are used
throughout the page, so styles allow ``'unsafe-inline'`` (CSS can't run
code).
"""
from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

from starlette.types import ASGIApp, Message, Receive, Scope, Send


def script_hashes(html: str) -> list[str]:
    return [
        "'sha256-" + base64.b64encode(hashlib.sha256(body.encode()).digest()).decode() + "'"
        for body in re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    ]


def content_security_policy(page: Path) -> str:
    hashes = " ".join(script_hashes(page.read_text(encoding="utf-8")))
    return "; ".join(
        [
            "default-src 'none'",
            f"script-src {hashes}",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data:",
            "connect-src 'self'",
            "form-action 'self'",
            "base-uri 'none'",
            "frame-ancestors 'none'",
        ]
    )


class SecurityHeaders:
    """ASGI middleware adding the headers below to every HTTP response;
    API and auth responses are also marked ``no-store`` so a shared cache
    never keeps one customer's data."""

    def __init__(self, app: ASGIApp, page: Path) -> None:
        self.app = app
        self.headers = [
            (b"content-security-policy", content_security_policy(page).encode()),
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"),
            (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
            (b"cross-origin-opener-policy", b"same-origin"),
            (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
        ]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        no_store = scope["path"] != "/"

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in self.headers if k not in present]
                if no_store and b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)
