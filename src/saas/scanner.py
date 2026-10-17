"""Malware scanning behind one interface.

``MALWARE_SCANNER=clamd`` streams each quarantined file to a ClamAV
daemon (``CLAMD_HOST``/``CLAMD_PORT``), which runs in its own container,
not in the web app. ``off`` records files as not scanned. The product
only accepts UTF-8 text (JSON, JSON Lines, CSV), so validation already
rejects executables and archives; scanning is the P1 second layer.
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Protocol

from saas.config import get_settings

_CHUNK = 1 << 16


@dataclass(frozen=True)
class ScanResult:
    clean: bool
    scanned: bool
    signature: str | None = None


class MalwareScanner(Protocol):
    def scan(self, data: bytes) -> ScanResult: ...


class NoScanner:
    def scan(self, data: bytes) -> ScanResult:
        return ScanResult(clean=True, scanned=False)


class ClamdScanner:
    def __init__(self, host: str, port: int, timeout_s: float = 60) -> None:
        self.host, self.port, self.timeout_s = host, port, timeout_s

    def scan(self, data: bytes) -> ScanResult:
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
            sock.sendall(b"zINSTREAM\0")
            for i in range(0, len(data), _CHUNK):
                chunk = data[i:i + _CHUNK]
                sock.sendall(struct.pack("!L", len(chunk)) + chunk)
            sock.sendall(struct.pack("!L", 0))
            reply = b""
            while not reply.endswith(b"\0"):
                part = sock.recv(4096)
                if not part:
                    break
                reply += part
        text = reply.rstrip(b"\0").decode(errors="replace")
        if text.endswith("OK"):
            return ScanResult(clean=True, scanned=True)
        if text.endswith("FOUND"):
            return ScanResult(clean=False, scanned=True, signature=text.split(":", 1)[-1].removesuffix("FOUND").strip())
        raise RuntimeError(f"clamd error: {text[:200]}")


def get_scanner() -> MalwareScanner:
    s = get_settings()
    return ClamdScanner(s.clamd_host, s.clamd_port) if s.malware_scanner == "clamd" else NoScanner()
