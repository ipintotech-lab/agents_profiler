"""Private object storage for customer files and results.

Keys are built only from server-generated ids --
``tenants/<tenant>/<area>/<object>`` -- and checked against that shape
before every operation, so no customer value (filename, path) ever
reaches the filesystem. ``LocalObjectStorage`` keeps objects under
``STORAGE_DIR`` (a private volume, files mode 0600); an S3-compatible
store with presigned one-object URLs is the P1 implementation of the same
interface.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

from saas.config import get_settings

AREAS = ("quarantine", "uploads", "results")
_KEY = re.compile(r"tenants/[0-9a-f]{32}/(quarantine|uploads|results)/[0-9a-f]{32}")


def object_key(tenant_id: str, area: str, object_id: str) -> str:
    key = f"tenants/{tenant_id}/{area}/{object_id}"
    check_key(key)
    return key


def check_key(key: str) -> None:
    if not _KEY.fullmatch(key):
        raise ValueError("invalid storage key")


class ObjectStorage(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def move(self, src: str, dst: str) -> None: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class LocalObjectStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        check_key(key)
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("invalid storage key")
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def move(self, src: str, dst: str) -> None:
        target = self._path(dst)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.replace(self._path(src), target)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


def get_storage() -> ObjectStorage:
    return LocalObjectStorage(get_settings().storage_dir)
