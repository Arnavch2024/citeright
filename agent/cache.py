"""Disk cache for search/fetch. Evals stay reproducible when CACHE_ENABLED=true."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diskcache import Cache


class JsonCache:
    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache: Cache | None = Cache(str(directory)) if enabled else None

    def get(self, key: str) -> Any | None:
        if self._cache is None:
            return None
        return self._cache.get(key)

    def set(self, key: str, value: Any, expire: int = 7 * 24 * 3600) -> None:
        if self._cache is None:
            return
        self._cache.set(key, value, expire=expire)

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
