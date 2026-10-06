"""Disk cache for LLM/VLM calls, keyed by input hash. Saves cost, makes tests deterministic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, directory: Path | str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def set(self, key: str, value: Any) -> None:
        self._path(key).write_text(json.dumps(value))