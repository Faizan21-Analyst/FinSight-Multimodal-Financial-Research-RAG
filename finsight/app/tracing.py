"""Request traces: one JSON file per question, so any answer can be audited after the fact.

A trace holds the question, the routing decision, every agent step (with timestamps), the evidence,
the claim verdicts, the confidence reasons, latency per step and LLM usage (calls, cache hits, tokens).
Files live in data/traces/ and are named by request id. Reading validates the id, so a request like
`/traces/../../.env` cannot escape the folder.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


def step_seconds(trace: list[dict], started: float | None) -> list[dict]:
    """Seconds spent in each step. Records carry the time each step FINISHED ('ts'), so a step's
    duration is its timestamp minus the previous one (the first is measured from `started`)."""
    out: list[dict] = []
    prev = started
    for rec in trace:
        ts = rec.get("ts")
        if ts is None:
            continue
        if prev is not None:
            out.append({"step": rec.get("step"), "seconds": round(ts - prev, 3)})
        prev = ts
    return out


class Tracer:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    @staticmethod
    def new_id() -> str:
        return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    def path_for(self, request_id: str) -> Path | None:
        return self.dir / f"{request_id}.json" if _ID.match(request_id or "") else None

    def write(self, request_id: str, payload: dict[str, Any]) -> Path | None:
        path = self.path_for(request_id)
        if path is None:
            return None
        if payload.get("trace") and payload.get("started_at") is not None:
            payload = {**payload, "step_seconds": step_seconds(payload["trace"], payload["started_at"])}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError:
            return None  # tracing must never break an answer
        return path

    def read(self, request_id: str) -> dict[str, Any] | None:
        path = self.path_for(request_id)
        if path is None or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
