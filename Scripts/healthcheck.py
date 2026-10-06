#!/usr/bin/env python3
"""Verify Qdrant and Postgres are reachable. Exit code 1 if either is down."""
import socket
import sys

import httpx

sys.path.insert(0, "src")
from finsight.core.config import get_settings  # noqa: E402


def check_qdrant(url: str) -> bool:
    try:
        return httpx.get(f"{url}/healthz", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def check_port(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    s = get_settings()
    results = {
        "qdrant": check_qdrant(s.qdrant_url),
        "postgres": check_port(s.postgres_host, s.postgres_port),
    }
    for name, ok in results.items():
        print(f"{'OK  ' if ok else 'FAIL'} {name}")
    sys.exit(0 if all(results.values()) else 1)