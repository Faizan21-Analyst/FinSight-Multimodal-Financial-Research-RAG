"""Polite EDGAR HTTP client: required User-Agent, rate limiting, retry with backoff."""
from __future__ import annotations

import time

import httpx

from finsight.core.exceptions import IngestionError
from finsight.core.logging import get_logger

log = get_logger(__name__)
RETRY_STATUS = {429, 500, 502, 503, 504}


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        min_interval: float = 0.15,  # SEC limit is 10 req/s; stay well under
        max_retries: int = 4,
        backoff_base: float = 1.0,
        client: httpx.Client | None = None,
    ):
        if not user_agent.strip():
            raise IngestionError("SEC_USER_AGENT is empty. EDGAR requires 'Name email@example.com'.")
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._last = 0.0
        self._client = client or httpx.Client(timeout=60, follow_redirects=True)
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get(self, url: str) -> httpx.Response:
        err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._client.get(url, headers=self._headers)
            except httpx.TransportError as e:
                err = e
            else:
                if resp.status_code not in RETRY_STATUS:
                    if resp.status_code >= 400:
                        raise IngestionError(f"HTTP {resp.status_code} for {url}")
                    return resp
                err = IngestionError(f"HTTP {resp.status_code} for {url}")
            if attempt < self.max_retries:
                delay = self.backoff_base * (2**attempt)
                log.warning("Retry %d for %s (%s) in %.1fs", attempt + 1, url, err, delay)
                time.sleep(delay)
        raise IngestionError(f"Failed after {self.max_retries} retries: {url}: {err}")

    def get_json(self, url: str) -> dict:
        return self.get(url).json()