"""Text LLM client for Groq (or any OpenAI-compatible server) over plain httpx.

Used by the router fallback, grader, planner, answerer and claim verifier (Modules 9-11).

What it does so the rest of the code does not have to:
  * disk cache keyed by the FULL request body (model, messages, params), so identical calls are free
    and reruns of an evaluation cost nothing;
  * retries 429/5xx with backoff, honouring Retry-After (Groq's free tier is rate limited);
  * strips <think>...</think> reasoning text some Qwen/DeepSeek setups emit;
  * JSON replies: complete_json() parses, optionally validates against a pydantic model, and retries
    once with the error message;
  * counts calls, cache hits, tokens and seconds (LLMStats) for the cost/latency columns of the eval.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from finsight.core.caching import DiskCache
from finsight.core.config import Settings, get_settings, load_app_config
from finsight.core.exceptions import ConfigError, FinSightError

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
RETRY_STATUS = (429, 500, 502, 503, 504)


class LLMError(FinSightError):
    """The LLM call failed or returned something unusable."""


_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)  # reply cut off mid-thought
_THINK_STRAY_CLOSE = re.compile(r"\A.*?</think>", re.DOTALL | re.IGNORECASE)  # opening tag was dropped
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _THINK_CLOSED.sub("", text)
    text = _THINK_OPEN.sub("", text)
    if "</think>" in text.lower():
        text = _THINK_STRAY_CLOSE.sub("", text)
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("reply contained no JSON object")
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise LLMError("JSON reply was not an object")
    return obj


@dataclass
class LLMStats:
    calls: int = 0  # real HTTP requests that succeeded
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class LLMResult:
    text: str
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class GroqLLM:
    def __init__(
        self,
        model: str,
        api_key: str = "",
        cache: DiskCache | None = None,
        base_url: str | None = None,
        extra_body: dict | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        retries: int = 4,
        timeout: float = 90.0,
        client: httpx.Client | None = None,
        sleep=time.sleep,
    ):
        self.model = model
        self.api_key = api_key
        self.cache = cache
        self.url = f"{base_url.rstrip('/')}/chat/completions" if base_url else GROQ_URL
        self.extra_body = extra_body or {}  # e.g. {"reasoning_effort": "none"} for Qwen 3.8 on Groq
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.timeout = timeout
        self._client = client
        self._sleep = sleep
        self.stats = LLMStats()

    # ---- one request ------------------------------------------------------------------

    def _post(self, body: dict) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        client = self._client or httpx.Client(timeout=self.timeout)
        resp = None
        for attempt in range(self.retries + 1):
            resp = client.post(self.url, json=body, headers=headers)
            if resp.status_code not in RETRY_STATUS:
                break
            if attempt < self.retries:
                try:
                    wait = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    wait = 2.0 ** (attempt + 1)
                self._sleep(min(wait, 60.0))
        return resp

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        if not self.api_key and "api.groq.com" in self.url:
            raise ConfigError("GROQ_API_KEY is not set (put it in .env)")
        user = prompt
        if json_mode and "json" not in f"{system or ''} {prompt}".lower():
            user += "\n\nRespond with a single JSON object."  # Groq's JSON mode requires the word
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)

        key = DiskCache.key_for("llm", self.model, body)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats.cache_hits += 1
                return LLMResult(hit["text"], True, hit.get("prompt_tokens", 0),
                                 hit.get("completion_tokens", 0))

        t0 = time.time()
        resp = self._post(body)
        elapsed = time.time() - t0
        if resp.status_code == 401:
            raise ConfigError("Groq rejected the API key (401). Check GROQ_API_KEY.")
        if resp.status_code != 200:
            raise LLMError(f"LLM error {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        choice = data["choices"][0]
        text = strip_thinking(choice["message"].get("content") or "")
        if not text:
            raise LLMError(f"empty reply (finish_reason={choice.get('finish_reason')}); "
                           "raise max_tokens or turn thinking off")
        usage = data.get("usage") or {}
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        self.stats.calls += 1
        self.stats.prompt_tokens += pt
        self.stats.completion_tokens += ct
        self.stats.seconds += elapsed
        if self.cache is not None:
            self.cache.set(key, {"text": text, "prompt_tokens": pt, "completion_tokens": ct})
        return LLMResult(text, False, pt, ct, elapsed)

    # ---- JSON replies -----------------------------------------------------------------

    def complete_json(
        self,
        prompt: str,
        system: str | None = None,
        model: type[BaseModel] | None = None,
        retries: int = 1,
        **kwargs: Any,
    ) -> dict[str, Any] | BaseModel:
        """JSON-mode call. Returns a dict, or a validated pydantic object if `model` is given.
        On a bad reply it retries (default once), telling the model what was wrong."""
        current = prompt
        last: Exception | None = None
        for _ in range(retries + 1):
            reply = self.complete(current, system, json_mode=True, **kwargs)
            try:
                obj = parse_json_object(reply.text)
                return model.model_validate(obj) if model else obj
            except (LLMError, ValidationError) as e:
                last = e
                current = (f"{prompt}\n\nYour previous reply was rejected ({str(e)[:200]}). "
                           "Reply with the JSON object only.")
        raise LLMError(f"no valid JSON after {retries + 1} attempts: {last}") from last


def build_llm(
    role: str = "default",
    cfg: dict | None = None,
    settings: Settings | None = None,
    **overrides: Any,
) -> GroqLLM:
    """Build the client from configs/settings.yaml. role: 'default' (answers, verification)
    or 'small' (routing, grading)."""
    cfg = load_app_config() if cfg is None else cfg
    s = settings or get_settings()
    llm_cfg = cfg.get("llm", {})
    model = cfg["models"]["llm_small" if role == "small" else "llm_default"]
    kwargs: dict[str, Any] = {
        "base_url": llm_cfg.get("base_url") or None,
        "extra_body": llm_cfg.get("extra_body") or None,
        "temperature": llm_cfg.get("temperature", 0.2),
        "max_tokens": llm_cfg.get("max_tokens", 1024),
    }
    kwargs.update(overrides)
    return GroqLLM(model, s.groq_api_key, cache=DiskCache(Path(s.cache_dir)), **kwargs)
