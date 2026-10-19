"""Chart -> structured data, using a vision-language model (VLM).

Flow: page image -> VLM (JSON only) -> validated ChartData -> markdown table / EvidenceItem.

Design rules:
  * The VLM only reports what is printed or drawn on the slide. No outside knowledge.
  * Unreadable values are null, never guessed.
  * `values_are_estimates` is True when numbers were read off bar heights instead of data labels.
    Module 10 uses it to widen the tolerance when comparing a chart to the 10-K.
  * The VLM is injected (VLMClient protocol), so tests use a fake and the real call is cached on
    disk by image hash (API cost control, rule 4 of the plan).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from finsight.core.caching import DiskCache
from finsight.core.exceptions import ConfigError, FinSightError
from finsight.core.schemas import DocType, EvidenceItem, Modality, SourceRef, make_id
from finsight.generation.numeric import UNIT_MULTIPLIERS, normalize_units
from finsight.vision.page_renderer import PageRecord

SYSTEM_PROMPT = (
    "You extract data from charts on company investor-presentation slides. "
    "Report only what is visibly printed or drawn on the slide. "
    "Never use outside knowledge to fill in or correct numbers."
)

EXTRACTION_PROMPT = """Look at this slide. If it contains a chart (bar, line, pie, stacked, waterfall) or a table-like graphic, extract its data.

Reply with ONE JSON object and nothing else (no prose, no markdown fences):
{
  "has_chart": true,
  "chart_type": "bar | line | pie | stacked_bar | waterfall | table | other",
  "title": "chart title as printed",
  "x_axis": "x-axis label, or what the categories are",
  "y_axis": "y-axis label",
  "unit_scale": "" or "thousand" or "million" or "billion",
  "unit_label": "e.g. USD, %, units",
  "period": "period covered, e.g. Q1 FY25 - Q4 FY25",
  "values_are_estimates": false,
  "series": [
    {"name": "series name", "points": [{"label": "Q1 FY25", "value": 26.0}]}
  ],
  "caveats": []
}

Rules:
- "value" is a plain number (no $, %, commas, or letters). Put the scale in unit_scale ("$26.0B" -> value 26.0, unit_scale "billion").
- Use the numbers printed on the chart. If a value is not printed and you must read it from bar height or line position, set values_are_estimates to true and note it in caveats.
- If a value cannot be read, use null. Never guess.
- If the slide has no chart or table-like graphic, return {"has_chart": false, "series": []}.
- Keep labels exactly as printed (including fiscal-period naming)."""


class ChartExtractionError(FinSightError):
    """The VLM reply could not be turned into valid ChartData."""


class ChartPoint(BaseModel):
    label: str
    value: float | None = None


class ChartSeries(BaseModel):
    name: str = ""
    points: list[ChartPoint] = Field(default_factory=list)


class ChartData(BaseModel):
    has_chart: bool = True
    chart_type: str = "other"
    title: str = ""
    x_axis: str = ""
    y_axis: str = ""
    unit_scale: str = ""
    unit_label: str = ""
    period: str = ""
    values_are_estimates: bool = False
    series: list[ChartSeries] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    @field_validator("unit_scale", mode="before")
    @classmethod
    def _check_scale(cls, v):
        v = str(v or "").strip().lower()
        if v not in UNIT_MULTIPLIERS:
            raise ValueError(f"unit_scale must be '', thousand, million or billion, got {v!r}")
        return v

    def absolute_values(self) -> list[tuple[str, str, Decimal]]:
        """(series, label, value in absolute units). Percentages are left unscaled.
        This is what the cross-modal verifier compares against XBRL."""
        scale = "" if "%" in self.unit_label else self.unit_scale
        return [
            (s.name, p.label, normalize_units(p.value, scale))
            for s in self.series
            for p in s.points
            if p.value is not None
        ]

    def to_markdown(self) -> str:
        if not self.has_chart or not self.series:
            return "(no chart data)"
        labels = list(dict.fromkeys(p.label for s in self.series for p in s.points))
        lookup = [{p.label: p.value for p in s.points} for s in self.series]
        header = "| " + " | ".join([self.x_axis or "label", *[s.name or "value" for s in self.series]]) + " |"
        sep = "|" + "---|" * (len(self.series) + 1)
        rows = []
        for lab in labels:
            cells = ["n/a" if lk.get(lab) is None else f"{lk[lab]:g}" for lk in lookup]
            rows.append("| " + " | ".join([lab, *cells]) + " |")
        return "\n".join([header, sep, *rows])


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_chart_json(raw: str) -> ChartData:
    text = _FENCE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ChartExtractionError("reply contained no JSON object")
    try:
        return ChartData.model_validate(json.loads(text[start : end + 1]))
    except json.JSONDecodeError as e:
        raise ChartExtractionError(f"invalid JSON: {e}") from e
    except ValidationError as e:
        raise ChartExtractionError(f"JSON did not match the schema: {str(e)[:300]}") from e


class VLMClient(Protocol):
    def complete(self, image_path: str, prompt: str, system: str) -> str: ...


class GroqVLM:
    """Vision call to Groq, or to any OpenAI-compatible server via `base_url` (plain httpx, no SDK).

    Examples: Groq (default) with meta-llama/llama-4-scout-17b-16e-instruct, or a local Ollama
    server (base_url="http://localhost:11434/v1", model="qwen2.5vl:7b", no API key needed).

    The free tier is rate limited, so 429 and 5xx replies are retried with backoff (honouring
    Retry-After). Results are cached on disk by (model, prompt, image hash), so each slide costs
    one request, ever. Images over Groq's ~4MB base64 limit are re-encoded as smaller JPEGs.
    """

    URL = "https://api.groq.com/openai/v1/chat/completions"
    MAX_B64_BYTES = 3_500_000
    MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

    def __init__(
        self,
        model: str,
        api_key: str = "",
        cache: DiskCache | None = None,
        max_tokens: int = 900,  # Groq free tier caps this model at 1000 output tokens/minute
        retries: int = 4,
        timeout: float = 90.0,
        client: httpx.Client | None = None,
        sleep=time.sleep,
        base_url: str | None = None,
        extra_body: dict | None = None,
    ):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions" if base_url else self.URL
        self.api_key = api_key
        self.extra_body = extra_body or {}  # per-model request params, e.g. reasoning_effort
        self.cache = cache
        self.max_tokens = max_tokens
        self.retries = retries
        self.timeout = timeout
        self._client = client
        self._sleep = sleep

    @classmethod
    def _shrink_to_jpeg(cls, data: bytes) -> bytes:
        try:
            from PIL import Image
        except ImportError as e:  # pragma: no cover
            raise ConfigError("Image is too large for Groq; pip install pillow to downscale it") from e
        img = Image.open(io.BytesIO(data)).convert("RGB")
        for max_side in (1600, 1280, 1024):
            scaled = img.copy()
            scaled.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            scaled.save(buf, format="JPEG", quality=85)
            if len(buf.getvalue()) * 4 // 3 <= cls.MAX_B64_BYTES:
                return buf.getvalue()
        raise ChartExtractionError("image is still too large for Groq after downscaling")

    def _data_url(self, path: Path) -> str:
        media_type = self.MEDIA_TYPES.get(path.suffix.lower())
        if media_type is None:
            raise ChartExtractionError(f"unsupported image type {path.suffix!r}")
        data = path.read_bytes()
        if len(data) * 4 // 3 > self.MAX_B64_BYTES:
            data, media_type = self._shrink_to_jpeg(data), "image/jpeg"
        return f"data:{media_type};base64,{base64.b64encode(data).decode()}"

    def complete(self, image_path: str, prompt: str, system: str) -> str:
        path = Path(image_path)
        key = DiskCache.key_for(
            "vlm", self.model, system, prompt, self.extra_body,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit["text"]
        if not self.api_key and "api.groq.com" in self.url:
            raise ConfigError("GROQ_API_KEY is not set (put it in .env)")

        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._data_url(path)}},
                    ],
                },
            ],
        }
        body.update(self.extra_body)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        client = self._client or httpx.Client(timeout=self.timeout)
        resp = None
        for attempt in range(self.retries + 1):
            resp = client.post(self.url, json=body, headers=headers)
            if resp.status_code not in (429, 500, 502, 503, 504) or (
                resp.status_code == 429 and "too large" in resp.text.lower()  # retrying cannot help
            ):
                break
            if attempt < self.retries:
                try:
                    wait = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    wait = 2.0 ** (attempt + 1)
                self._sleep(min(wait, 60.0))
        if resp.status_code == 401:
            raise ConfigError("Groq rejected the API key (401). Check GROQ_API_KEY.")
        if resp.status_code != 200:
            raise ChartExtractionError(f"Groq error {resp.status_code}: {resp.text[:200]}")
        text = resp.json()["choices"][0]["message"]["content"] or ""
        if self.cache is not None:
            self.cache.set(key, {"text": text})
        return text


class ChartExtractor:
    def __init__(self, vlm: VLMClient, max_retries: int = 1):
        self.vlm = vlm
        self.max_retries = max_retries

    def extract(self, image_path: str) -> ChartData:
        prompt = EXTRACTION_PROMPT
        last: ChartExtractionError | None = None
        for _ in range(self.max_retries + 1):
            raw = self.vlm.complete(image_path, prompt, SYSTEM_PROMPT)
            try:
                return parse_chart_json(raw)
            except ChartExtractionError as e:
                last = e
                prompt = (
                    f"{EXTRACTION_PROMPT}\n\nYour previous reply was rejected ({e}). "
                    "Reply with the JSON object only."
                )
        raise ChartExtractionError(f"chart extraction failed for {image_path}: {last}") from last


def chart_evidence(chart: ChartData, page: PageRecord) -> EvidenceItem:
    """Wrap an extracted chart as evidence so it flows through fusion and citation like any other."""
    meta = [f"Chart: {chart.title or '(untitled)'}"]
    if chart.period:
        meta.append(f"Period: {chart.period}")
    unit = " ".join(x for x in (chart.unit_scale, chart.unit_label) if x)
    if unit:
        meta.append(f"Units: {unit}")
    if chart.values_are_estimates:
        meta.append("Values were read from the graphic and are approximate.")
    return EvidenceItem(
        evidence_id=make_id("chart", page.page_id),
        modality=Modality.CHART,
        content="\n".join([*meta, "", chart.to_markdown()]),
        source=SourceRef(
            doc_id=page.doc_id,
            ticker=page.ticker,
            doc_type=DocType.PRESENTATION,
            fiscal_year=page.fiscal_year,
            fiscal_period=page.fiscal_period,
            page=page.page,
        ),
        retriever="vlm",
        image_path=page.image_path,
    )
