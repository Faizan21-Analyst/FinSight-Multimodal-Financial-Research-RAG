"""Typed contracts shared by every module. Components talk only through these."""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    PAGE_IMAGE = "page_image"
    SQL = "sql"


class DocType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    PRESENTATION = "presentation"
    TRANSCRIPT = "transcript"
    XBRL = "xbrl"


class Verdict(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partially_supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient_evidence"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceRef(BaseModel):
    """Where a piece of evidence comes from. Drives citations."""

    doc_id: str
    ticker: str
    doc_type: DocType
    fiscal_year: int | None = None
    fiscal_period: str | None = None  # "FY", "Q1", "Q2", "Q3"
    page: int | None = None
    section: str | None = None  # e.g. "Item 7"


def make_id(*parts: object) -> str:
    """Deterministic short id from parts, so re-ingestion is idempotent."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Chunk(BaseModel):
    chunk_id: str
    text: str
    modality: Modality = Modality.TEXT
    source: SourceRef
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    evidence_id: str
    modality: Modality
    content: str  # text, table markdown, chart description, or SQL result
    source: SourceRef
    score: float = 0.0
    retriever: str = ""  # "dense", "sparse", "hybrid", "colpali", "sql"
    image_path: str | None = None


class Citation(BaseModel):
    evidence_id: str
    source: SourceRef
    snippet: str = ""


class Claim(BaseModel):
    text: str
    supported: bool | None = None  # None = not yet verified
    evidence_ids: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    verdict: Verdict | None = None  # set for cross-modal verification queries
    trace: list[dict[str, Any]] = Field(default_factory=list)