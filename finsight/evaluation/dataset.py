"""Evaluation questions with PREDICATE-based gold labels.

A gold label is not a chunk id; it is a description of what an answer-bearing chunk looks like:
"a table chunk from an NVDA filing that contains the number 8,675 and the phrase
'research and development'". The harness resolves it against the current corpus, so
re-chunking, re-indexing or changing the chunk size never invalidates the benchmark.

A question has one or more GoldSpecs. Every spec needs at least one hit, which lets a
comparison question ("NVIDIA vs AMD R&D") demand evidence for BOTH companies.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

_NUMBER_PHRASE = re.compile(r"^[\d,.]+$")


def normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).lower()


def _has_phrase(norm_text: str, phrase: str) -> bool:
    p = normalize(phrase)
    if _NUMBER_PHRASE.match(p):  # whole-number match: "8,675" must not hit "18,675" or "8,6751"
        return re.search(rf"(?<![\d,.]){re.escape(p)}(?![\d,]|\.\d)", norm_text) is not None
    return p in norm_text


class GoldSpec(BaseModel):
    ticker: str | None = None
    doc_type: str | None = None  # "10-K" | "10-Q"
    fiscal_year: int | None = None
    fiscal_period: str | None = None  # "FY" | "Q1" ...
    section: str | None = None  # prefix match: "Item 1A", "Part I"
    modality: str | None = None  # "text" | "table"
    all_of: list[str] = Field(default_factory=list)  # every phrase must appear
    any_of: list[str] = Field(default_factory=list)  # at least one must appear (if non-empty)

    def matches(self, chunk: dict) -> bool:
        for field in ("ticker", "doc_type", "fiscal_year", "fiscal_period", "modality"):
            want = getattr(self, field)
            if want is not None and chunk.get(field) != want:
                return False
        if self.section is not None and not (chunk.get("section") or "").startswith(self.section):
            return False
        text = normalize(chunk.get("text", ""))
        if any(not _has_phrase(text, p) for p in self.all_of):
            return False
        return not self.any_of or any(_has_phrase(text, p) for p in self.any_of)


class OracleFilter(BaseModel):
    """What a perfect query-understanding step would extract. Used for the 'oracle' runs that
    show how much a router/filter step could gain, without building the router yet."""

    tickers: list[str] | None = None
    fiscal_years: list[int] | None = None
    doc_types: list[str] | None = None


class Question(BaseModel):
    id: str
    category: str
    question: str
    gold: list[GoldSpec]
    oracle: OracleFilter | None = None


def resolve_gold(corpus: Iterable[dict], spec: GoldSpec) -> set[str]:
    """Chunk ids (payload dicts as stored in Qdrant) satisfying the spec."""
    return {c["chunk_id"] for c in corpus if spec.matches(c)}


def load_questions(*paths: Path) -> list[Question]:
    out: list[Question] = []
    for path in paths:
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    out.append(Question.model_validate_json(line))
                except Exception as e:  # noqa: BLE001
                    raise ValueError(f"{path}:{n}: {e}") from e
    ids = [q.id for q in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate question ids: {sorted(dupes)}")
    return out


def save_questions(path: Path, questions: Iterable[Question]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(q.model_dump_json(exclude_none=True) for q in questions) + "\n", encoding="utf-8"
    )