"""Claims: split an answer into checkable statements, read its [E#] citations, find its numbers.

All deterministic (no LLM). The answerer is told to cite every claim as [E1], [E2], ... and to quote
figures exactly as they appear in the evidence; this module recovers the structure so the verifier can
check each claim against the evidence it cites.

A sentence is a claim when it carries a citation label or a checkable number. Headings, transitions
("Here is the comparison") and other filler carry neither and are dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from finsight.core.schemas import Verdict
from finsight.generation.numeric import UNIT_MULTIPLIERS

# ---- citations --------------------------------------------------------------------------

_LABEL_GROUP = re.compile(r"\[(E\d+(?:\s*[,;]\s*E?\d+)*)\]", re.I)
_LABELS_AFTER_PERIOD = re.compile(r"([.!?])\s+((?:\[E\d+(?:\s*[,;]\s*E?\d+)*\]\s*)+)", re.I)


def parse_labels(text: str) -> list[str]:
    """['E1', 'E2'] for '... [E1] ... [E2, E3]' style citations, in order, deduplicated."""
    out: list[str] = []
    for m in _LABEL_GROUP.finditer(text):
        for n in re.findall(r"\d+", m.group(1)):
            lab = f"E{int(n)}"
            if lab not in out:
                out.append(lab)
    return out


def strip_labels(text: str) -> str:
    return re.sub(r"\s+", " ", _LABEL_GROUP.sub("", text)).strip()


# ---- splitting --------------------------------------------------------------------------

_BULLET = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s+")
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])(?<!\b[A-Z]\.)\s+(?=[A-Z"“*])')


def _clean_line(line: str) -> str:
    line = _BULLET.sub("", line)
    return line.replace("**", "").replace("__", "").replace("`", "").strip()


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        # "... 2024. [E1]" -> "... 2024 [E1]." so the label stays with its sentence
        line = _LABELS_AFTER_PERIOD.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", line)
        out.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip())
    return out


# ---- numbers ----------------------------------------------------------------------------

_NUM = re.compile(
    r"(?P<dollar>\$)?\s?(?P<num>\d[\d,]*(?:\.\d+)?)\s?"
    r"(?P<unit>trillion|billion|million|thousand|bn|mm|%|percent|[kmb](?![a-z]))?",
    re.I,
)
_NOISE = re.compile(r"\b(?:FY|Q)\s?\d+\b|\bE\d+\b", re.I)  # Q2, FY2024, E1 are not quantities


@dataclass(frozen=True)
class ClaimNumber:
    raw: str
    value: Decimal   # absolute units (dollars), or the plain number for a percentage
    half: Decimal    # half a unit of the last stated digit: how loosely a rounded figure may match
    kind: str        # "money" | "pct" | "count"


def checkable_numbers(text: str) -> list[ClaimNumber]:
    """Quantities worth checking: anything with $, %, or a scale word, plus bare numbers >= 1000.
    Years (1900-2100), small bare integers (dates, counts like '2 companies') and period tokens
    (Q2, FY2024) are skipped: they identify, they do not assert."""
    t = _NOISE.sub(" ", _LABEL_GROUP.sub(" ", text))
    out: list[ClaimNumber] = []
    for m in _NUM.finditer(t):
        raw = m.group("num").replace(",", "")
        unit = (m.group("unit") or "").lower()
        base = Decimal(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        half = Decimal("0.5") * (Decimal(10) ** -decimals)
        if unit in ("%", "percent"):
            kind, mult = "pct", Decimal(1)
        elif unit:
            kind, mult = "money", UNIT_MULTIPLIERS[unit]
        elif m.group("dollar"):
            kind, mult = "money", Decimal(1)
        else:
            if decimals == 0 and 1900 <= base <= 2100:
                continue
            if abs(base) < 1000:
                continue
            kind, mult = "count", Decimal(1)
        out.append(ClaimNumber(m.group(0).strip(), base * mult, half * mult, kind))
    return out


def extract_claims(answer: str) -> list[str]:
    return [s for s in split_sentences(answer) if parse_labels(s) or checkable_numbers(s)]


# ---- result type shared by the verifier and the confidence rules ------------------------

# severe flags mean the answer cannot be trusted as written
SEVERE_FLAGS = {"invalid_citation", "number_not_in_evidence", "uncited"}


@dataclass
class ClaimCheck:
    text: str
    labels: list[str]
    evidence_ids: list[str]
    verdict: Verdict
    flags: list[str] = field(default_factory=list)
    reason: str = ""
    judged_by: str = "rules"   # "rules" | "numbers" | "llm"
