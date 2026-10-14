"""Assign each block to a filing section ("Item 7", "Part I Item 2", ...).

The table of contents also contains "Item 1A ..." lines. We pick, for every (part, item), the
heading occurrence that owns the LONGEST span of text before the next heading: a real section
body, never a one-line table-of-contents entry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from finsight.ingestion.parsing.html_parser import Block

FRONT = "Front matter"
_ITEM = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*[\.\:\-\u2013\u2014]?\s*(.*)$", re.I)
_PART = re.compile(r"^\s*part\s+(iv|i{1,3})\b\s*[\.\:\-\u2013\u2014]?\s*(.*)$", re.I)
MAX_HEADING_CHARS = 250

EXPECTED = {
    "10-K": ["Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8"],
    "10-Q": ["Part I Item 1", "Part I Item 2"],
}


@dataclass(frozen=True)
class Candidate:
    index: int  # block index
    key: str  # "Item 7" or "Part I Item 2"
    title: str


def find_candidates(blocks: list[Block], with_part: bool) -> list[Candidate]:
    out: list[Candidate] = []
    part = ""
    for i, b in enumerate(blocks):
        if b.kind != "text" or len(b.text) > MAX_HEADING_CHARS:
            continue
        if (m := _PART.match(b.text)):
            part = f"Part {m.group(1).upper()}"
            continue
        if (m := _ITEM.match(b.text)):
            item = f"Item {m.group(1).upper()}"
            key = f"{part} {item}".strip() if with_part and part else item
            out.append(Candidate(i, key, m.group(2).strip(" .:-")))
    return out


def _span_chars(blocks: list[Block], start: int, end: int) -> int:
    return sum(len(b.text) for b in blocks[start:end] if b.kind == "text")


def assign_sections(blocks: list[Block], form: str) -> list[tuple[str, str]]:
    """Return (section_key, section_title) per block, same length/order as blocks."""
    cands = find_candidates(blocks, with_part=(form == "10-Q"))
    spans: list[tuple[Candidate, int]] = []
    for n, c in enumerate(cands):
        end = cands[n + 1].index if n + 1 < len(cands) else len(blocks)
        spans.append((c, _span_chars(blocks, c.index, end)))

    best: dict[str, tuple[Candidate, int]] = {}
    for c, size in spans:
        if c.key not in best or size > best[c.key][1]:
            best[c.key] = (c, size)
    chosen = sorted((c for c, _ in best.values()), key=lambda c: c.index)

    labels = [(FRONT, "")] * len(blocks)
    for n, c in enumerate(chosen):
        end = chosen[n + 1].index if n + 1 < len(chosen) else len(blocks)
        for i in range(c.index, end):
            labels[i] = (c.key, c.title)
    return labels


def missing_sections(labels: list[tuple[str, str]], form: str) -> list[str]:
    present = {k for k, _ in labels}
    return [s for s in EXPECTED.get(form, []) if s not in present]