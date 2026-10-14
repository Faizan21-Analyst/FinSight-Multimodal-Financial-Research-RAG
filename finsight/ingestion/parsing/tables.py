"""Turn messy EDGAR HTML table rows into clean rows and markdown. Pure functions."""
from __future__ import annotations

import re

_TRAILING = {")", "%", ")%", "%)"}
_LEADING = {"$", "(", "($", "$("}
_NUMERIC_CELL = re.compile(r"^[\s$()\d,.%]+$")


def approx_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def clean_row(cells: list[str]) -> list[str]:
    """Drop empty cells and re-attach the "$", "(", ")" and "%" fragments EDGAR splits off.

    ["Revenue", "$", "60,922", "", "(1,234", ")"]  ->  ["Revenue", "$60,922", "(1,234)"]
    """
    out: list[str] = []
    for raw in cells:
        c = re.sub(r"\s+", " ", raw.replace("\xa0", " ")).strip()
        if not c:
            continue
        if _NUMERIC_CELL.match(c) and any(ch.isdigit() for ch in c):
            c = re.sub(r"\s+", "", c)  # "( 45 )" -> "(45)", "$ 1,234" -> "$1,234", "74.0 %" -> "74.0%"
        if out and c in _TRAILING:
            out[-1] += c
        elif out and out[-1] in _LEADING:
            out[-1] += c
        else:
            out.append(c)
    return out

def clean_table(rows: list[list[str]]) -> list[list[str]]:
    cleaned = [clean_row(r) for r in rows]
    return [r for r in cleaned if r]


def is_data_table(rows: list[list[str]]) -> bool:
    """>=2 non-empty rows and at least one row with >=2 cells. Everything else is layout."""
    return len(rows) >= 2 and max(len(r) for r in rows) >= 2


def _pad(row: list[str], width: int) -> list[str]:
    """Pad a short row to `width`. A row starting with a digit (a column header such as
    "Jan 28, 2024" whose blank label cell was dropped) is left-padded so values stay under
    their columns; label rows are right-padded."""
    gap = width - len(row)
    if gap <= 0:
        return row
    if any(ch.isdigit() for ch in row[0]):
        return [""] * gap + row
    return row + [""] * gap


def to_markdown(rows: list[list[str]]) -> str:
    esc = lambda c: c.replace("|", "\\|")  # noqa: E731
    width = max(len(r) for r in rows)
    padded = [_pad(r, width) for r in rows]
    lines = ["| " + " | ".join(esc(c) for c in padded[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in padded[1:]]
    return "\n".join(lines)


def split_table(rows: list[list[str]], max_tokens: int) -> list[list[list[str]]]:
    """Split a long table by rows; every part repeats the header row."""
    if approx_tokens(to_markdown(rows)) <= max_tokens or len(rows) <= 2:
        return [rows]
    header, body = rows[0], rows[1:]
    parts: list[list[list[str]]] = []
    cur: list[list[str]] = []
    for r in body:
        trial = [header, *cur, r]
        if cur and approx_tokens(to_markdown(trial)) > max_tokens:
            parts.append([header, *cur])
            cur = []
        cur.append(r)
    if cur:
        parts.append([header, *cur])
    return parts