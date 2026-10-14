"""Parse an EDGAR 10-K/10-Q HTML (inline XBRL) file into an ordered list of Blocks.

A Block is either a paragraph of text or a data table, tagged with a page number.

Notes (log surprises in docs/failures.md):
  * EDGAR HTML has no real pages. "page" = 1 + number of CSS page-break markers seen so far.
    It tracks the printed page closely but is not guaranteed identical; treat it as a locator.
  * The hidden <ix:header> block (XBRL contexts) and display:none elements are removed.
  * Tables are atomic. Layout tables (single column / single row) become plain text blocks.
  * Page furniture ("Table of Contents", bare page numbers) is dropped.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning
from finsight.ingestion.parsing.tables import clean_table, is_data_table

# Some EDGAR files start with an <?xml ...?> declaration; we parse them as HTML on purpose.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
BLOCK_TAGS = ["div", "p", "table", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]
SKIP_TAGS = {"script", "style", "head", "title", "meta", "link", "ix:header", "noscript"}
_BREAK_BEFORE = re.compile(r"page-break-before\s*:\s*always|break-before\s*:\s*page")
_BREAK_AFTER = re.compile(r"page-break-after\s*:\s*always|break-after\s*:\s*page")
_HIDDEN = re.compile(r"display\s*:\s*none")
_NOISE = re.compile(
    r"^(\d{1,3}|page \d+( of \d+)?|table of contents|back to (contents|index))$", re.I
)


@dataclass
class Block:
    kind: str  # "text" | "table"
    page: int
    text: str = ""  # paragraph text (kind == "text")
    rows: list[list[str]] = field(default_factory=list)  # cleaned rows (kind == "table")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _style(el: Tag) -> str:
    return (el.get("style") or "").lower()


def _table_rows(table: Tag) -> list[list[str]]:
    rows = []
    for tr in table.find_all("tr"):
        rows.append([normalize(td.get_text(" ")) for td in tr.find_all(["td", "th"])])
    return rows


def parse_html(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    root = soup.body or soup
    blocks: list[Block] = []
    page = 1

    def emit_text(text: str) -> None:
        text = normalize(text)
        if text and not _NOISE.match(text):
            blocks.append(Block("text", page, text=text))

    stack: list[tuple[str, object]] = [("enter", root)]
    while stack:
        action, item = stack.pop()
        if action == "text":
            emit_text(str(item))
            continue
        el: Tag = item  # type: ignore[assignment]
        if action == "exit":
            if _BREAK_AFTER.search(_style(el)):
                page += 1
            continue

        name = (el.name or "").lower()
        style = _style(el)
        if name in SKIP_TAGS or _HIDDEN.search(style):
            continue
        if _BREAK_BEFORE.search(style):
            page += 1
        if name == "hr":
            if _BREAK_AFTER.search(style) or _BREAK_BEFORE.search(style):
                page += 1
            continue

        stack.append(("exit", el))

        if name == "table":
            rows = clean_table(_table_rows(el))
            if is_data_table(rows):
                blocks.append(Block("table", page, rows=rows))
            else:  # layout table: keep its words as text
                for r in rows:
                    emit_text(" ".join(r))
            continue

        if el.find(BLOCK_TAGS) is None:  # leaf block: one paragraph
            emit_text(el.get_text(" "))
            continue

        for child in reversed(list(el.children)):
            if isinstance(child, Tag):
                stack.append(("enter", child))
            elif type(child) is NavigableString and child.strip():  # excludes Comment/Doctype
                stack.append(("text", str(child)))
    return blocks