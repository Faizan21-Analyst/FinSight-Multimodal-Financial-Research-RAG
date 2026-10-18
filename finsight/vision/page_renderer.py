"""Render investor-deck PDFs into page images plus a text layer, and describe each page as a PageRecord.

Convention for deck files (so metadata never has to be typed twice):

    data/raw/decks/<anything>/TICKER_FY2025_Q4.pdf              -> NVDA, FY2025, Q4
    data/raw/decks/<anything>/TICKER_FY2025_Q4_cfo-commentary.pdf  (optional free-text suffix)
    data/raw/decks/<anything>/TICKER_FY2025_investor-day.pdf    -> no fiscal period

Rendering is idempotent: images that already exist are not redrawn.
The text layer (PDF text, not OCR) feeds the OCR-then-text baseline that visual retrieval is
compared against. Text inside raster charts is NOT in it; that is exactly the gap ColPali closes.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from finsight.core.exceptions import ConfigError, IngestionError
from finsight.core.logging import get_logger
from finsight.core.schemas import DocType, make_id

log = get_logger(__name__)

DECK_NAME_RE = re.compile(
    r"^(?P<ticker>[A-Za-z]{1,6})_FY(?P<fy>\d{4})"
    r"(?:_(?P<period>Q[1-4]|FY)(?=_|$))?"
    r"(?:_(?P<rest>.+))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageRecord:
    page_id: str  # 16 hex chars; the indexer turns it into an integer point id
    doc_id: str
    ticker: str
    doc_type: str  # DocType.PRESENTATION.value
    fiscal_year: int | None
    fiscal_period: str | None
    page: int  # 1-based
    image_path: str
    text: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> PageRecord:
        return cls(**json.loads(line))


@dataclass(frozen=True)
class DeckMeta:
    pdf_path: Path
    doc_id: str
    ticker: str
    fiscal_year: int
    fiscal_period: str | None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_deck_filename(pdf_path: Path | str) -> DeckMeta:
    path = Path(pdf_path)
    m = DECK_NAME_RE.match(path.stem)
    if not m:
        raise IngestionError(
            f"Cannot read metadata from deck file name {path.name!r}. "
            "Expected TICKER_FY2025_Q4.pdf (period and a text suffix are optional)."
        )
    period = m.group("period")
    return DeckMeta(
        pdf_path=path,
        doc_id=f"{_slug(path.stem)}-deck",
        ticker=m.group("ticker").upper(),
        fiscal_year=int(m.group("fy")),
        fiscal_period=period.upper() if period else None,
    )


def discover_decks(decks_dir: Path | str, tickers: Sequence[str] | None = None) -> list[DeckMeta]:
    wanted = {t.upper() for t in tickers} if tickers else None
    decks: list[DeckMeta] = []
    for pdf in sorted(Path(decks_dir).rglob("*.pdf")):
        try:
            meta = parse_deck_filename(pdf)
        except IngestionError as e:
            log.warning("skipping %s: %s", pdf, e)
            continue
        if wanted and meta.ticker not in wanted:
            continue
        decks.append(meta)
    return decks


def render_pdf(
    deck: DeckMeta, out_dir: Path | str, dpi: int = 120, max_pages: int | None = None
) -> list[PageRecord]:
    """Render every page to <out_dir>/<doc_id>/pNNN.png and return one PageRecord per page."""
    try:
        import pymupdf as fitz  # current name
    except ImportError:
        try:
            import fitz  # PyMuPDF older than 1.24.3
        except ImportError as e:  # pragma: no cover
            raise ConfigError("PyMuPDF is required for page rendering: pip install pymupdf") from e

    folder = Path(out_dir) / deck.doc_id
    folder.mkdir(parents=True, exist_ok=True)
    records: list[PageRecord] = []
    with fitz.open(str(deck.pdf_path)) as doc:
        for number, page in enumerate(doc, start=1):
            if max_pages and number > max_pages:
                break
            img = folder / f"p{number:03d}.png"
            if not img.exists():
                page.get_pixmap(dpi=dpi).save(str(img))
            records.append(
                PageRecord(
                    page_id=make_id("page", deck.doc_id, number),
                    doc_id=deck.doc_id,
                    ticker=deck.ticker,
                    doc_type=DocType.PRESENTATION.value,
                    fiscal_year=deck.fiscal_year,
                    fiscal_period=deck.fiscal_period,
                    page=number,
                    image_path=img.as_posix(),
                    text=" ".join(page.get_text("text").split()),
                )
            )
    log.info("rendered %s: %d pages", deck.pdf_path.name, len(records))
    return records


def write_manifest(records: Iterable[PageRecord], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")
            n += 1
    return n


def read_manifest(path: Path | str, tickers: Sequence[str] | None = None) -> list[PageRecord]:
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"{path} not found; run scripts/build_pages.py first")
    wanted = {t.upper() for t in tickers} if tickers else None
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = PageRecord.from_json(line)
            if not wanted or rec.ticker in wanted:
                out.append(rec)
    return out
