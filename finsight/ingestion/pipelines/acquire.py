"""Module 1 pipeline: download filings + XBRL, write manifest and data inventory."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from finsight.core.config import get_settings, load_companies
from finsight.core.logging import get_logger
from finsight.ingestion.edgar.client import EdgarClient
from finsight.ingestion.edgar.filings import (
    FilingRecord,
    download_filing,
    parse_recent_filings,
    submissions_url,
)
from finsight.ingestion.edgar.xbrl import download_companyfacts

log = get_logger(__name__)


def load_manifest(path: Path) -> dict[str, FilingRecord]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["doc_id"]: FilingRecord(**r) for r in rows}


def save_manifest(path: Path, records: dict[str, FilingRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda r: (r.ticker, r.report_date, r.form))
    path.write_text("\n".join(r.model_dump_json() for r in ordered) + "\n")


def _count_manual(raw_dir: Path, ticker: str, sub: str, pattern: str) -> int:
    d = raw_dir / ticker / sub
    return len(list(d.glob(pattern))) if d.exists() else 0


def write_inventory(
    path: Path, records: dict[str, FilingRecord], raw_dir: Path, tickers: list[str]
) -> None:
    lines = [
        "# Data inventory",
        "",
        f"Generated {date.today().isoformat()} by `scripts/download_data.py`.",
        "",
        "## SEC filings (automatic)",
        "",
        "| Ticker | Form | Period end | Filed | Size (KB) | File |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(records.values(), key=lambda r: (r.ticker, r.report_date, r.form)):
        kb = (r.size_bytes or 0) // 1024
        lines.append(
            f"| {r.ticker} | {r.form} | {r.report_date} | {r.filing_date} | {kb} | `{Path(r.path or '').name}` |"
        )
    lines += [
        "",
        "## Manual downloads (investor presentations, transcripts)",
        "",
        "Put PDFs in `data/raw/<TICKER>/presentations/` and `data/raw/<TICKER>/transcripts/`.",
        "",
        "| Ticker | Presentations (PDF) | Transcripts (PDF/TXT) | XBRL companyfacts |",
        "|---|---|---|---|",
    ]
    for t in tickers:
        pres = _count_manual(raw_dir, t, "presentations", "*.pdf")
        trans = _count_manual(raw_dir, t, "transcripts", "*.*")
        facts = "yes" if (raw_dir / t / "companyfacts.json").exists() else "MISSING"
        lines.append(f"| {t} | {pres} | {trans} | {facts} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(years: int = 3, tickers: list[str] | None = None, refresh: bool = False) -> None:
    settings = get_settings()
    client = EdgarClient(settings.require_sec_user_agent())
    raw_dir = Path(settings.data_dir) / "raw"
    manifest_path = Path(settings.data_dir) / "manifests" / "filings.jsonl"

    companies = load_companies()
    if tickers:
        wanted = {t.upper() for t in tickers}
        companies = [c for c in companies if c.ticker in wanted]

    cutoff = date.today() - timedelta(days=365 * years + 45)
    manifest = load_manifest(manifest_path)

    for c in companies:
        log.info("=== %s (CIK %s) ===", c.ticker, c.cik)
        subs = client.get_json(submissions_url(c.cik))
        for rec in parse_recent_filings(subs, c.ticker, c.cik, cutoff=cutoff):
            manifest[rec.doc_id] = download_filing(client, rec, raw_dir)
        download_companyfacts(client, c.ticker, c.cik, raw_dir, refresh=refresh)
        save_manifest(manifest_path, manifest)  # save per company so a crash loses nothing

    write_inventory(Path("docs") / "data_inventory.md", manifest, raw_dir, [c.ticker for c in companies])
    n10k = sum(1 for r in manifest.values() if r.form == "10-K")
    n10q = sum(1 for r in manifest.values() if r.form == "10-Q")
    log.info("Done: %d 10-K, %d 10-Q in manifest", n10k, n10q)