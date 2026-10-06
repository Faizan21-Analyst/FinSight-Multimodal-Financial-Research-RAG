"""List and download 10-K / 10-Q filings from EDGAR."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel

from finsight.core.logging import get_logger
from finsight.ingestion.edgar.client import EdgarClient

log = get_logger(__name__)


class FilingRecord(BaseModel):
    doc_id: str
    ticker: str
    cik: str
    form: str  # "10-K" or "10-Q"
    accession: str
    filing_date: str
    report_date: str  # period end date; fiscal year is resolved later from XBRL
    primary_document: str
    url: str
    path: str | None = None
    size_bytes: int | None = None


def submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"


def filing_url(cik: str, accession: str, primary_document: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession.replace('-', '')}/{primary_document}"
    )


def parse_recent_filings(
    submissions: dict,
    ticker: str,
    cik: str,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    cutoff: date | None = None,
) -> list[FilingRecord]:
    """Pure function: submissions JSON -> filing records. Amendments (10-K/A) are excluded."""
    recent = submissions["filings"]["recent"]
    records: list[FilingRecord] = []
    for i in range(len(recent["accessionNumber"])):
        form = recent["form"][i]
        report_date = recent["reportDate"][i]
        if form not in forms or not report_date:
            continue
        if cutoff and date.fromisoformat(report_date) < cutoff:
            continue
        acc = recent["accessionNumber"][i]
        doc = recent["primaryDocument"][i]
        records.append(
            FilingRecord(
                doc_id=f"{ticker.lower()}-{form.lower()}-{report_date}",
                ticker=ticker,
                cik=cik,
                form=form,
                accession=acc,
                filing_date=recent["filingDate"][i],
                report_date=report_date,
                primary_document=doc,
                url=filing_url(cik, acc, doc),
            )
        )
    return sorted(records, key=lambda r: r.report_date)


def download_filing(client: EdgarClient, rec: FilingRecord, raw_dir: Path) -> FilingRecord:
    """Download one filing. Idempotent: existing non-empty files are skipped."""
    suffix = Path(rec.primary_document).suffix or ".htm"
    dest = raw_dir / rec.ticker / f"{rec.form}_{rec.report_date}_{rec.accession.replace('-', '')}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("skip   %s", dest.name)
    else:
        log.info("fetch  %s", dest.name)
        dest.write_bytes(client.get(rec.url).content)
    return rec.model_copy(update={"path": str(dest), "size_bytes": dest.stat().st_size})