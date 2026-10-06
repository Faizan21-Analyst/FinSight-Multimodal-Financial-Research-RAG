"""Download XBRL companyfacts (structured financial data) for a company."""
from __future__ import annotations

import json
from pathlib import Path

from finsight.core.exceptions import IngestionError
from finsight.core.logging import get_logger
from finsight.ingestion.edgar.client import EdgarClient

log = get_logger(__name__)


def companyfacts_url(cik: str) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"


def download_companyfacts(
    client: EdgarClient, ticker: str, cik: str, raw_dir: Path, refresh: bool = False
) -> Path:
    dest = raw_dir / ticker / "companyfacts.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        log.info("skip   %s/companyfacts.json", ticker)
        return dest
    log.info("fetch  %s/companyfacts.json", ticker)
    data = client.get_json(companyfacts_url(cik))
    if "facts" not in data:
        raise IngestionError(f"companyfacts for {ticker} has no 'facts' key")
    dest.write_text(json.dumps(data))
    return dest