import httpx
import pytest

from finsight.core.exceptions import IngestionError
from finsight.ingestion.edgar.client import EdgarClient
from finsight.ingestion.edgar.filings import (
    FilingRecord, download_filing, filing_url, parse_recent_filings, submissions_url,
)
from datetime import date

SUBS = {
    "filings": {
        "recent": {
            "accessionNumber": ["0001-24-000001", "0001-24-000002", "0001-23-000003", "0001-24-000004"],
            "form": ["10-K", "10-Q", "10-K/A", "8-K"],
            "filingDate": ["2024-02-21", "2024-05-22", "2023-03-01", "2024-06-01"],
            "reportDate": ["2024-01-28", "2024-04-28", "2022-12-31", ""],
            "primaryDocument": ["a10k.htm", "a10q.htm", "a10ka.htm", "x8k.htm"],
        }
    }
}


def _client(handler, **kw):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return EdgarClient("Test t@example.com", min_interval=0, backoff_base=0, client=http, **kw)


def test_urls():
    assert submissions_url("1045449").endswith("CIK0001045449.json")
    assert filing_url("0001045449", "0001-24-000001", "a.htm").endswith("/1045449/000124000001/a.htm")


def test_parse_filters_forms_and_amendments():
    recs = parse_recent_filings(SUBS, "NVDA", "0001045449")
    assert [r.form for r in recs] == ["10-K", "10-Q"]
    assert recs[0].doc_id == "nvda-10-k-2024-01-28"


def test_parse_respects_cutoff():
    recs = parse_recent_filings(SUBS, "NVDA", "0001045449", cutoff=date(2024, 3, 1))
    assert [r.form for r in recs] == ["10-Q"]


def test_client_retries_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503 if calls["n"] < 3 else 200, json={"ok": True})

    assert _client(handler).get_json("https://x.test/a") == {"ok": True}
    assert calls["n"] == 3


def test_client_raises_on_404_without_retry():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(404)

    with pytest.raises(IngestionError):
        _client(handler).get("https://x.test/a")
    assert calls["n"] == 1


def test_empty_user_agent_rejected():
    with pytest.raises(IngestionError):
        EdgarClient("  ")


def test_download_is_idempotent(tmp_path):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(200, content=b"<html>filing</html>")

    rec = parse_recent_filings(SUBS, "NVDA", "0001045449")[0]
    c = _client(handler)
    r1 = download_filing(c, rec, tmp_path)
    r2 = download_filing(c, rec, tmp_path)
    assert calls["n"] == 1
    assert r1.size_bytes == r2.size_bytes == len(b"<html>filing</html>")