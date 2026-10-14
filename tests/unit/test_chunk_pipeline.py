import json

from finsight.ingestion.pipelines.chunk_filings import fiscal_labels, fye_months, run


def _manifest(tmp_path, tenk_html, tenq_html):
    k = tmp_path / "k.htm"; k.write_text(tenk_html, encoding="utf-8")
    q = tmp_path / "q.htm"; q.write_text(tenq_html, encoding="utf-8")
    rows = [
        dict(doc_id="acme-10-k-2024-01-28", ticker="ACME", form="10-K", report_date="2024-01-28", path=str(k)),
        dict(doc_id="acme-10-q-2024-04-28", ticker="ACME", form="10-Q", report_date="2024-04-28", path=str(q)),
        dict(doc_id="acme-10-q-2024-07-28", ticker="ACME", form="10-Q", report_date="2024-07-28", path=str(tmp_path / "gone.htm")),
    ]
    m = tmp_path / "filings.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return m, rows


def test_fiscal_labels(tmp_path, tenk_html, tenq_html):
    _, rows = _manifest(tmp_path, tenk_html, tenq_html)
    fye = fye_months(rows)
    assert fye == {"ACME": 1}
    assert fiscal_labels(rows[0], 1) == (2024, "FY")
    assert fiscal_labels(rows[1], 1) == (2025, "Q1")


def test_pipeline_writes_jsonl_skips_missing_and_is_idempotent(tmp_path, tenk_html, tenq_html):
    m, _ = _manifest(tmp_path, tenk_html, tenq_html)
    out = tmp_path / "chunks"
    s = run(m, out)
    assert {x.doc_id for x in s} == {"acme-10-k-2024-01-28", "acme-10-q-2024-04-28"}
    assert (out / "acme-10-k-2024-01-28.jsonl").exists()
    assert not (out / "acme-10-q-2024-07-28.jsonl").exists()
    first = json.loads((out / "acme-10-q-2024-04-28.jsonl").read_text().splitlines()[0])
    assert first["source"]["fiscal_year"] == 2025 and first["source"]["fiscal_period"] == "Q1"
    assert run(m, out) == []                               # second run: everything skipped
    assert len(run(m, out, force=True)) == 2