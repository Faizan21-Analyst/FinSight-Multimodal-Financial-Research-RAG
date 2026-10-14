from datetime import date
from decimal import Decimal

from finsight.retrieval.sql_retriever import format_value, period_label, row_to_evidence

ROW = dict(ticker="NVDA", metric="rd_expense", tag="ResearchAndDevelopmentExpense", unit="USD",
           period_type="FY", fiscal_year=2024, fiscal_quarter=None,
           period_start=date(2023, 1, 30), period_end=date(2024, 1, 28),
           value=Decimal("8675000000"), accession="0001045810-24-000029", form="10-K",
           filed=date(2024, 2, 21))


def test_format_value():
    assert format_value(Decimal("8675000000"), "USD") == "$8.675 billion"
    assert format_value(Decimal("26044000000"), "USD") == "$26.044 billion"
    assert format_value(Decimal("450000"), "USD") == "$450,000"
    assert format_value(Decimal("1.19"), "USD/shares") == "$1.19 per share"


def test_period_label():
    assert period_label(ROW) == "FY2024"
    assert period_label({**ROW, "period_type": "Q", "fiscal_quarter": 1}) == "Q1 FY2024"
    assert period_label({**ROW, "period_type": "instant", "fiscal_quarter": 4}) == "FY2024 year-end"


def test_row_to_evidence_is_traceable_and_stable():
    ev = row_to_evidence(ROW)
    assert ev.modality.value == "sql" and ev.retriever == "sql"
    assert "8675000000" in ev.content and "10-K" in ev.content and "0001045810-24-000029" in ev.content
    assert ev.source.ticker == "NVDA" and ev.source.fiscal_year == 2024 and ev.source.fiscal_period == "FY"
    assert row_to_evidence(ROW).evidence_id == ev.evidence_id


RESTATED = {**ROW, "value": Decimal("9000000000"), "accession": "0001045810-25-000020",
            "form": "10-K", "filed": date(2025, 2, 26), "restated": True,
            "original_value": Decimal("8675000000"), "original_accession": "0001045810-24-000029",
            "original_form": "10-K", "original_filed": date(2024, 2, 21)}
UNRESTATED = {**ROW, "accession": "0001045810-26-000021", "filed": date(2026, 2, 25),
              "restated": False, "original_value": Decimal("8675000000"),
              "original_accession": "0001045810-24-000029", "original_form": "10-K",
              "original_filed": date(2024, 2, 21)}


def test_unrestated_value_cites_the_original_filing():
    ev = row_to_evidence(UNRESTATED)
    assert "0001045810-24-000029" in ev.content and "0001045810-26-000021" not in ev.content
    assert "NOTE" not in ev.content
    assert ev.source.doc_id.endswith("0001045810-24-000029")


def test_restated_value_cites_latest_filing_and_notes_the_original():
    ev = row_to_evidence(RESTATED)
    assert "$9.000 billion" in ev.content and "0001045810-25-000020" in ev.content
    assert "NOTE: restated" in ev.content and "$8.675 billion" in ev.content
    assert "0001045810-24-000029" in ev.content          # original accession mentioned in the note


def test_as_reported_returns_original_value_and_filing():
    ev = row_to_evidence(RESTATED, as_reported=True)
    assert "$8.675 billion" in ev.content and "0001045810-24-000029" in ev.content
    assert "NOTE" not in ev.content and "$9.000" not in ev.content
    assert row_to_evidence(RESTATED).evidence_id != ev.evidence_id


def test_legacy_rows_without_original_columns_still_work():
    ev = row_to_evidence(ROW)                              # ROW has no original_* keys
    assert "8675000000" in ev.content and "0001045810-24-000029" in ev.content