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