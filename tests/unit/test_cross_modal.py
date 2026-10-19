"""Module 10 tests: period parsing, tolerance, segment-vs-consolidated guard, verdict aggregation."""
from decimal import Decimal

from finsight.agents.cross_modal import (
    CrossModalVerifier,
    PointCheck,
    amounts_in_text,
    decimals_of,
    fact_value,
    is_consolidated,
    overall_verdict,
    parse_period,
    text_supports,
    tolerance,
)
from finsight.core.schemas import DocType, EvidenceItem, Modality, SourceRef, Verdict
from finsight.vision.chart_extractor import ChartData, ChartPoint, ChartSeries
from finsight.vision.page_renderer import PageRecord


def page(ticker="AMD"):
    return PageRecord(page_id="p1", doc_id="amd-deck", ticker=ticker, doc_type="presentation",
                      fiscal_year=2026, fiscal_period="Q2", page=5, image_path="x.png", text="")


def chart(series, unit_scale="million", title="", estimates=False, unit_label="USD"):
    return ChartData(title=title, unit_scale=unit_scale, unit_label=unit_label,
                     values_are_estimates=estimates,
                     series=[ChartSeries(name=n, points=[ChartPoint(label=l, value=v) for l, v in pts])
                             for n, pts in series])


def xbrl_item(value, metric="revenue", fy=2025):
    return EvidenceItem(
        evidence_id=f"x-{metric}-{fy}-{value}", modality=Modality.SQL,
        content=f"AMD {metric} for FY{fy}: x (raw value {value} USD; period a to b)",
        source=SourceRef(doc_id="d", ticker="AMD", doc_type=DocType("xbrl"), fiscal_year=fy))


class FakeSql:
    def __init__(self, facts):  # {(metric, fy, quarter): value}
        self.facts, self.calls = facts, []

    def get_metric(self, ticker, metric, fiscal_year, period_type="FY", fiscal_quarter=None, **kw):
        self.calls.append((metric, fiscal_year, period_type, fiscal_quarter))
        v = self.facts.get((metric, fiscal_year, fiscal_quarter))
        return None if v is None else xbrl_item(v, metric, fiscal_year)


class FakeText:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, mode="hybrid", limit=8, filt=None):
        return [EvidenceItem(evidence_id=f"t{i}", modality=Modality.TEXT, content=c,
                             source=SourceRef(doc_id="d", ticker="AMD", doc_type=DocType("10-Q")))
                for i, c in enumerate(self.chunks)]


# ---- helpers ----------------------------------------------------------------------------

def test_parse_period():
    assert parse_period("Q2 FY25") == (2025, 2)
    assert parse_period("Q2 2026") == (2026, 2)
    assert parse_period("Q3'24") == (2024, 3)
    assert parse_period("FY2024") == (2024, None)
    assert parse_period("2023") == (2023, None)
    assert parse_period("2026 Q1") == (2026, 1)
    assert parse_period("Data Center") == (None, None)


def test_decimals_and_tolerance():
    assert decimals_of(26.0) == 1 and decimals_of(6.72) == 2 and decimals_of(3240.0) == 0
    # $26.0B printed vs 26.04B in the filing -> within half a tenth of a billion
    assert abs(Decimal("26.0e9") - Decimal("26.04e9")) <= tolerance(26.0, "billion", Decimal("26.04e9"), 0.005)
    assert abs(Decimal("25.6e9") - Decimal("26.04e9")) > tolerance(26.0, "billion", Decimal("26.04e9"), 0.005)


def test_segment_names_are_not_consolidated():
    assert is_consolidated("Revenue", "Q2 FY25", "AMD Revenue by Quarter")
    assert is_consolidated("Total revenue", "FY2024", "")
    assert not is_consolidated("Data Center", "Q2 FY25", "Revenue")
    assert not is_consolidated("Revenue", "Q2 FY25", "Record Quarterly Revenue")


def test_amount_parsing_and_text_match():
    assert Decimal("6.7e9") in amounts_in_text("revenue was $6.7 billion")
    assert Decimal("6718e6") in amounts_in_text("Data Center 6,718 3,240")
    assert text_supports("Data Center 6,718", Decimal("6.7e9"), Decimal("5e7")) is not None
    assert text_supports("Data Center 9,999", Decimal("6.7e9"), Decimal("5e7")) is None


def test_fact_value_reads_raw_value():
    assert fact_value(xbrl_item(8675000000)) == Decimal("8675000000")


# ---- regressions from the first AMD run ------------------------------------------------

def test_exact_figure_is_not_satisfied_by_a_neighbour_from_another_quarter():
    text = FakeText(["Embedded net revenue of $823 million for the three months ended March 29, 2025"])
    r = CrossModalVerifier(text=text).verify(chart([("Embedded", [("Q2'25", 824)])], "million"), page())
    assert r.verdict == Verdict.INSUFFICIENT          # 823 is Q1, not 824


def test_closest_number_across_chunks_wins_not_the_first_in_range():
    text = FakeText(["Embedded net revenue of $823 million (Q1)", "Embedded net revenue of $824 million (Q2)"])
    r = CrossModalVerifier(text=text).verify(chart([("Embedded", [("Q2'25", 824)])], "million"), page())
    assert r.verdict == Verdict.SUPPORTED and r.checks[0].filing_value == Decimal("824e6")


def test_loss_matches_by_magnitude_and_says_sign_not_checked():
    text = FakeText(["Data Center operating loss of $155 million"])
    r = CrossModalVerifier(text=text).verify(
        chart([("Data Center Operating Income", [("Q2'25", -155)])], "million"), page())
    assert r.verdict == Verdict.SUPPORTED and "sign not checked" in r.checks[0].note


def test_rounded_filing_figure_supports_but_is_labelled_rounded():
    text = FakeText(["Data Center net revenue was $6.7 billion"])
    r = CrossModalVerifier(text=text).verify(chart([("Data Center", [("Q2'26", 6718)])], "million"), page())
    assert r.verdict == Verdict.SUPPORTED and "rounded" in r.checks[0].note


# ---- regressions from the first NVIDIA run ------------------------------------------------

def nvda_page(text):
    return PageRecord(page_id="p2", doc_id="nvda-deck", ticker="NVDA", doc_type="presentation",
                      fiscal_year=2027, fiscal_period="Q2", page=2, image_path="x.png", text=text)


def test_total_series_is_checked_against_xbrl_even_when_title_names_the_breakdown():
    sql = FakeSql({("revenue", 2027, 2): 96221000000})
    c = chart([("Total", [("Q2 FY27", 96221)])], "million", title="Market Platform segments")
    r = CrossModalVerifier(sql=sql).verify(c, nvda_page("Revenue by Market Platform"))
    assert r.verdict == Verdict.SUPPORTED and r.checks[0].method == "xbrl"


def test_wrong_total_is_contradicted():
    sql = FakeSql({("revenue", 2027, 2): 96221000000})
    c = chart([("Total", [("Q2 FY27", 99000)])], "million", title="Market Platform segments")
    r = CrossModalVerifier(sql=sql).verify(c, nvda_page("Revenue by Market Platform"))
    assert r.verdict == Verdict.CONTRADICTED


def test_total_with_ambiguous_metric_is_not_compared():
    sql = FakeSql({("revenue", 2027, 2): 96221000000})
    c = chart([("Total", [("Q2 FY27", 99000)])], "million", title="Market Platform segments")
    r = CrossModalVerifier(sql=sql).verify(c, nvda_page("Revenue and net income"))
    assert r.verdict == Verdict.INSUFFICIENT and sql.calls == []


class RecordingText(FakeText):
    def search(self, query, mode="hybrid", limit=8, filt=None):
        self.filt = filt
        return super().search(query, mode, limit, filt)


def test_prior_period_search_includes_next_fiscal_year_for_recast_comparatives():
    text = RecordingText(["Hyperscale revenue 24,168"])
    c = chart([("Hyperscale", [("Q2 FY26", 24168)])], "million")
    r = CrossModalVerifier(text=text).verify(c, nvda_page(""))
    assert 2027 in text.filt.fiscal_years and 2026 in text.filt.fiscal_years
    assert r.verdict == Verdict.SUPPORTED


# ---- verdicts ---------------------------------------------------------------------------

def test_consolidated_point_matching_xbrl_is_supported():
    v = CrossModalVerifier(sql=FakeSql({("revenue", 2025, None): 25785000000}))
    r = v.verify(chart([("Revenue", [("FY2025", 25.8)])], "billion"), page())
    assert r.verdict == Verdict.SUPPORTED and r.checks[0].method == "xbrl"


def test_consolidated_point_that_differs_is_contradicted():
    v = CrossModalVerifier(sql=FakeSql({("revenue", 2025, None): 25785000000}))
    r = v.verify(chart([("Revenue", [("FY2025", 28.0)])], "billion"), page())
    assert r.verdict == Verdict.CONTRADICTED
    assert r.checks[0].note.startswith("chart is +")


def test_quarter_labels_use_quarterly_lookup():
    sql = FakeSql({("revenue", 2026, 2): 7685000000})
    CrossModalVerifier(sql=sql).verify(chart([("Revenue", [("Q2 FY26", 7.7)])], "billion"), page())
    assert sql.calls[0] == ("revenue", 2026, "Q", 2)


def test_segment_series_never_contradicted_by_consolidated_total():
    """The key safety rule: 'Data Center revenue' must not be compared with total revenue."""
    sql = FakeSql({("revenue", 2025, None): 25785000000})
    r = CrossModalVerifier(sql=sql).verify(chart([("Data Center", [("FY2025", 12.6)])], "billion"), page())
    assert r.checks[0].verdict != Verdict.CONTRADICTED
    assert sql.calls == []  # XBRL was not even consulted


def test_segment_point_supported_by_filing_text():
    text = FakeText(["Data Center segment net revenue of $6,718 million in the quarter"])
    r = CrossModalVerifier(text=text).verify(chart([("Data Center", [("Q2 FY26", 6.7)])], "billion"), page())
    assert r.verdict == Verdict.SUPPORTED and r.checks[0].method == "text"


def test_segment_point_not_in_text_is_insufficient_not_contradicted():
    text = FakeText(["Data Center segment net revenue of $9,000 million"])
    r = CrossModalVerifier(text=text).verify(chart([("Data Center", [("Q2 FY26", 6.7)])], "billion"), page())
    assert r.verdict == Verdict.INSUFFICIENT


def test_text_match_requires_series_name_in_chunk():
    text = FakeText(["Gaming revenue was $6,718 million"])  # right number, wrong segment
    r = CrossModalVerifier(text=text).verify(chart([("Data Center", [("Q2 FY26", 6.7)])], "billion"), page())
    assert r.verdict == Verdict.INSUFFICIENT


def test_estimated_values_get_wider_tolerance():
    sql = FakeSql({("revenue", 2025, None): 25785000000})
    est = CrossModalVerifier(sql=sql).verify(chart([("Revenue", [("FY2025", 26.5)])], "billion", estimates=True), page())
    assert est.verdict == Verdict.SUPPORTED          # 2.8% off, within the 5% estimate tolerance
    exact = CrossModalVerifier(sql=sql).verify(chart([("Revenue", [("FY2025", 26.5)])], "billion"), page())
    assert exact.verdict == Verdict.CONTRADICTED


def test_percent_series_is_unverified():
    r = CrossModalVerifier(sql=FakeSql({})).verify(
        chart([("Revenue growth", [("Q2", 107.0)])], "", unit_label="%"), page())
    assert r.verdict == Verdict.INSUFFICIENT


def test_no_chart_is_insufficient():
    r = CrossModalVerifier().verify(ChartData(has_chart=False), page())
    assert r.verdict == Verdict.INSUFFICIENT and r.checks == []


def test_overall_verdict_rules():
    def pc(v):
        return PointCheck("s", "l", Decimal(1), v)
    S, C, I = Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT
    assert overall_verdict([pc(S), pc(S)]) == Verdict.SUPPORTED
    assert overall_verdict([pc(S), pc(I)]) == Verdict.PARTIAL
    assert overall_verdict([pc(S), pc(C)]) == Verdict.PARTIAL
    assert overall_verdict([pc(C), pc(I)]) == Verdict.CONTRADICTED
    assert overall_verdict([pc(I)]) == Verdict.INSUFFICIENT
    assert overall_verdict([]) == Verdict.INSUFFICIENT


def test_report_evidence_includes_chart_and_matched_fact():
    v = CrossModalVerifier(sql=FakeSql({("revenue", 2025, None): 25785000000}))
    r = v.verify(chart([("Revenue", [("FY2025", 25.8)])], "billion"), page())
    kinds = {e.modality for e in r.evidence}
    assert kinds == {Modality.CHART, Modality.SQL}
    assert r.coverage == 1.0
