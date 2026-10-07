from datetime import date

from finsight.indexing.xbrl_parse import (
    classify_period, fiscal_position, infer_fye_month, parse_companyfacts,
)


def d(s):
    return date.fromisoformat(s)


def test_classify_period():
    assert classify_period(None, d("2024-01-28")) == "instant"
    assert classify_period(d("2023-01-30"), d("2024-01-28")) == "FY"      # 364 days
    assert classify_period(d("2023-01-30"), d("2023-04-30")) == "Q"       # 90 days
    assert classify_period(d("2023-01-30"), d("2023-07-30")) == "YTD"     # 6 months
    assert classify_period(d("2023-01-30"), d("2023-10-29")) == "YTD"     # 9 months
    assert classify_period(d("2023-01-30"), d("2023-03-01")) is None      # odd length


def test_fiscal_position_jan_fye():  # NVIDIA-style
    assert fiscal_position(d("2024-01-28"), 1) == (2024, 4)
    assert fiscal_position(d("2024-04-28"), 1) == (2025, 1)
    assert fiscal_position(d("2024-07-28"), 1) == (2025, 2)
    assert fiscal_position(d("2024-10-27"), 1) == (2025, 3)


def test_fiscal_position_sep_fye():  # Apple-style
    assert fiscal_position(d("2023-09-30"), 9) == (2023, 4)
    assert fiscal_position(d("2023-12-30"), 9) == (2024, 1)
    assert fiscal_position(d("2024-06-29"), 9) == (2024, 3)


def test_fiscal_position_dec_fye_and_52_53_week():
    assert fiscal_position(d("2024-12-28"), 12) == (2024, 4)
    assert fiscal_position(d("2024-03-30"), 12) == (2024, 1)
    assert fiscal_position(d("2022-01-01"), 12) == (2021, 4)  # 53-week year ending Jan 1


def test_infer_fye_month():
    assert infer_fye_month([d("2023-01-29"), d("2024-01-28"), d("2025-01-26")]) == 1
    assert infer_fye_month([d("2023-12-30"), d("2024-12-28")]) == 12


def _entry(start, end, val, form, filed, accn, fy=2099, fp="FY"):
    e = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        e["start"] = start
    return e


COMPANYFACTS = {
    "cik": 1045810,
    "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            _entry("2023-01-30", "2024-01-28", 60922000000, "10-K", "2024-02-21", "A1"),
            # same period re-reported as a comparative in the next 10-K (restated slightly)
            _entry("2023-01-30", "2024-01-28", 60923000000, "10-K", "2025-02-26", "A2"),
            _entry("2024-01-29", "2025-01-26", 130497000000, "10-K", "2025-02-26", "A2"),
            _entry("2024-01-29", "2024-04-28", 26044000000, "10-Q", "2024-05-29", "B1"),
            _entry("2024-01-29", "2024-07-28", 56000000000, "10-Q", "2024-08-28", "B2"),  # YTD
            _entry("2024-01-29", "2024-03-01", 1, "10-Q", "2024-05-29", "B1"),            # odd, dropped
            _entry("2024-01-29", "2025-01-26", 999, "10-K/A", "2025-03-01", "X"),         # amendment, dropped
        ]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            _entry("2023-01-30", "2024-01-28", 111, "10-K", "2024-02-21", "A1"),  # lower priority tag
        ]}},
        "Assets": {"units": {"USD": [
            _entry(None, "2024-01-28", 65728000000, "10-K", "2024-02-21", "A1"),
        ]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            _entry("2023-01-30", "2024-01-28", 1.19, "10-K", "2024-02-21", "A1"),
        ]}},
    }},
}


def _by(facts, metric, ptype):
    return [f for f in facts if f.metric == metric and f.period_type == ptype]


def test_parse_dedup_priority_and_filters():
    facts = parse_companyfacts(COMPANYFACTS, "NVDA")
    fy = _by(facts, "revenue", "FY")
    assert [f.fiscal_year for f in fy] == [2024, 2025]
    fy2024 = fy[0]
    assert fy2024.value == 60923000000          # latest filing wins
    assert fy2024.accession == "A2"
    assert fy2024.tag == "Revenues"             # higher-priority tag beats the lower-priority one (111)
    assert fy2024.fiscal_quarter is None
    assert len(_by(facts, "revenue", "Q")) == 1
    assert _by(facts, "revenue", "Q")[0].fiscal_year == 2025 and _by(facts, "revenue", "Q")[0].fiscal_quarter == 1
    ytd = _by(facts, "revenue", "YTD")
    assert len(ytd) == 1 and ytd[0].fiscal_quarter == 2
    assert all(f.value != 999 and f.value != 1 for f in facts)   # amendment + odd duration gone


def test_parse_instant_and_eps_units():
    facts = parse_companyfacts(COMPANYFACTS, "NVDA")
    a = _by(facts, "total_assets", "instant")[0]
    assert (a.fiscal_year, a.fiscal_quarter, a.period_start) == (2024, 4, None)
    e = _by(facts, "eps_diluted", "FY")[0]
    assert e.unit == "USD/shares" and e.value == 1.19
    assert e.cik == "0001045810"


def test_parse_empty_returns_empty():
    assert parse_companyfacts({"cik": 1, "facts": {}}, "X") == []