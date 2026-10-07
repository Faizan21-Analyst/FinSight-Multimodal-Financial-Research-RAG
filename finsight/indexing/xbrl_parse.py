"""Parse SEC XBRL `companyfacts` JSON into clean, de-duplicated, fiscal-period-tagged facts.

Pure functions only (no I/O, no DB) so everything here is unit-testable offline.

Gotchas this module handles (log them in docs/failures.md as you meet them):
  * An entry's own `fy`/`fp` fields describe the FILING that reported it, not the period the
    number covers. A FY2022 figure shown as a comparative in the FY2024 10-K has fy=2024.
    We therefore derive fiscal year/quarter from the period END DATE instead.
  * The same period is reported many times (original + later comparatives/restatements).
    We keep one row per (metric, period_type, period_end): preferred tag, then latest filing.
  * Companies switch tags over time (e.g. SalesRevenueNet -> Revenues), so each metric has
    an ordered list of candidate tags.
  * companyfacts contains only NON-dimensional facts. Segment / product-line numbers
    (e.g. NVIDIA "Data Center" revenue) are NOT here; they come from filing tables (Module 3).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

# metric name -> us-gaap tags, in priority order
METRICS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "gross_profit": ("GrossProfit",),
    "rd_expense": ("ResearchAndDevelopmentExpense",),
    "sga_expense": ("SellingGeneralAndAdministrativeExpense",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "stockholders_equity": ("StockholdersEquity",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
}


@dataclass(frozen=True)
class Fact:
    ticker: str
    cik: str
    metric: str
    tag: str
    unit: str
    period_type: str  # "FY" | "Q" | "YTD" | "instant"
    fiscal_year: int
    fiscal_quarter: int | None  # None for FY; quarter of period end for Q / YTD / instant
    period_start: date | None
    period_end: date
    value: float
    accession: str
    form: str
    filed: date


def classify_period(start: date | None, end: date) -> str | None:
    """Duration -> period type. Tolerates 52/53-week fiscal calendars. None = drop."""
    if start is None:
        return "instant"
    days = (end - start).days
    if 350 <= days <= 380:
        return "FY"
    if 80 <= days <= 100:
        return "Q"
    if 170 <= days <= 200 or 260 <= days <= 290:  # 6-month and 9-month year-to-date
        return "YTD"
    return None


def fiscal_position(end: date, fye_month: int) -> tuple[int, int]:
    """(fiscal_year, fiscal_quarter) of a period-end date.

    We shift back 10 days so 52/53-week calendars that end a few days into the next month
    (e.g. Dec calendar ending Jan 1) still land in the right month.
    Fiscal year is named by the calendar year in which it ENDS (NVIDIA fiscal 2024 ends Jan 2024).
    """
    d = end - timedelta(days=10)
    fy = d.year if d.month <= fye_month else d.year + 1
    q = ((d.month - fye_month - 1) % 12) // 3 + 1
    return fy, q


def infer_fye_month(annual_ends: list[date]) -> int:
    """Fiscal-year-end month = most common end month of annual (10-K) periods."""
    if not annual_ends:
        raise ValueError("No annual periods found; cannot infer fiscal year end")
    months = Counter((e - timedelta(days=10)).month for e in annual_ends)
    return months.most_common(1)[0][0]


def parse_companyfacts(
    data: dict, ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q")
) -> list[Fact]:
    gaap = data.get("facts", {}).get("us-gaap", {})
    cik = str(data.get("cik", "")).zfill(10)

    # pass 1: collect candidate rows
    raw: list[tuple] = []
    for metric, tags in METRICS.items():
        for priority, tag in enumerate(tags):
            node = gaap.get(tag)
            if not node:
                continue
            for unit, entries in node.get("units", {}).items():
                for e in entries:
                    if e.get("form") not in forms:
                        continue
                    end = date.fromisoformat(e["end"])
                    start = date.fromisoformat(e["start"]) if e.get("start") else None
                    ptype = classify_period(start, end)
                    if ptype is None:
                        continue
                    raw.append((metric, priority, tag, unit, ptype, start, end, e))

    annual_ends = [r[6] for r in raw if r[4] == "FY" and r[7]["form"] == "10-K"]
    if not raw or not annual_ends:
        return []
    fye = infer_fye_month(annual_ends)

    # pass 2: de-duplicate -> best (lowest priority index, then latest filed) per period
    best: dict[tuple, tuple[tuple, Fact]] = {}
    for metric, priority, tag, unit, ptype, start, end, e in raw:
        filed = date.fromisoformat(e["filed"])
        fy, q = fiscal_position(end, fye)
        fact = Fact(
            ticker=ticker,
            cik=cik,
            metric=metric,
            tag=tag,
            unit=unit,
            period_type=ptype,
            fiscal_year=fy,
            fiscal_quarter=None if ptype == "FY" else q,
            period_start=start,
            period_end=end,
            value=e["val"],
            accession=e["accn"],
            form=e["form"],
            filed=filed,
        )
        key = (metric, ptype, end)
        rank = (priority, -filed.toordinal())
        if key not in best or rank < best[key][0]:
            best[key] = (rank, fact)

    return sorted(
        (f for _, f in best.values()),
        key=lambda f: (f.metric, f.period_type, f.period_end),
    )