"""Numeric layer: every number shown to the user is computed here, never by the LLM.

Covers growth rates, CAGR, margins, ratios, unit normalisation and fiscal-period alignment.
Pure functions with no I/O, so they are cheap to unit test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from finsight.core.exceptions import FinSightError

Number = Decimal | float | int | str

UNIT_MULTIPLIERS: dict[str, Decimal] = {
    "": Decimal(1),
    "unit": Decimal(1),
    "thousand": Decimal(10) ** 3,
    "thousands": Decimal(10) ** 3,
    "k": Decimal(10) ** 3,
    "million": Decimal(10) ** 6,
    "millions": Decimal(10) ** 6,
    "mm": Decimal(10) ** 6,
    "m": Decimal(10) ** 6,
    "billion": Decimal(10) ** 9,
    "billions": Decimal(10) ** 9,
    "bn": Decimal(10) ** 9,
    "b": Decimal(10) ** 9,
    "trillion": Decimal(10) ** 12,
    "trillions": Decimal(10) ** 12,
    "t": Decimal(10) ** 12,
}


class NumericError(FinSightError):
    """Raised when a calculation is undefined (e.g. growth from a zero base)."""


def to_decimal(x: Number) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def normalize_units(value: Number, unit: str = "") -> Decimal:
    """Convert a value stated in 'millions', 'billions', ... to absolute units."""
    key = unit.strip().lower()
    if key not in UNIT_MULTIPLIERS:
        raise NumericError(f"unknown unit scale: {unit!r}")
    return to_decimal(value) * UNIT_MULTIPLIERS[key]


_NUM = re.compile(
    r"(?P<open>\()?\s*\$?\s*(?P<num>-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>trillion|billion|million|thousand|bn|mm|[kmbt])?(?P<close>\))?",
    re.IGNORECASE,
)


def parse_amount(text: str) -> Decimal:
    """'$26.97 billion' -> 26970000000, '(1,234)' -> -1234, '4.5M' -> 4500000."""
    m = _NUM.search(text)
    if not m:
        raise NumericError(f"no number found in {text!r}")
    num = Decimal(m.group("num").replace(",", ""))
    if m.group("open") and m.group("close"):  # accounting-style negative
        num = -abs(num)
    return normalize_units(num, (m.group("scale") or "").lower())


@dataclass(frozen=True)
class Growth:
    start: Decimal
    end: Decimal
    absolute_change: Decimal
    pct_change: Decimal  # percent, e.g. 12.5 means +12.5%

    def describe(self, digits: int = 1) -> str:
        sign = "+" if self.pct_change >= 0 else ""
        return f"{sign}{self.pct_change:.{digits}f}%"


def growth(start: Number, end: Number) -> Growth:
    """Percent change from start to end. Uses |start| as base so a loss narrowing reads as positive."""
    s, e = to_decimal(start), to_decimal(end)
    if s == 0:
        raise NumericError("growth from a zero base is undefined")
    change = e - s
    return Growth(s, e, change, change / abs(s) * 100)


def cagr(start: Number, end: Number, years: float) -> Decimal:
    """Compound annual growth rate in percent. Both values must be positive."""
    s, e = to_decimal(start), to_decimal(end)
    if years <= 0:
        raise NumericError("years must be positive")
    if s <= 0 or e <= 0:
        raise NumericError("CAGR needs positive start and end values")
    return (Decimal(float(e / s) ** (1.0 / years)) - 1) * 100


def ratio(numerator: Number, denominator: Number) -> Decimal:
    d = to_decimal(denominator)
    if d == 0:
        raise NumericError("division by zero")
    return to_decimal(numerator) / d


def margin(profit: Number, revenue: Number) -> Decimal:
    """Profit as a percent of revenue (gross, operating, net - caller picks the profit line)."""
    return ratio(profit, revenue) * 100


def margin_change_bps(profit0: Number, revenue0: Number, profit1: Number, revenue1: Number) -> Decimal:
    """Change in margin between two periods, in basis points."""
    return (margin(profit1, revenue1) - margin(profit0, revenue0)) * 100


def share_of_total(part: Number, total: Number) -> Decimal:
    return ratio(part, total) * 100


def approx_equal(a: Number, b: Number, rel_tol: float = 0.005) -> bool:
    """True if a and b agree within a relative tolerance (default 0.5%). Used when checking a
    chart's rounded figure against an exact XBRL value."""
    x, y = to_decimal(a), to_decimal(b)
    if x == y:
        return True
    base = max(abs(x), abs(y))
    return abs(x - y) / base <= Decimal(str(rel_tol))


# ---- fiscal-period alignment -------------------------------------------------------------

@dataclass(frozen=True)
class FiscalCalendar:
    """Fiscal year ends in `end_month` (1-12). A year is named by the calendar year it ENDS in
    (NVIDIA FY2024 ends Jan 2024; Apple FY2024 ends Sep 2024), matching the loader's convention."""

    ticker: str
    end_month: int

    def fiscal_year_of(self, d: date) -> int:
        return d.year if d.month <= self.end_month else d.year + 1

    def fiscal_quarter_of(self, d: date) -> int:
        months_into_year = (d.month - self.end_month - 1) % 12  # 0 = first month of the FY
        return months_into_year // 3 + 1

    def fiscal_year_end(self, fiscal_year: int) -> date:
        """Approximate last day of the fiscal year (day 28 is safe for every month)."""
        return date(fiscal_year, self.end_month, 28)


def align_fiscal_years(cal_a: FiscalCalendar, fy_a: int, cal_b: FiscalCalendar) -> tuple[int, str]:
    """Company B's fiscal year that ends closest to A's `fy_a`, plus a warning when the two
    year-ends differ by more than a month (so a comparison can say 'not the same 12 months')."""
    end_a = cal_a.fiscal_year_end(fy_a)
    # candidate B years: the one containing A's year-end and its neighbours; pick the nearest end
    best_fy = min(
        (cal_b.fiscal_year_of(end_a) + d for d in (-1, 0, 1)),
        key=lambda fy: abs((cal_b.fiscal_year_end(fy) - end_a).days),
    )
    gap_months = round(abs((cal_b.fiscal_year_end(best_fy) - end_a).days) / 30.4)
    note = ""
    if gap_months > 1:
        note = (
            f"{cal_a.ticker} FY{fy_a} ends in month {cal_a.end_month}; {cal_b.ticker} FY{best_fy} "
            f"ends in month {cal_b.end_month}. The periods are about {gap_months} months apart."
        )
    return best_fy, note


def format_money(value: Number, digits: int = 2) -> str:
    v = to_decimal(value)
    a = abs(v)
    sign = "-" if v < 0 else ""
    for scale, label in (
        (Decimal(10) ** 12, "trillion"),
        (Decimal(10) ** 9, "billion"),
        (Decimal(10) ** 6, "million"),
    ):
        if a >= scale:
            return f"{sign}${a / scale:,.{digits}f} {label}"
    return f"{sign}${a:,.0f}"
