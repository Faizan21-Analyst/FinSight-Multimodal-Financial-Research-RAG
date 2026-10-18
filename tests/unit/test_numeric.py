from datetime import date
from decimal import Decimal

import pytest

from finsight.generation.numeric import (
    FiscalCalendar,
    NumericError,
    align_fiscal_years,
    approx_equal,
    cagr,
    format_money,
    growth,
    margin,
    margin_change_bps,
    normalize_units,
    parse_amount,
)

NVDA = FiscalCalendar("NVDA", 1)   # FY ends late January
AAPL = FiscalCalendar("AAPL", 9)   # FY ends late September
TSLA = FiscalCalendar("TSLA", 12)


def test_growth_basic():
    g = growth(100, 125)
    assert g.pct_change == Decimal(25)
    assert g.describe() == "+25.0%"


def test_growth_decline_and_zero_base():
    assert growth(200, 150).pct_change == Decimal(-25)
    with pytest.raises(NumericError):
        growth(0, 10)


def test_growth_loss_narrowing_is_positive():
    assert growth(-100, -50).pct_change == Decimal(50)


def test_cagr():
    assert round(cagr(100, 121, 2), 4) == Decimal("10.0000")
    with pytest.raises(NumericError):
        cagr(-1, 5, 2)


def test_margin_and_bps():
    assert margin(25, 100) == Decimal(25)
    assert margin_change_bps(20, 100, 25, 100) == Decimal(500)


def test_units():
    assert normalize_units(26.97, "billion") == Decimal("26970000000.00")
    with pytest.raises(NumericError):
        normalize_units(1, "gazillion")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$26.97 billion", Decimal("26970000000")),
        ("1,234 million", Decimal("1234000000")),
        ("(1,234)", Decimal("-1234")),
        ("$450", Decimal("450")),
    ],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == expected


def test_approx_equal():
    assert approx_equal(26.97e9, 26.9712e9)
    assert not approx_equal(26.97e9, 28e9)


def test_fiscal_year_of():
    assert NVDA.fiscal_year_of(date(2024, 1, 28)) == 2024
    assert NVDA.fiscal_year_of(date(2023, 6, 30)) == 2024
    assert AAPL.fiscal_year_of(date(2024, 9, 28)) == 2024
    assert AAPL.fiscal_year_of(date(2024, 10, 5)) == 2025
    assert TSLA.fiscal_year_of(date(2024, 12, 31)) == 2024


def test_fiscal_quarter_of():
    assert AAPL.fiscal_quarter_of(date(2023, 10, 15)) == 1
    assert AAPL.fiscal_quarter_of(date(2024, 9, 15)) == 4
    assert TSLA.fiscal_quarter_of(date(2024, 2, 1)) == 1


def test_align_warns_when_year_ends_differ():
    fy, note = align_fiscal_years(NVDA, 2024, TSLA)
    # NVDA FY2024 (ends Jan 2024) overlaps Tesla's calendar 2023; one month apart, so no warning
    assert fy == 2023 and note == ""
    _, note2 = align_fiscal_years(AAPL, 2024, TSLA)
    assert "months apart" in note2


def test_format_money():
    assert format_money(26_970_000_000) == "$26.97 billion"
    assert format_money(-4_500_000) == "-$4.50 million"
