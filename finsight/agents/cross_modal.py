"""Cross-modal verification: is a chart from an investor deck supported by the company's filings?

Input : ChartData (extracted from a slide by the VLM) + the PageRecord it came from.
Output: a CrossModalReport with a verdict per data point and an overall verdict.

Two ways to check a point, in order of trust:
  1. XBRL  - the point is a CONSOLIDATED metric (revenue, net income, R&D, operating income) for a
             parseable period. Compare with the exact XBRL fact. Only this path can say CONTRADICTED.
  2. Text  - everything else (segments like "Data Center"). Search the 10-K/10-Q and look for the
             number next to the series name. A hit says SUPPORTED; no hit says INSUFFICIENT, never
             CONTRADICTED, because absence from retrieved text is not evidence of a different value.

Design rules (they follow the failure modes seen in earlier modules and in the first AMD run):
  * A wrong "contradicted" is worse than "insufficient". Segment series are never compared with
    consolidated XBRL totals, and anything ambiguous falls through to INSUFFICIENT.
  * Tolerance reflects how the number was printed: half a unit of its last printed digit, at least
    0.1% (printed figures), or 5% when the VLM had to read values off bar heights. An exact figure
    ("824" million) must not be satisfied by a neighbour ("823"), which is usually another quarter.
  * Text matching takes the CLOSEST number across all retrieved chunks, not the first one in range. A
    rounded filing figure ("$6.7 billion") may support an exact chart value within its own rounding;
    such matches are labelled as rounded.
  * Signs are not checked in text (a loss is printed as "loss of $155 million" or "(155)"); the note says so.
  * All arithmetic (differences, growth) goes through finsight.generation.numeric.
  * Percent series are not verified; they are reported as unverified with a note.
  * The period inside a text chunk is NOT checked (only the fiscal year is filtered), so text support
    means "this figure appears next to this name in the filing", not "for exactly this quarter".
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal

from finsight.agents.router import TICKER_ALIASES, extract_metrics
from finsight.core.exceptions import FinSightError
from finsight.core.schemas import EvidenceItem, Verdict
from finsight.generation.numeric import UNIT_MULTIPLIERS, growth, normalize_units
from finsight.retrieval.text_retriever import SearchFilter
from finsight.vision.chart_extractor import ChartData, chart_evidence
from finsight.vision.page_renderer import PageRecord

log = logging.getLogger(__name__)

# words that name a consolidated metric or the period, not a business segment
_GENERIC = {
    "total", "consolidated", "company", "quarterly", "annual", "quarter", "year", "fiscal", "by", "in",
    "of", "the", "and", "for", "usd", "us", "dollars", "billion", "million", "thousand", "billions",
    "millions", "revenue", "revenues", "net", "sales", "income", "operating", "profit", "research",
    "development", "expense", "expenses", "r", "d", "fy", "yoy", "ttm", "gaap", "non", "b", "m",
}
_ALIAS_WORDS = {w for alias in TICKER_ALIASES for w in alias.split()}
_PERIOD_TOKEN = re.compile(r"^(q[1-4]|fy\d*|20\d{2}|\d+)$")


def tokenize(text: str) -> list[str]:
    """Plain lowercase words. Deliberately no stemming or stopword removal: the segment-vs-consolidated
    guard must see exactly the words printed on the slide."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---- parsing helpers --------------------------------------------------------------------

def _year(s: str) -> int:
    y = int(s)
    return 2000 + y if y < 100 else y


def parse_period(label: str) -> tuple[int | None, int | None]:
    """Chart label -> (fiscal_year, fiscal_quarter). Quarter is None for a full year.
    'Q2 FY25' -> (2025, 2); 'FY2024' -> (2024, None); 'Q3 2026' -> (2026, 3); '2024' -> (2024, None)."""
    t = label.strip()
    m = re.search(r"Q([1-4])\s*(?:FY|fiscal)?\s*['’]?\s*(\d{2,4})\b", t, re.I)
    if m:
        return _year(m.group(2)), int(m.group(1))
    m = re.search(r"\b(20\d{2})\s*Q([1-4])\b", t, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\bFY\s*['’]?\s*(\d{2,4})\b", t, re.I)
    if m:
        return _year(m.group(1)), None
    m = re.fullmatch(r"\s*(20\d{2})\s*", t)
    if m:
        return int(m.group(1)), None
    return None, None


def decimals_of(value: float) -> int:
    """Digits after the decimal point as printed (trailing zeros ignored; at least 1 below 100,
    since '26.0' on a slide is a one-decimal figure)."""
    s = f"{value:.10f}".rstrip("0").rstrip(".")
    d = len(s.split(".")[1]) if "." in s else 0
    return max(d, 1) if abs(value) < 100 else d


def tolerance(printed: float, scale: str, reference: Decimal, rel: float) -> Decimal:
    """Allowed absolute difference between the chart value and the filing value."""
    half_unit = Decimal("0.5") * (Decimal(10) ** -decimals_of(printed)) * UNIT_MULTIPLIERS[scale]
    return max(abs(reference) * Decimal(str(rel)), half_unit)


def is_consolidated(series: str, label: str, title: str) -> bool:
    """True when nothing in the names points at a business segment, so an XBRL total is comparable.
    Conservative: any unexplained word (e.g. 'Data Center', 'Record') makes it a non-match."""
    for text in (series, label, title):
        for tok in tokenize(text):
            if tok in _GENERIC or tok in _ALIAS_WORDS or _PERIOD_TOKEN.match(tok):
                continue
            return False
    return True


_RAW = re.compile(r"raw value (-?\d+(?:\.\d+)?)")


def fact_value(item: EvidenceItem) -> Decimal | None:
    """Exact value stored in an XBRL evidence item (see sql_retriever.row_to_evidence)."""
    m = _RAW.search(item.content)
    return Decimal(m.group(1)) if m else None


_AMOUNT = re.compile(r"\$?\(?(\d[\d,]*(?:\.\d+)?)\)?\s*(trillion|billion|million|thousand|bn|mm|[kmb])?\b", re.I)


def candidates_in_text(text: str) -> list[tuple[Decimal, Decimal]]:
    """Every number in `text` as (absolute-dollar value, half-unit of its own rounding).
    A number with a scale word has one reading; a bare number is ambiguous (10-K tables are usually
    in millions or thousands), so it yields several. Signs are dropped (see module docstring)."""
    out: list[tuple[Decimal, Decimal]] = []
    for m in _AMOUNT.finditer(text):
        raw = m.group(1).replace(",", "")
        base = Decimal(raw)
        d = len(raw.split(".")[1]) if "." in raw else 0
        half = Decimal("0.5") * (Decimal(10) ** -d)
        scale = (m.group(2) or "").lower()
        scales = (scale,) if scale else ("", "thousand", "million")
        for s in scales:
            mult = UNIT_MULTIPLIERS[s]
            out.append((base * mult, half * mult))
    return out


def amounts_in_text(text: str) -> list[Decimal]:
    return [v for v, _ in candidates_in_text(text)]


def text_supports(text: str, target: Decimal, tol: Decimal) -> tuple[Decimal, Decimal, bool] | None:
    """Closest number in `text` to |target|, as (value, difference, was_rounded), or None when nothing is
    close enough. A candidate is close enough within max(tol, half a unit of its own rounding);
    was_rounded is True when it needed that extra slack (the filing figure itself is rounded)."""
    goal = abs(target)
    best: tuple[Decimal, Decimal, bool] | None = None
    for value, half in candidates_in_text(text):
        diff = abs(value - goal)
        if diff <= max(tol, half) and (best is None or diff < best[1]):
            best = (value, diff, diff > tol)
    return best


def _name_tokens(series: str, title: str) -> list[str]:
    toks = tokenize(series) or tokenize(title)
    return [t for t in toks if t not in _GENERIC and t not in _ALIAS_WORDS and not _PERIOD_TOKEN.match(t)
            and len(t) > 2]


# ---- results ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PointCheck:
    series: str
    label: str
    chart_value: Decimal | None
    verdict: Verdict
    method: str = "none"          # "xbrl" | "text" | "none"
    filing_value: Decimal | None = None
    tolerance: Decimal | None = None
    evidence_id: str | None = None
    note: str = ""


@dataclass
class CrossModalReport:
    page_id: str
    chart_title: str
    verdict: Verdict
    checks: list[PointCheck] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)   # chart + matched filing evidence

    @property
    def counts(self) -> dict[str, int]:
        c = {"supported": 0, "contradicted": 0, "unverified": 0}
        for k in self.checks:
            if k.verdict == Verdict.SUPPORTED:
                c["supported"] += 1
            elif k.verdict == Verdict.CONTRADICTED:
                c["contradicted"] += 1
            else:
                c["unverified"] += 1
        return c

    @property
    def coverage(self) -> float:
        """Share of chart points that could be checked against the filings."""
        n = len(self.checks)
        return 0.0 if n == 0 else (self.counts["supported"] + self.counts["contradicted"]) / n

    def summary(self) -> str:
        c = self.counts
        lines = [f"Verdict: {self.verdict.value}  ({c['supported']} supported, {c['contradicted']} "
                 f"contradicted, {c['unverified']} unverified; coverage {self.coverage:.0%})"]
        for k in self.checks:
            val = "n/a" if k.chart_value is None else f"{k.chart_value:,.0f}"
            flt = f" | filing {k.filing_value:,.0f}" if k.filing_value is not None else ""
            lines.append(f"  [{k.verdict.value:>21}] {k.series or '-'} / {k.label}: chart {val}{flt}"
                         f" ({k.method}) {k.note}".rstrip())
        return "\n".join(lines)


def overall_verdict(checks: list[PointCheck]) -> Verdict:
    sup = sum(c.verdict == Verdict.SUPPORTED for c in checks)
    con = sum(c.verdict == Verdict.CONTRADICTED for c in checks)
    unv = len(checks) - sup - con
    if sup + con == 0:
        return Verdict.INSUFFICIENT
    if con and not sup:
        return Verdict.CONTRADICTED
    if con:
        return Verdict.PARTIAL
    return Verdict.SUPPORTED if unv == 0 else Verdict.PARTIAL


# ---- verifier ---------------------------------------------------------------------------

class CrossModalVerifier:
    def __init__(self, sql=None, text=None, rel_tol: float = 0.001, estimate_tol: float = 0.05,
                 text_limit: int = 8):
        """sql: SqlRetriever-like (.get_metric); text: TextRetriever-like (.search). Either may be None."""
        self.sql, self.text = sql, text
        self.rel_tol, self.estimate_tol, self.text_limit = rel_tol, estimate_tol, text_limit

    def verify(self, chart: ChartData, page: PageRecord) -> CrossModalReport:
        evidence: list[EvidenceItem] = [chart_evidence(chart, page)] if chart.has_chart else []
        if not chart.has_chart or not chart.series:
            return CrossModalReport(page.page_id, chart.title, Verdict.INSUFFICIENT, [], evidence)

        rel = self.estimate_tol if chart.values_are_estimates else self.rel_tol
        percent = "%" in chart.unit_label
        scale = "" if percent else chart.unit_scale
        checks: list[PointCheck] = []
        for s in chart.series:
            for p in s.points:
                if p.value is None:
                    continue
                absolute = normalize_units(p.value, scale)
                if percent:
                    checks.append(PointCheck(s.name, p.label, absolute, Verdict.INSUFFICIENT,
                                             note="percentage series not verified"))
                    continue
                check, ev = self._check_point(chart, page, s.name, p.label, p.value, scale, absolute, rel)
                checks.append(check)
                if ev is not None and all(e.evidence_id != ev.evidence_id for e in evidence):
                    evidence.append(ev)
        return CrossModalReport(page.page_id, chart.title, overall_verdict(checks), checks, evidence)

    # -- one point ------------------------------------------------------------------------

    def _check_point(self, chart, page, series, label, printed, scale, absolute, rel):
        note = ""
        fy, quarter = parse_period(label)
        if fy is None:
            fy, quarter = parse_period(series)
        tot = self._total_metric(chart, page, series)
        metric = tot or (extract_metrics(series) or extract_metrics(label) or extract_metrics(chart.title)
                  or (None,))[0]

        if (self.sql is not None and metric and fy and (tot or is_consolidated(series, label, chart.title))):
            try:
                item = self.sql.get_metric(page.ticker, metric, fy,
                                           "Q" if quarter else "FY", quarter)
            except FinSightError as e:
                item, note = None, f"XBRL lookup failed: {e}"
            if item is not None:
                filing = fact_value(item)
                if filing is not None:
                    tol = tolerance(printed, scale, filing, rel)
                    if abs(absolute - filing) <= tol:
                        return PointCheck(series, label, absolute, Verdict.SUPPORTED, "xbrl",
                                          filing, tol, item.evidence_id), item
                    diff = growth(filing, absolute).describe()
                    return PointCheck(series, label, absolute, Verdict.CONTRADICTED, "xbrl", filing, tol,
                                      item.evidence_id,
                                      f"chart is {diff} vs the filing"), item
            else:
                note = note or "no XBRL fact for this period"

        return self._check_text(chart, page, series, label, absolute, printed, scale, fy, rel, note)

    @staticmethod
    def _total_metric(chart, page, series) -> str | None:
        """A series named just 'Total' is the consolidated total of whatever the chart breaks down, so it
        is comparable with an XBRL fact even though the chart title names the breakdown (e.g. 'Market
        Platform'). The metric comes from the chart or slide wording, and only when unambiguous."""
        words = set(tokenize(series))
        if not words or not words <= {"total", "consolidated", "company"}:
            return None
        for text in (f"{chart.title} {chart.y_axis}", page.text or ""):
            found = extract_metrics(text)
            if len(found) == 1:
                return found[0]
        return None

    def _check_text(self, chart, page, series, label, absolute, printed, scale, fy, rel, note):
        if self.text is None:
            return PointCheck(series, label, absolute, Verdict.INSUFFICIENT,
                              note=note or "no filing text available"), None
        names = _name_tokens(series, chart.title)
        if not names:
            return PointCheck(series, label, absolute, Verdict.INSUFFICIENT,
                              note=note or "series has no distinctive name to match in the filing"), None
        # prior-period figures reappear as comparatives (often recast) in the NEXT year's filings
        years = tuple(sorted({y for y in (fy, fy + 1 if fy else None, page.fiscal_year) if y})) or None
        filt = SearchFilter(tickers=(page.ticker,), fiscal_years=years,
                            doc_types=("10-K", "10-Q"))
        query = f"{page.ticker} {series} {label} {chart.title}".strip()
        try:
            hits = self.text.search(query, mode="hybrid", limit=self.text_limit, filt=filt)
        except FinSightError as e:
            return PointCheck(series, label, absolute, Verdict.INSUFFICIENT,
                              note=f"filing search failed: {e}"), None

        tol = tolerance(printed, scale, absolute, rel)
        best = None  # (diff, value, rounded, chunk)
        for h in hits:
            low = h.content.lower()
            if not all(t in low for t in names):
                continue
            found = text_supports(h.content, absolute, tol)
            if found is not None and (best is None or found[1] < best[0]):
                best = (found[1], found[0], found[2], h)
        if best is not None:
            _, value, rounded, h = best
            notes = ["matches a rounded figure in the filing" if rounded else "number found in filing text"]
            if absolute < 0:
                notes.append("sign not checked")
            return PointCheck(series, label, absolute, Verdict.SUPPORTED, "text", value, tol,
                              h.evidence_id, "; ".join(notes)), h
        return PointCheck(series, label, absolute, Verdict.INSUFFICIENT, "text",
                          note=(note + "; " if note else "") + "value not found in retrieved filing text"), None
