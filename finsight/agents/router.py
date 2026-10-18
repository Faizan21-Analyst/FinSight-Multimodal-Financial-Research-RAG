"""Intent router: turns a question into a QueryPlan (intent, entities, which retrievers to run).

Rule-based on purpose: deterministic, free, instant and unit-testable. An LLM fallback for
low-confidence cases can be layered on top later (Module 9) and judged against this baseline.

Intent priority (first match wins):
  cross_modal > visual > comparison > numeric > narrative
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from finsight.retrieval.text_retriever import SearchFilter


class Intent(str, Enum):
    NARRATIVE = "narrative"
    NUMERIC = "numeric"
    VISUAL = "visual"
    COMPARISON = "comparison"
    CROSS_MODAL = "cross_modal"


# alias (lowercase, matched on word boundaries) -> ticker
TICKER_ALIASES: dict[str, str] = {
    "nvidia": "NVDA", "nvda": "NVDA",
    "amd": "AMD", "advanced micro devices": "AMD",
    "apple": "AAPL", "aapl": "AAPL",
    "tesla": "TSLA", "tsla": "TSLA",
}

# phrase -> XBRL metric name used in finsight.xbrl_facts
METRIC_KEYWORDS: dict[str, str] = {
    "research and development": "rd_expense",
    "r&d": "rd_expense",
    "net income": "net_income",
    "operating income": "operating_income",
    "operating profit": "operating_income",
    "revenue": "revenue",
    "net sales": "revenue",
    "net revenue": "revenue",
}

VISUAL_WORDS = r"\b(chart|graph|figure|slide|deck|plot|diagram|infographic|visuali[sz]ation)\b"
PRESENTATION_WORDS = r"\b(investor (deck|presentation)|presentation|slide|deck)\b"
FILING_WORDS = r"\b(10-?k|10-?q|filing|annual report|sec)\b"
VERIFY_WORDS = r"\b(supported|support|verify|consistent|contradict\w*|match\w*|agree\w*|reconcile|true|accurate)\b"
COMPARE_WORDS = r"\b(compare|comparison|versus|vs\.?|higher|lower|more than|less than|differ\w*)\b"
NUMERIC_WORDS = r"\b(how much|how many|what (was|were|is|are)|total|amount|value)\b"
WHY_WORDS = r"\b(why|how did|explain|reason|driver|risk|describe|discuss|strategy|outlook)\b"


@dataclass(frozen=True)
class QueryPlan:
    question: str
    intent: Intent
    tickers: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    metrics: tuple[str, ...] = ()
    retrievers: tuple[str, ...] = ("text",)  # subset of: text, visual, sql
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def search_filter(self) -> SearchFilter | None:
        """Metadata filter for the text retriever. Only what the question actually names."""
        if not (self.tickers or self.fiscal_years):
            return None
        return SearchFilter(
            tickers=self.tickers or None,
            fiscal_years=self.fiscal_years or None,
        )


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def extract_tickers(text: str) -> tuple[str, ...]:
    found: list[str] = []
    low = text.lower()
    for alias, ticker in TICKER_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", low) and ticker not in found:
            found.append(ticker)
    return tuple(found)


def extract_years(text: str) -> tuple[int, ...]:
    years = {int(y) for y in re.findall(r"\b(?:fy|fiscal(?: year)?\s*)?\s*(20\d{2})\b", text, re.IGNORECASE)}
    return tuple(sorted(years))


def extract_metrics(text: str) -> tuple[str, ...]:
    low = text.lower()
    out: list[str] = []
    for phrase, metric in METRIC_KEYWORDS.items():
        if phrase in low and metric not in out:
            out.append(metric)
    return tuple(out)


def route(question: str) -> QueryPlan:
    q = question.strip()
    tickers, years, metrics = extract_tickers(q), extract_years(q), extract_metrics(q)

    visual = _has(VISUAL_WORDS, q)
    verify = _has(VERIFY_WORDS, q) or _has(FILING_WORDS, q)
    comparing = len(tickers) >= 2 or _has(COMPARE_WORDS, q)
    asks_why = _has(WHY_WORDS, q)
    numeric = bool(metrics) and (_has(NUMERIC_WORDS, q) or comparing) and not asks_why

    reasons: list[str] = []
    if visual and (_has(PRESENTATION_WORDS, q) or _has(FILING_WORDS, q)) and verify:
        intent, retrievers = Intent.CROSS_MODAL, ("visual", "text", "sql")
        reasons.append("mentions a chart/slide and asks to check it against filings")
    elif visual:
        intent, retrievers = Intent.VISUAL, ("visual", "text")
        reasons.append("asks about a chart, slide or figure")
    elif comparing:
        intent = Intent.COMPARISON
        retrievers = ("sql", "text") if metrics else ("text",)
        reasons.append("compares companies or periods")
    elif numeric:
        intent, retrievers = Intent.NUMERIC, ("sql", "text")
        reasons.append("asks for a specific financial figure")
    else:
        intent, retrievers = Intent.NARRATIVE, ("text",)
        reasons.append("open-ended or explanatory question")

    return QueryPlan(q, intent, tickers, years, metrics, retrievers, tuple(reasons))
