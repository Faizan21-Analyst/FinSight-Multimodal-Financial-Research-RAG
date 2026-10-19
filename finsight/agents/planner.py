"""Planner: turns a QueryPlan into concrete retrieval requests.

Why this exists (Module 5 finding): comparison questions miss because one blended query
retrieves chunks for only one company. The fix is structural, not a better ranker:
  * comparison questions get ONE sub-query PER COMPANY, each filtered to that ticker;
  * numeric questions with ticker + metric + year get exact SQL lookups (XBRL = source of truth).
Deterministic and free; the LLM is only used later, to grade and rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass

from finsight.agents.router import Intent, QueryPlan, extract_tickers
from finsight.retrieval.text_retriever import SearchFilter


@dataclass(frozen=True)
class SubQuery:
    text: str
    tickers: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()

    def search_filter(self, relax_years: bool = False) -> SearchFilter | None:
        years = () if relax_years else self.fiscal_years
        if not (self.tickers or years):
            return None
        return SearchFilter(tickers=self.tickers or None, fiscal_years=years or None)


@dataclass(frozen=True)
class SqlRequest:
    ticker: str
    metric: str
    fiscal_year: int


def build_subqueries(plan: QueryPlan, text: str | None = None) -> list[SubQuery]:
    """`text` overrides the question wording (used when the agent rewrites the query).

    For per-company sub-queries a rewrite only replaces the wording of the company it is about:
    a rewrite that names NVIDIA must not be run for AMD. Companies the rewrite does not name
    keep the original question (a rewrite naming no company applies to all).
    """
    q = text or plan.question
    if plan.intent in (Intent.COMPARISON, Intent.CROSS_MODAL) and len(plan.tickers) >= 2:
        named = extract_tickers(q) if text else ()
        return [
            SubQuery(q if (not named or t in named) else plan.question, (t,), plan.fiscal_years)
            for t in plan.tickers
        ]
    return [SubQuery(q, plan.tickers, plan.fiscal_years)]


def sql_requests(plan: QueryPlan) -> list[SqlRequest]:
    """Exact lookups, only when the question names ticker, metric and fiscal year."""
    if "sql" not in plan.retrievers:
        return []
    return [
        SqlRequest(t, m, y)
        for t in plan.tickers
        for m in plan.metrics
        for y in plan.fiscal_years
    ]
