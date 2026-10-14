"""Auto-generate numeric lookup questions with OBJECTIVE gold from XBRL (Modules 2 and 3).

For every (company, metric, period) in the XBRL facts we ask "What was X's <metric> in fiscal
<year>?" and define gold as: a TABLE chunk of that company's filings that contains the exact
number as originally reported AND a line-item label for the metric.

Gold is deliberately lenient about WHICH filing: FY2024 R&D also appears as a comparative column
in the FY2025 10-K, and a table showing it is a valid answer. The question names the year, so a
retriever that finds the right year-specific table is rewarded, and the oracle-filter runs show
how much knowing (company, year, form) would add.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from finsight.evaluation.dataset import GoldSpec, OracleFilter, Question, resolve_gold
from finsight.ingestion.pipelines.verify_chunks import number_token

NAMES = {"NVDA": "NVIDIA", "AMD": "AMD", "AAPL": "Apple", "TSLA": "Tesla"}

# metric -> (phrase used in the question, labels a filing table may use for that line item)
METRIC_TEXT: dict[str, tuple[str, list[str]]] = {
    "revenue": ("total revenue", ["revenue", "net sales"]),
    "rd_expense": ("research and development expense", ["research and development"]),
    "operating_income": ("operating income", ["operating income", "income from operations"]),
    "net_income": ("net income", ["net income"]),
}
ORDINAL = {1: "first", 2: "second", 3: "third"}
MIN_TOKEN_CHARS = 3  # "5" or "12" would match too many unrelated tables
MAX_GOLD = 150


def make_numeric_questions(
    rows: Sequence[dict],
    corpus: Sequence[dict],
    max_quarterly: int = 24,
    seed: int = 7,
) -> tuple[list[Question], list[str]]:
    """rows: dicts with ticker, metric, period_type ('FY'|'Q'), fiscal_year, fiscal_quarter,
    value (the ORIGINALLY reported value in dollars). Returns (questions, skipped_reasons)."""
    questions: list[Question] = []
    skipped: list[str] = []

    def build(r: dict) -> Question | None:
        if r["metric"] not in METRIC_TEXT or r["ticker"] not in NAMES:
            return None
        label, labels = METRIC_TEXT[r["metric"]]
        token = number_token(r["value"])
        tag = f"{r['ticker']} {r['metric']} FY{r['fiscal_year']}" + (
            f" Q{r['fiscal_quarter']}" if r["period_type"] == "Q" else "")
        if len(token.replace(",", "")) < MIN_TOKEN_CHARS:
            skipped.append(f"{tag}: number {token} too short to be a reliable needle")
            return None
        spec = GoldSpec(ticker=r["ticker"], modality="table", all_of=[token], any_of=labels)
        n = len(resolve_gold(corpus, spec))
        if n == 0:
            skipped.append(f"{tag}: no table chunk contains {token}")
            return None
        if n > MAX_GOLD:
            skipped.append(f"{tag}: {n} gold chunks (too generic)")
            return None
        name = NAMES[r["ticker"]]
        if r["period_type"] == "FY":
            text = f"What was {name}'s {label} in fiscal {r['fiscal_year']}?"
            oracle = OracleFilter(tickers=[r["ticker"]], fiscal_years=[r["fiscal_year"]], doc_types=["10-K"])
            qid = f"num-{r['ticker']}-{r['metric']}-fy{r['fiscal_year']}"
        else:
            q = r["fiscal_quarter"]
            text = f"What was {name}'s {label} in the {ORDINAL[q]} quarter of fiscal {r['fiscal_year']}?"
            oracle = OracleFilter(tickers=[r["ticker"]], fiscal_years=[r["fiscal_year"]], doc_types=["10-Q"])
            qid = f"num-{r['ticker']}-{r['metric']}-fy{r['fiscal_year']}-q{q}"
        return Question(id=qid, category="numeric_lookup", question=text, gold=[spec], oracle=oracle)

    annual = [r for r in rows if r["period_type"] == "FY"]
    quarterly = [r for r in rows if r["period_type"] == "Q" and r.get("fiscal_quarter") in ORDINAL]
    random.Random(seed).shuffle(quarterly)

    for r in sorted(annual, key=lambda r: (r["ticker"], r["metric"], r["fiscal_year"])):
        if (q := build(r)):
            questions.append(q)
    kept = 0
    for r in quarterly:
        if kept >= max_quarterly:
            break
        if (q := build(r)):
            questions.append(q)
            kept += 1
    seen: set[str] = set()
    unique = []
    for q in questions:
        if q.id not in seen:
            seen.add(q.id)
            unique.append(q)
    return unique, skipped