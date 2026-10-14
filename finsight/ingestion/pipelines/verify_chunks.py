"""Cross-check Module 3 (parsed tables) against Module 2 (XBRL numbers in Postgres).

For each filing we ask: does the XBRL value of revenue / R&D / operating income / net income
for this filing's own period appear as a number in that filing's TABLE chunks?
All four companies report in millions, so XBRL 60922000000 must show up as "60,922".

A FAIL means one of: table cells were mangled by the parser, the section/table was dropped,
or XBRL holds a restated value that differs from the original filing (worth knowing too).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol

METRICS = ("revenue", "operating_income", "net_income", "rd_expense")


class SupportsRunSql(Protocol):
    def run_sql(self, sql: str) -> list[dict]: ...


@dataclass
class Check:
    doc_id: str
    metric: str
    expected: str  # e.g. "60,922"; "" when skipped
    status: str  # PASS | FAIL | SKIP


def number_token(value: Decimal | float, scale: int = 1_000_000) -> str:
    """XBRL raw dollars -> the string a filing reported in millions, e.g. '60,922'."""
    return f"{round(abs(float(value)) / scale):,}"


def contains_number(text: str, token: str) -> bool:
    """Whole-number match: '60,922' must not match inside '160,922' or '60,9225'."""
    return re.search(rf"(?<![\d,.]){re.escape(token)}(?![\d,]|\.\d)", text) is not None


def load_chunks(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _sql_value(r: SupportsRunSql, ticker: str, metric: str, ptype: str, fy: int,
               q: int | None) -> Decimal | None:
    if not ticker.isalnum():
        raise ValueError(f"unsafe ticker {ticker!r}")
    sql = (
        "SELECT COALESCE(original_value, value) AS value"
        " FROM finsight.xbrl_facts"
        f" WHERE ticker = '{ticker}' AND metric = '{metric}'"
        f" AND period_type = '{ptype}' AND fiscal_year = {int(fy)}"
    )
    if q:
        sql += f" AND fiscal_quarter = {int(q)}"
    rows = r.run_sql(sql)
    return rows[0]["value"] if rows else None


def verify_doc(retriever: SupportsRunSql, rows: list[dict]) -> list[Check]:
    if not rows:
        return []
    src = rows[0]["source"]
    doc_id, ticker, fy, fp = src["doc_id"], src["ticker"], src["fiscal_year"], src["fiscal_period"]
    if src["doc_type"] == "10-K":
        ptype, q = "FY", None
    else:
        ptype, q = "Q", int(fp[1:])
    tables = "\n".join(r["text"] for r in rows if r["modality"] == "table")

    checks = []
    for metric in METRICS:
        val = _sql_value(retriever, ticker, metric, ptype, fy, q)
        if val is None:
            checks.append(Check(doc_id, metric, "", "SKIP"))
            continue
        token = number_token(val)
        ok = contains_number(tables, token)
        checks.append(Check(doc_id, metric, token, "PASS" if ok else "FAIL"))
    return checks