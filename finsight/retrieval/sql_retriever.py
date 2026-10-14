"""Structured-data retrieval over XBRL facts in Postgres.

Two entry points:
  * get_metric / get_series : deterministic parameterised lookups (no LLM). Exact numbers.
  * run_sql                 : guarded free-form SQL, for the text-to-SQL path added in Module 7.
Every result becomes an EvidenceItem so downstream modules treat SQL like any other evidence.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row

from finsight.core.exceptions import RetrievalError
from finsight.core.schemas import DocType, EvidenceItem, Modality, SourceRef, make_id
from finsight.retrieval.sql_guard import validate_sql

COLUMNS = (
    "ticker, metric, tag, unit, period_type, fiscal_year, fiscal_quarter, "
    "period_start, period_end, value, accession, form, filed, "
    "restated, original_value, original_accession, original_form, original_filed"
)


def period_label(row: dict[str, Any]) -> str:
    fy, q, pt = row["fiscal_year"], row["fiscal_quarter"], row["period_type"]
    if pt == "FY":
        return f"FY{fy}"
    if pt == "Q":
        return f"Q{q} FY{fy}"
    if pt == "YTD":
        return f"first {6 if q == 2 else 9} months of FY{fy}"
    return f"FY{fy} year-end" if q == 4 else f"end of Q{q} FY{fy}"


def format_value(value: Decimal | float, unit: str) -> str:
    v = float(value)
    if unit == "USD":
        if abs(v) >= 1e9:
            return f"${v / 1e9:,.3f} billion"
        if abs(v) >= 1e6:
            return f"${v / 1e6:,.0f} million"
        return f"${v:,.0f}"
    if unit == "USD/shares":
        return f"${v:.2f} per share"
    return f"{v:,.2f} {unit}"


def row_to_evidence(row: dict[str, Any], as_reported: bool = False) -> EvidenceItem:
    """Build evidence from a fact row.

    The value shown and the filing cited always agree:
      * not restated, or as_reported=True -> ORIGINAL value, cites the ORIGINAL filing
      * restated and as_reported=False    -> LATEST value, cites the LATEST filing, and the
                                             original figure is mentioned in a note
    """
    restated = bool(row.get("restated"))
    has_original = row.get("original_accession") is not None
    use_original = has_original and (as_reported or not restated)
    if use_original:
        value, accn = row["original_value"], row["original_accession"]
        form, filed = row["original_form"], row["original_filed"]
    else:
        value, accn, form, filed = row["value"], row["accession"], row["form"], row["filed"]

    label = period_label(row)
    start = f"{row['period_start']} to " if row["period_start"] else "as of "
    content = (
        f"{row['ticker']} {row['metric']} for {label}: {format_value(value, row['unit'])} "
        f"(raw value {value} {row['unit']}; period {start}{row['period_end']}; "
        f"XBRL tag {row['tag']}; reported in {form} filed {filed}, accession {accn})"
    )
    if restated and has_original and not as_reported:
        content += (
            f" NOTE: restated. Originally reported as "
            f"{format_value(row['original_value'], row['unit'])} "
            f"in {row['original_form']} filed {row['original_filed']}, "
            f"accession {row['original_accession']}."
        )
    fp = "FY" if row["period_type"] == "FY" else f"Q{row['fiscal_quarter']}"
    return EvidenceItem(
        evidence_id=make_id("xbrl", row["ticker"], row["metric"], row["period_type"],
                            row["period_end"], "orig" if use_original else "latest"),
        modality=Modality.SQL,
        content=content,
        source=SourceRef(
            doc_id=f"{row['ticker'].lower()}-xbrl-{accn}",
            ticker=row["ticker"],
            doc_type=DocType.XBRL,
            fiscal_year=row["fiscal_year"],
            fiscal_period=fp,
        ),
        score=1.0,
        retriever="sql",
    )

class SqlRetriever:
    def __init__(self, dsn: str, max_rows: int = 200, timeout_ms: int = 5000):
        self.dsn = dsn
        self.max_rows = max_rows
        self.timeout_ms = timeout_ms

    def _query(self, sql: str, params: tuple | None = None) -> list[dict[str, Any]]:
        try:
            with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
                conn.read_only = True  # the real safety net
                with conn.cursor() as cur:
                    cur.execute(f"SET LOCAL statement_timeout = {int(self.timeout_ms)}")
                    cur.execute(sql, params)
                    return cur.fetchall()
        except psycopg.Error as e:
            raise RetrievalError(f"SQL execution failed: {e}") from e

    def run_sql(self, sql: str) -> list[dict[str, Any]]:
        """Validate then execute free-form SQL (read-only, row-capped, timed out)."""
        return self._query(validate_sql(sql, self.max_rows))

    def get_metric(
        self,
        ticker: str,
        metric: str,
        fiscal_year: int,
        period_type: str = "FY",
        fiscal_quarter: int | None = None,
        as_reported: bool = False,
    ) -> EvidenceItem | None:
        sql = (
            f"SELECT {COLUMNS} FROM finsight.xbrl_facts "
            "WHERE ticker = %s AND metric = %s AND period_type = %s AND fiscal_year = %s "
            "AND (%s::int IS NULL OR fiscal_quarter = %s::int) LIMIT 1"
        )
        rows = self._query(sql, (ticker.upper(), metric, period_type, fiscal_year,
                                 fiscal_quarter, fiscal_quarter))
        return row_to_evidence(rows[0], as_reported) if rows else None

    def get_series(
        self,
        ticker: str,
        metric: str,
        period_type: str = "FY",
        start_year: int | None = None,
        end_year: int | None = None,
        as_reported: bool = False,
    ) -> list[EvidenceItem]:
        sql = (
            f"SELECT {COLUMNS} FROM finsight.xbrl_facts "
            "WHERE ticker = %s AND metric = %s AND period_type = %s "
            "AND (%s::int IS NULL OR fiscal_year >= %s::int) "
            "AND (%s::int IS NULL OR fiscal_year <= %s::int) "
            "ORDER BY period_end"
        )
        rows = self._query(sql, (ticker.upper(), metric, period_type,
                                 start_year, start_year, end_year, end_year))
        return [row_to_evidence(r, as_reported) for r in rows]