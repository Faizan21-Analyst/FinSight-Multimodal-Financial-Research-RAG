"""Create the xbrl_facts table and upsert parsed facts. Idempotent: safe to re-run."""
from __future__ import annotations

from decimal import Decimal

import psycopg

from finsight.indexing.xbrl_parse import Fact

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS finsight;

CREATE TABLE IF NOT EXISTS finsight.xbrl_facts (
    id             BIGSERIAL PRIMARY KEY,
    ticker         TEXT    NOT NULL,
    cik            TEXT    NOT NULL,
    metric         TEXT    NOT NULL,
    tag            TEXT    NOT NULL,
    unit           TEXT    NOT NULL,
    period_type    TEXT    NOT NULL CHECK (period_type IN ('FY','Q','YTD','instant')),
    fiscal_year    INT     NOT NULL,
    fiscal_quarter INT,
    period_start   DATE,
    period_end     DATE    NOT NULL,
    value          NUMERIC NOT NULL,
    accession      TEXT    NOT NULL,
    form           TEXT    NOT NULL,
    filed          DATE    NOT NULL,
    UNIQUE (ticker, metric, period_type, period_end)
);

CREATE INDEX IF NOT EXISTS ix_xbrl_lookup
    ON finsight.xbrl_facts (ticker, metric, period_type, fiscal_year);
"""

UPSERT_SQL = """
INSERT INTO finsight.xbrl_facts
    (ticker, cik, metric, tag, unit, period_type, fiscal_year, fiscal_quarter,
     period_start, period_end, value, accession, form, filed)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, metric, period_type, period_end) DO UPDATE SET
    cik = EXCLUDED.cik, tag = EXCLUDED.tag, unit = EXCLUDED.unit,
    fiscal_year = EXCLUDED.fiscal_year, fiscal_quarter = EXCLUDED.fiscal_quarter,
    period_start = EXCLUDED.period_start, value = EXCLUDED.value,
    accession = EXCLUDED.accession, form = EXCLUDED.form, filed = EXCLUDED.filed;
"""


def fact_to_row(f: Fact) -> tuple:
    return (
        f.ticker, f.cik, f.metric, f.tag, f.unit, f.period_type, f.fiscal_year,
        f.fiscal_quarter, f.period_start, f.period_end, Decimal(str(f.value)),
        f.accession, f.form, f.filed,
    )


def ensure_schema(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        conn.execute(SCHEMA_SQL)


def load_facts(dsn: str, facts: list[Fact]) -> int:
    """Upsert facts; returns number of rows written."""
    rows = [fact_to_row(f) for f in facts]
    with psycopg.connect(dsn) as conn:
        conn.execute(SCHEMA_SQL)
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, rows)
    return len(rows)