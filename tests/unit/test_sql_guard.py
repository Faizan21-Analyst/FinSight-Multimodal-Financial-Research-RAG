import pytest

from finsight.core.exceptions import UnsafeQueryError
from finsight.retrieval.sql_guard import validate_sql

OK = "SELECT value FROM finsight.xbrl_facts WHERE ticker = 'NVDA' AND metric = 'rd_expense'"


def test_accepts_select_and_adds_limit():
    out = validate_sql(OK)
    assert out.endswith("LIMIT 200")


def test_keeps_existing_limit_and_trailing_semicolon():
    out = validate_sql(OK + " LIMIT 5;")
    assert out.endswith("LIMIT 5") and ";" not in out


def test_accepts_unqualified_table_cte_and_aggregates():
    validate_sql("SELECT ticker, SUM(value) FROM xbrl_facts GROUP BY ticker")
    validate_sql("WITH a AS (SELECT * FROM finsight.xbrl_facts) SELECT * FROM a")


def test_semicolon_inside_literal_is_fine():
    validate_sql("SELECT * FROM finsight.xbrl_facts WHERE tag = 'a;b'")


@pytest.mark.parametrize("bad", [
    "DELETE FROM finsight.xbrl_facts",
    "DROP TABLE finsight.xbrl_facts",
    "UPDATE finsight.xbrl_facts SET value = 0",
    "SELECT 1; DROP TABLE finsight.xbrl_facts",
    "SELECT * FROM pg_user",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM other_table",
    "SELECT * FROM finsight.xbrl_facts, secrets",
    "SELECT pg_sleep(10)",
    "SELECT * FROM finsight.xbrl_facts -- hi",
    "SELECT * FROM finsight.xbrl_facts /* x */",
    "SELECT * INTO newtable FROM finsight.xbrl_facts",
    'SELECT * FROM "finsight"."xbrl_facts"',
    "SELECT * FROM finsight.xbrl_facts LIMIT 100000",
    "EXPLAIN SELECT 1",
    "",
])
def test_rejects_unsafe(bad):
    with pytest.raises(UnsafeQueryError):
        validate_sql(bad)