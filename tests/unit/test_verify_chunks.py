from decimal import Decimal
from finsight.retrieval.sql_guard import validate_sql
from finsight.ingestion.pipelines.verify_chunks import contains_number, number_token, verify_doc


def test_number_token_formats_millions_with_commas():
    assert number_token(Decimal("60922000000")) == "60,922"
    assert number_token(Decimal("-1234000000")) == "1,234"          # losses print in parentheses
    assert number_token(Decimal("383285000000")) == "383,285"
    assert number_token(Decimal("950000")) == "1"


def test_contains_number_is_whole_number_match():
    assert contains_number("| Revenue | $60,922 | $26,974 |", "60,922")
    assert contains_number("| Net loss | $(1,234) |", "1,234")
    assert not contains_number("| Revenue | $160,922 |", "60,922")
    assert not contains_number("| Revenue | 60,9225 |", "60,922")
    assert not contains_number("| Margin | 60,922.5 |", "60,922")


class FakeRetriever:
    def __init__(self, values):
        self.values, self.sql = values, []

    def run_sql(self, sql):
        validate_sql(sql)          # every generated query must pass the real SQL guard
        self.sql.append(sql)
        for metric, v in self.values.items():
            if f"metric = '{metric}'" in sql:
                return [{"value": v}]
        return []


def _rows(doc_type="10-K", fp="FY", table="| Revenue | $60,922 | $26,974 |\n| Net income | $29,760 |"):
    src = dict(doc_id="nvda-x", ticker="NVDA", doc_type=doc_type, fiscal_year=2024, fiscal_period=fp)
    return [
        {"modality": "text", "text": "R&D was 8,675 in text only", "source": src},
        {"modality": "table", "text": table, "source": src},
    ]


def test_verify_doc_pass_fail_skip():
    r = FakeRetriever({"revenue": Decimal("60922000000"), "net_income": Decimal("29760000000"),
                       "rd_expense": Decimal("8675000000")})
    got = {c.metric: c.status for c in verify_doc(r, _rows())}
    assert got == {"revenue": "PASS", "net_income": "PASS",
                   "rd_expense": "FAIL",          # number only in prose, not in a table chunk
                   "operating_income": "SKIP"}    # not in XBRL


def test_verify_doc_10q_uses_quarter_period():
    r = FakeRetriever({"revenue": Decimal("26044000000")})
    verify_doc(r, _rows("10-Q", "Q1", "| Revenue | $26,044 | $7,192 |"))
    assert "period_type = 'Q'" in r.sql[0] and "fiscal_quarter = 1" in r.sql[0]


def test_verify_doc_empty():
    assert verify_doc(FakeRetriever({}), []) == []