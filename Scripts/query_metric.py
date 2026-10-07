#!/usr/bin/env python3
"""Module 2 done-test.  Usage: python scripts/query_metric.py NVDA rd_expense 2024"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.core.config import get_settings  # noqa: E402
from finsight.retrieval.sql_retriever import SqlRetriever  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: query_metric.py TICKER METRIC FISCAL_YEAR [PERIOD_TYPE] [QUARTER]")
    ticker, metric, fy = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ptype = sys.argv[4] if len(sys.argv) > 4 else "FY"
    q = int(sys.argv[5]) if len(sys.argv) > 5 else None
    ev = SqlRetriever(get_settings().postgres_dsn).get_metric(ticker, metric, fy, ptype, q)
    print(ev.content if ev else "NOT FOUND")