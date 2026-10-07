#!/usr/bin/env python3
"""Usage: python scripts/load_xbrl.py [--tickers NVDA AMD]
Parses data/raw/<T>/companyfacts.json and upserts into Postgres."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.core.config import get_settings, load_companies  # noqa: E402
from finsight.core.logging import setup_logging, get_logger  # noqa: E402
from finsight.indexing.sql_loader import load_facts  # noqa: E402
from finsight.indexing.xbrl_parse import METRICS, parse_companyfacts  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*")
    a = p.parse_args()
    s = get_settings()
    setup_logging(s.log_level)
    log = get_logger("load_xbrl")
    companies = load_companies()
    if a.tickers:
        wanted = {t.upper() for t in a.tickers}
        companies = [c for c in companies if c.ticker in wanted]
    for c in companies:
        path = Path(s.data_dir) / "raw" / c.ticker / "companyfacts.json"
        if not path.exists():
            log.warning("%s: %s missing, run download_data.py first", c.ticker, path)
            continue
        facts = parse_companyfacts(json.loads(path.read_text()), c.ticker)
        n = load_facts(s.postgres_dsn, facts)
        by_type = Counter(f.period_type for f in facts)
        by_metric = Counter(f.metric for f in facts)
        log.info("%s: loaded %d facts %s", c.ticker, n, dict(by_type))
        missing = sorted(set(METRICS) - set(by_metric))
        if missing:
            log.warning("%s: no data for metrics: %s", c.ticker, ", ".join(missing))