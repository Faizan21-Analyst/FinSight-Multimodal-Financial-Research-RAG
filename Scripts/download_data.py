#!/usr/bin/env python3
"""Usage: python scripts/download_data.py [--years 3] [--tickers NVDA AMD] [--refresh]"""
import argparse
import sys

sys.path.insert(0, "src")

from finsight.core.config import get_settings  # noqa: E402
from finsight.core.logging import setup_logging  # noqa: E402
from finsight.ingestion.pipelines.acquire import run  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--refresh", action="store_true", help="re-download companyfacts")
    a = p.parse_args()
    setup_logging(get_settings().log_level)
    run(years=a.years, tickers=a.tickers, refresh=a.refresh)