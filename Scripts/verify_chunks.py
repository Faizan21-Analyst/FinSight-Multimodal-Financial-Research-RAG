#!/usr/bin/env python3
"""Module 3 quality gate: do XBRL numbers appear in the parsed table chunks?
Usage: python scripts/verify_chunks.py [--tickers NVDA] [--show-pass]
Needs Postgres up and XBRL loaded (Module 2)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.core.config import get_settings  # noqa: E402
from finsight.ingestion.pipelines.verify_chunks import load_chunks, verify_doc  # noqa: E402
from finsight.retrieval.sql_retriever import SqlRetriever  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--show-pass", action="store_true")
    a = p.parse_args()

    s = get_settings()
    retriever = SqlRetriever(s.postgres_dsn)
    files = sorted((Path(s.data_dir) / "processed" / "chunks").glob("*.jsonl"))
    if a.tickers:
        keep = tuple(t.lower() + "-" for t in a.tickers)
        files = [f for f in files if f.stem.startswith(keep)]
    if not files:
        sys.exit("no chunk files found; run scripts/build_chunks.py")

    totals = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    print(f"{'doc_id':30} {'pass':>4} {'fail':>4} {'skip':>4}   failures (metric: expected number)")
    for f in files:
        checks = verify_doc(retriever, load_chunks(f))
        n = {k: sum(c.status == k for c in checks) for k in totals}
        for k in totals:
            totals[k] += n[k]
        fails = ", ".join(f"{c.metric}: {c.expected}" for c in checks if c.status == "FAIL")
        if a.show_pass or fails:
            print(f"{f.stem:30} {n['PASS']:>4} {n['FAIL']:>4} {n['SKIP']:>4}   {fails}")

    done = totals["PASS"] + totals["FAIL"]
    rate = totals["PASS"] / done if done else 0.0
    print(f"\nPASS {totals['PASS']}  FAIL {totals['FAIL']}  SKIP {totals['SKIP']}  "
          f"-> {rate:.0%} of checkable numbers found in parsed tables")
    sys.exit(0 if rate >= 0.9 else 1)