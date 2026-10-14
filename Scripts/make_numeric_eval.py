#!/usr/bin/env python3
"""Generate numeric lookup questions with objective gold from XBRL facts + parsed chunks.

Usage: python scripts/make_numeric_eval.py [--max-quarterly 24] [--out data/eval/numeric.jsonl]
Needs Postgres up with XBRL loaded (Module 2) and chunk files built (Module 3)."""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.core.config import get_settings  # noqa: E402
from finsight.evaluation.dataset import save_questions  # noqa: E402
from finsight.evaluation.numeric_questions import METRIC_TEXT, make_numeric_questions  # noqa: E402
from finsight.indexing.text_index import chunk_payload, load_chunks  # noqa: E402
from finsight.retrieval.sql_retriever import SqlRetriever  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/eval/numeric.jsonl")
    p.add_argument("--max-quarterly", type=int, default=24)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    s = get_settings()
    metrics = ", ".join(f"'{m}'" for m in METRIC_TEXT)
    rows = SqlRetriever(s.postgres_dsn, max_rows=5000).run_sql(
        "SELECT ticker, metric, period_type, fiscal_year, fiscal_quarter,"
        " COALESCE(original_value, value) AS value"
        f" FROM finsight.xbrl_facts WHERE metric IN ({metrics}) AND period_type IN ('FY', 'Q')"
    )
    corpus = [chunk_payload(c) for c in load_chunks(Path(s.data_dir) / "processed" / "chunks")]
    print(f"{len(rows)} XBRL facts, {len(corpus)} chunks")

    questions, skipped = make_numeric_questions(rows, corpus, a.max_quarterly, a.seed)
    save_questions(Path(a.out), questions)

    print(f"\nwrote {len(questions)} questions to {a.out}")
    by = Counter((q.gold[0].ticker, "quarterly" if q.id.endswith(("-q1", "-q2", "-q3")) else "annual")
                 for q in questions)
    for (ticker, kind), n in sorted(by.items()):
        print(f"  {ticker} {kind:9} {n}")
    if skipped:
        print(f"\nskipped {len(skipped)} (first 15):")
        for line in skipped[:15]:
            print("  -", line)