#!/usr/bin/env python3
"""Module 3 done-test: coverage table + random chunks to eyeball.
Usage: python scripts/inspect_chunks.py [--n 20] [--kind table|text] [--doc DOC_ID] [--seed 1]"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.core.config import get_settings  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--kind", choices=["text", "table"])
    p.add_argument("--doc")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    d = Path(get_settings().data_dir) / "processed" / "chunks"
    files = sorted(d.glob("*.jsonl"))
    if a.doc:
        files = [f for f in files if a.doc in f.stem]
    if not files:
        sys.exit(f"no chunk files in {d}; run scripts/build_chunks.py")

    allc = []
    print(f"{'doc_id':34} {'chunks':>6} {'text':>5} {'table':>5} {'sections':>8}  top sections")
    for f in files:
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]
        kinds = Counter(r["modality"] for r in rows)
        secs = Counter(r["source"]["section"] for r in rows)
        top = ", ".join(f"{k}:{v}" for k, v in secs.most_common(4))
        print(f"{f.stem:34} {len(rows):>6} {kinds['text']:>5} {kinds['table']:>5} "
              f"{len(secs):>8}  {top}")
        allc += rows

    pool = [r for r in allc if not a.kind or r["modality"] == a.kind]
    random.Random(a.seed).shuffle(pool)
    print(f"\n=== {min(a.n, len(pool))} random chunks ===")
    for r in pool[: a.n]:
        s = r["source"]
        print(f"\n--- {r['modality'].upper()} | {s['ticker']} {s['doc_type']} FY{s['fiscal_year']} "
              f"{s['fiscal_period']} | {s['section']} | page {s['page']} | {r['chunk_id']}")
        body = r["text"]
        print(body[:600] + (" ..." if len(body) > 600 else ""))