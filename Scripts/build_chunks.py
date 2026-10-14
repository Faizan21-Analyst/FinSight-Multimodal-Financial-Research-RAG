#!/usr/bin/env python3
"""Usage: python scripts/build_chunks.py [--tickers NVDA] [--force]
Parses every filing in data/manifests/filings.jsonl into data/processed/chunks/<doc_id>.jsonl"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.core.config import get_settings, load_app_config  # noqa: E402
from finsight.core.logging import setup_logging  # noqa: E402
from finsight.ingestion.chunking.chunker import ChunkConfig  # noqa: E402
from finsight.ingestion.pipelines.chunk_filings import run  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--force", action="store_true", help="re-chunk even if output exists")
    a = p.parse_args()
    s = get_settings()
    setup_logging(s.log_level)
    c = load_app_config()["chunking"]
    cfg = ChunkConfig(max_tokens=c["max_tokens"], overlap_tokens=c["overlap_tokens"])
    data = Path(s.data_dir)
    summaries = run(data / "manifests" / "filings.jsonl", data / "processed" / "chunks",
                    cfg, a.tickers, a.force)
    bad = [x for x in summaries if x.missing or x.chunks == 0]
    print(f"\n{len(summaries)} documents chunked, {len(bad)} with missing sections or no chunks")
    for x in bad:
        print(f"  CHECK {x.doc_id}: missing {x.missing}")