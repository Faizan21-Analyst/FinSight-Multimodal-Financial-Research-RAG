#!/usr/bin/env python3
"""Module 4 done-test: query the index from the command line.

Examples:
  python scripts/search.py "Data Center revenue" --ticker NVDA --fy 2025 --kind table
  python scripts/search.py "export controls China" --mode all --k 5
  python scripts/search.py "22,563" --mode sparse
Use --model to pick the collection (must match what you passed to build_index.py)."""
import argparse
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from qdrant_client import QdrantClient  # noqa: E402

from finsight.core.config import get_settings, load_app_config  # noqa: E402
from finsight.indexing.embedders import build_embedder, collection_name  # noqa: E402
from finsight.retrieval.text_retriever import MODES, SearchFilter, TextRetriever  # noqa: E402


def show(title, hits, width=150):
    print(f"\n=== {title} ===")
    if not hits:
        print("  (no results)")
    for i, h in enumerate(hits, 1):
        s = h.source
        where = f"{s.ticker} {s.doc_type.value} FY{s.fiscal_year} {s.fiscal_period} | {s.section} | p{s.page}"
        snippet = h.content.replace("\n", " ")[:width]
        print(f"{i:>2}. {h.score:7.4f}  [{h.modality.value:5}] {where}\n      {snippet}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--mode", choices=[*MODES, "all"], default="hybrid")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model")
    p.add_argument("--ticker", nargs="*")
    p.add_argument("--fy", nargs="*", type=int, help="fiscal year(s)")
    p.add_argument("--doc-type", nargs="*", choices=["10-K", "10-Q"])
    p.add_argument("--period", nargs="*", help="FY Q1 Q2 Q3")
    p.add_argument("--section", nargs="*", help='e.g. "Item 7" "Part I Item 2"')
    p.add_argument("--kind", nargs="*", choices=["text", "table"])
    a = p.parse_args()

    s = get_settings()
    cfg = load_app_config()
    model = a.model or cfg["models"]["embedding"]
    coll = collection_name(cfg["qdrant"]["text_collection"], model)
    r = cfg.get("retrieval", {})
    retriever = TextRetriever(
        QdrantClient(url=s.qdrant_url, timeout=60), build_embedder(model), coll,
        top_k_dense=r.get("top_k_dense", 30), top_k_sparse=r.get("top_k_sparse", 30),
        rrf_k=r.get("rrf_k", 60),
    )
    filt = SearchFilter(
        tickers=tuple(t.upper() for t in a.ticker) if a.ticker else None,
        fiscal_years=tuple(a.fy) if a.fy else None,
        doc_types=tuple(a.doc_type) if a.doc_type else None,
        fiscal_periods=tuple(a.period) if a.period else None,
        sections=tuple(a.section) if a.section else None,
        modalities=tuple(a.kind) if a.kind else None,
    )
    print(f"collection: {coll}\nquery     : {a.query}")
    for mode in (MODES if a.mode == "all" else (a.mode,)):
        show(mode.upper(), retriever.search(a.query, mode=mode, limit=a.k, filt=filt))