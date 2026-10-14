"""Embed every chunk in data/processed/chunks and upsert into Qdrant.

Usage:
  python scripts/build_index.py --model BAAI/bge-small-en-v1.5
  python scripts/build_index.py
  python scripts/build_index.py --tickers NVDA --recreate
  python scripts/build_index.py --sparse-only

Re-running is cheap: chunks already in the collection are skipped (no re-embedding).
Each model gets its own collection, e.g. finsight_text__bge-small-en-v1.5.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from qdrant_client import QdrantClient  # noqa: E402

from finsight.core.config import get_settings, load_app_config  # noqa: E402
from finsight.core.logging import setup_logging  # noqa: E402
from finsight.indexing.embedders import build_embedder, collection_name  # noqa: E402
from finsight.indexing.text_index import TextIndexer, load_chunks  # noqa: E402


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="override models.embedding from settings.yaml")
    p.add_argument("--tickers", nargs="*")
    p.add_argument(
        "--recreate",
        action="store_true",
        help="drop and rebuild the collection",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-embed even chunks already indexed",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", help="cpu | cuda | mps (default: auto)")
    p.add_argument(
        "--sparse-only",
        action="store_true",
        help="rebuild only the BM25 vectors of already-indexed chunks (seconds, no model)",
    )
    a = p.parse_args()

    s = get_settings()
    setup_logging(s.log_level)
    cfg = load_app_config()
    model = a.model or cfg["models"]["embedding"]
    coll = collection_name(cfg["qdrant"]["text_collection"], model)

    chunks = load_chunks(
        Path(s.data_dir) / "processed" / "chunks",
        a.tickers,
    )
    if not chunks:
        sys.exit("no chunks found; run scripts/build_chunks.py first")

    embedder = (
        build_embedder(model, batch_size=a.batch_size, device=a.device)
        if not model.startswith("hashing")
        else build_embedder(model)
    )

    client = QdrantClient(url=s.qdrant_url, timeout=60)
    indexer = TextIndexer(
        client,
        embedder,
        coll,
        batch_size=a.batch_size,
    )

    if a.sparse_only:
        if not client.collection_exists(coll):
            sys.exit(
                f"collection {coll} does not exist; run a normal build first"
            )

        def sparse_progress(done: int, total: int) -> None:
            if done == total or done % 1600 < a.batch_size:
                print(f"  {done}/{total}")

        n = indexer.refresh_sparse(chunks, progress=sparse_progress)
        print(
            f"\nrefreshed sparse vectors of {n} chunks in {coll} "
            "(dense vectors untouched)"
        )
        sys.exit(0)

    indexer.ensure_collection(recreate=a.recreate)

    print(
        f"model      : {model}\n"
        f"collection : {coll}\n"
        f"chunks     : {len(chunks)}"
    )
    t0 = time.time()

    def progress(st):
        if (
            st.total % (a.batch_size * 10) < a.batch_size
            or st.total == len(chunks)
        ):
            rate = st.indexed / max(time.time() - t0, 1e-9)
            left = (
                (len(chunks) - st.total) / rate / 60
                if rate and st.indexed
                else 0
            )
            print(
                f"  {st.total:>6}/{len(chunks)}  "
                f"indexed {st.indexed}  skipped {st.skipped}"
                f"  {rate:5.1f} chunks/s"
                + (f"  ~{left:.0f} min left" if left else "")
            )

    stats = indexer.index(chunks, force=a.force, progress=progress)
    print(
        f"\ndone in {time.time() - t0:.0f}s: "
        f"indexed {stats.indexed}, skipped {stats.skipped}, "
        f"collection now holds {client.count(coll).count} points"
    )
