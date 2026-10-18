"""Embed rendered deck pages and upsert them into Qdrant (multi-vector, MaxSim).

Usage:
  python scripts/build_visual_index.py --model hashing        # plumbing test, no GPU, no download
  python scripts/build_visual_index.py --model vidore/colSmol-256M   # small, CPU-friendly
  python scripts/build_visual_index.py                         # settings.yaml model (ColQwen2, wants a GPU)
  python scripts/build_visual_index.py --tickers NVDA --recreate

Each model gets its own collection. Re-running skips pages already indexed.
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
from finsight.indexing.embedders import collection_name  # noqa: E402
from finsight.indexing.visual_embedders import build_visual_embedder  # noqa: E402
from finsight.indexing.visual_index import VisualIndexer  # noqa: E402
from finsight.vision.page_renderer import read_manifest  # noqa: E402

if __name__ == "__main__":
    cfg = load_app_config()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=cfg["models"]["visual_retriever"])
    p.add_argument("--pages-dir", default=cfg.get("vision", {}).get("pages_dir", "data/processed/pages"))
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--recreate", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--batch-size", type=int, default=2, help="pages per forward pass (lower if out of memory)")
    p.add_argument("--device", help="cpu | cuda | mps (default: auto)")
    a = p.parse_args()

    s = get_settings()
    setup_logging(s.log_level)
    pages = read_manifest(Path(a.pages_dir) / "pages.jsonl", a.tickers)
    if not pages:
        sys.exit("no pages in the manifest; run scripts/build_pages.py first")

    kwargs = {} if a.model.startswith("hashing") else {"device": a.device, "batch_size": a.batch_size}
    embedder = build_visual_embedder(a.model, **kwargs)
    coll = collection_name(cfg["qdrant"]["visual_collection"], a.model)

    client = QdrantClient(url=s.qdrant_url, timeout=120)
    indexer = VisualIndexer(client, embedder, coll, batch_size=a.batch_size)
    indexer.ensure_collection(recreate=a.recreate)
    print(f"model      : {a.model}\ncollection : {coll}\npages      : {len(pages)}")

    t0 = time.time()

    def progress(st):
        print(f"  {st.total:>5}/{len(pages)}  indexed {st.indexed}  skipped {st.skipped}"
              f"  {time.time() - t0:5.0f}s")

    stats = indexer.index(pages, force=a.force, progress=progress)
    print(f"\ndone in {time.time() - t0:.0f}s: indexed {stats.indexed}, skipped {stats.skipped}, "
          f"collection holds {client.count(coll).count} pages")
