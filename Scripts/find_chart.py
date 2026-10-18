"""Find the slide for a query (visual retrieval, OCR-text baseline, or both), optionally
extract the chart's data with a VLM.

Usage:
  python scripts/find_chart.py "data center revenue by quarter" --ticker NVDA
  python scripts/find_chart.py "gross margin trend" --mode ocr
  python scripts/find_chart.py "data center revenue" --ticker NVDA --extract
  python scripts/find_chart.py "..." --model hashing        # if you indexed with the test embedder
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from qdrant_client import QdrantClient  # noqa: E402

from finsight.core.caching import DiskCache  # noqa: E402
from finsight.core.config import get_settings, load_app_config  # noqa: E402
from finsight.core.logging import setup_logging  # noqa: E402
from finsight.indexing.embedders import collection_name  # noqa: E402
from finsight.indexing.visual_embedders import build_visual_embedder  # noqa: E402
from finsight.retrieval.text_retriever import SearchFilter  # noqa: E402
from finsight.retrieval.visual_retriever import PageTextRetriever, VisualRetriever  # noqa: E402
from finsight.vision.chart_extractor import ChartExtractor, GroqVLM, chart_evidence  # noqa: E402
from finsight.vision.page_renderer import read_manifest  # noqa: E402


def show(label, hits):
    print(f"\n== {label} ==")
    if not hits:
        print("  (no results)")
    for i, h in enumerate(hits, 1):
        s = h.source
        print(f"  {i}. {s.ticker} FY{s.fiscal_year} {s.fiscal_period or ''} p.{s.page}  "
              f"score {h.score:.3f}\n     {h.image_path}\n     {h.content[:110]!r}")


if __name__ == "__main__":
    cfg = load_app_config()
    vcfg = cfg.get("vision", {})
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--mode", choices=["visual", "ocr", "both"], default="both")
    p.add_argument("--model", default=cfg["models"]["visual_retriever"])
    p.add_argument("--pages-dir", default=vcfg.get("pages_dir", "data/processed/pages"))
    p.add_argument("--ticker", nargs="*")
    p.add_argument("--year", nargs="*", type=int)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--extract", action="store_true", help="run the VLM on the best hit")
    p.add_argument("--device")
    a = p.parse_args()

    s = get_settings()
    setup_logging(s.log_level)
    filt = SearchFilter(tickers=tuple(t.upper() for t in a.ticker) if a.ticker else None,
                        fiscal_years=tuple(a.year) if a.year else None)
    if not (filt.tickers or filt.fiscal_years):
        filt = None

    best = None
    if a.mode in ("ocr", "both"):
        pages = read_manifest(Path(a.pages_dir) / "pages.jsonl")
        hits = PageTextRetriever(pages).search(a.query, a.top, filt)
        show("OCR/text-layer BM25 (baseline)", hits)
        best = best or (hits[0] if hits else None)
    if a.mode in ("visual", "both"):
        kwargs = {} if a.model.startswith("hashing") else {"device": a.device}
        embedder = build_visual_embedder(a.model, **kwargs)
        coll = collection_name(cfg["qdrant"]["visual_collection"], a.model)
        retriever = VisualRetriever(QdrantClient(url=s.qdrant_url, timeout=120), embedder, coll)
        hits = retriever.search(a.query, a.top, filt)
        show(f"Visual retrieval ({a.model})", hits)
        best = hits[0] if hits else best

    if a.extract:
        if best is None:
            sys.exit("\nnothing to extract: no hits")
        cache = DiskCache(Path(s.cache_dir))
        vlm = GroqVLM(vcfg.get("vlm_model", "qwen/qwen3.8-27b"),
                      s.groq_api_key, cache, base_url=vcfg.get("vlm_base_url") or None,
                      extra_body=vcfg.get("vlm_extra_body") or None)
        chart = ChartExtractor(vlm).extract(best.image_path)
        print(f"\n== Extracted chart ({best.source.ticker} p.{best.source.page}) ==")
        print(chart.to_markdown())
        if chart.caveats:
            print("caveats:", "; ".join(chart.caveats))
        out = Path(best.image_path).with_suffix(".chart.json")
        out.write_text(json.dumps(chart.model_dump(), indent=2), encoding="utf-8")
        print(f"\nsaved {out}")
