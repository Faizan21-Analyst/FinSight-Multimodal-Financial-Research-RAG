"""Signature demo: "Here's a chart from the investor deck. Is it supported by the 10-K?"

Finds a slide, extracts its chart with the VLM, and checks every plotted value against XBRL facts
(GAAP totals) and 10-K/10-Q text (segments, percentages).

Usage:
  python scripts/verify_slide.py "Data Center segment revenue" --ticker AMD
  python scripts/verify_slide.py --ticker AMD --page 7                 # a specific slide, no search
  python scripts/verify_slide.py "revenue by quarter" --ticker NVDA --mode visual
  python scripts/verify_slide.py "..." --no-sql                         # text evidence only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from qdrant_client import QdrantClient  # noqa: E402

from finsight.agents.cross_modal import CrossModalVerifier  # noqa: E402
from finsight.core.caching import DiskCache  # noqa: E402
from finsight.core.config import get_settings, load_app_config  # noqa: E402
from finsight.indexing.embedders import build_embedder, collection_name  # noqa: E402
from finsight.retrieval.text_retriever import SearchFilter, TextRetriever  # noqa: E402
from finsight.retrieval.visual_retriever import PageTextRetriever  # noqa: E402
from finsight.vision.chart_extractor import ChartExtractor, GroqVLM  # noqa: E402
from finsight.vision.page_renderer import read_manifest  # noqa: E402

if __name__ == "__main__":
    cfg = load_app_config()
    vcfg = cfg.get("vision", {})
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--ticker", required=True)
    p.add_argument("--year", type=int)
    p.add_argument("--page", type=int, help="verify this slide number directly (skips search)")
    p.add_argument("--mode", choices=["ocr", "visual"], default="ocr")
    p.add_argument("--pages-dir", default=vcfg.get("pages_dir", "data/processed/pages"))
    p.add_argument("--no-sql", action="store_true")
    p.add_argument("--device")
    a = p.parse_args()
    if not (a.query or a.page):
        sys.exit("give a query, or --page N")

    s, ticker = get_settings(), a.ticker.upper()
    pages = [pg for pg in read_manifest(Path(a.pages_dir) / "pages.jsonl") if pg.ticker == ticker]
    if a.year:
        pages = [pg for pg in pages if pg.fiscal_year == a.year]
    if not pages:
        sys.exit(f"no rendered slides for {ticker}. Run scripts/build_pages.py first.")

    # 1. pick the slide
    if a.page:
        match = [pg for pg in pages if pg.page == a.page]
        if not match:
            sys.exit(f"{ticker} has no slide {a.page}")
        slide = match[0]
    else:
        filt = SearchFilter(tickers=(ticker,), fiscal_years=(a.year,) if a.year else None)
        if a.mode == "visual":
            from finsight.indexing.visual_embedders import build_visual_embedder
            from finsight.retrieval.visual_retriever import VisualRetriever
            vm = cfg["models"]["visual_retriever"]
            hits = VisualRetriever(
                QdrantClient(url=s.qdrant_url, timeout=120),
                build_visual_embedder(vm, device=a.device),
                collection_name(cfg["qdrant"]["visual_collection"], vm),
            ).search(a.query, 1, filt)
        else:
            hits = PageTextRetriever(pages).search(a.query, 1, filt)
        if not hits:
            sys.exit("no slide matched that query")
        slide = next(pg for pg in pages if pg.page_id == hits[0].evidence_id)
    print(f"slide: {slide.ticker} FY{slide.fiscal_year} {slide.fiscal_period or ''} p.{slide.page}\n"
          f"       {slide.image_path}")

    # 2. read the chart
    vlm = GroqVLM(vcfg.get("vlm_model", "qwen/qwen3.8-27b"), s.groq_api_key, DiskCache(Path(s.cache_dir)),
                  base_url=vcfg.get("vlm_base_url") or None, extra_body=vcfg.get("vlm_extra_body") or None)
    chart = ChartExtractor(vlm).extract(slide.image_path)
    if not chart.has_chart or not chart.series:
        sys.exit("\nno chart found on this slide")
    print(f"\nchart: {chart.title or '(untitled)'}  [{chart.chart_type}, "
          f"{' '.join(x for x in (chart.unit_scale, chart.unit_label) if x) or 'no units'}"
          f"{', values estimated' if chart.values_are_estimates else ''}]")
    print(chart.to_markdown())

    # 3. verify against the filings
    sql = None
    if not a.no_sql:
        from finsight.retrieval.sql_retriever import SqlRetriever
        dsn = next((getattr(s, n) for n in ("postgres_dsn", "database_url", "pg_dsn") if getattr(s, n, None)), None)
        if not dsn:
            sys.exit("no Postgres DSN in settings (use --no-sql for text evidence only)")
        sql = SqlRetriever(dsn)
    model, r = cfg["models"]["embedding"], cfg["retrieval"]
    text = TextRetriever(
        QdrantClient(url=s.qdrant_url, timeout=60), build_embedder(model, device=a.device),
        collection_name(cfg["qdrant"]["text_collection"], model),
        top_k_dense=r["top_k_dense"], top_k_sparse=r["top_k_sparse"], rrf_k=r["rrf_k"],
    )
    report = CrossModalVerifier(sql, text).verify(chart, ticker, slide.fiscal_year, slide.fiscal_period)

    print(f"\n== verification ==\n{report.to_markdown()}")
    notes = [(c.claim.label, c.note) for c in report.checks if c.note]
    for label, note in notes:
        print(f"  - {label}: {note}")
    print(f"\nOVERALL: {report.verdict.value.upper()}   {report.counts()}")
