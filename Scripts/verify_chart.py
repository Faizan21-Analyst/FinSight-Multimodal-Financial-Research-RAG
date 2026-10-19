"""The signature demo: "Here's a chart from the investor deck. Is it supported by the 10-K?"

Finds the slide (by query, or by exact ticker + page number), reads its chart with the VLM, and checks
every data point against the filings (XBRL facts, then 10-K/10-Q text).

Usage:
  python scripts/verify_chart.py "Data Center segment revenue" --ticker AMD
  python scripts/verify_chart.py --ticker AMD --page 5                  # skip retrieval, use this slide
  python scripts/verify_chart.py "revenue by quarter" --ticker NVDA --no-sql
  python scripts/verify_chart.py "..." --mode visual                    # use ColPali retrieval instead of OCR BM25
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
    p.add_argument("--ticker", nargs="*")
    p.add_argument("--year", nargs="*", type=int)
    p.add_argument("--page", type=int, help="use this page number of the matching deck (no retrieval)")
    p.add_argument("--mode", choices=["ocr", "visual"], default="ocr")
    p.add_argument("--tamper", action="append", metavar="SERIES|LABEL=VALUE",
                   help="demo: overwrite a chart value before verifying, e.g. --tamper \"Total|Q2 FY27=99000\" "
                        "(value in the chart's own units). Repeatable.")
    p.add_argument("--no-sql", action="store_true")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--device")
    a = p.parse_args()
    if not a.query and a.page is None:
        sys.exit("give a query, or --ticker with --page")

    s = get_settings()
    pages = read_manifest(Path(vcfg.get("pages_dir", "data/processed/pages")) / "pages.jsonl")
    tickers = tuple(t.upper() for t in a.ticker) if a.ticker else None
    filt = SearchFilter(tickers=tickers, fiscal_years=tuple(a.year) if a.year else None)
    filt = filt if (filt.tickers or filt.fiscal_years) else None

    if a.page is not None:
        match = [pg for pg in pages if pg.page == a.page and (not tickers or pg.ticker in tickers)
                 and (not a.year or pg.fiscal_year in a.year)]
        if not match:
            sys.exit(f"no page {a.page} for {tickers or 'any ticker'} in the manifest")
        page = match[0]
    else:
        if a.mode == "visual":
            from finsight.indexing.visual_embedders import build_visual_embedder
            from finsight.retrieval.visual_retriever import VisualRetriever
            model = cfg["models"]["visual_retriever"]
            ret = VisualRetriever(QdrantClient(url=s.qdrant_url, timeout=120),
                                  build_visual_embedder(model, device=a.device),
                                  collection_name(cfg["qdrant"]["visual_collection"], model))
        else:
            ret = PageTextRetriever(pages)
        hits = ret.search(a.query, 1, filt)
        if not hits:
            sys.exit("no slide matched the query")
        page = next(pg for pg in pages if pg.page_id == hits[0].evidence_id)
    print(f"slide: {page.ticker} {page.doc_id} p.{page.page}  {page.image_path}")

    vlm = GroqVLM(vcfg.get("vlm_model", "qwen/qwen3.8-27b"), s.groq_api_key, DiskCache(Path(s.cache_dir)),
                  base_url=vcfg.get("vlm_base_url") or None, extra_body=vcfg.get("vlm_extra_body") or None)
    chart = ChartExtractor(vlm).extract(page.image_path)
    for spec in a.tamper or []:
        key, _, val = spec.partition("=")
        t_series, _, t_label = key.partition("|")
        hit = False
        for s_ in chart.series:
            for pt in s_.points:
                if s_.name.strip().lower() == t_series.strip().lower() and pt.label.strip().lower() == t_label.strip().lower():
                    print(f"TAMPERED for demo: {s_.name} / {pt.label}: {pt.value:g} -> {float(val):g}")
                    pt.value, hit = float(val), True
        if not hit:
            sys.exit(f"--tamper: no point '{t_series}' / '{t_label}' in the chart (names must match the table below)")
    print("\n== chart as read from the slide ==")
    print(chart.to_markdown())
    if chart.caveats:
        print("caveats:", "; ".join(chart.caveats))

    sql = None
    if not a.no_sql:
        from finsight.retrieval.sql_retriever import SqlRetriever
        dsn = next((getattr(s, n) for n in ("postgres_dsn", "database_url", "pg_dsn") if getattr(s, n, None)), None)
        if not dsn:
            sys.exit("no Postgres DSN in settings; use --no-sql to skip XBRL checks")
        sql = SqlRetriever(dsn)
    text = None
    if not a.no_text:
        model = cfg["models"]["embedding"]
        r = cfg["retrieval"]
        text = TextRetriever(QdrantClient(url=s.qdrant_url, timeout=60), build_embedder(model, device=a.device),
                             collection_name(cfg["qdrant"]["text_collection"], model),
                             top_k_dense=r["top_k_dense"], top_k_sparse=r["top_k_sparse"], rrf_k=r["rrf_k"])

    report = CrossModalVerifier(sql=sql, text=text).verify(chart, page)
    print("\n== verification against the filings ==")
    print(report.summary())
    cited = [e for e in report.evidence if e.modality.value in ("sql", "text", "table")]
    if cited:
        print("\n== filing evidence used ==")
        for e in cited:
            print(f"  {e.source.ticker} {e.source.doc_type.value} FY{e.source.fiscal_year}: {e.content[:140]!r}")
