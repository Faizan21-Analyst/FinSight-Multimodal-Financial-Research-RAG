"""Render investor-deck PDFs to page images and write the page manifest.

Put decks in data/raw/decks/ named TICKER_FY2025_Q4.pdf (see finsight/vision/page_renderer.py).

Usage:
  python scripts/build_pages.py
  python scripts/build_pages.py --tickers NVDA --dpi 150
Re-running is cheap: existing page images are not redrawn.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.core.config import get_settings, load_app_config  # noqa: E402
from finsight.core.logging import setup_logging  # noqa: E402
from finsight.vision.page_renderer import discover_decks, render_pdf, write_manifest  # noqa: E402

if __name__ == "__main__":
    cfg = load_app_config().get("vision", {})
    p = argparse.ArgumentParser()
    p.add_argument("--decks-dir", default=cfg.get("decks_dir", "data/raw/decks"))
    p.add_argument("--pages-dir", default=cfg.get("pages_dir", "data/processed/pages"))
    p.add_argument("--dpi", type=int, default=cfg.get("render_dpi", 120))
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--max-pages", type=int, help="render only the first N pages of each deck (testing)")
    a = p.parse_args()

    setup_logging(get_settings().log_level)
    decks = discover_decks(a.decks_dir, a.tickers)
    if not decks:
        sys.exit(
            f"no decks found in {a.decks_dir}. Save investor presentations there as "
            "TICKER_FY2025_Q4.pdf (e.g. NVDA_FY2025_Q4.pdf)."
        )

    records = []
    for d in decks:
        recs = render_pdf(d, a.pages_dir, dpi=a.dpi, max_pages=a.max_pages)
        records.extend(recs)
        empty = sum(1 for r in recs if not r.text)
        print(f"{d.pdf_path.name:<40} {len(recs):>4} pages  ({empty} with no text layer)")

    manifest = Path(a.pages_dir) / "pages.jsonl"
    n = write_manifest(records, manifest)
    print(f"\nwrote {n} pages to {manifest}")
