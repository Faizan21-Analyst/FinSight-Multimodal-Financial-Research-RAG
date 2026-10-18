"""Download one investor-deck PDF into data/raw/decks/ under the right name.

You find the PDF link on the company's investor-relations site (open the quarterly results page,
right-click the slides/CFO-commentary PDF, copy link address), then:

  python scripts/fetch_deck.py --url "https://.../NVIDIA-Q2-FY26-CFO-Commentary.pdf" --name NVDA_FY2026_Q2
  python scripts/fetch_deck.py --url "https://.../AMD-Q2-2026-slides.pdf" --name AMD_FY2026_Q2

--name is TICKER_FY<year>_<Q1-Q4> (the same convention scripts/build_pages.py reads).
If a site blocks the download (403), open the link in your browser, save the PDF into
data/raw/decks/ yourself, and give it the same name.
"""
import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.core.exceptions import IngestionError  # noqa: E402
from finsight.core.config import get_settings  # noqa: E402
from finsight.vision.page_renderer import parse_deck_filename  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--name", required=True, help="e.g. NVDA_FY2026_Q2 (no .pdf)")
    p.add_argument("--out-dir", default="data/raw/decks")
    a = p.parse_args()

    name = a.name[:-4] if a.name.lower().endswith(".pdf") else a.name
    try:
        parse_deck_filename(f"{name}.pdf")
    except IngestionError as e:
        sys.exit(str(e))

    ua = get_settings().sec_user_agent or "Mozilla/5.0 (FinSight research project)"
    out = Path(a.out_dir) / f"{name}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(a.url, headers={"User-Agent": ua}, follow_redirects=True, timeout=60)
    if r.status_code != 200:
        sys.exit(f"download failed: HTTP {r.status_code}. Save the PDF manually as {out}")
    if not r.content.startswith(b"%PDF"):
        sys.exit("that link is not a PDF (it returned a web page). Use the direct PDF link.")
    out.write_bytes(r.content)
    print(f"saved {out} ({len(r.content) / 1e6:.1f} MB)")
