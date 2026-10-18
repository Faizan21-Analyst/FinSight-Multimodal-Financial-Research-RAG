"""Download every deck listed in configs/decks.yaml into data/raw/decks/.

Usage:
  python scripts/fetch_decks.py            # skips files that already exist
  python scripts/fetch_decks.py --force    # re-download

If a site refuses the download (HTTP 403), open the URL in your browser, save the PDF into
data/raw/decks/ and give it the name from decks.yaml. Then run scripts/build_pages.py.
"""
import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.core.config import get_settings, load_yaml  # noqa: E402
from finsight.core.exceptions import IngestionError  # noqa: E402
from finsight.vision.page_renderer import parse_deck_filename  # noqa: E402

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/raw/decks")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    decks = load_yaml("decks.yaml").get("decks", [])
    if not decks:
        sys.exit("configs/decks.yaml has no decks")
    ua = get_settings().sec_user_agent  # SEC-style UA is not needed here; browser UA works better on IR sites
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for d in decks:
        name = d["name"].removesuffix(".pdf")
        try:
            parse_deck_filename(f"{name}.pdf")
        except IngestionError as e:
            print(f"SKIP  {name}: {e}")
            failed.append(name)
            continue
        out = out_dir / f"{name}.pdf"
        if out.exists() and not a.force:
            print(f"have  {out}")
            continue
        try:
            r = httpx.get(d["url"], headers={"User-Agent": BROWSER_UA}, follow_redirects=True, timeout=120)
        except httpx.HTTPError as e:
            print(f"FAIL  {name}: {e}")
            failed.append(name)
            continue
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            print(f"FAIL  {name}: HTTP {r.status_code}, "
                  f"{'not a PDF' if r.status_code == 200 else 'download refused'}")
            failed.append(name)
            continue
        out.write_bytes(r.content)
        print(f"saved {out}  ({len(r.content) / 1e6:.1f} MB)")

    if failed:
        print(f"\n{len(failed)} deck(s) failed. Download them in your browser from the URLs in "
              f"configs/decks.yaml and save into {out_dir}/ with the listed names.")
        sys.exit(1)
    print("\nnext: python scripts/build_pages.py")
