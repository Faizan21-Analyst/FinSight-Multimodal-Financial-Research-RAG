#!/usr/bin/env python3
"""Show how a filing's headings were detected. Use when build_chunks reports MISSING sections.
Usage: python scripts/debug_sections.py data/raw/NVDA/10-K_2024-01-28_000104581024000029.htm [10-K|10-Q]"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.ingestion.parsing.html_parser import parse_html  # noqa: E402
from finsight.ingestion.parsing.sections import assign_sections, find_candidates  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: debug_sections.py FILE.htm [10-K|10-Q]")
    form = sys.argv[2] if len(sys.argv) > 2 else ("10-Q" if "10-Q" in sys.argv[1] else "10-K")
    blocks = parse_html(Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace"))
    n_tab = sum(b.kind == "table" for b in blocks)
    print(f"{len(blocks)} blocks ({n_tab} tables), {blocks[-1].page if blocks else 0} pages\n")

    cands = find_candidates(blocks, with_part=(form == "10-Q"))
    print("heading candidates (block index, key, title):")
    for c in cands:
        print(f"  {c.index:6d}  {c.key:16s} {c.title[:60]}")

    labels = assign_sections(blocks, form)
    sizes: dict[str, int] = {}
    for b, (k, _) in zip(blocks, labels):
        if b.kind == "text":
            sizes[k] = sizes.get(k, 0) + len(b.text)
    print("\nchosen sections (characters of text):")
    for k, v in sizes.items():
        print(f"  {k:16s} {v:>9,}")

    print("\nfirst 25 text blocks (check for page furniture / noise):")
    for b in [x for x in blocks if x.kind == "text"][:25]:
        print(f"  p{b.page:<3d} {b.text[:100]}")