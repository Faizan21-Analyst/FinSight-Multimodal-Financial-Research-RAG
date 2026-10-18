"""Visual retrieval vs OCR-then-text retrieval on hand-labelled slide questions.

You label the gold slides yourself (open the deck, find the right page). One JSON per line in
data/eval/visual/visual_questions.jsonl (a subfolder, so scripts/run_eval.py, which loads
data/eval/*.jsonl, does not mistake these for text-retrieval questions):

  {"id": "v01", "question": "Quarterly data center revenue", "ticker": "NVDA",
   "doc_id": "nvda-fy2025-q4-deck", "pages": [14]}

`pages` lists every page that would count as a correct answer. Aim for 15-20 questions, mixing
slides whose numbers live only inside a chart image with slides that have a text layer.

Usage:
  python scripts/eval_visual.py --model vidore/colSmol-256M
  python scripts/eval_visual.py --model hashing --no-visual      # OCR baseline only
"""
import argparse
import json
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
from finsight.retrieval.text_retriever import SearchFilter  # noqa: E402
from finsight.retrieval.visual_retriever import PageTextRetriever, VisualRetriever  # noqa: E402
from finsight.vision.page_renderer import read_manifest  # noqa: E402

KS = (1, 3, 5)


def first_hit_rank(hits, doc_id, pages):
    for i, h in enumerate(hits, 1):
        if h.source.doc_id == doc_id and h.source.page in pages:
            return i
    return None


def summarize(ranks):
    n = len(ranks)
    row = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    row["MRR"] = sum(1 / r for r in ranks if r) / n
    return row


if __name__ == "__main__":
    cfg = load_app_config()
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="data/eval/visual/visual_questions.jsonl")
    p.add_argument("--model", default=cfg["models"]["visual_retriever"])
    p.add_argument("--pages-dir", default=cfg.get("vision", {}).get("pages_dir", "data/processed/pages"))
    p.add_argument("--no-visual", action="store_true")
    p.add_argument("--device")
    a = p.parse_args()

    s = get_settings()
    setup_logging(s.log_level)
    qpath = Path(a.questions)
    if not qpath.exists():
        sys.exit(f"{qpath} not found. See the docstring of this script for the format.")
    questions = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines() if line.strip()]

    systems = {"ocr-bm25": PageTextRetriever(read_manifest(Path(a.pages_dir) / "pages.jsonl"))}
    if not a.no_visual:
        kwargs = {} if a.model.startswith("hashing") else {"device": a.device}
        systems[a.model] = VisualRetriever(
            QdrantClient(url=s.qdrant_url, timeout=120),
            build_visual_embedder(a.model, **kwargs),
            collection_name(cfg["qdrant"]["visual_collection"], a.model),
        )

    ranks = {name: [] for name in systems}
    misses = []
    for q in questions:
        filt = SearchFilter(tickers=(q["ticker"],)) if q.get("ticker") else None
        for name, r in systems.items():
            rank = first_hit_rank(r.search(q["question"], 5, filt), q["doc_id"], set(q["pages"]))
            ranks[name].append(rank)
            if rank is None:
                misses.append((name, q["id"], q["question"]))

    cols = [f"hit@{k}" for k in KS] + ["MRR"]
    lines = [f"# Visual retrieval ablation ({len(questions)} questions)", "",
             "| retriever | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, rk in ranks.items():
        row = summarize(rk)
        lines.append(f"| {name} | " + " | ".join(f"{row[c]:.2f}" for c in cols) + " |")
    if misses:
        lines += ["", "## Misses (not in top 5)", ""] + [f"- {n}: {i} {t}" for n, i, t in misses]
    report = "\n".join(lines)
    print(report)

    out = Path("evaluation/reports") / f"visual_{time.strftime('%Y%m%d-%H%M')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nsaved {out}")
