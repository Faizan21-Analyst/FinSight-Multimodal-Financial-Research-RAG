"""Retrieval evaluation: dense vs BM25 vs hybrid (vs hybrid + rerank),
with and without an oracle filter.

Usage:
  python scripts/run_eval.py --validate
  python scripts/run_eval.py
  python scripts/run_eval.py --rerank --reranker BAAI/bge-reranker-base
  python scripts/run_eval.py --oracle off --limit 40
  python scripts/run_eval.py --split
  python scripts/run_eval.py --split --oracle on

Reports are written to evaluation/reports/ (markdown + JSON).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from qdrant_client import QdrantClient  # noqa: E402

from finsight.core.caching import DiskCache  # noqa: E402
from finsight.core.config import get_settings, load_app_config  # noqa: E402
from finsight.evaluation.dataset import load_questions, resolve_gold  # noqa: E402
from finsight.evaluation.runner import (  # noqa: E402
    EvalConfig,
    evaluate,
    format_markdown,
    to_jsonable,
)
from finsight.indexing.embedders import build_embedder, collection_name  # noqa: E402
from finsight.indexing.text_index import chunk_payload, load_chunks  # noqa: E402
from finsight.retrieval.reranker import (  # noqa: E402
    CrossEncoderReranker,
    RerankingBackend,
)
from finsight.retrieval.text_retriever import TextRetriever  # noqa: E402


BROAD = 400  # A gold set this large means the predicate is probably too loose.


def validate(questions, corpus) -> int:
    print(f"{'id':34} {'category':16} gold chunks per spec")
    bad = 0

    for q in questions:
        sizes = [len(resolve_gold(corpus, spec)) for spec in q.gold]

        flag = (
            "  <-- ZERO: fix or delete"
            if 0 in sizes
            else ("  <-- very broad" if max(sizes) > BROAD else "")
        )

        bad += bool(flag and "ZERO" in flag)
        print(f"{q.id:34} {q.category:16} {sizes}{flag}")

    print(
        f"\n{len(questions)} questions, {bad} with an empty gold set "
        "(these are dropped from scoring)"
    )
    return bad


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--model", help="embedding model (default: settings.yaml)")
    p.add_argument(
        "--questions",
        nargs="*",
        help="jsonl files (default: data/eval/*.jsonl)",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="only check gold predicates, then exit",
    )
    p.add_argument(
        "--configs",
        nargs="*",
        default=["dense", "sparse", "hybrid"],
        choices=["dense", "sparse", "hybrid", "hybrid+rerank"],
    )
    p.add_argument(
        "--rerank",
        action="store_true",
        help="also run hybrid+rerank",
    )
    p.add_argument(
        "--reranker",
        help="cross-encoder model (default: settings.yaml models.reranker)",
    )
    p.add_argument(
        "--candidates",
        type=int,
        default=30,
        help="hybrid candidates passed to the reranker",
    )
    p.add_argument(
        "--oracle",
        choices=["off", "on", "both"],
        default="both",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help=(
            "also run oracle per-company split retrieval "
            "(ceiling for comparison questions)"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        help="random subset of N questions (seed 0)",
    )
    p.add_argument("--device")
    p.add_argument("--out", default="evaluation/reports")

    a = p.parse_args()

    s = get_settings()
    cfg = load_app_config()
    data = Path(s.data_dir)

    files = (
        [Path(f) for f in a.questions]
        if a.questions
        else sorted((data / "eval").glob("*.jsonl"))
    )

    if not files:
        sys.exit(
            "no question files in data/eval/. "
            "Run scripts/make_numeric_eval.py first."
        )

    questions = load_questions(*files)
    corpus = [
        chunk_payload(chunk)
        for chunk in load_chunks(data / "processed" / "chunks")
    ]

    print(
        f"{len(questions)} questions from "
        f"{[f.name for f in files]}, {len(corpus)} chunks"
    )

    if a.validate:
        sys.exit(1 if validate(questions, corpus) else 0)

    if a.limit and a.limit < len(questions):
        questions = random.Random(0).sample(questions, a.limit)

    model = a.model or cfg["models"]["embedding"]
    coll = collection_name(cfg["qdrant"]["text_collection"], model)
    r = cfg.get("retrieval", {})

    client = QdrantClient(url=s.qdrant_url, timeout=60)

    base = TextRetriever(
        client,
        (
            build_embedder(model, device=a.device)
            if not model.startswith("hashing")
            else build_embedder(model)
        ),
        coll,
        top_k_dense=r.get("top_k_dense", 30),
        top_k_sparse=r.get("top_k_sparse", 30),
        rrf_k=r.get("rrf_k", 60),
    )

    modes = list(a.configs) + (
        ["hybrid+rerank"]
        if a.rerank and "hybrid+rerank" not in a.configs
        else []
    )

    reranker_name = None
    backend = base

    if "hybrid+rerank" in modes:
        reranker_name = a.reranker or cfg["models"]["reranker"]

        backend = RerankingBackend(
            base,
            CrossEncoderReranker(reranker_name, device=a.device),
            candidates=a.candidates,
            cache=DiskCache(data / "processed" / "rerank_cache"),
        )

    oracles = {
        "off": [False],
        "on": [True],
        "both": [False, True],
    }[a.oracle]

    # Original configurations: each selected mode with the requested
    # oracle-filter setting(s).
    configs = [
        EvalConfig(m, o)
        for o in oracles
        for m in modes
    ]

    # Additional configurations: oracle-filtered, per-company split retrieval.
    if a.split:
        configs += [
            EvalConfig(m, True, True)
            for m in modes
        ]

    # Include split labels in the Markdown report when requested.
    labels = modes + (
        [f"{m}+split" for m in modes]
        if a.split
        else []
    )

    # Load the embedding model before measuring retrieval latency.
    if not model.startswith("hashing"):
        base.embedder.embed_query("warm up")

    def progress(c, n, total):
        if n % 10 == 0 or n == total:
            print(
                f"  {c.name:22} {n}/{total}",
                flush=True,
            )

    t0 = time.time()

    result = evaluate(
        questions,
        corpus,
        backend,
        configs,
        progress=progress,
    )

    meta = {
        "embedding model": model,
        "collection": f"{coll} ({client.count(coll).count} points)",
        "chunks in corpus": len(corpus),
        "reranker": reranker_name or "-",
        "hybrid": (
            f"RRF k={r.get('rrf_k', 60)}, "
            f"{r.get('top_k_dense', 30)} dense + "
            f"{r.get('top_k_sparse', 30)} sparse candidates"
        ),
        "question files": ", ".join(f.name for f in files),
        "run time": f"{time.time() - t0:.0f}s",
    }

    md = format_markdown(result, meta, labels)

    stamp = time.strftime("%Y%m%d-%H%M")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    slug = coll.split("__")[-1]

    (out / f"retrieval_{slug}_{stamp}.md").write_text(
        md,
        encoding="utf-8",
    )

    (out / f"retrieval_{slug}_{stamp}.json").write_text(
        json.dumps(to_jsonable(result)),
        encoding="utf-8",
    )

    print("\n" + md)
    print(f"saved to {out}/retrieval_{slug}_{stamp}.(md|json)")
