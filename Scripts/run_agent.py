"""Run the corrective retrieval agent on one question and print the full trace.

Usage:
  python scripts/run_agent.py "Compare NVIDIA and AMD research and development expense in fiscal 2024"
  python scripts/run_agent.py "Why did Tesla margins change?" --no-llm      # heuristic grader, no API calls
  python scripts/run_agent.py "What was Apple net income in 2024?" --sql    # also exact XBRL lookups
  python scripts/run_agent.py "..." --answer                                 # also generate a cited answer
"""
import argparse
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from qdrant_client import QdrantClient  # noqa: E402

from finsight.agents.graph import FinSightAgent  # noqa: E402
from finsight.agents.grader import EvidenceGrader  # noqa: E402
from finsight.core.config import get_settings, load_app_config  # noqa: E402
from finsight.indexing.embedders import build_embedder, collection_name  # noqa: E402
from finsight.llm.client import build_llm  # noqa: E402
from finsight.retrieval.text_retriever import TextRetriever  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--no-llm", action="store_true", help="heuristic grader only")
    p.add_argument("--sql", action="store_true", help="enable exact XBRL lookups (needs Postgres)")
    p.add_argument("--answer", action="store_true", help="generate a cited answer")
    p.add_argument("--device")
    a = p.parse_args()

    s, cfg = get_settings(), load_app_config()
    model = cfg["models"]["embedding"]
    r, ag = cfg["retrieval"], cfg["agent"]
    text = TextRetriever(
        QdrantClient(url=s.qdrant_url, timeout=60),
        build_embedder(model, device=a.device),
        collection_name(cfg["qdrant"]["text_collection"], model),
        top_k_dense=r["top_k_dense"], top_k_sparse=r["top_k_sparse"], rrf_k=r["rrf_k"],
    )

    sql = None
    if a.sql:
        from finsight.retrieval.sql_retriever import SqlRetriever
        dsn = next((getattr(s, n) for n in ("postgres_dsn", "database_url", "pg_dsn") if getattr(s, n, None)), None)
        if not dsn:
            sys.exit("no Postgres DSN found in settings (looked for postgres_dsn / database_url / pg_dsn)")
        sql = SqlRetriever(dsn)

    agent = FinSightAgent(
        text=text,
        grader=EvidenceGrader(None if a.no_llm else build_llm("small")),
        sql=sql,
        answer_llm=build_llm("default") if a.answer else None,
        max_iterations=ag["max_iterations"],
        threshold=ag["grader_threshold"],
    )
    res = agent.run(a.question)

    print(f"\nintent={res.plan.intent.value} tickers={res.plan.tickers} years={res.plan.fiscal_years} "
          f"metrics={res.plan.metrics}")
    print("\n--- trace ---")
    for t in res.trace:
        print({k: v for k, v in t.items() if v not in ("", [], (), None)})
    print(f"\n--- evidence ({len(res.package)} items, {res.iterations} iteration(s), "
          f"final score {res.grade.score:.2f}, {res.seconds:.1f}s) ---")
    for n, item in enumerate(res.package.items):
        src = item.source
        print(f"[E{n + 1}] {src.ticker} {src.doc_type.value} FY{src.fiscal_year} {src.section or ''} "
              f"({item.modality.value}): {item.content[:110]!r}")
    if res.answer:
        print("\n--- answer ---\n" + res.answer)
