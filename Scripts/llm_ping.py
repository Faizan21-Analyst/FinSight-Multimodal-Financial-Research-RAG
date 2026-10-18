"""Smoke-test the text LLM: one real call to Groq with the settings.yaml model and parameters.

Usage:
  python scripts/llm_ping.py
  python scripts/llm_ping.py "Why do margins fall when R&D grows faster than revenue?" --role small
  python scripts/llm_ping.py "Give NVIDIA's fiscal year end month as JSON" --json
"""
import argparse
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from finsight.core.config import get_settings  # noqa: E402
from finsight.core.logging import setup_logging  # noqa: E402
from finsight.llm.client import build_llm  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="In one sentence, what is a 10-K filing?")
    p.add_argument("--role", choices=["default", "small"], default="default")
    p.add_argument("--json", action="store_true", help="JSON mode")
    a = p.parse_args()

    setup_logging(get_settings().log_level)
    llm = build_llm(a.role)
    print(f"model: {llm.model}   extra_body: {llm.extra_body}\n")
    out = llm.complete_json(a.prompt) if a.json else llm.complete(a.prompt).text
    print(out)
    print(f"\nstats: {llm.stats.as_dict()}")
