"""Ranking metrics over a ranked list of binary relevance flags (True = relevant). Pure functions."""
from __future__ import annotations

import math
from collections.abc import Sequence


def first_rank(flags: Sequence[bool]) -> int | None:
    """1-based rank of the first relevant result, or None."""
    for i, f in enumerate(flags, start=1):
        if f:
            return i
    return None


def hit_at_k(flags: Sequence[bool], k: int) -> float:
    """1.0 if any relevant result is in the top k. (Called 'recall@k' in many RAG papers.)"""
    return 1.0 if any(flags[:k]) else 0.0


def reciprocal_rank(flags: Sequence[bool], k: int = 10) -> float:
    r = first_rank(flags[:k])
    return 1.0 / r if r else 0.0


def ndcg_at_k(flags: Sequence[bool], k: int, n_relevant: int) -> float:
    """Binary-gain nDCG. n_relevant = how many relevant chunks exist in the corpus, which
    sets the ideal ranking (capped at k)."""
    if n_relevant <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, f in enumerate(flags[:k], start=1) if f)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_relevant, k) + 1))
    return dcg / ideal