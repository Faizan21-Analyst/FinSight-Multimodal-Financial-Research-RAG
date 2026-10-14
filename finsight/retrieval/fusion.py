"""Rank fusion. Pure functions.

Module 4 uses reciprocal_rank_fusion to merge dense and sparse result lists.
Module 7 reuses it to merge text, visual and SQL evidence.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[Hashable, float]]:
    """Merge ranked lists. score(item) = sum over lists of weight / (k + rank), rank from 1.

    * Only ranks matter, so scores from different retrievers (cosine vs BM25) need no
      calibration. That is the whole point of RRF.
    * A larger k flattens the advantage of top ranks; k=60 is the standard default.
    * An item repeated within one list counts once, at its best rank.
    * Ties break on best rank across lists, then on the item itself, so output is deterministic.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")
    scores: dict[Hashable, float] = {}
    best_rank: dict[Hashable, int] = {}
    for i, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[i]
        seen: set[Hashable] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
            best_rank[item] = min(rank, best_rank.get(item, rank))
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], str(kv[0])))