"""Second-stage reranking.

Retrieval compares a query vector to chunk vectors computed independently: fast, but coarse.
A cross-encoder reads the query and the passage TOGETHER, which is much more precise but far too
slow for the whole corpus. So: hybrid retrieval picks ~30 candidates, the reranker re-orders them.

* CrossEncoderReranker : real model (sentence-transformers CrossEncoder), lazy-loaded.
* OverlapReranker      : deterministic test double (query-term overlap). Plumbing tests only.
* Scores are cached on disk per (model, query, candidate set), so re-running an evaluation
  never pays for the same reranking twice.
"""
from __future__ import annotations

from typing import Protocol

from finsight.core.caching import DiskCache
from finsight.core.schemas import EvidenceItem
from finsight.indexing.sparse import tokenize


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


def evidence_text(item: EvidenceItem) -> str:
    """What the reranker reads: a short context header plus the chunk, like the embedding text."""
    s = item.source
    head = " | ".join(p for p in (s.ticker, s.doc_type.value,
                                  f"FY{s.fiscal_year} {s.fiscal_period or ''}".strip() if s.fiscal_year else "",
                                  s.section or "") if p)
    return f"{head}\n{item.content}"


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 512,
                 device: str | None = None):
        self.name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # heavy import, kept lazy

            self._model = CrossEncoder(self.name, max_length=self.max_length, device=self.device)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        out = self.model.predict(
            [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(x) for x in out]


class OverlapReranker:
    """Fraction of distinct query terms present in the passage."""

    name = "overlap"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(tokenize(t))) / len(q) for t in texts]


def rerank(
    query: str,
    items: list[EvidenceItem],
    reranker: Reranker,
    top_n: int | None = None,
    cache: DiskCache | None = None,
) -> list[EvidenceItem]:
    if not items:
        return []
    key = None
    scores: list[float] | None = None
    if cache is not None:
        key = DiskCache.key_for(reranker.name, query, sorted(i.evidence_id for i in items))
        hit = cache.get(key)
        if hit is not None:
            scores = [hit[i.evidence_id] for i in items]
    if scores is None:
        scores = reranker.score(query, [evidence_text(i) for i in items])
        if cache is not None and key is not None:
            cache.set(key, {i.evidence_id: s for i, s in zip(items, scores)})
    ranked = sorted(zip(items, scores), key=lambda p: -p[1])  # stable: ties keep retrieval order
    out = [i.model_copy(update={"score": s, "retriever": "hybrid+rerank"}) for i, s in ranked]
    return out[:top_n] if top_n else out


class RerankingBackend:
    """Adds a 'hybrid+rerank' mode to any retriever exposing search(query, mode, limit, filt)."""

    def __init__(self, base, reranker: Reranker, candidates: int = 30, cache: DiskCache | None = None):
        self.base = base
        self.reranker = reranker
        self.candidates = candidates
        self.cache = cache

    def search(self, query: str, mode: str = "hybrid", limit: int = 10, filt=None):
        if mode != "hybrid+rerank":
            return self.base.search(query, mode=mode, limit=limit, filt=filt)
        cands = self.base.search(query, mode="hybrid", limit=self.candidates, filt=filt)
        return rerank(query, cands, self.reranker, top_n=limit, cache=self.cache)