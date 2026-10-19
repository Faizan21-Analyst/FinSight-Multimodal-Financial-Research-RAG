"""Text retrieval over the Qdrant collection: dense, sparse (BM25) or hybrid (RRF).

All three modes share one code path and one return type, so Module 5 can run the
dense vs BM25 vs hybrid ablation by changing a single argument.

Hybrid = fetch top_k_dense and top_k_sparse candidates separately (same metadata filter),
then merge with reciprocal rank fusion (finsight.retrieval.fusion). Fusion is done here rather
than server-side so rrf_k and weights are ours to tune and test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from finsight.core.exceptions import RetrievalError
from finsight.core.schemas import DocType, EvidenceItem, Modality, SourceRef
from finsight.indexing.embedders import DenseEmbedder
from finsight.indexing.sparse import query_vector
from finsight.indexing.text_index import DENSE, SPARSE
from finsight.retrieval.fusion import reciprocal_rank_fusion

MODES = ("dense", "sparse", "hybrid")


@dataclass(frozen=True)
class SearchFilter:
    """Metadata filter. Values within one field are OR-ed; different fields are AND-ed."""

    tickers: tuple[str, ...] | None = None
    fiscal_years: tuple[int, ...] | None = None
    doc_types: tuple[str, ...] | None = None  # "10-K", "10-Q"
    fiscal_periods: tuple[str, ...] | None = None  # "FY", "Q1", ...
    sections: tuple[str, ...] | None = None  # "Item 7", "Part I Item 2", ...
    modalities: tuple[str, ...] | None = None  # "text", "table"

    def to_qdrant(self) -> models.Filter | None:
        spec = {
            "ticker": self.tickers,
            "fiscal_year": self.fiscal_years,
            "doc_type": self.doc_types,
            "fiscal_period": self.fiscal_periods,
            "section": self.sections,
            "modality": self.modalities,
        }
        must = [
            models.FieldCondition(key=k, match=models.MatchAny(any=list(v)))
            for k, v in spec.items()
            if v
        ]
        return models.Filter(must=must) if must else None


def payload_to_evidence(payload: dict[str, Any], score: float, retriever: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=payload["chunk_id"],
        modality=Modality(payload["modality"]),
        content=payload["text"],
        source=SourceRef(
            doc_id=payload["doc_id"],
            ticker=payload["ticker"],
            doc_type=DocType(payload["doc_type"]),
            fiscal_year=payload.get("fiscal_year"),
            fiscal_period=payload.get("fiscal_period"),
            page=payload.get("page"),
            section=payload.get("section"),
        ),
        score=score,
        retriever=retriever,
    )


class TextRetriever:
    def __init__(
        self,
        client: QdrantClient,
        embedder: DenseEmbedder,
        collection: str,
        top_k_dense: int = 30,
        top_k_sparse: int = 30,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),  # (dense, sparse)
    ):
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.rrf_k = rrf_k
        self.weights = weights

    def _query(self, query, using: str, limit: int, flt: models.Filter | None) -> list:
        try:
            res = self.client.query_points(
                self.collection, query=query, using=using, limit=limit,
                query_filter=flt, with_payload=True,
            )
        except ResponseHandlingException as e:
            raise RetrievalError(
                f"Cannot reach Qdrant ({e}). Is it running? Try: docker compose up -d qdrant"
            ) from e
        except (UnexpectedResponse, ValueError) as e:
            raise RetrievalError(
                f"Qdrant query failed on {self.collection!r} ({e}). "
                "Did you run scripts/build_index.py for this model?"
            ) from e
        return res.points

    def _dense(self, query: str, limit: int, flt: models.Filter | None) -> list:
        return self._query(self.embedder.embed_query(query), DENSE, limit, flt)

    def _sparse(self, query: str, limit: int, flt: models.Filter | None) -> list:
        idx, val = query_vector(query)
        if not idx:  # query was only stopwords/punctuation
            return []
        return self._query(models.SparseVector(indices=idx, values=val), SPARSE, limit, flt)

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        filt: SearchFilter | None = None,
    ) -> list[EvidenceItem]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        flt = filt.to_qdrant() if filt else None

        if mode == "dense":
            return [payload_to_evidence(p.payload, p.score, "dense")
                    for p in self._dense(query, limit, flt)]
        if mode == "sparse":
            return [payload_to_evidence(p.payload, p.score, "sparse")
                    for p in self._sparse(query, limit, flt)]

        dense = self._dense(query, self.top_k_dense, flt)
        sparse = self._sparse(query, self.top_k_sparse, flt)
        payloads = {p.payload["chunk_id"]: p.payload for p in [*sparse, *dense]}
        fused = reciprocal_rank_fusion(
            [[p.payload["chunk_id"] for p in dense], [p.payload["chunk_id"] for p in sparse]],
            k=self.rrf_k, weights=self.weights,
        )
        return [payload_to_evidence(payloads[cid], score, "hybrid")
                for cid, score in fused[:limit]]