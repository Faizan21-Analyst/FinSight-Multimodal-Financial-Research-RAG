"""Page retrieval: ColPali-style visual search, plus an OCR-then-text baseline to compare against.

Both expose search(query, limit, filt) and return EvidenceItem (modality PAGE_IMAGE) with the
page image path, so the rest of the system treats them identically and Module 5-style ablations
("visual vs OCR-text") are a one-argument change.

* VisualRetriever   : MaxSim over page-image multi-vectors in Qdrant.
* PageTextRetriever : BM25 over each page's text layer, in memory. This is the "just extract the
                      text and search it" approach. It cannot see numbers inside charts.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from qdrant_client import QdrantClient

from finsight.core.exceptions import RetrievalError
from finsight.core.schemas import DocType, EvidenceItem, Modality, SourceRef
from finsight.indexing.sparse import tokenize
from finsight.indexing.visual_embedders import VisualEmbedder
from finsight.indexing.visual_index import PAGE_VEC
from finsight.retrieval.text_retriever import SearchFilter
from finsight.vision.page_renderer import PageRecord

SNIPPET_CHARS = 1500


def page_evidence(
    page_id: str, doc_id: str, ticker: str, fiscal_year: int | None, fiscal_period: str | None,
    page: int, image_path: str, text: str, score: float, retriever: str,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=page_id,
        modality=Modality.PAGE_IMAGE,
        content=text[:SNIPPET_CHARS] or "(slide with no extractable text)",
        source=SourceRef(
            doc_id=doc_id, ticker=ticker, doc_type=DocType.PRESENTATION,
            fiscal_year=fiscal_year, fiscal_period=fiscal_period, page=page,
        ),
        score=score,
        retriever=retriever,
        image_path=image_path,
    )


class VisualRetriever:
    def __init__(self, client: QdrantClient, embedder: VisualEmbedder, collection: str):
        self.client = client
        self.embedder = embedder
        self.collection = collection

    def search(
        self, query: str, limit: int = 5, filt: SearchFilter | None = None
    ) -> list[EvidenceItem]:
        try:
            res = self.client.query_points(
                self.collection,
                query=self.embedder.embed_query(query),
                using=PAGE_VEC,
                limit=limit,
                query_filter=filt.to_qdrant() if filt else None,
                with_payload=True,
            )
        except Exception as e:  # qdrant raises several unrelated types here
            raise RetrievalError(
                f"visual query failed on {self.collection!r} ({e}). "
                "Did you run scripts/build_visual_index.py?"
            ) from e
        return [
            page_evidence(
                p.payload["page_id"], p.payload["doc_id"], p.payload["ticker"],
                p.payload.get("fiscal_year"), p.payload.get("fiscal_period"), p.payload["page"],
                p.payload["image_path"], p.payload.get("text", ""), p.score, "colpali",
            )
            for p in res.points
        ]


def _matches(page: PageRecord, f: SearchFilter | None) -> bool:
    if f is None:
        return True
    return (
        (not f.tickers or page.ticker in f.tickers)
        and (not f.fiscal_years or page.fiscal_year in f.fiscal_years)
        and (not f.fiscal_periods or page.fiscal_period in f.fiscal_periods)
        and (not f.doc_types or page.doc_type in f.doc_types)
    )


class PageTextRetriever:
    """In-memory BM25 over page text layers (the OCR-then-text baseline)."""

    def __init__(self, pages: Sequence[PageRecord], k1: float = 1.5, b: float = 0.75):
        self.pages = list(pages)
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(p.text)) for p in self.pages]
        self._len = [sum(c.values()) for c in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for c in self._tf:
            df.update(c.keys())
        n = len(self.pages)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def _score(self, i: int, q_terms: list[str]) -> float:
        tf, dl = self._tf[i], self._len[i]
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f:
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg or 1.0))
                s += self._idf[t] * f * (self.k1 + 1) / denom
        return s

    def search(
        self, query: str, limit: int = 5, filt: SearchFilter | None = None
    ) -> list[EvidenceItem]:
        q_terms = list(dict.fromkeys(tokenize(query)))
        scored = [
            (self._score(i, q_terms), i) for i, p in enumerate(self.pages) if _matches(p, filt)
        ]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda si: (-si[0], si[1]))
        out = []
        for s, i in scored[:limit]:
            p = self.pages[i]
            out.append(
                page_evidence(
                    p.page_id, p.doc_id, p.ticker, p.fiscal_year, p.fiscal_period, p.page,
                    p.image_path, p.text, s, "page-bm25",
                )
            )
        return out
