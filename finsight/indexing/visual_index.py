"""Index rendered pages into Qdrant as multi-vectors (one vector per image patch).

Same conventions as the text index: point id = page id as an integer (idempotent), re-runs skip
pages already indexed, one collection per embedding model. HNSW is disabled (m=0): a deck
collection holds hundreds of pages, so exact MaxSim search is fast and avoids a costly
multi-vector graph.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from qdrant_client import QdrantClient, models

from finsight.core.exceptions import ConfigError
from finsight.core.logging import get_logger
from finsight.indexing.text_index import IndexStats
from finsight.indexing.visual_embedders import VisualEmbedder
from finsight.vision.page_renderer import PageRecord

log = get_logger(__name__)

PAGE_VEC = "page"

PAYLOAD_INDEXES = {
    "ticker": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
    "fiscal_period": models.PayloadSchemaType.KEYWORD,
}


def page_point_id(page_id: str) -> int:
    return int(page_id, 16)


def page_payload(p: PageRecord) -> dict:
    return {
        "page_id": p.page_id,
        "doc_id": p.doc_id,
        "ticker": p.ticker,
        "doc_type": p.doc_type,
        "fiscal_year": p.fiscal_year,
        "fiscal_period": p.fiscal_period,
        "page": p.page,
        "image_path": p.image_path,
        "text": p.text[:4000],
    }


class VisualIndexer:
    def __init__(
        self,
        client: QdrantClient,
        embedder: VisualEmbedder,
        collection: str,
        batch_size: int = 4,
    ):
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.batch_size = batch_size

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            size = self.client.get_collection(self.collection).config.params.vectors[PAGE_VEC].size  # type: ignore[index]
            if size != self.embedder.dim:
                raise ConfigError(
                    f"Collection {self.collection} has dim {size} but {self.embedder.name} "
                    f"produces {self.embedder.dim}. Re-run with --recreate."
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                PAGE_VEC: models.VectorParams(
                    size=self.embedder.dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                )
            },
        )
        for field, schema in PAYLOAD_INDEXES.items():
            self.client.create_payload_index(self.collection, field, schema)
        log.info("created collection %s (dim %d)", self.collection, self.embedder.dim)

    def _existing(self, ids: list[int]) -> set[int]:
        found = self.client.retrieve(self.collection, ids=ids, with_payload=False, with_vectors=False)
        return {int(p.id) for p in found}

    def index(
        self,
        pages: Iterable[PageRecord],
        force: bool = False,
        progress: Callable[[IndexStats], None] | None = None,
    ) -> IndexStats:
        pages = list(pages)
        stats = IndexStats()
        for i in range(0, len(pages), self.batch_size):
            batch = pages[i : i + self.batch_size]
            stats.total += len(batch)
            todo = batch
            if not force:
                have = self._existing([page_point_id(p.page_id) for p in batch])
                todo = [p for p in batch if page_point_id(p.page_id) not in have]
            stats.skipped += len(batch) - len(todo)
            if todo:
                vectors = self.embedder.embed_pages(todo)
                self.client.upsert(
                    self.collection,
                    points=[
                        models.PointStruct(
                            id=page_point_id(p.page_id), vector={PAGE_VEC: v}, payload=page_payload(p)
                        )
                        for p, v in zip(todo, vectors)
                    ],
                    wait=True,
                )
                stats.indexed += len(todo)
            if progress:
                progress(stats)
        return stats
