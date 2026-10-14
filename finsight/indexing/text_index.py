"""Index chunks into Qdrant: one dense vector + one BM25 sparse vector per chunk.

Design points:
  * Point id = the chunk id read as a 64-bit integer, so re-indexing is idempotent.
  * Re-runs skip chunks already in the collection (no re-embedding), so a crash during a long
    embedding run costs nothing. Use force=True to re-embed.
  * Each embedding model has its own collection (see embedders.collection_name).
  * What gets embedded is "context header + chunk text"; what is STORED is the clean chunk text.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from finsight.core.exceptions import ConfigError
from finsight.core.logging import get_logger
from finsight.core.schemas import Chunk
from finsight.indexing.embedders import DenseEmbedder
from finsight.indexing.sparse import document_vector

log = get_logger(__name__)

DENSE = "dense"
SPARSE = "sparse"

PAYLOAD_INDEXES = {
    "ticker": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
    "fiscal_period": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "modality": models.PayloadSchemaType.KEYWORD,
}


def point_id(chunk_id: str) -> int:
    """Chunk ids are 16 hex chars = 64 bits = a valid Qdrant unsigned-integer point id."""
    return int(chunk_id, 16)


def embedding_text(chunk: Chunk) -> str:
    """Context header + text. The header lets a bare table ('| Revenue | $26,044 |') carry its
    company, period and section into both the dense and the sparse representation."""
    s = chunk.source
    parts = [s.ticker, s.doc_type.value]
    if s.fiscal_year:
        parts.append(f"FY{s.fiscal_year} {s.fiscal_period or ''}".strip())
    if s.section:
        parts.append(s.section)
    title = chunk.metadata.get("section_title")
    if title:
        parts.append(title)
    return f"{' | '.join(parts)}\n{chunk.text}"


def chunk_payload(chunk: Chunk) -> dict:
    s = chunk.source
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "modality": chunk.modality.value,
        "doc_id": s.doc_id,
        "ticker": s.ticker,
        "doc_type": s.doc_type.value,
        "fiscal_year": s.fiscal_year,
        "fiscal_period": s.fiscal_period,
        "section": s.section,
        "section_title": chunk.metadata.get("section_title"),
        "page": s.page,
        "caption": chunk.metadata.get("caption"),
    }


def load_chunks(chunk_dir: Path, tickers: list[str] | None = None) -> list[Chunk]:
    files = sorted(chunk_dir.glob("*.jsonl"))
    if tickers:
        prefixes = tuple(t.lower() + "-" for t in tickers)
        files = [f for f in files if f.stem.startswith(prefixes)]
    chunks: list[Chunk] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))
    return chunks


@dataclass
class IndexStats:
    total: int = 0
    indexed: int = 0
    skipped: int = 0


class TextIndexer:
    def __init__(
        self,
        client: QdrantClient,
        embedder: DenseEmbedder,
        collection: str,
        batch_size: int = 64,
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
            info = self.client.get_collection(self.collection)
            size = info.config.params.vectors[DENSE].size  # type: ignore[index]
            if size != self.embedder.dim:
                raise ConfigError(
                    f"Collection {self.collection} has dim {size} but {self.embedder.name} "
                    f"produces {self.embedder.dim}. Re-run with --recreate."
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field, schema in PAYLOAD_INDEXES.items():
            self.client.create_payload_index(self.collection, field, schema)
        log.info("created collection %s (dim %d)", self.collection, self.embedder.dim)

    def existing_ids(self, ids: list[int]) -> set[int]:
        found = self.client.retrieve(
            self.collection, ids=ids, with_payload=False, with_vectors=False
        )
        return {int(p.id) for p in found}

    
    def refresh_sparse(
        self,
        chunks: Iterable[Chunk],
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Rebuild ONLY the sparse vectors of chunks already indexed.

        Dense vectors, payloads and ids remain untouched. No dense embedding
        model runs, so tokenizer or weighting changes can be applied without
        re-embedding the entire corpus.
        """
        chunks = list(chunks)
        updated = 0

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]

            have = self.existing_ids(
                [point_id(chunk.chunk_id) for chunk in batch]
            )

            points = []

            for chunk in batch:
                chunk_point_id = point_id(chunk.chunk_id)

                if chunk_point_id not in have:
                    continue

                idx, val = document_vector(embedding_text(chunk))

                points.append(
                    models.PointVectors(
                        id=chunk_point_id,
                        vector={
                            SPARSE: models.SparseVector(
                                indices=idx,
                                values=val,
                            )
                        },
                    )
                )

            if points:
                self.client.update_vectors(
                    self.collection,
                    points=points,
                    wait=True,
                )
                updated += len(points)

            if progress:
                progress(
                    min(i + self.batch_size, len(chunks)),
                    len(chunks),
                )

        return updated

    def index(
        self,
        chunks: Iterable[Chunk],
        force: bool = False,
        progress: Callable[[IndexStats], None] | None = None,
    ) -> IndexStats:
        stats = IndexStats()
        chunks = list(chunks)
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            stats.total += len(batch)
            if not force:
                have = self.existing_ids([point_id(c.chunk_id) for c in batch])
                todo = [c for c in batch if point_id(c.chunk_id) not in have]
            else:
                todo = batch
            stats.skipped += len(batch) - len(todo)
            if todo:
                texts = [embedding_text(c) for c in todo]
                dense = self.embedder.embed_documents(texts)
                points = []
                for c, text, d in zip(todo, texts, dense):
                    idx, val = document_vector(text)
                    points.append(models.PointStruct(
                        id=point_id(c.chunk_id),
                        vector={DENSE: d, SPARSE: models.SparseVector(indices=idx, values=val)},
                        payload=chunk_payload(c),
                    ))
                self.client.upsert(self.collection, points=points, wait=True)
                stats.indexed += len(todo)
            if progress:
                progress(stats)
        return stats