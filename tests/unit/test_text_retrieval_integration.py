"""Needs a running Qdrant (docker compose up -d qdrant). Skipped automatically otherwise.
Uses throwaway collections named test_finsight_<random>; never touches the real index."""
import os
import uuid

import pytest
from qdrant_client import QdrantClient

from finsight.core.exceptions import ConfigError, RetrievalError
from finsight.core.schemas import Chunk, DocType, Modality, SourceRef, make_id
from finsight.indexing.embedders import HashingEmbedder
from finsight.indexing.text_index import TextIndexer, point_id
from finsight.retrieval.text_retriever import SearchFilter, TextRetriever


@pytest.fixture(scope="module")
def client():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    c = QdrantClient(url=url, timeout=10)
    try:
        c.get_collections()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qdrant not reachable at {url}: {e}")
    return c


class CountingEmbedder(HashingEmbedder):
    def __init__(self, dim=64):
        super().__init__(dim)
        self.doc_calls = 0

    def embed_documents(self, texts):
        self.doc_calls += len(texts)
        return super().embed_documents(texts)


def _chunk(key, ticker, doc_type, fy, period, section, modality, text, caption=None):
    return Chunk(
        chunk_id=make_id(key), text=text, modality=modality,
        source=SourceRef(doc_id=f"{ticker.lower()}-{doc_type.value}-{fy}", ticker=ticker,
                         doc_type=doc_type, fiscal_year=fy, fiscal_period=period, page=7, section=section),
        metadata={"section_title": "", "caption": caption or ""},
    )


K, Q = DocType.TEN_K, DocType.TEN_Q
T, TB = Modality.TEXT, Modality.TABLE
CHUNKS = [
    _chunk("c1", "NVDA", K, 2024, "FY", "Item 7", TB, "| Data Center | $22,563 | $4,284 |", "revenue by market"),
    _chunk("c2", "NVDA", K, 2024, "FY", "Item 1A", T, "export controls on advanced computing China license requirements"),
    _chunk("c3", "NVDA", Q, 2025, "Q1", "Part I Item 2", T, "Data Center revenue increased due to Hopper demand"),
    _chunk("c4", "AMD", K, 2024, "FY", "Item 7", T, "Instinct MI300 accelerators data center segment revenue"),
    _chunk("c5", "AMD", K, 2024, "FY", "Item 8", TB, "| Research and development | $5,872 | $5,872 |"),
    _chunk("c6", "AAPL", K, 2024, "FY", "Item 1", T, "iPhone Mac iPad wearables and services"),
    _chunk("c7", "TSLA", K, 2023, "FY", "Item 7", T, "automotive regulatory credits revenue"),
    _chunk("c8", "TSLA", K, 2024, "FY", "Item 1A", T, "supply chain battery cells risk"),
]
C1, C2, C3, C4, C5, C6, C7, C8 = (c.chunk_id for c in CHUNKS)


@pytest.fixture()
def env(client):
    name = f"test_finsight_{uuid.uuid4().hex[:8]}"
    emb = CountingEmbedder(64)
    indexer = TextIndexer(client, emb, name, batch_size=3)  # small batches exercise the loop
    indexer.ensure_collection()
    yield client, emb, indexer, name
    if client.collection_exists(name):
        client.delete_collection(name)


def _retriever(env):
    client, emb, _, name = env
    return TextRetriever(client, emb, name)


def test_index_is_idempotent_and_skips_embedding(env):
    client, emb, indexer, name = env
    s1 = indexer.index(CHUNKS)
    assert (s1.total, s1.indexed, s1.skipped) == (8, 8, 0)
    assert client.count(name).count == 8 and emb.doc_calls == 8

    s2 = indexer.index(CHUNKS)                       # nothing new: no embedding work at all
    assert (s2.indexed, s2.skipped) == (0, 8) and emb.doc_calls == 8

    s3 = indexer.index(CHUNKS, force=True)           # force re-embeds but never duplicates
    assert s3.indexed == 8 and emb.doc_calls == 16 and client.count(name).count == 8
    assert client.retrieve(name, ids=[point_id(C1)])[0].payload["ticker"] == "NVDA"


def test_sparse_finds_exact_number_regardless_of_formatting(env):
    env[2].index(CHUNKS)
    r = _retriever(env)
    for q in ("22,563", "$22563", "22563 million"):
        assert r.search(q, mode="sparse", limit=3)[0].evidence_id == C1


def test_all_three_modes_return_sorted_evidence(env):
    env[2].index(CHUNKS)
    r = _retriever(env)
    for mode in ("dense", "sparse", "hybrid"):
        hits = r.search("data center revenue", mode=mode, limit=5)
        assert hits and all(h.retriever == mode for h in hits)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
    top3 = {h.evidence_id for h in r.search("data center revenue", mode="hybrid", limit=3)}
    assert top3 == {C1, C3, C4}


def test_hybrid_rescues_what_one_retriever_misses(env):
    env[2].index(CHUNKS)
    r = _retriever(env)
    # "MI300" only exists as a rare lexical token; hybrid must surface it at the top
    assert r.search("MI300", mode="hybrid", limit=3)[0].evidence_id == C4


def test_metadata_filters(env):
    env[2].index(CHUNKS)
    r = _retriever(env)
    for mode in ("dense", "sparse", "hybrid"):
        amd = r.search("revenue", mode=mode, limit=10, filt=SearchFilter(tickers=("AMD",)))
        assert amd and {h.source.ticker for h in amd} == {"AMD"}
    fy23 = r.search("revenue", limit=10, filt=SearchFilter(fiscal_years=(2023,)))
    assert [h.evidence_id for h in fy23] == [C7]
    tables = r.search("revenue", limit=10, filt=SearchFilter(modalities=("table",)))
    assert {h.evidence_id for h in tables} == {C1, C5}
    both = r.search("revenue", limit=10, filt=SearchFilter(tickers=("NVDA",), sections=("Item 7",)))
    assert [h.evidence_id for h in both] == [C1]
    q1 = r.search("revenue", limit=10, filt=SearchFilter(doc_types=("10-Q",), fiscal_periods=("Q1",)))
    assert [h.evidence_id for h in q1] == [C3]
    assert r.search("revenue", limit=10, filt=SearchFilter(tickers=("NVDA",), fiscal_years=(1999,))) == []


def test_evidence_carries_full_source(env):
    env[2].index(CHUNKS)
    ev = _retriever(env).search("22,563", mode="hybrid", limit=1)[0]
    assert ev.modality == Modality.TABLE and ev.content.startswith("| Data Center")
    s = ev.source
    assert (s.ticker, s.doc_type, s.fiscal_year, s.fiscal_period, s.section, s.page) == (
        "NVDA", K, 2024, "FY", "Item 7", 7)


def test_stopword_only_query_does_not_crash_sparse_or_hybrid(env):
    env[2].index(CHUNKS)
    r = _retriever(env)
    assert r.search("the of and", mode="sparse") == []
    assert isinstance(r.search("the of and", mode="hybrid"), list)


def test_dimension_mismatch_and_recreate(env):
    client, _, _, name = env
    env[2].index(CHUNKS)
    other = TextIndexer(client, HashingEmbedder(32), name)
    with pytest.raises(ConfigError, match="--recreate"):
        other.ensure_collection()
    other.ensure_collection(recreate=True)
    assert client.count(name).count == 0
    assert client.get_collection(name).config.params.vectors["dense"].size == 32


def test_missing_collection_gives_clear_error(client):
    r = TextRetriever(client, HashingEmbedder(64), "test_finsight_does_not_exist")
    with pytest.raises(RetrievalError, match="build_index"):
        r.search("anything")


def test_invalid_mode_rejected(env):
    with pytest.raises(ValueError):
        _retriever(env).search("x", mode="magic")

def test_refresh_sparse_replaces_only_sparse_vectors(env):
    from finsight.indexing.sparse import document_vector
    from finsight.indexing.text_index import SPARSE, DENSE, embedding_text

    client, emb, indexer, name = env
    indexer.index(CHUNKS)
    before = client.retrieve(name, ids=[point_id(C1)], with_vectors=True, with_payload=True)[0]

    changed = CHUNKS[0].model_copy(update={"text": "| Totally different words about turbines | $1 |"})
    calls_before = emb.doc_calls
    n = indexer.refresh_sparse([changed, CHUNKS[1]])
    assert n == 2 and emb.doc_calls == calls_before          # no embedding model ran

    after = client.retrieve(name, ids=[point_id(C1)], with_vectors=True, with_payload=True)[0]
    assert after.vector[DENSE] == before.vector[DENSE]        # dense untouched
    assert after.payload == before.payload                    # payload untouched
    idx, val = document_vector(embedding_text(changed))
    assert list(after.vector[SPARSE].indices) == idx and list(after.vector[SPARSE].values) == pytest.approx(val)
    assert list(after.vector[SPARSE].indices) != list(before.vector[SPARSE].indices)

    r = TextRetriever(client, emb, name)
    assert r.search("turbines", mode="sparse", limit=1)[0].evidence_id == C1   # now findable by the new text


def test_refresh_sparse_ignores_chunks_that_were_never_indexed(env):
    client, _, indexer, name = env
    indexer.index(CHUNKS[:3])
    assert indexer.refresh_sparse(CHUNKS) == 3                 # the other 5 are skipped, not errors
    assert client.count(name).count == 3