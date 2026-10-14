import math

import pytest
from qdrant_client import models

from finsight.core.schemas import Chunk, DocType, Modality, SourceRef
from finsight.indexing.embedders import HashingEmbedder, SentenceTransformerEmbedder, build_embedder, collection_name
from finsight.indexing.sparse import document_vector, query_vector, term_index, tokenize
from finsight.indexing.text_index import chunk_payload, embedding_text, point_id
from finsight.retrieval.fusion import reciprocal_rank_fusion
from finsight.retrieval.text_retriever import SearchFilter, payload_to_evidence


# ---------- sparse ----------
def test_numbers_are_normalised_across_formats():
    assert tokenize("Revenue was $22,563 million") == ["revenue", "22563", "million"]
    assert tokenize("(1,234.5)") == ["1234.5"]
    assert tokenize("in 2024, revenue") == ["2024", "revenue"]
    assert tokenize("74.0%") == ["74.0"]
    assert tokenize("22,563") == tokenize("$22563")


def test_words_stopwords_and_identifiers():
    assert tokenize("The Data Center and the Gaming segment") == ["data", "center", "gaming", "segment"]
    assert "fy2024" in tokenize("FY2024 results")
    assert tokenize("Item 1A") == ["item", "1"]      # the lone "a" is dropped; use section filters for "1A"
    assert tokenize("") == [] and tokenize("the of and") == []


def test_term_index_is_stable_u32():
    assert term_index("revenue") == term_index("revenue")
    assert 0 <= term_index("revenue") < 2**32
    assert term_index("revenue") != term_index("revenues")


def test_document_vector_bm25_properties():
    idx, val = document_vector("revenue revenue revenue growth")
    assert idx == sorted(idx) and len(idx) == len(set(idx)) == 2
    w = dict(zip(idx, val))
    assert w[term_index("revenue")] > w[term_index("growth")] > 0
    # saturation: 3x the frequency must be well under 3x the weight
    assert w[term_index("revenue")] < 3 * w[term_index("growth")]
    # longer documents dilute the same term
    short = dict(zip(*document_vector("margin")))[term_index("margin")]
    long_ = dict(zip(*document_vector("margin " + "filler " * 400)))[term_index("margin")]
    assert long_ < short
    assert document_vector("") == ([], [])


def test_query_vector_unique_terms_weight_one():
    idx, val = query_vector("revenue revenue growth")
    assert len(idx) == 2 and val == [1.0, 1.0]
    assert query_vector("the of") == ([], [])


# ---------- fusion ----------
def test_rrf_rewards_agreement_over_single_list_wins():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "d"]], k=60)
    order = [x for x, _ in fused]
    # b and c were found by BOTH lists, so both outrank a and d (found by one list each)
    assert set(order[:2]) == {"b", "c"} and set(order[2:]) == {"a", "d"}
    # c (ranks 3 and 1) edges b (ranks 2 and 2): RRF is convex, one top rank counts for a lot
    assert order[0] == "c"
    scores = dict(fused)
    assert scores["b"] == pytest.approx(2 / 62)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert set(order) == {"a", "b", "c", "d"}


def test_rrf_weights_k_dedupe_and_empty():
    assert reciprocal_rank_fusion([[], []]) == []
    w = reciprocal_rank_fusion([["a"], ["b"]], weights=[2.0, 1.0])
    assert [x for x, _ in w] == ["a", "b"]
    dup = dict(reciprocal_rank_fusion([["a", "a", "a"]]))
    assert dup["a"] == pytest.approx(1 / 61)     # repeated item counted once, at best rank
    big_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    small_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    assert (small_k["a"] / small_k["b"]) > (big_k["a"] / big_k["b"])   # small k favours the top
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


def test_rrf_ties_are_deterministic():
    r1 = reciprocal_rank_fusion([["x"], ["y"]])
    r2 = reciprocal_rank_fusion([["x"], ["y"]])
    assert r1 == r2 and [i for i, _ in r1] == ["x", "y"]


# ---------- embedders ----------
def test_hashing_embedder_deterministic_normalised_and_lexical():
    e = HashingEmbedder(64)
    a, b = e.embed_documents(["data center revenue", "data center revenue"])
    assert a == b and len(a) == e.dim == 64
    assert math.sqrt(sum(x * x for x in a)) == pytest.approx(1.0)
    q = e.embed_query("data center revenue")
    same = sum(x * y for x, y in zip(q, a))
    other = sum(x * y for x, y in zip(q, e.embed_documents(["gaming graphics gpu"])[0]))
    assert same > other


def test_collection_name_and_factory():
    assert collection_name("finsight_text", "BAAI/bge-m3") == "finsight_text__bge-m3"
    assert collection_name("finsight_text", "BAAI/bge-small-en-v1.5") == "finsight_text__bge-small-en-v1.5"
    assert isinstance(build_embedder("hashing-32"), HashingEmbedder) and build_embedder("hashing-32").dim == 32
    st = build_embedder("BAAI/bge-small-en-v1.5")          # must not load the model
    assert isinstance(st, SentenceTransformerEmbedder) and st._model is None


def test_bge_query_instruction_only_where_expected():
    assert SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5").query_instruction.startswith("Represent")
    assert SentenceTransformerEmbedder("BAAI/bge-m3").query_instruction == ""
    assert SentenceTransformerEmbedder("BAAI/bge-m3", query_instruction="x ").query_instruction == "x "


# ---------- index helpers ----------
def _chunk(**kw):
    src = SourceRef(doc_id="nvda-10-k-2024-01-28", ticker="NVDA", doc_type=DocType.TEN_K,
                    fiscal_year=2024, fiscal_period="FY", page=41, section="Item 7")
    base = dict(chunk_id="f51c7a8fb2a34844", text="| Data Center | $22,563 |", modality=Modality.TABLE,
                source=src, metadata={"section_title": "MD&A", "caption": "Revenue by market"})
    base.update(kw)
    return Chunk(**base)


def test_point_id_is_valid_u64_and_stable():
    assert point_id("f51c7a8fb2a34844") == int("f51c7a8fb2a34844", 16) < 2**64
    assert point_id("0000000000000001") == 1


def test_embedding_text_carries_context_but_payload_text_is_clean():
    c = _chunk()
    et = embedding_text(c)
    assert et.startswith("NVDA | 10-K | FY2024 FY | Item 7 | MD&A\n")
    assert et.endswith(c.text)
    assert chunk_payload(c)["text"] == c.text                # stored text has no header
    p = chunk_payload(c)
    assert p["ticker"] == "NVDA" and p["fiscal_year"] == 2024 and p["modality"] == "table"
    assert p["caption"] == "Revenue by market" and p["page"] == 41


def test_search_filter_builds_and_or_conditions():
    assert SearchFilter().to_qdrant() is None
    f = SearchFilter(tickers=("NVDA", "AMD"), fiscal_years=(2024,), modalities=("table",)).to_qdrant()
    got = {c.key: c.match.any for c in f.must}
    assert got == {"ticker": ["NVDA", "AMD"], "fiscal_year": [2024], "modality": ["table"]}


def test_payload_roundtrip_to_evidence():
    ev = payload_to_evidence(chunk_payload(_chunk()), 0.5, "hybrid")
    assert ev.evidence_id == "f51c7a8fb2a34844" and ev.modality == Modality.TABLE
    assert ev.source.section == "Item 7" and ev.source.page == 41 and ev.retriever == "hybrid"
    assert isinstance(models.Filter(must=[]), models.Filter)

def test_possessives_and_question_words_do_not_pollute_the_sparse_query():
    assert tokenize("NVIDIA's revenue") == ["nvidia", "revenue"]
    assert tokenize("customers' orders") == ["customers", "orders"]
    assert tokenize("NVIDIA\u2019s revenue") == ["nvidia", "revenue"]          # curly apostrophe
    assert tokenize("How does Tesla account for its digital assets?") == ["tesla", "account", "digital", "assets"]
    assert tokenize("it's 22,563") == ["22563"]
    # a possessive query now overlaps the unpossessed text it should match
    assert set(query_vector("What was NVIDIA's revenue?")[0]) <= set(document_vector("NVIDIA revenue grew")[0])