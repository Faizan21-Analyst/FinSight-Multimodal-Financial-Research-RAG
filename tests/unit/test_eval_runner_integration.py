"""Needs a running Qdrant. Skipped automatically otherwise. Uses a throwaway collection."""
import json
import os
import uuid

import pytest
from qdrant_client import QdrantClient

from finsight.core.schemas import Chunk, DocType, Modality, SourceRef, make_id
from finsight.evaluation.dataset import GoldSpec, OracleFilter, Question
from finsight.evaluation.runner import EvalConfig, evaluate, format_markdown, misses, summarize, to_jsonable
from finsight.indexing.embedders import HashingEmbedder
from finsight.indexing.text_index import TextIndexer, chunk_payload
from finsight.retrieval.reranker import OverlapReranker, RerankingBackend
from finsight.retrieval.text_retriever import TextRetriever

K, Q = DocType.TEN_K, DocType.TEN_Q
T, TB = Modality.TEXT, Modality.TABLE


def _c(key, ticker, dt, fy, period, section, modality, text):
    return Chunk(chunk_id=make_id(key), text=text, modality=modality,
                 source=SourceRef(doc_id=f"{ticker}-{dt.value}-{fy}", ticker=ticker, doc_type=dt,
                                  fiscal_year=fy, fiscal_period=period, page=1, section=section),
                 metadata={"section_title": ""})


CHUNKS = [
    _c("n1", "NVDA", K, 2024, "FY", "Item 7", TB, "| Research and development | $8,675 | $7,339 |"),
    _c("n2", "NVDA", K, 2023, "FY", "Item 7", TB, "| Research and development | $7,339 | $5,268 |"),
    _c("a1", "AMD", K, 2024, "FY", "Item 8", TB, "| Research and development | $5,872 | $5,872 |"),
    _c("a2", "AMD", K, 2024, "FY", "Item 7", T, "data center segment revenue growth driven by Instinct MI300"),
    _c("n3", "NVDA", Q, 2025, "Q1", "Part I Item 2", TB, "| Data Center | $22,563 | $4,284 |"),
    _c("n4", "NVDA", K, 2024, "FY", "Item 1A", T, "export controls China advanced computing license requirements"),
    _c("p1", "AAPL", K, 2024, "FY", "Item 1", T, "iPhone Mac iPad wearables services"),
    _c("p2", "TSLA", K, 2024, "FY", "Item 7", T, "automotive regulatory credits revenue"),
]
N1, N2, A1, A2, N3, N4, P1, P2 = (c.chunk_id for c in CHUNKS)
CORPUS = [chunk_payload(c) for c in CHUNKS]

QUESTIONS = [
    Question(id="q1", category="numeric_lookup", question="NVDA research and development expense fiscal 2024",
             gold=[GoldSpec(ticker="NVDA", modality="table", all_of=["8,675"], any_of=["research and development"])],
             oracle=OracleFilter(tickers=["NVDA"], fiscal_years=[2024], doc_types=["10-K"])),
    Question(id="q2", category="numeric_lookup", question="Data Center revenue 22,563",
             gold=[GoldSpec(ticker="NVDA", modality="table", all_of=["22,563"])],
             oracle=OracleFilter(tickers=["NVDA"])),
    Question(id="q3", category="comparison", question="Compare NVDA and AMD research and development spending",
             gold=[GoldSpec(ticker="NVDA", modality="table", all_of=["research and development"]),
                   GoldSpec(ticker="AMD", modality="table", all_of=["research and development"])],
             oracle=OracleFilter(tickers=["NVDA", "AMD"])),
    Question(id="q-missing", category="narrative", question="Something not in the corpus",
             gold=[GoldSpec(all_of=["quantum teleportation"])]),
]


@pytest.fixture()
def backend():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, timeout=10)
    try:
        client.get_collections()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qdrant not reachable at {url}: {e}")
    name, emb = f"test_finsight_{uuid.uuid4().hex[:8]}", HashingEmbedder(64)
    TextIndexer(client, emb, name).ensure_collection()
    TextIndexer(client, emb, name).index(CHUNKS)
    yield RerankingBackend(TextRetriever(client, emb, name), OverlapReranker(), candidates=30)
    client.delete_collection(name)


def _run(backend, modes=("dense", "sparse", "hybrid", "hybrid+rerank")):
    cfgs = [EvalConfig(m, oracle) for m in modes for oracle in (False, True)]
    return evaluate(QUESTIONS, CORPUS, backend, cfgs)


def test_questions_without_gold_are_dropped_with_a_reason(backend):
    r = _run(backend, modes=("hybrid",))
    assert r.n_questions == 3
    assert [qid for qid, _ in r.dropped] == ["q-missing"] and "no gold chunk" in r.dropped[0][1]
    assert {rec.question_id for rec in r.records} == {"q1", "q2", "q3"}
    assert r.gold_sizes == {"q1": [1], "q2": [1], "q3": [2, 1]}


def test_every_config_scores_every_question(backend):
    r = _run(backend)
    assert len(r.records) == 4 * 2 * 3                      # modes x oracle x questions
    s = summarize(r.records)
    assert all(v["n"] == 3 for v in s.values()) and len(s) == 8
    assert all(rec.latency_ms > 0 for rec in r.records)


def test_exact_number_found_at_rank_one_by_sparse(backend):
    r = _run(backend, modes=("sparse",))
    q2 = next(x for x in r.records if x.question_id == "q2" and not x.oracle)
    assert q2.hit1 == 1.0 and q2.mrr == 1.0 and q2.first_ranks == [1]


def test_oracle_filter_never_hurts_and_fixes_the_confusable_question(backend):
    r = _run(backend, modes=("hybrid",))
    q1 = {x.oracle: x for x in r.records if x.question_id == "q1"}
    assert q1[True].hit1 == 1.0                             # only NVDA FY2024 10-K is searched
    s = summarize(r.records)
    assert s[("hybrid", True)]["mrr"] >= s[("hybrid", False)]["mrr"]


def test_multi_spec_question_reports_coverage_across_specs(backend):
    r = _run(backend, modes=("hybrid",))
    q3 = next(x for x in r.records if x.question_id == "q3" and x.oracle)
    assert len(q3.first_ranks) == 2 and all(fr is not None for fr in q3.first_ranks)
    assert q3.hit10 == 1.0                                  # both companies' evidence retrieved


def test_misses_report_and_markdown_and_json(backend):
    r = _run(backend)
    modes = ["dense", "sparse", "hybrid", "hybrid+rerank"]
    md = format_markdown(r, {"embedding model": "hashing-64"}, modes)
    for needle in ("# Retrieval evaluation", "## Overall: Unfiltered", "## Overall: With oracle filter",
                   "## By category", "numeric_lookup", "comparison", "hybrid+rerank",
                   "## Dropped questions", "q-missing", "embedding model: hashing-64"):
        assert needle in md, needle
    assert isinstance(misses(r.records, "sparse", False), list)
    blob = json.dumps(to_jsonable(r))                       # must be JSON-serialisable
    assert json.loads(blob)["n_questions"] == 3

def test_split_config_requires_oracle_and_interleaves_companies():
    with pytest.raises(ValueError, match="oracle"):
        EvalConfig("hybrid", oracle=False, split=True)
    from finsight.evaluation.runner import interleave
    from finsight.core.schemas import EvidenceItem

    def ev(i):
        return EvidenceItem(evidence_id=i, modality=Modality.TEXT, content="", score=0, retriever="x",
                            source=SourceRef(doc_id="d", ticker="T", doc_type=DocType.TEN_K))
    out = interleave([[ev("a1"), ev("a2"), ev("a3")], [ev("b1"), ev("b2")], [ev("a1"), ev("c1")]], 5)
    assert [e.evidence_id for e in out] == ["a1", "b1", "a2", "b2", "c1"]      # round-robin, de-duplicated
    assert [e.evidence_id for e in interleave([[ev("a1")], []], 3)] == ["a1"]


@pytest.fixture()
def skewed():
    """NVDA has six strongly matching chunks and AMD one weaker match, so a single
    two-company query is dominated by NVDA. Exactly the failure split retrieval should fix."""
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, timeout=10)
    try:
        client.get_collections()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qdrant not reachable at {url}: {e}")
    chunks = [_c(f"sn{i}", "NVDA", K, 2020 + i, "FY", "Item 7", TB,
                 "| Research and development spending compare | $%d | $1 |" % (8000 + i)) for i in range(6)]
    chunks.append(_c("sa", "AMD", K, 2024, "FY", "Item 8", TB, "| Research and development | $5,872 |"))
    name, emb = f"test_finsight_{uuid.uuid4().hex[:8]}", HashingEmbedder(64)
    TextIndexer(client, emb, name).ensure_collection()
    TextIndexer(client, emb, name).index(chunks)
    # no company names in the question text: the oracle supplies them, and a rare name token
    # (high IDF) would otherwise hand one company an arbitrary lexical advantage
    q = Question(id="cmp", category="comparison", question="Compare the two companies research and development spending",
                 gold=[GoldSpec(ticker="NVDA", modality="table", all_of=["research and development"]),
                       GoldSpec(ticker="AMD", modality="table", all_of=["research and development"])],
                 oracle=OracleFilter(tickers=["NVDA", "AMD"]))
    yield q, [chunk_payload(c) for c in chunks], TextRetriever(client, emb, name)
    client.delete_collection(name)


def test_split_rescues_the_company_a_single_query_buries(skewed):
    q, corpus, retriever = skewed
    plain = evaluate([q], corpus, retriever, [EvalConfig("hybrid", True)]).records[0]
    split = evaluate([q], corpus, retriever, [EvalConfig("hybrid", True, split=True)]).records[0]
    assert split.mode == "hybrid+split" and plain.mode == "hybrid"
    assert plain.first_ranks == [1, 7]                                # AMD buried below all 6 NVDA chunks
    assert split.first_ranks == [1, 2]                                # both companies at the top
    assert split.mrr > plain.mrr and split.hit3 == 1.0 and plain.hit3 == 0.5


def test_split_equals_plain_oracle_on_single_company_questions(backend):
    q1 = [q for q in QUESTIONS if q.id == "q1"]
    a = evaluate(q1, CORPUS, backend, [EvalConfig("hybrid", True)]).records[0]
    b = evaluate(q1, CORPUS, backend, [EvalConfig("hybrid", True, split=True)]).records[0]
    assert (a.hit1, a.mrr, a.first_ranks) == (b.hit1, b.mrr, b.first_ranks)


def test_report_has_by_category_tables_for_both_filter_settings(backend):
    cfgs = [EvalConfig("hybrid", False), EvalConfig("hybrid", True), EvalConfig("hybrid", True, True)]
    r = evaluate(QUESTIONS, CORPUS, backend, cfgs)
    md = format_markdown(r, {}, ["hybrid", "hybrid+split"])
    assert "## By category (unfiltered)" in md and "## By category (oracle filter)" in md
    oracle_block = md.split("## By category (oracle filter)")[1]
    assert "hybrid+split" in oracle_block.split("\n")[2]          # column header row