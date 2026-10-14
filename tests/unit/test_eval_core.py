import math
from decimal import Decimal

import pytest

from finsight.core.caching import DiskCache
from finsight.core.schemas import DocType, EvidenceItem, Modality, SourceRef
from finsight.evaluation.dataset import (
    GoldSpec, OracleFilter, Question, load_questions, normalize, resolve_gold, save_questions,
)
from finsight.evaluation.metrics import first_rank, hit_at_k, ndcg_at_k, reciprocal_rank
from finsight.evaluation.numeric_questions import make_numeric_questions
from finsight.retrieval.reranker import OverlapReranker, RerankingBackend, evidence_text, rerank


def chunk(cid, text, ticker="NVDA", doc_type="10-K", fy=2024, period="FY", section="Item 7", modality="table"):
    return dict(chunk_id=cid, text=text, ticker=ticker, doc_type=doc_type, fiscal_year=fy,
                fiscal_period=period, section=section, modality=modality)


# ---------- gold predicates ----------
def test_number_phrases_match_whole_numbers_only():
    s = GoldSpec(all_of=["8,675"])
    assert s.matches(chunk("a", "| R&D | $8,675 | $7,339 |"))
    assert not s.matches(chunk("b", "| R&D | $18,675 |"))
    assert not s.matches(chunk("c", "| R&D | 8,6751 |"))
    assert not s.matches(chunk("d", "| R&D | 8,675.5 |"))


def test_all_of_any_of_and_normalisation():
    s = GoldSpec(all_of=["export"], any_of=["China", "Hong Kong"])
    assert s.matches(chunk("a", "Export controls on sales to CHINA apply."))
    assert not s.matches(chunk("b", "Export controls apply."))          # any_of unmet
    assert not s.matches(chunk("c", "Sales to China grew."))            # all_of unmet
    assert GoldSpec(all_of=["nvidia's"]).matches(chunk("d", "NVIDIA\u2019s revenue"))   # curly apostrophe
    assert GoldSpec(all_of=["data center"]).matches(chunk("e", "Data\xa0Center   revenue"))


def test_metadata_fields_and_section_prefix():
    c = chunk("a", "text", ticker="AMD", doc_type="10-Q", fy=2025, period="Q1",
              section="Part I Item 2", modality="text")
    assert GoldSpec(ticker="AMD", doc_type="10-Q", fiscal_year=2025, fiscal_period="Q1",
                    section="Part I", modality="text").matches(c)
    assert not GoldSpec(ticker="NVDA").matches(c)
    assert not GoldSpec(section="Item 2").matches(c)                    # prefix, not substring
    assert not GoldSpec(modality="table").matches(c)


def test_resolve_gold_returns_ids():
    corpus = [chunk("a", "revenue 1,000"), chunk("b", "revenue 2,000"), chunk("c", "revenue 1,000", ticker="AMD")]
    assert resolve_gold(corpus, GoldSpec(ticker="NVDA", all_of=["1,000"])) == {"a"}
    assert resolve_gold(corpus, GoldSpec(all_of=["revenue"])) == {"a", "b", "c"}
    assert resolve_gold(corpus, GoldSpec(all_of=["nothing here"])) == set()
    assert normalize("A\u2019s  B") == "a's b"


# ---------- dataset io ----------
def test_dataset_roundtrip_comments_and_duplicate_ids(tmp_path):
    qs = [Question(id="q1", category="c", question="Q?", gold=[GoldSpec(all_of=["x"])],
                   oracle=OracleFilter(tickers=["NVDA"], fiscal_years=[2024]))]
    p = tmp_path / "q.jsonl"
    save_questions(p, qs)
    p.write_text("# a comment line\n\n" + p.read_text())
    got = load_questions(p)
    assert got == qs and got[0].oracle.fiscal_years == [2024]
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(p, p)
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_questions(bad)


# ---------- metrics (hand-computed) ----------
def test_rank_metrics():
    flags = [False, True, False, True]
    assert first_rank(flags) == 2 and first_rank([False, False]) is None
    assert hit_at_k(flags, 1) == 0.0 and hit_at_k(flags, 2) == 1.0
    assert reciprocal_rank(flags) == 0.5
    assert reciprocal_rank(flags, k=1) == 0.0                   # first hit is outside the cutoff
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(flags, 10, n_relevant=2) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([True, True], 10, n_relevant=2) == pytest.approx(1.0)
    assert ndcg_at_k([True], 10, n_relevant=50) < 1.0           # many relevant chunks exist
    assert ndcg_at_k([False], 10, n_relevant=0) == 0.0


# ---------- numeric question generation ----------
CORPUS = [
    chunk("t1", "| Research and development | $8,675 | $7,339 |"),
    chunk("t2", "| Total net sales | $391,035 | $383,285 |", ticker="AAPL"),
    chunk("t3", "| Total revenue | $26,044 | $7,192 |", doc_type="10-Q", period="Q1", fy=2025),
    chunk("t4", "| Net income | $12 | $9 |"),
]


def _row(ticker, metric, fy, value, pt="FY", q=None):
    return dict(ticker=ticker, metric=metric, period_type=pt, fiscal_year=fy, fiscal_quarter=q, value=Decimal(value))


def test_numeric_questions_have_objective_gold_and_skips_are_explained():
    rows = [
        _row("NVDA", "rd_expense", 2024, "8675000000"),
        _row("AAPL", "revenue", 2024, "391035000000"),                  # "net sales" label
        _row("NVDA", "revenue", 2025, "26044000000", "Q", 1),
        _row("NVDA", "net_income", 2024, "12000000"),                   # token "12": too short
        _row("NVDA", "operating_income", 2024, "99999000000"),          # appears in no chunk
        _row("NVDA", "capex", 2024, "5000000000"),                      # metric we don't ask about
        _row("XYZ", "revenue", 2024, "1000000000"),                     # unknown company
    ]
    qs, skipped = make_numeric_questions(rows, CORPUS)
    by_id = {q.id: q for q in qs}
    assert set(by_id) == {"num-NVDA-rd_expense-fy2024", "num-AAPL-revenue-fy2024", "num-NVDA-revenue-fy2025-q1"}
    q = by_id["num-NVDA-rd_expense-fy2024"]
    assert q.question == "What was NVIDIA's research and development expense in fiscal 2024?"
    assert q.gold[0].all_of == ["8,675"] and q.gold[0].modality == "table" and q.gold[0].ticker == "NVDA"
    assert q.oracle.tickers == ["NVDA"] and q.oracle.fiscal_years == [2024] and q.oracle.doc_types == ["10-K"]
    assert resolve_gold(CORPUS, q.gold[0]) == {"t1"}
    assert by_id["num-AAPL-revenue-fy2024"].question == "What was Apple's total revenue in fiscal 2024?"
    qq = by_id["num-NVDA-revenue-fy2025-q1"]
    assert "first quarter of fiscal 2025" in qq.question and qq.oracle.doc_types == ["10-Q"]
    reasons = " ".join(skipped)
    assert "too short" in reasons and "no table chunk contains 99,999" in reasons


def test_quarterly_sampling_is_capped_and_deterministic():
    rows = [_row("NVDA", "revenue", fy, "26044000000", "Q", q) for fy in (2024, 2025, 2026) for q in (1, 2, 3)]
    a, _ = make_numeric_questions(rows, CORPUS, max_quarterly=4, seed=1)
    b, _ = make_numeric_questions(rows, CORPUS, max_quarterly=4, seed=1)
    assert len(a) == 4 and [q.id for q in a] == [q.id for q in b]


# ---------- reranker ----------
def _ev(cid, text):
    return EvidenceItem(evidence_id=cid, modality=Modality.TEXT, content=text, score=0.1, retriever="hybrid",
                        source=SourceRef(doc_id="d", ticker="NVDA", doc_type=DocType.TEN_K,
                                         fiscal_year=2024, fiscal_period="FY", section="Item 7"))


class CountingReranker(OverlapReranker):
    def __init__(self):
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        return super().score(query, texts)


def test_rerank_orders_by_score_and_relabels():
    items = [_ev("a", "gaming graphics"), _ev("b", "data center revenue growth"), _ev("c", "data center")]
    out = rerank("data center revenue", items, OverlapReranker())
    assert [i.evidence_id for i in out] == ["b", "c", "a"]
    assert all(i.retriever == "hybrid+rerank" for i in out)
    assert out[0].score == pytest.approx(1.0) and items[0].score == 0.1       # originals untouched
    assert [i.evidence_id for i in rerank("data center revenue", items, OverlapReranker(), top_n=1)] == ["b"]
    assert rerank("q", [], OverlapReranker()) == []


def test_rerank_ties_keep_retrieval_order():
    items = [_ev("x", "zzz"), _ev("y", "qqq"), _ev("z", "www")]
    assert [i.evidence_id for i in rerank("unrelated", items, OverlapReranker())] == ["x", "y", "z"]


def test_rerank_cache_avoids_recomputation(tmp_path):
    items = [_ev("a", "alpha"), _ev("b", "beta alpha")]
    rr, cache = CountingReranker(), DiskCache(tmp_path)
    first = rerank("alpha beta", items, rr, cache=cache)
    again = rerank("alpha beta", list(reversed(items)), rr, cache=cache)       # same candidate SET
    assert rr.calls == 1 and [i.evidence_id for i in first] == [i.evidence_id for i in again]
    rerank("different query", items, rr, cache=cache)
    assert rr.calls == 2


def test_evidence_text_has_context_header():
    assert evidence_text(_ev("a", "body")).startswith("NVDA | 10-K | FY2024 FY | Item 7\nbody")


def test_reranking_backend_adds_mode_and_delegates_others():
    class Base:
        def __init__(self):
            self.seen = []

        def search(self, query, mode="hybrid", limit=10, filt=None):
            self.seen.append((mode, limit))
            return [_ev("a", "gaming"), _ev("b", "data center revenue")][:limit]

    base = Base()
    be = RerankingBackend(base, OverlapReranker(), candidates=30)
    out = be.search("data center revenue", mode="hybrid+rerank", limit=1)
    assert [i.evidence_id for i in out] == ["b"] and base.seen == [("hybrid", 30)]
    be.search("x", mode="dense", limit=5)
    assert base.seen[-1] == ("dense", 5)