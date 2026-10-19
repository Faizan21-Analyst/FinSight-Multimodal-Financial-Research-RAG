"""Module 9 tests: planner, grader and the corrective loop, all with fakes (no Qdrant, no LLM)."""
import pytest

pytest.importorskip("langgraph")

from finsight.agents.graph import FinSightAgent  # noqa: E402
from finsight.agents.grader import EvidenceGrader, GradeModel, heuristic_grade  # noqa: E402
from finsight.agents.planner import build_subqueries, sql_requests  # noqa: E402
from finsight.agents.router import route  # noqa: E402
from finsight.core.exceptions import RetrievalError  # noqa: E402
from finsight.core.schemas import DocType, EvidenceItem, Modality, SourceRef  # noqa: E402
from finsight.generation.evidence import EvidencePackage  # noqa: E402


def ev(eid, ticker="NVDA", year=2025, modality=Modality.TEXT, text="some text"):
    return EvidenceItem(
        evidence_id=eid, modality=modality, content=text,
        source=SourceRef(doc_id=f"{ticker}-doc", ticker=ticker, doc_type=DocType("10-K"), fiscal_year=year),
        score=1.0, retriever="fake",
    )


class FakeText:
    """Returns chunks per ticker; `empty_when_years` simulates a wrong fiscal-year filter."""

    def __init__(self, empty_when_years=False, fail=False):
        self.calls, self.empty_when_years, self.fail = [], empty_when_years, fail

    def search(self, query, mode="hybrid", limit=10, filt=None):
        self.calls.append((query, filt))
        if self.fail:
            raise RetrievalError("down")
        if self.empty_when_years and filt is not None and filt.fiscal_years:
            return []
        tickers = (filt.tickers if filt and filt.tickers else ("NVDA",))
        return [ev(f"{t}-{abs(hash(query)) % 1000}-{n}", t) for t in tickers for n in range(2)]


class FakeSql:
    def get_metric(self, ticker, metric, fiscal_year, **kw):
        return ev(f"sql-{ticker}-{metric}-{fiscal_year}", ticker, fiscal_year, Modality.SQL, "value")


class ScriptedLLM:
    def __init__(self, grades):
        self.grades, self.n = list(grades), 0

    def complete_json(self, prompt, system=None, model=None, retries=1, **kw):
        g = self.grades[min(self.n, len(self.grades) - 1)]
        self.n += 1
        return GradeModel(**g)


# ---- planner ----------------------------------------------------------------------------

def test_comparison_gets_one_subquery_per_company():
    plan = route("Compare NVIDIA and AMD research and development expense in 2024")
    subs = build_subqueries(plan)
    assert [s.tickers for s in subs] == [("NVDA",), ("AMD",)]


def test_single_company_one_subquery():
    assert len(build_subqueries(route("Why did Tesla margins fall?"))) == 1


def test_sql_requests_need_ticker_metric_year():
    plan = route("What was Apple net income in 2024?")
    assert [(r.ticker, r.metric, r.fiscal_year) for r in sql_requests(plan)] == [("AAPL", "net_income", 2024)]
    assert sql_requests(route("What was Apple net income?")) == []  # no year -> no exact lookup


def test_rewrite_naming_one_company_does_not_leak_to_the_other():
    plan = route("Compare NVIDIA and AMD research and development expense in 2024")
    subs = build_subqueries(plan, "NVIDIA 10-K FY2024 Research and Development expense")
    assert subs[0].text.startswith("NVIDIA")          # NVDA sub-query gets the rewrite
    assert subs[1].text == plan.question               # AMD keeps the original wording


# ---- grader -----------------------------------------------------------------------------

def test_grader_empty_evidence_scores_zero_without_llm():
    g = EvidenceGrader(llm=None).grade(route("x"), EvidencePackage([]))
    assert g.score == 0 and g.source == "empty"


def test_heuristic_penalises_missing_company():
    plan = route("Compare NVIDIA and AMD revenue")
    g = heuristic_grade(plan, EvidencePackage([ev("a", "NVDA")]))
    assert g.score < 0.6 and "AMD" in g.missing


def test_grader_falls_back_when_llm_errors():
    class Boom:
        def complete_json(self, *a, **k):
            from finsight.llm.client import LLMError
            raise LLMError("bad")

    g = EvidenceGrader(Boom()).grade(route("Why did NVIDIA grow?"), EvidencePackage([ev("a")]))
    assert g.source == "heuristic"


# ---- loop -------------------------------------------------------------------------------

def make_agent(text, llm, sql=None, **kw):
    return FinSightAgent(text=text, grader=EvidenceGrader(llm), sql=sql, **kw)


def test_good_evidence_stops_after_one_iteration():
    agent = make_agent(FakeText(), ScriptedLLM([{"score": 0.9}]))
    r = agent.run("Why did NVIDIA data center revenue grow?")
    assert r.iterations == 1
    assert [t["step"] for t in r.trace] == ["plan", "retrieve", "grade", "answer", "verify"]


def test_weak_evidence_rewrites_then_stops_when_good():
    text = FakeText()
    llm = ScriptedLLM([{"score": 0.3, "missing": "segment detail", "rewritten_query": "NVIDIA segment revenue"},
                       {"score": 0.8}])
    r = make_agent(text, llm).run("Why did NVIDIA data center revenue grow?")
    assert r.iterations == 2
    assert [t["step"] for t in r.trace].count("rewrite") == 1
    assert text.calls[1][0] == "NVIDIA segment revenue"


def test_loop_is_capped_at_max_iterations():
    llm = ScriptedLLM([{"score": 0.1, "rewritten_query": "again"}])
    r = make_agent(FakeText(), llm, max_iterations=3).run("Why did NVIDIA data center revenue grow?")
    assert r.iterations == 3
    assert [t["step"] for t in r.trace][-2:] == ["answer", "verify"]


class SameResultsText(FakeText):
    def search(self, query, mode="hybrid", limit=10, filt=None):
        self.calls.append((query, filt))
        return [ev("fixed-1"), ev("fixed-2")]


def test_stops_when_retry_finds_nothing_new():
    llm = ScriptedLLM([{"score": 0.2, "rewritten_query": "again"}])
    r = make_agent(SameResultsText(), llm, max_iterations=3).run("Why did NVIDIA data center revenue grow?")
    assert r.iterations == 2   # not 3: round 2 added nothing
    assert [t for t in r.trace if t["step"] == "retrieve"][1]["new_items"] == 0


def test_empty_result_relaxes_year_filter():
    text = FakeText(empty_when_years=True)
    llm = ScriptedLLM([{"score": 0.8}])
    r = make_agent(text, llm).run("Why did NVIDIA revenue grow in 2031?")
    assert r.iterations == 2
    assert text.calls[0][1].fiscal_years == (2031,)
    assert text.calls[1][1] is not None and not text.calls[1][1].fiscal_years
    assert len(r.package) > 0


def test_comparison_covers_both_companies():
    r = make_agent(FakeText(), ScriptedLLM([{"score": 0.9}])).run(
        "Compare NVIDIA and AMD research and development expense")
    assert {i.source.ticker for i in r.package.items} == {"NVDA", "AMD"}


def test_sql_facts_are_pinned_first():
    r = make_agent(FakeText(), ScriptedLLM([{"score": 0.9}]), sql=FakeSql()).run(
        "What was NVIDIA revenue in 2025?")
    assert r.package.items[0].modality == Modality.SQL


def test_rewrite_naming_one_company_does_not_change_the_other_subquery():
    plan = route("Compare NVIDIA and AMD research and development expense in 2024")
    subs = build_subqueries(plan, "NVIDIA 10-K FY2024 Research and Development expense")
    assert subs[0].text.startswith("NVIDIA 10-K")
    assert subs[1].text == plan.question  # AMD keeps the original wording


class FixedText(FakeText):
    """Same chunks whatever the query, like the real run where retries returned identical evidence."""

    def search(self, query, mode="hybrid", limit=10, filt=None):
        self.calls.append((query, filt))
        return [ev("fixed-1"), ev("fixed-2")]


def test_stops_early_when_a_retry_finds_nothing_new():
    llm = ScriptedLLM([{"score": 0.3, "rewritten_query": "NVIDIA again"}])
    r = make_agent(FixedText(), llm, max_iterations=5).run("Why did NVIDIA data center revenue grow?")
    assert r.iterations == 2  # round 2 added no new evidence, so no round 3


def test_retriever_failure_is_logged_not_raised():
    r = make_agent(FakeText(fail=True), ScriptedLLM([{"score": 0.9}]), max_iterations=2).run("Why did NVIDIA grow?")
    retrieve = [t for t in r.trace if t["step"] == "retrieve"][0]
    assert retrieve["errors"] and len(r.package) == 0
