import pytest

from finsight.agents.router import Intent, route
from finsight.core.schemas import DocType, EvidenceItem, Modality, SourceRef
from finsight.generation.evidence import assemble


@pytest.mark.parametrize(
    "question,intent",
    [
        ("Why did NVIDIA's operating margin fall in fiscal 2024?", Intent.NARRATIVE),
        ("What was Apple's net income in fiscal 2023?", Intent.NUMERIC),
        ("Compare NVIDIA and AMD research and development spending in fiscal 2024", Intent.COMPARISON),
        ("Which has higher data center revenue, NVIDIA or AMD?", Intent.COMPARISON),
        ("Find the data center revenue chart in NVIDIA's investor deck", Intent.VISUAL),
        ("Is the revenue chart from the investor deck supported by the 10-K?", Intent.CROSS_MODAL),
        ("Describe the export-control risks Tesla mentions", Intent.NARRATIVE),
    ],
)
def test_intents(question, intent):
    assert route(question).intent == intent


def test_entities_and_filter():
    plan = route("What was Apple's research and development expense in fiscal 2022?")
    assert plan.tickers == ("AAPL",)
    assert plan.fiscal_years == (2022,)
    assert plan.metrics == ("rd_expense",)
    assert plan.retrievers == ("sql", "text")
    f = plan.search_filter()
    assert f.tickers == ("AAPL",) and f.fiscal_years == (2022,)


def test_no_entities_means_no_filter():
    assert route("What are the main supply chain risks?").search_filter() is None


def test_cross_modal_uses_all_retrievers():
    plan = route("Does the slide showing data center growth match the 10-K?")
    assert plan.intent == Intent.CROSS_MODAL
    assert set(plan.retrievers) == {"visual", "text", "sql"}


def _ev(eid, modality=Modality.TEXT, content="x", retriever="hybrid"):
    return EvidenceItem(
        evidence_id=eid, modality=modality, content=content, retriever=retriever,
        source=SourceRef(doc_id="d", ticker="NVDA", doc_type=DocType.TEN_K, fiscal_year=2024),
    )


def test_sql_is_pinned_first_and_duplicates_collapse():
    sql = [_ev("s1", Modality.SQL, "NVDA R&D FY2024: $8.675 billion", "sql")]
    text = [_ev("t1"), _ev("t2"), _ev("t1")]
    visual = [_ev("t2", Modality.PAGE_IMAGE), _ev("v1", Modality.PAGE_IMAGE)]
    pkg = assemble({"sql": sql, "text": text, "visual": visual})
    ids = [i.evidence_id for i in pkg.items]
    assert ids[0] == "s1"
    assert len(ids) == len(set(ids)) == 4
    assert ids[1] == "t2"  # appears in two lists, so RRF ranks it above t1


def test_budget_and_labels():
    big = [_ev(f"t{i}", content="y" * 1000) for i in range(10)]
    pkg = assemble({"text": big}, max_items=8, max_chars=3500)
    assert len(pkg) == 3
    assert list(pkg.by_label()) == ["E1", "E2", "E3"]
    assert "[E1] NVDA | 10-K | FY2024" in pkg.render_context()


def test_keeps_one_item_even_over_budget():
    pkg = assemble({"text": [_ev("t1", content="z" * 5000)]}, max_chars=100)
    assert len(pkg) == 1


def test_empty_results():
    assert len(assemble({})) == 0
