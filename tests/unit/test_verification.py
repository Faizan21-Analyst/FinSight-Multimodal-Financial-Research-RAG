"""Module 11 tests: claim splitting, number/citation checks, LLM judging, confidence rules."""
from decimal import Decimal

import pytest

from finsight.core.exceptions import FinSightError
from finsight.core.schemas import Confidence, DocType, EvidenceItem, Modality, SourceRef, Verdict
from finsight.generation.claim_verifier import (
    ClaimVerifier,
    JudgementBatch,
    answer_from_report,
    number_in_text,
    overall_verdict,
)
from finsight.generation.claims import (
    ClaimCheck,
    checkable_numbers,
    extract_claims,
    parse_labels,
    split_sentences,
    strip_labels,
)
from finsight.generation.confidence import report_confidence, score_confidence
from finsight.generation.evidence import EvidencePackage


def ev(eid, content, ticker="NVDA", modality=Modality.TEXT):
    return EvidenceItem(evidence_id=eid, modality=modality, content=content,
                        source=SourceRef(doc_id="d", ticker=ticker, doc_type=DocType("10-K"), fiscal_year=2024))


E1 = "NVDA rd_expense for FY2024: $8.675 billion (raw value 8675000000 USD; period 2023-01-30 to 2024-01-28)"
E2 = "AMD rd_expense for FY2024: $6.456 billion (raw value 6456000000 USD; period 2023-12-31 to 2024-12-28)"
E3 = "AMD R&D expenses were $6.5 billion in 2024, up 10% from $5.9 billion in 2023."


def pkg():
    return EvidencePackage([ev("a", E1), ev("b", E2, "AMD"), ev("c", E3, "AMD")])


class FakeJudge:
    def __init__(self, verdicts=None, fail=False):
        self.verdicts, self.fail, self.calls = verdicts or {}, fail, 0

    def complete_json(self, prompt, system=None, model=None, retries=1, **kw):
        self.calls += 1
        if self.fail:
            raise FinSightError("down")
        ids = [int(s.split(":")[0].split()[1]) for s in prompt.splitlines() if s.startswith("Claim ")]
        return model.model_validate({"judgements": [
            {"id": i, "verdict": self.verdicts.get(i, "supported"), "reason": "ok"} for i in ids]})


# ---- splitting, labels, numbers ---------------------------------------------------------

def test_parse_labels_variants():
    assert parse_labels("a [E1] b [E2, E3] c [E1]") == ["E1", "E2", "E3"]
    assert parse_labels("no citations here") == []
    assert strip_labels("R&D was $8B [E1].") == "R&D was $8B ."


def test_split_keeps_label_with_its_sentence_even_after_the_period():
    s = split_sentences("NVIDIA spent $8.675 billion. [E1] AMD spent $6.456 billion [E2].")
    assert len(s) == 2 and "[E1]" in s[0] and "[E2]" in s[1]


def test_bullets_headings_and_filler_are_handled():
    answer = ("Based on the evidence, here is the comparison:\n\n**NVIDIA**\n"
              "*   **R&D Expense:** $8.675 billion [E1].\n*   Fiscal Period: January 30, 2023 [E1].\n")
    claims = extract_claims(answer)
    assert len(claims) == 2 and all("[E1]" in c for c in claims)


def test_abbreviations_do_not_split_sentences():
    assert len(split_sentences("The U.S. Government imposed export controls [E1].")) == 1


def test_checkable_numbers_skip_years_dates_and_small_integers():
    nums = checkable_numbers("In fiscal 2024 (Q2 FY27), on January 30, 2023, NVIDIA had 2 segments and $8.675 billion")
    assert [n.raw for n in nums] == ["$8.675 billion"]
    assert nums[0].value == Decimal("8675000000")
    assert [n.kind for n in checkable_numbers("up 10% to 6,718 and $5.9B")] == ["pct", "count", "money"]


def test_number_in_text_allows_rounding_both_ways():
    n = checkable_numbers("$6.5 billion")[0]
    assert number_in_text(n, E2)                       # 6.456B is within half a tenth of a billion
    assert not number_in_text(checkable_numbers("$7.5 billion")[0], E2)
    assert number_in_text(checkable_numbers("$8.675 billion")[0], E1)


# ---- verifier (no LLM) ------------------------------------------------------------------

def verify(text, judge=None, package=None, **kw):
    return ClaimVerifier(judge).verify("q", text, package or pkg(), **kw)


def test_correct_numbers_are_supported_but_flagged_numbers_only_without_llm():
    v = verify("NVIDIA R&D was $8.675 billion [E1]. AMD R&D was $6.456 billion [E2].")
    assert v.answer.verdict == Verdict.SUPPORTED
    assert all("numbers_only" in c.flags for c in v.checks)
    assert v.answer.confidence == Confidence.MEDIUM    # capped: meaning was not judged


def test_number_cited_to_wrong_item_is_miscited_partial():
    v = verify("AMD R&D was $6.456 billion [E1].")
    c = v.checks[0]
    assert c.verdict == Verdict.PARTIAL and "miscited" in c.flags


def test_number_in_no_evidence_is_flagged_severe_and_low_confidence():
    v = verify("NVIDIA R&D was $9.999 billion [E1].")
    c = v.checks[0]
    assert "number_not_in_evidence" in c.flags and c.verdict == Verdict.INSUFFICIENT
    assert v.answer.confidence == Confidence.LOW


def test_invalid_citation_label():
    v = verify("NVIDIA R&D was $8.675 billion [E9].")
    assert "invalid_citation" in v.checks[0].flags and v.answer.confidence == Confidence.LOW


def test_uncited_claim_cannot_be_supported():
    v = verify("NVIDIA R&D was $8.675 billion.")
    c = v.checks[0]
    assert "uncited" in c.flags and c.verdict == Verdict.PARTIAL and v.answer.confidence == Confidence.LOW


def test_derived_figure_the_model_computed_is_caught():
    v = verify("NVIDIA spent $2.219 billion more than AMD [E1][E2].")
    assert "number_not_in_evidence" in v.checks[0].flags


def test_no_claims_is_insufficient_and_low():
    v = verify("Here is some filler with no figures or citations.")
    assert v.checks == [] and v.answer.verdict == Verdict.INSUFFICIENT and v.answer.confidence == Confidence.LOW


# ---- verifier (LLM judge) ---------------------------------------------------------------

def test_llm_judges_meaning_and_one_batched_call():
    judge = FakeJudge()
    v = verify("NVIDIA R&D was $8.675 billion [E1]. AMD R&D was $6.456 billion [E2].", judge)
    assert judge.calls == 1 and all(c.judged_by == "llm" for c in v.checks)
    assert v.answer.confidence == Confidence.HIGH and v.answer.verdict == Verdict.SUPPORTED


def test_llm_contradiction_makes_confidence_low():
    judge = FakeJudge({2: "contradicted"})
    v = verify("NVIDIA R&D was $8.675 billion [E1]. AMD R&D was $6.456 billion [E2].", judge)
    assert v.answer.verdict == Verdict.PARTIAL and v.answer.confidence == Confidence.LOW
    assert v.answer.claims[1].supported is False


def test_llm_is_not_asked_about_claims_whose_numbers_failed():
    judge = FakeJudge()
    verify("NVIDIA R&D was $9.999 billion [E1].", judge)
    assert judge.calls == 0


def test_judge_failure_falls_back_to_numbers_only():
    v = verify("NVIDIA R&D was $8.675 billion [E1].", FakeJudge(fail=True))
    assert v.checks[0].verdict == Verdict.SUPPORTED and "numbers_only" in v.checks[0].flags


def test_claim_without_numbers_needs_the_judge():
    v = verify("The fiscal years end on different dates [E1][E2].")
    assert "unjudged" in v.checks[0].flags and v.checks[0].verdict == Verdict.INSUFFICIENT
    v2 = verify("The fiscal years end on different dates [E1][E2].", FakeJudge())
    assert v2.checks[0].verdict == Verdict.SUPPORTED


def test_the_unverified_context_claim_from_module_9_is_now_checked():
    """AMD '$6.5 billion ... 10% increase from $5.9 billion' cited to the right chunk passes; to the wrong one it does not."""
    ok = verify("AMD R&D was $6.5 billion, a 10% increase from $5.9 billion [E3].")
    assert ok.checks[0].verdict == Verdict.SUPPORTED
    bad = verify("AMD R&D was $6.5 billion, a 10% increase from $5.9 billion [E1].")
    assert bad.checks[0].flags and bad.checks[0].verdict != Verdict.SUPPORTED


def test_answer_object_has_citations_and_trace():
    v = verify("NVIDIA R&D was $8.675 billion [E1].", FakeJudge())
    a = v.answer
    assert [c.evidence_id for c in a.citations] == ["a"] and a.trace[0]["judged_by"] == "llm"


# ---- confidence rules -------------------------------------------------------------------

def check(verdict=Verdict.SUPPORTED, flags=()):
    return ClaimCheck("c", ["E1"], ["a"], verdict, list(flags))


def test_confidence_levels():
    assert score_confidence([check()] * 10).level == Confidence.HIGH
    assert score_confidence([check()] * 7 + [check(Verdict.INSUFFICIENT)] * 3).level == Confidence.MEDIUM
    assert score_confidence([check()] * 5 + [check(Verdict.INSUFFICIENT)] * 5).level == Confidence.LOW
    assert score_confidence([check(Verdict.CONTRADICTED)] + [check()] * 9).level == Confidence.LOW
    assert score_confidence([]).level == Confidence.LOW


def test_confidence_caps_and_severe_flags():
    assert score_confidence([check(flags=["miscited"])] * 10).level == Confidence.MEDIUM
    assert score_confidence([check()] * 10, grade_score=0.4, threshold=0.6).level == Confidence.MEDIUM
    assert score_confidence([check()] * 9 + [check(flags=["uncited"])]).level == Confidence.LOW


def test_overall_verdict():
    S, C, I, P = Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT, Verdict.PARTIAL

    def mk(*vs):
        return [check(v) for v in vs]
    assert overall_verdict(mk(S, S)) == S
    assert overall_verdict(mk(S, I)) == P
    assert overall_verdict(mk(S, C)) == P
    assert overall_verdict(mk(C, I)) == C
    assert overall_verdict(mk(I)) == I
    assert overall_verdict([]) == I


# ---- cross-modal report -> answer -------------------------------------------------------

def report(checks, verdict, coverage_ok=True):
    from finsight.agents.cross_modal import CrossModalReport, PointCheck
    pcs = [PointCheck("s", f"l{i}", Decimal(1_000_000), v, m, Decimal(1_000_000), None, f"x{i}")
           for i, (v, m) in enumerate(checks)]
    return CrossModalReport("p", "title", verdict, pcs, [])


def test_report_confidence_rules():
    S, C, I = Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT
    assert report_confidence(report([(S, "xbrl")] * 4, S)).level == Confidence.HIGH
    assert report_confidence(report([(S, "text")] * 4, S)).level == Confidence.MEDIUM
    assert report_confidence(report([(C, "xbrl")] + [(S, "text")] * 9, Verdict.PARTIAL)).level == Confidence.HIGH
    assert report_confidence(report([(S, "text")] + [(I, "none")] * 4, Verdict.PARTIAL)).level == Confidence.LOW
    assert report_confidence(report([], I)).level == Confidence.LOW


def test_answer_from_report_carries_verdict_and_claims():
    r = report([(Verdict.SUPPORTED, "xbrl"), (Verdict.CONTRADICTED, "xbrl")], Verdict.PARTIAL)
    a = answer_from_report("is the chart supported?", r)
    assert a.verdict == Verdict.PARTIAL and [c.supported for c in a.claims] == [True, False]
    assert "1 points supported, 1 contradicted" in a.answer


# ---- agent integration ------------------------------------------------------------------

def test_agent_runs_the_verifier_after_answering():
    pytest.importorskip("langgraph")
    from finsight.agents.graph import FinSightAgent
    from finsight.agents.grader import EvidenceGrader, GradeModel

    class Text:
        def search(self, query, mode="hybrid", limit=8, filt=None):
            return [ev("a", E1)]

    class Grader:
        def complete_json(self, prompt, system=None, model=None, retries=1, **kw):
            return GradeModel(score=0.9)

    class Answerer:
        def complete(self, prompt, system=None, **kw):
            from finsight.llm.client import LLMResult
            return LLMResult("NVIDIA R&D was $8.675 billion [E1].")

    agent = FinSightAgent(text=Text(), grader=EvidenceGrader(Grader()), answer_llm=Answerer(),
                          verifier=ClaimVerifier(FakeJudge()))
    r = agent.run("Why did NVIDIA R&D grow?")
    assert r.verified is not None and r.verified.answer.confidence == Confidence.HIGH
    assert [t["step"] for t in r.trace][-2:] == ["answer", "verify"]
