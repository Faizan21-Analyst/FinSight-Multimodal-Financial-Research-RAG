"""Claim and citation verification for a generated answer.

For every claim in the answer:
  1. Citations   - every [E#] must exist in the evidence package; a factual claim must cite something.
  2. Numbers     - every checkable number in the claim must appear in the evidence it cites (rounded
                   figures may match within their own rounding). Numbers are never judged by an LLM.
  3. Meaning     - only if 1 and 2 pass, an LLM judge (small model, one batched JSON call) says whether the
                   cited evidence supports the claim: supported / partially_supported / contradicted /
                   insufficient_evidence. Without an LLM, a claim whose numbers all match is accepted on
                   numbers alone and flagged "numbers_only" (confidence is then capped at medium).

A number found only in a DIFFERENT evidence item is "miscited" (partial); a number found nowhere is
"number_not_in_evidence" (insufficient, severe: it may be invented or computed by the model).
Confidence comes from finsight.generation.confidence.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from finsight.core.exceptions import FinSightError
from finsight.core.schemas import Answer, Citation, Claim, Verdict
from finsight.generation.claims import (
    ClaimCheck,
    ClaimNumber,
    checkable_numbers,
    extract_claims,
    strip_labels,
    parse_labels,
)
from finsight.generation.confidence import ConfidenceResult, report_confidence, score_confidence
from finsight.generation.evidence import EvidencePackage

log = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You check whether evidence supports a claim. Use ONLY the evidence shown for that claim. "
    "supported = the evidence states it; partially_supported = part of it is stated; "
    "contradicted = the evidence states something different; insufficient_evidence = the evidence does not say. "
    "Reply with JSON only."
)


class Judgement(BaseModel):
    id: int
    verdict: Verdict
    reason: str = ""


class JudgementBatch(BaseModel):
    judgements: list[Judgement]


# Words that make a claim more than "this fact says this number": comparisons, trends, arithmetic.
_NEEDS_JUDGE = re.compile(
    r"\b(higher|lower|more|less|greater|larger|smaller|bigger|exceed\w*|increas\w*|decreas\w*|grew|grow\w*|"
    r"declin\w*|fell|rose|than|versus|vs|ratio|growth|margin|total|combined|sum|difference|gap|change[sd]?)\b",
    re.I)


def sql_settled(text: str, cited) -> bool:
    """True when a claim whose numbers already matched needs no LLM judge: it cites only XBRL facts (the
    source of truth, structured), makes no comparison or arithmetic, and names no company other than the
    one the cited facts belong to. Saves a Groq call on the common 'X's metric was $N [E1]' sentence."""
    if not cited or any(i.modality.value != "sql" for i in cited):
        return False
    if _NEEDS_JUDGE.search(strip_labels(text)):
        return False
    from finsight.agents.router import TICKER_ALIASES  # local import: agents imports generation

    tickers = {i.source.ticker for i in cited}
    low = text.lower()
    return not any(tk not in tickers and re.search(rf"\b{re.escape(alias.lower())}\b", low)
                   for alias, tk in TICKER_ALIASES.items())


def number_in_text(n: ClaimNumber, text: str) -> bool:
    """True if `text` states this quantity, allowing for rounding on either side."""
    from finsight.agents.cross_modal import candidates_in_text  # local import: agents imports generation

    return any(abs(v - n.value) <= max(n.half, h) for v, h in candidates_in_text(text))


@dataclass
class VerifiedAnswer:
    answer: Answer
    checks: list[ClaimCheck] = field(default_factory=list)
    confidence: ConfidenceResult | None = None

    def summary(self) -> str:
        a = self.answer
        lines = [f"verdict: {a.verdict.value if a.verdict else 'n/a'}   confidence: {a.confidence.value}"]
        if self.confidence:
            lines += [f"  - {r}" for r in self.confidence.reasons]
        for c in self.checks:
            flags = f" flags={c.flags}" if c.flags else ""
            lines.append(f"  [{c.verdict.value:>21}] ({c.judged_by}) {c.text[:90]}{flags}"
                         + (f"  -- {c.reason}" if c.reason else ""))
        return "\n".join(lines)


def overall_verdict(checks: list[ClaimCheck]) -> Verdict:
    sup = sum(c.verdict == Verdict.SUPPORTED for c in checks)
    con = sum(c.verdict == Verdict.CONTRADICTED for c in checks)
    if not checks or (sup == 0 and con == 0 and not any(c.verdict == Verdict.PARTIAL for c in checks)):
        return Verdict.INSUFFICIENT
    if con and not sup:
        return Verdict.CONTRADICTED
    if con:
        return Verdict.PARTIAL
    return Verdict.SUPPORTED if sup == len(checks) else Verdict.PARTIAL


class ClaimVerifier:
    def __init__(self, llm=None, batch_size: int = 6, snippet_chars: int = 700):
        """llm: GroqLLM-like with .complete_json(prompt, system, model=...). None = rules and numbers only."""
        self.llm, self.batch_size, self.snippet_chars = llm, batch_size, snippet_chars

    # -- public ---------------------------------------------------------------------------

    def verify(
        self,
        question: str,
        answer_text: str,
        package: EvidencePackage,
        grade_score: float | None = None,
        threshold: float = 0.6,
    ) -> VerifiedAnswer:
        by_label = package.by_label()
        all_text = "\n".join(i.content for i in package.items)
        checks: list[ClaimCheck] = []
        to_judge: list[int] = []

        for text in extract_claims(answer_text):
            labels = parse_labels(text)
            valid = [l for l in labels if l in by_label]
            cited = [by_label[l] for l in valid]
            check = ClaimCheck(text, labels, [i.evidence_id for i in cited], Verdict.INSUFFICIENT)
            nums = checkable_numbers(text)
            if [l for l in labels if l not in by_label]:
                check.flags.append("invalid_citation")
            if not labels:
                check.flags.append("uncited")

            if not cited:
                found_elsewhere = bool(nums) and all(number_in_text(n, all_text) for n in nums)
                check.reason = ("numbers appear in the evidence but the claim cites nothing" if found_elsewhere
                                else "no valid citation")
                if found_elsewhere and not labels:
                    check.verdict = Verdict.PARTIAL
            else:
                cited_text = "\n".join(i.content for i in cited)
                missing = [n for n in nums if not number_in_text(n, cited_text)]
                if missing:
                    where = [n for n in missing if number_in_text(n, all_text)]
                    raws = ", ".join(n.raw for n in missing)
                    if len(where) == len(missing):
                        check.flags.append("miscited")
                        check.verdict = Verdict.PARTIAL
                        check.reason = f"{raws} appear in other evidence, not in the cited item"
                    else:
                        check.flags.append("number_not_in_evidence")
                        check.reason = f"{raws} not found in any evidence (invented or computed?)"
                elif nums and sql_settled(text, cited):
                    check.verdict, check.judged_by = Verdict.SUPPORTED, "sql"
                    check.reason = "numbers match the cited XBRL fact"
                elif self.llm is not None:
                    to_judge.append(len(checks))
                elif nums:
                    check.verdict, check.judged_by = Verdict.SUPPORTED, "numbers"
                    check.flags.append("numbers_only")
                    check.reason = "all numbers found in the cited evidence; meaning not judged"
                else:
                    check.flags.append("unjudged")
                    check.reason = "no checkable numbers and no LLM judge"
            checks.append(check)

        self._judge(checks, to_judge, package)

        verdict = overall_verdict(checks)
        conf = score_confidence(checks, grade_score, threshold)
        return VerifiedAnswer(self._to_answer(question, answer_text, checks, package, verdict, conf),
                              checks, conf)

    # -- LLM meaning check ----------------------------------------------------------------

    def _judge(self, checks: list[ClaimCheck], idx: list[int], package: EvidencePackage) -> None:
        by_label = package.by_label()
        for start in range(0, len(idx), self.batch_size):
            batch = idx[start : start + self.batch_size]
            parts = []
            for i in batch:
                c = checks[i]
                ev = "\n".join(f"[{l}] {by_label[l].content[: self.snippet_chars]}" for l in c.labels if l in by_label)
                parts.append(f"Claim {i + 1}: {c.text}\nEvidence for claim {i + 1}:\n{ev}")
            prompt = ("\n\n".join(parts) + '\n\nReturn {"judgements": [{"id": <claim number>, "verdict": '
                      '"supported|partially_supported|contradicted|insufficient_evidence", '
                      '"reason": "at most 12 words"}]}')
            try:
                result = self.llm.complete_json(prompt, JUDGE_SYSTEM, model=JudgementBatch,
                                                max_tokens=700, temperature=0.0)
                got = {j.id: j for j in result.judgements}
            except FinSightError as e:
                log.warning("claim judge failed (%s); falling back to numbers only", e)
                got = {}
            for i in batch:
                c, j = checks[i], got.get(i + 1)
                if j is not None:
                    c.verdict, c.reason, c.judged_by = j.verdict, j.reason, "llm"
                else:  # judge unavailable for this claim: numbers matched, so accept on numbers alone
                    nums = checkable_numbers(c.text)
                    if nums:
                        c.verdict, c.judged_by = Verdict.SUPPORTED, "numbers"
                        c.flags.append("numbers_only")
                        c.reason = "judge unavailable; all numbers found in the cited evidence"
                    else:
                        c.flags.append("unjudged")
                        c.reason = "judge unavailable and no checkable numbers"

    # -- assemble the Answer --------------------------------------------------------------

    @staticmethod
    def _to_answer(question, text, checks, package, verdict, conf: ConfidenceResult) -> Answer:
        by_id = {i.evidence_id: i for i in package.items}
        cited_ids = list(dict.fromkeys(e for c in checks for e in c.evidence_ids))
        claims = [
            Claim(text=c.text,
                  supported=True if c.verdict == Verdict.SUPPORTED
                  else False if c.verdict == Verdict.CONTRADICTED else None,
                  evidence_ids=c.evidence_ids)
            for c in checks
        ]
        citations = [Citation(evidence_id=e, source=by_id[e].source, snippet=by_id[e].content[:160])
                     for e in cited_ids if e in by_id]
        trace = [{"claim": c.text[:120], "verdict": c.verdict.value, "flags": c.flags,
                  "judged_by": c.judged_by, "reason": c.reason} for c in checks]
        return Answer(question=question, answer=text, claims=claims, citations=citations,
                      confidence=conf.level, verdict=verdict, trace=trace)


def answer_from_report(question: str, report) -> Answer:
    """Turn a CrossModalReport into an Answer: the chart-vs-filing verdict with per-point claims."""
    conf = report_confidence(report)
    by_id = {e.evidence_id: e for e in report.evidence}
    claims, cited = [], []
    for k in report.checks:
        val = "n/a" if k.chart_value is None else f"{k.chart_value:,.0f}"
        claims.append(Claim(
            text=f"{k.series or 'value'} / {k.label}: chart shows {val}" + (f" ({k.note})" if k.note else ""),
            supported=True if k.verdict == Verdict.SUPPORTED
            else False if k.verdict == Verdict.CONTRADICTED else None,
            evidence_ids=[k.evidence_id] if k.evidence_id else []))
        if k.evidence_id and k.evidence_id in by_id and k.evidence_id not in cited:
            cited.append(k.evidence_id)
    citations = [Citation(evidence_id=e, source=by_id[e].source, snippet=by_id[e].content[:160]) for e in cited]
    c = report.counts
    text = (f"Chart '{report.chart_title or 'untitled'}': {c['supported']} points supported, "
            f"{c['contradicted']} contradicted, {c['unverified']} not verifiable against the filings.")
    return Answer(question=question, answer=text, claims=claims, citations=citations,
                  confidence=conf.level, verdict=report.verdict,
                  trace=[{"confidence_reasons": list(conf.reasons)}])
