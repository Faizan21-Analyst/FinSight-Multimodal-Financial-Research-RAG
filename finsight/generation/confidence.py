"""Confidence: an explicit, testable rule set, not a model's self-reported certainty.

For an answer (list of ClaimChecks):
  LOW    no verifiable claims, or any contradicted claim, or any SEVERE flag
         (invalid citation, a number that appears in no evidence, an uncited claim)
  HIGH   at least 90% of claims supported, and none of the cautions below
  MEDIUM at least 60% supported; also HIGH downgraded by a caution
  LOW    otherwise
Cautions that cap HIGH at MEDIUM: a claim cited to the wrong evidence ("miscited"), a claim accepted on
its numbers alone with no meaning check ("numbers_only"/"unjudged"), or an agent grade below threshold.

For a cross-modal report (chart vs filings):
  contradicted by an exact XBRL fact          -> HIGH (an exact figure disagrees)
  supported, full coverage, >= half via XBRL  -> HIGH
  supported with text-only support            -> MEDIUM (text matching does not check the quarter)
  coverage below 60%, or nothing checkable    -> LOW
Every result carries its reasons, so the number on screen can always be explained.
"""
from __future__ import annotations

from dataclasses import dataclass

from finsight.core.schemas import Confidence, Verdict
from finsight.generation.claims import SEVERE_FLAGS, ClaimCheck

CAUTION_FLAGS = {"miscited", "numbers_only", "unjudged"}


@dataclass(frozen=True)
class ConfidenceResult:
    level: Confidence
    reasons: tuple[str, ...]


def score_confidence(
    checks: list[ClaimCheck], grade_score: float | None = None, threshold: float = 0.6
) -> ConfidenceResult:
    if not checks:
        return ConfidenceResult(Confidence.LOW, ("no verifiable claims in the answer",))
    reasons: list[str] = []
    contradicted = [c for c in checks if c.verdict == Verdict.CONTRADICTED]
    severe = sorted({f for c in checks for f in c.flags if f in SEVERE_FLAGS})
    if contradicted:
        reasons.append(f"{len(contradicted)} claim(s) contradicted by the evidence")
    if severe:
        reasons.append("citation/number problems: " + ", ".join(severe))
    if contradicted or severe:
        return ConfidenceResult(Confidence.LOW, tuple(reasons))

    supported = sum(c.verdict == Verdict.SUPPORTED for c in checks)
    frac = supported / len(checks)
    reasons.append(f"{supported}/{len(checks)} claims supported")
    level = Confidence.HIGH if frac >= 0.9 else Confidence.MEDIUM if frac >= 0.6 else Confidence.LOW

    cautions = sorted({f for c in checks for f in c.flags if f in CAUTION_FLAGS})
    if grade_score is not None and grade_score < threshold:
        cautions.append(f"evidence grade {grade_score:.2f} below {threshold}")
    if cautions and level == Confidence.HIGH:
        level = Confidence.MEDIUM
        reasons.append("capped at medium: " + ", ".join(cautions))
    elif cautions:
        reasons.append("cautions: " + ", ".join(cautions))
    return ConfidenceResult(level, tuple(reasons))


def report_confidence(report) -> ConfidenceResult:
    """Confidence for a CrossModalReport (duck-typed: .verdict, .checks, .coverage)."""
    if not report.checks:
        return ConfidenceResult(Confidence.LOW, ("no chart data to verify",))
    checked = [k for k in report.checks if k.verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED)]
    xbrl_share = (sum(k.method == "xbrl" for k in checked) / len(checked)) if checked else 0.0
    xbrl_contradiction = any(k.verdict == Verdict.CONTRADICTED and k.method == "xbrl" for k in report.checks)
    cov = report.coverage
    base = (f"coverage {cov:.0%}", f"{xbrl_share:.0%} of checked points verified against XBRL")

    if report.verdict == Verdict.INSUFFICIENT or cov < 0.6:
        return ConfidenceResult(Confidence.LOW, ("too few points could be checked", *base))
    if xbrl_contradiction and (report.verdict == Verdict.CONTRADICTED or cov >= 0.9):
        return ConfidenceResult(Confidence.HIGH, ("an exact XBRL fact disagrees with the chart", *base))
    if report.verdict == Verdict.SUPPORTED and cov == 1.0 and xbrl_share >= 0.5:
        return ConfidenceResult(Confidence.HIGH, base)
    return ConfidenceResult(
        Confidence.MEDIUM,
        ("support rests mostly on filing text, which is not matched to the exact quarter", *base),
    )
