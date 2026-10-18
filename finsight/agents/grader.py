"""Evidence grader: is the retrieved evidence enough to answer the question?

One LLM call (small model, JSON) returns a score in [0, 1], what is missing, and a rewritten
search query. Cheap shortcuts avoid the call when the answer is obvious:
  * no evidence at all -> score 0;
  * if the LLM fails or returns junk -> heuristic fallback (are all named companies covered?),
    so the agent loop degrades instead of crashing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from finsight.agents.router import QueryPlan
from finsight.core.exceptions import FinSightError
from finsight.generation.evidence import EvidencePackage

log = logging.getLogger(__name__)

GRADER_SYSTEM = (
    "You grade whether retrieved evidence is sufficient to answer a financial question. "
    "Be strict: every company, period and metric the question names must be supported by the evidence. "
    "Reply with JSON only."
)

GRADER_PROMPT = """Question: {question}

Evidence:
{context}

Return JSON with keys:
  "score": number 0-1 (1 = fully answerable from this evidence, 0 = nothing relevant),
  "missing": short string naming what is missing (empty if nothing),
  "rewritten_query": a better search query that would find the missing evidence (empty if score >= 0.8).
"""


class GradeModel(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    missing: str = ""
    rewritten_query: str = ""


@dataclass(frozen=True)
class GradeResult:
    score: float
    missing: str = ""
    rewritten_query: str = ""
    source: str = "llm"  # "llm" | "empty" | "heuristic"


class JsonLLM(Protocol):
    def complete_json(self, prompt: str, system: str | None = None, model: type[BaseModel] | None = None,
                      retries: int = 1, **kwargs): ...


def heuristic_grade(plan: QueryPlan, package: EvidencePackage) -> GradeResult:
    """Coverage check without an LLM: every ticker named in the question needs evidence."""
    if not len(package):
        return GradeResult(0.0, "no evidence retrieved", "", "empty")
    if not plan.tickers:
        return GradeResult(0.6, "", "", "heuristic")
    have = {i.source.ticker for i in package.items}
    missing = [t for t in plan.tickers if t not in have]
    if not missing:
        return GradeResult(0.7, "", "", "heuristic")
    covered = 1 - len(missing) / len(plan.tickers)
    return GradeResult(round(0.5 * covered, 2), f"no evidence for {', '.join(missing)}", "", "heuristic")


class EvidenceGrader:
    def __init__(self, llm: JsonLLM | None, max_context_chars: int = 6000):
        self.llm = llm
        self.max_context_chars = max_context_chars

    def grade(self, plan: QueryPlan, package: EvidencePackage) -> GradeResult:
        if not len(package):
            return GradeResult(0.0, "no evidence retrieved", "", "empty")
        if self.llm is None:
            return heuristic_grade(plan, package)
        prompt = GRADER_PROMPT.format(
            question=plan.question, context=package.render_context()[: self.max_context_chars]
        )
        try:
            g = self.llm.complete_json(prompt, GRADER_SYSTEM, model=GradeModel, max_tokens=300, temperature=0.0)
        except FinSightError as e:
            log.warning("grader LLM failed (%s); using heuristic", e)
            return heuristic_grade(plan, package)
        return GradeResult(g.score, g.missing.strip(), g.rewritten_query.strip(), "llm")
