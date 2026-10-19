"""Corrective retrieval loop (LangGraph).

    plan -> retrieve -> grade --(good enough, out of iterations, or stalled)--> answer -> END
                          ^            |
                          +-- rewrite <-+ (weak)

* plan     : route() + planner (per-company sub-queries, exact SQL lookups)
* retrieve : run every sub-query and SQL lookup, accumulate results across rounds, assemble evidence
* grade    : EvidenceGrader scores sufficiency (threshold from settings: agent.grader_threshold)
* rewrite  : use the grader's rewritten query; if almost nothing came back, also drop the fiscal-year
             filter (a wrong year inference is the usual cause of an empty result)
* stall    : if a retry adds no new evidence, stop: another round would repeat itself
* answer   : optional; one LLM call over the labelled evidence. Skipped when no answer LLM is given.
Every step appends a record to `trace`, returned with the result, so a run can be audited.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from finsight.agents.grader import EvidenceGrader, GradeResult
from finsight.agents.planner import SubQuery, build_subqueries, sql_requests
from finsight.agents.router import QueryPlan, route
from finsight.core.exceptions import FinSightError
from finsight.core.schemas import EvidenceItem
from finsight.generation.evidence import EvidencePackage, assemble

log = logging.getLogger(__name__)

ANSWER_SYSTEM = (
    "You answer financial questions using ONLY the labelled evidence. Cite labels like [E1] after each "
    "claim. Never do arithmetic yourself: quote figures exactly as they appear. If the evidence does not "
    "support an answer, say what is missing."
)

VisualSearch = Callable[[str, Any], list[EvidenceItem]]  # (query, SearchFilter | None) -> evidence


class AgentState(TypedDict, total=False):
    question: str
    plan: QueryPlan
    query_text: str | None
    relax_years: bool
    results: dict[str, list[EvidenceItem]]
    package: EvidencePackage
    grade: GradeResult
    iteration: int
    new_items: int
    verified: Any
    answer: str
    trace: list[dict]


@dataclass
class AgentResult:
    plan: QueryPlan
    package: EvidencePackage
    grade: GradeResult
    iterations: int
    trace: list[dict] = field(default_factory=list)
    answer: str = ""
    verified: Any = None   # VerifiedAnswer when a verifier is configured
    seconds: float = 0.0


def _merge(old: list[EvidenceItem], new: list[EvidenceItem]) -> list[EvidenceItem]:
    seen = {i.evidence_id for i in old}
    return old + [i for i in new if i.evidence_id not in seen]


class FinSightAgent:
    def __init__(
        self,
        text,                      # TextRetriever-like: .search(query, mode, limit, filt)
        grader: EvidenceGrader,
        sql=None,                  # SqlRetriever-like: .get_metric(ticker, metric, fiscal_year)
        visual: VisualSearch | None = None,
        answer_llm=None,           # GroqLLM-like: .complete(prompt, system)
        verifier=None,             # ClaimVerifier-like: .verify(question, answer, package, grade_score, threshold)
        max_iterations: int = 3,
        threshold: float = 0.6,
        per_query_limit: int = 8,
        max_items: int = 8,
        max_chars: int = 12_000,
    ):
        self.text, self.sql, self.visual = text, sql, visual
        self.grader, self.answer_llm, self.verifier = grader, answer_llm, verifier
        self.max_iterations, self.threshold = max_iterations, threshold
        self.per_query_limit, self.max_items, self.max_chars = per_query_limit, max_items, max_chars
        self.graph = self._build()

    # ---- nodes ------------------------------------------------------------------------

    @staticmethod
    def _log(state: AgentState, **rec: Any) -> list[dict]:
        trace = list(state.get("trace", []))
        trace.append(rec)
        log.info("agent step: %s", rec)
        return trace

    def _plan(self, state: AgentState) -> AgentState:
        plan = route(state["question"])
        trace = self._log(
            state, step="plan", intent=plan.intent.value, tickers=plan.tickers, years=plan.fiscal_years,
            metrics=plan.metrics, retrievers=plan.retrievers,
            subqueries=len(build_subqueries(plan)), sql_lookups=len(sql_requests(plan)),
            sql_enabled=self.sql is not None,
        )
        return {"plan": plan, "iteration": 0, "relax_years": False, "query_text": None,
                "results": {}, "trace": trace}

    def _retrieve(self, state: AgentState) -> AgentState:
        plan, results = state["plan"], {k: list(v) for k, v in state.get("results", {}).items()}
        relax = state.get("relax_years", False)
        subs: list[SubQuery] = build_subqueries(plan, state.get("query_text"))
        errors: list[str] = []
        counts: dict[str, int] = {}

        if "text" in plan.retrievers or not plan.retrievers:
            for n, sq in enumerate(subs):
                try:
                    got = self.text.search(sq.text, mode="hybrid", limit=self.per_query_limit,
                                           filt=sq.search_filter(relax))
                except FinSightError as e:
                    errors.append(f"text[{n}]: {e}")
                    continue
                key = f"text:{n}"
                results[key] = _merge(results.get(key, []), got)
                counts[key] = len(got)

        if "visual" in plan.retrievers and self.visual is not None:
            for n, sq in enumerate(subs):
                try:
                    got = self.visual(sq.text, sq.search_filter(relax))
                except FinSightError as e:
                    errors.append(f"visual[{n}]: {e}")
                    continue
                key = f"visual:{n}"
                results[key] = _merge(results.get(key, []), got)
                counts[key] = len(got)

        if self.sql is not None and state.get("iteration", 0) == 0:  # exact lookups never change on retry
            found = []
            for r in sql_requests(plan):
                try:
                    item = self.sql.get_metric(r.ticker, r.metric, r.fiscal_year)
                except FinSightError as e:
                    errors.append(f"sql {r.ticker}/{r.metric}/{r.fiscal_year}: {e}")
                    continue
                if item is not None:
                    found.append(item)
                else:
                    errors.append(f"sql {r.ticker}/{r.metric}/{r.fiscal_year}: no row")
            results["sql"] = _merge(results.get("sql", []), found)
            counts["sql"] = len(found)

        package = assemble(results, max_items=self.max_items, max_chars=self.max_chars)
        iteration = state.get("iteration", 0) + 1
        before = {i.evidence_id for i in state["package"].items} if "package" in state else set()
        new_items = len({i.evidence_id for i in package.items} - before)
        trace = self._log(state, step="retrieve", iteration=iteration, counts=counts,
                          package_items=len(package), new_items=new_items, relaxed_years=relax,
                          errors=errors)
        return {"results": results, "package": package, "iteration": iteration,
                "new_items": new_items, "trace": trace}

    def _grade(self, state: AgentState) -> AgentState:
        g = self.grader.grade(state["plan"], state["package"])
        trace = self._log(state, step="grade", iteration=state["iteration"], score=g.score,
                          source=g.source, missing=g.missing, rewritten_query=g.rewritten_query)
        return {"grade": g, "trace": trace}

    def _decide(self, state: AgentState) -> str:
        if state["grade"].score >= self.threshold or state["iteration"] >= self.max_iterations:
            return "answer"
        if state["iteration"] >= 2 and state.get("new_items", 1) == 0:
            return "answer"  # the retry found nothing new: more rounds would repeat themselves
        return "rewrite"

    def _rewrite(self, state: AgentState) -> AgentState:
        g, plan = state["grade"], state["plan"]
        text = g.rewritten_query or (f"{plan.question} {g.missing}".strip() if g.missing else plan.question)
        relax = state.get("relax_years", False) or g.score < 0.2 or len(state["package"]) == 0
        trace = self._log(state, step="rewrite", iteration=state["iteration"], query=text,
                          relax_years=relax)
        return {"query_text": text, "relax_years": relax, "trace": trace}

    def _answer(self, state: AgentState) -> AgentState:
        pkg, text = state["package"], ""
        if self.answer_llm is not None and len(pkg):
            prompt = f"Question: {state['question']}\n\nEvidence:\n{pkg.render_context()}"
            try:
                text = self.answer_llm.complete(prompt, ANSWER_SYSTEM, max_tokens=700).text
            except FinSightError as e:
                text = ""
                log.warning("answer LLM failed: %s", e)
        trace = self._log(state, step="answer", answered=bool(text), final_score=state["grade"].score,
                          iterations=state["iteration"])
        return {"answer": text, "trace": trace}

    def _verify(self, state: AgentState) -> AgentState:
        v = None
        if self.verifier is not None and state.get("answer"):
            v = self.verifier.verify(state["question"], state["answer"], state["package"],
                                     grade_score=state["grade"].score, threshold=self.threshold)
        trace = self._log(state, step="verify", verified=v is not None,
                          verdict=v.answer.verdict.value if v and v.answer.verdict else None,
                          confidence=v.answer.confidence.value if v else None,
                          claims=len(v.checks) if v else 0)
        return {"verified": v, "trace": trace}

    # ---- graph ------------------------------------------------------------------------

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("rewrite", self._rewrite)
        g.add_node("answer", self._answer)
        g.set_entry_point("plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._decide, {"answer": "answer", "rewrite": "rewrite"})
        g.add_edge("rewrite", "retrieve")
        g.add_node("verify", self._verify)
        g.add_edge("answer", "verify")
        g.add_edge("verify", END)
        return g.compile()

    def run(self, question: str) -> AgentResult:
        t0 = time.time()
        s = self.graph.invoke({"question": question, "trace": []})
        return AgentResult(
            plan=s["plan"], package=s["package"], grade=s["grade"], iterations=s["iteration"],
            trace=s["trace"], answer=s.get("answer", ""), seconds=time.time() - t0,
            verified=s.get("verified"),
        )
