"""The service layer: one entry point that routes a question to the right pipeline.

  text question                        -> FinSightAgent (plan, retrieve, grade, rewrite, answer, verify)
  "is this chart supported?" question  -> find the slide, read its chart with the VLM, verify every
                                          point against XBRL and the filings (cross_modal), score confidence
  uploaded chart image + ticker        -> the same chart pipeline, on the image the user supplied

Every call returns one JSON-ready dict (answer, verdict, confidence + reasons, claims, evidence, trace,
latency, LLM usage), and writes it to data/traces/<request_id>.json. A FinSightError never escapes ask():
it comes back as `error` / `error_type` in the payload with the trace id, so the UI can show it.
All collaborators are injected, so tests run with fakes; `build_service()` wires the real ones.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any

from finsight.agents.router import TICKER_ALIASES, Intent, route
from finsight.app.tracing import Tracer, step_seconds
from finsight.core.exceptions import ConfigError, FinSightError, RetrievalError
from finsight.generation.claim_verifier import answer_from_report
from finsight.generation.confidence import report_confidence
from finsight.vision.page_renderer import PageRecord

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
KNOWN_TICKERS = set(TICKER_ALIASES.values())
DEFAULT_CHART_QUESTION = "Is this chart supported by the company's filings?"


class BadRequestError(FinSightError):
    """The caller sent something unusable (unknown ticker, not an image)."""


def evidence_rows(items, limit_chars: int = 1500) -> list[dict[str, Any]]:
    rows = []
    for n, i in enumerate(items):
        s = i.source
        rows.append({
            "label": f"E{n + 1}", "evidence_id": i.evidence_id, "modality": i.modality.value,
            "ticker": s.ticker, "doc_type": s.doc_type.value, "fiscal_year": s.fiscal_year,
            "fiscal_period": s.fiscal_period, "page": s.page, "section": s.section,
            "retriever": i.retriever, "image_path": i.image_path, "content": i.content[:limit_chars],
        })
    return rows


class FinSightService:
    def __init__(
        self,
        agent,                       # FinSightAgent-like: .run(question) -> AgentResult
        tracer: Tracer,
        cross=None,                  # CrossModalVerifier-like: .verify(chart, page) -> report
        extractor=None,              # ChartExtractor-like: .extract(image_path) -> ChartData
        page_search=None,            # PageTextRetriever-like: .search(query, limit, filt); .pages
        usage_sources=(),            # objects with .stats (LLMStats), summed into per-request usage
        uploads_dir: Path = Path("data/uploads"),
    ):
        self.agent, self.tracer = agent, tracer
        self.cross, self.extractor, self.page_search = cross, extractor, page_search
        self.usage_sources = list(usage_sources)
        self.uploads_dir = Path(uploads_dir)
        self._pages = {p.page_id: p for p in getattr(page_search, "pages", [])}
        self._lock = threading.Lock()  # models and the local index are not built for concurrent use

    def warm_up(self) -> float:
        """Pay the one-off costs (embedding model load, Qdrant connection) before the first real request.
        Never raises: a failed warm-up only means the first request is slow, or reports the real error."""
        t = time.time()
        text = getattr(self.agent, "text", None)
        if text is not None:
            try:
                text.search("research and development expense", limit=1)
            except Exception as e:  # noqa: BLE001 - warm-up must not stop the API from starting
                log.warning("warm-up search failed: %s", e)
        seconds = round(time.time() - t, 2)
        log.info("warm-up finished in %.2fs", seconds)
        return seconds

    # ---- public ---------------------------------------------------------------------------

    def ask(self, question: str) -> dict[str, Any]:
        return self._run(question, lambda: self._dispatch(question))

    def ask_chart_image(self, image_bytes: bytes, ticker: str, filename: str = "chart.png",
                        question: str = DEFAULT_CHART_QUESTION) -> dict[str, Any]:
        return self._run(question, lambda: self._verify_upload(question, image_bytes, ticker, filename))

    # ---- plumbing -------------------------------------------------------------------------

    def _usage(self) -> dict[str, float]:
        total: dict[str, float] = {}
        for src in self.usage_sources:
            for k, v in src.stats.as_dict().items():
                total[k] = total.get(k, 0) + v
        return total

    def _run(self, question: str, work) -> dict[str, Any]:
        rid, started = self.tracer.new_id(), time.time()
        with self._lock:
            before = self._usage()
            try:
                payload = work()
                payload.update(error=None, error_type=None)
            except FinSightError as e:
                log.warning("request %s failed: %s", rid, e)
                payload = {"mode": None, "intent": None, "answer": "", "verdict": None, "confidence": None,
                           "confidence_reasons": [], "claims": [], "evidence": [], "chart": None, "trace": [],
                           "error": str(e), "error_type": type(e).__name__}
            after = self._usage()
        payload.update(
            request_id=rid, question=question, started_at=started,
            total_seconds=round(time.time() - started, 3),
            step_seconds=step_seconds(payload.get("trace", []), started),
            usage={k: round(after.get(k, 0) - before.get(k, 0), 3) for k in after},
        )
        self.tracer.write(rid, payload)
        return payload

    def _dispatch(self, question: str) -> dict[str, Any]:
        plan = route(question)
        if plan.intent == Intent.CROSS_MODAL:
            return self._verify_deck_slide(question, plan)
        res = self.agent.run(question)
        v = res.verified
        return {
            "mode": "agent", "intent": plan.intent.value, "answer": res.answer,
            "verdict": v.answer.verdict.value if v and v.answer.verdict else None,
            "confidence": v.answer.confidence.value if v else None,
            "confidence_reasons": list(v.confidence.reasons) if v and v.confidence else [],
            "claims": [{"text": c.text, "verdict": c.verdict.value, "flags": c.flags, "reason": c.reason,
                        "judged_by": c.judged_by} for c in v.checks] if v else [],
            "evidence": evidence_rows(res.package.items),
            "chart": None,
            "trace": res.trace,
            "grade": {"score": res.grade.score, "missing": res.grade.missing, "iterations": res.iterations},
        }

    # ---- chart pipeline -------------------------------------------------------------------

    def _require_chart_pipeline(self) -> None:
        if not (self.cross and self.extractor):
            raise ConfigError("chart verification is not configured (needs the vision model and filing index)")

    def _verify_deck_slide(self, question: str, plan) -> dict[str, Any]:
        self._require_chart_pipeline()
        if self.page_search is None:
            raise ConfigError("no investor-deck pages are indexed; run scripts/build_pages.py")
        t = time.time()
        hits = self.page_search.search(question, 1, plan.search_filter())
        if not hits:
            raise RetrievalError("no slide in the indexed decks matched the question; "
                                 "upload the chart image instead")
        page = self._pages[hits[0].evidence_id]
        steps = [{"step": "find_slide", "page_id": page.page_id, "ticker": page.ticker, "page": page.page,
                  "ts": time.time(), "seconds": round(time.time() - t, 3)}]
        return self._verify_page(question, page, plan.intent.value, steps)

    def _verify_upload(self, question: str, data: bytes, ticker: str, filename: str) -> dict[str, Any]:
        self._require_chart_pipeline()
        ticker = (ticker or "").strip().upper()
        if ticker not in KNOWN_TICKERS:
            raise BadRequestError(f"unknown ticker {ticker!r}; supported: {', '.join(sorted(KNOWN_TICKERS))}")
        suffix = Path(filename or "chart.png").suffix.lower()
        if suffix not in IMAGE_SUFFIXES or not data:
            raise BadRequestError(f"upload a {'/'.join(sorted(IMAGE_SUFFIXES))} image")
        digest = hashlib.sha256(data).hexdigest()[:12]
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        path = self.uploads_dir / f"{digest}{suffix}"
        path.write_bytes(data)
        page = PageRecord(page_id=f"upload-{digest}", doc_id=f"upload-{ticker.lower()}", ticker=ticker,
                          doc_type="presentation", fiscal_year=None, fiscal_period=None, page=1,
                          image_path=str(path), text="")
        steps = [{"step": "upload", "ticker": ticker, "bytes": len(data), "ts": time.time()}]
        return self._verify_page(question, page, "cross_modal", steps)

    def _verify_page(self, question: str, page: PageRecord, intent: str, steps: list[dict]) -> dict[str, Any]:
        chart = self.extractor.extract(page.image_path)
        steps.append({"step": "extract_chart", "has_chart": chart.has_chart, "series": len(chart.series),
                      "estimates": chart.values_are_estimates, "ts": time.time()})
        report = self.cross.verify(chart, page)
        conf = report_confidence(report)
        answer = answer_from_report(question, report)
        c = report.counts
        steps.append({"step": "verify_chart", "verdict": report.verdict.value, "coverage": round(report.coverage, 3),
                      **c, "confidence": conf.level.value, "ts": time.time()})
        return {
            "mode": "chart", "intent": intent, "answer": answer.answer,
            "verdict": report.verdict.value, "confidence": conf.level.value,
            "confidence_reasons": list(conf.reasons),
            "claims": [{"text": f"{k.series or 'value'} / {k.label}", "verdict": k.verdict.value,
                        "flags": [k.method] if k.method != "none" else [], "reason": k.note,
                        "judged_by": k.method,
                        "chart_value": float(k.chart_value) if k.chart_value is not None else None,
                        "filing_value": float(k.filing_value) if k.filing_value is not None else None}
                       for k in report.checks],
            "evidence": evidence_rows(report.evidence),
            "chart": {"title": chart.title, "markdown": chart.to_markdown(), "caveats": chart.caveats,
                      "coverage": round(report.coverage, 3),
                      "slide": {"ticker": page.ticker, "doc_id": page.doc_id, "page": page.page,
                                "image_path": page.image_path}},
            "trace": steps,
        }


def build_service(cfg: dict | None = None, settings=None, with_sql: bool = True) -> FinSightService:
    """Wire the real retrievers, models and verifiers from configs/settings.yaml and .env."""
    from qdrant_client import QdrantClient

    from finsight.agents.cross_modal import CrossModalVerifier
    from finsight.agents.grader import EvidenceGrader
    from finsight.agents.graph import FinSightAgent
    from finsight.core.caching import DiskCache
    from finsight.core.config import get_settings, load_app_config
    from finsight.generation.claim_verifier import ClaimVerifier
    from finsight.indexing.embedders import build_embedder, collection_name
    from finsight.llm.client import build_llm
    from finsight.retrieval.sql_retriever import SqlRetriever
    from finsight.retrieval.text_retriever import TextRetriever
    from finsight.retrieval.visual_retriever import PageTextRetriever
    from finsight.vision.chart_extractor import ChartExtractor, GroqVLM
    from finsight.vision.page_renderer import read_manifest

    cfg = cfg or load_app_config()
    s = settings or get_settings()
    r, ag, vcfg = cfg["retrieval"], cfg["agent"], cfg.get("vision", {})
    model = cfg["models"]["embedding"]
    text = TextRetriever(
        QdrantClient(url=s.qdrant_url, timeout=60), build_embedder(model),
        collection_name(cfg["qdrant"]["text_collection"], model),
        top_k_dense=r["top_k_dense"], top_k_sparse=r["top_k_sparse"], rrf_k=r["rrf_k"],
    )
    sql = SqlRetriever(s.postgres_dsn) if with_sql else None
    small, default = build_llm("small", cfg, s), build_llm("default", cfg, s)
    agent = FinSightAgent(
        text=text, grader=EvidenceGrader(small), sql=sql, answer_llm=default, verifier=ClaimVerifier(small),
        max_iterations=ag["max_iterations"], threshold=ag["grader_threshold"],
    )

    manifest = Path(vcfg.get("pages_dir", "data/processed/pages")) / "pages.jsonl"
    pages = read_manifest(manifest) if manifest.exists() else []
    vlm = GroqVLM(vcfg.get("vlm_model", "qwen/qwen3.8-27b"), s.groq_api_key, DiskCache(Path(s.cache_dir)),
                  base_url=vcfg.get("vlm_base_url") or None, extra_body=vcfg.get("vlm_extra_body") or None)
    data = Path(s.data_dir)
    return FinSightService(
        agent=agent, tracer=Tracer(data / "traces"),
        cross=CrossModalVerifier(sql=sql, text=text), extractor=ChartExtractor(vlm),
        page_search=PageTextRetriever(pages) if pages else None,
        usage_sources=[small, default, vlm], uploads_dir=data / "uploads",
    )
