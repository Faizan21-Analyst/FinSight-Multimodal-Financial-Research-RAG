"""Module 12 tests: tracer, service routing, chart upload, API status codes and path safety."""
import base64
import json
from decimal import Decimal

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from finsight.agents.cross_modal import CrossModalVerifier  # noqa: E402
from finsight.agents.grader import EvidenceGrader, GradeModel  # noqa: E402
from finsight.agents.graph import FinSightAgent  # noqa: E402
from finsight.app.api import create_app  # noqa: E402
from finsight.app.service import FinSightService  # noqa: E402
from finsight.app.tracing import Tracer, step_seconds  # noqa: E402
from finsight.core.exceptions import RetrievalError  # noqa: E402
from finsight.core.schemas import DocType, EvidenceItem, Modality, SourceRef  # noqa: E402
from finsight.generation.claim_verifier import ClaimVerifier  # noqa: E402
from finsight.llm.client import LLMResult, LLMStats  # noqa: E402
from finsight.vision.chart_extractor import ChartData, ChartPoint, ChartSeries  # noqa: E402
from finsight.vision.page_renderer import PageRecord  # noqa: E402

E1 = "NVDA rd_expense for FY2024: $8.675 billion (raw value 8675000000 USD; period 2023-01-30 to 2024-01-28)"


def ev(eid, content, ticker="NVDA"):
    return EvidenceItem(evidence_id=eid, modality=Modality.TEXT, content=content,
                        source=SourceRef(doc_id="d", ticker=ticker, doc_type=DocType("10-K"), fiscal_year=2024))


class Counter:
    """Stands in for a Groq client: has .stats like GroqLLM."""
    def __init__(self):
        self.stats = LLMStats()

    def bump(self):
        self.stats.calls += 1
        self.stats.completion_tokens += 10


class Text:
    def search(self, query, mode="hybrid", limit=8, filt=None):
        return [ev("a", E1)]


class Grader:
    def __init__(self, llm):
        self.llm = llm

    def complete_json(self, prompt, system=None, model=None, retries=1, **kw):
        self.llm.bump()
        return GradeModel(score=0.9)


class Answerer:
    def __init__(self, llm):
        self.llm = llm

    def complete(self, prompt, system=None, **kw):
        self.llm.bump()
        return LLMResult("NVIDIA R&D was $8.675 billion [E1].")


class Judge:
    def complete_json(self, prompt, system=None, model=None, retries=1, **kw):
        ids = [int(s.split(":")[0].split()[1]) for s in prompt.splitlines() if s.startswith("Claim ")]
        return model.model_validate({"judgements": [{"id": i, "verdict": "supported", "reason": "ok"} for i in ids]})


class FakeSql:
    def get_metric(self, ticker, metric, fiscal_year, period_type="FY", fiscal_quarter=None, **kw):
        if (metric, fiscal_year, fiscal_quarter) != ("revenue", 2026, 2):
            return None
        return EvidenceItem(evidence_id="x1", modality=Modality.SQL,
                            content="AMD revenue for Q2 FY2026: x (raw value 7685000000 USD)",
                            source=SourceRef(doc_id="d", ticker=ticker, doc_type=DocType("xbrl"), fiscal_year=2026))


class Extractor:
    def __init__(self, value=7685):
        self.value, self.paths = value, []

    def extract(self, image_path):
        self.paths.append(image_path)
        return ChartData(title="Revenue", unit_scale="million", unit_label="USD",
                         series=[ChartSeries(name="Revenue", points=[ChartPoint(label="Q2 FY26", value=self.value)])])


class Pages:
    def __init__(self, hit=True, img="slide.png"):
        self.pages = [PageRecord(page_id="p1", doc_id="amd-deck", ticker="AMD", doc_type="presentation",
                                 fiscal_year=2026, fiscal_period="Q2", page=5, image_path=img, text="revenue")]
        self.hit = hit

    def search(self, query, limit=5, filt=None):
        return [ev("p1", "slide", "AMD")] if self.hit else []


def make_service(tmp_path, chart=True, hit=True, value=7685):
    llm = Counter()
    agent = FinSightAgent(text=Text(), grader=EvidenceGrader(Grader(llm)), answer_llm=Answerer(llm),
                          verifier=ClaimVerifier(Judge()))
    kwargs = {}
    if chart:
        kwargs = dict(cross=CrossModalVerifier(sql=FakeSql()), extractor=Extractor(value), page_search=Pages(hit))
    return FinSightService(agent, Tracer(tmp_path / "traces"), usage_sources=[llm],
                           uploads_dir=tmp_path / "uploads", **kwargs)


CHART_Q = "Here is a chart from the AMD investor deck showing revenue. Is it supported by the 10-Q?"


# ---- tracer -----------------------------------------------------------------------------

def test_tracer_roundtrip_and_id_validation(tmp_path):
    t = Tracer(tmp_path)
    rid = t.new_id()
    t.write(rid, {"a": 1})
    assert t.read(rid) == {"a": 1}
    assert t.read("../../.env") is None and t.read("nope") is None and t.path_for("a/b") is None


def test_step_seconds_from_finish_times():
    trace = [{"step": "plan", "ts": 10.5}, {"step": "retrieve", "ts": 13.0}, {"step": "x"}]
    assert step_seconds(trace, 10.0) == [{"step": "plan", "seconds": 0.5}, {"step": "retrieve", "seconds": 2.5}]


# ---- service ----------------------------------------------------------------------------

def test_text_question_goes_through_the_agent_with_verification_and_usage(tmp_path):
    svc = make_service(tmp_path)
    r = svc.ask("Why did NVIDIA R&D grow?")
    assert r["mode"] == "agent" and r["error"] is None
    assert r["verdict"] == "supported" and r["confidence"] == "high"
    assert r["claims"][0]["judged_by"] == "llm" and r["evidence"][0]["label"] == "E1"
    assert r["usage"]["calls"] == 2                      # grader + answerer for this request only
    saved = json.loads((tmp_path / "traces" / f"{r['request_id']}.json").read_text())
    assert saved["request_id"] == r["request_id"] and saved["step_seconds"]


def test_usage_is_per_request_not_cumulative(tmp_path):
    svc = make_service(tmp_path)
    svc.ask("Why did NVIDIA R&D grow?")
    assert svc.ask("Why did NVIDIA R&D grow again?")["usage"]["calls"] == 2


def test_chart_question_routes_to_the_chart_pipeline(tmp_path):
    r = make_service(tmp_path).ask(CHART_Q)
    assert r["mode"] == "chart" and r["verdict"] == "supported"
    assert r["chart"]["slide"]["page"] == 5 and r["claims"][0]["judged_by"] == "xbrl"
    assert [s["step"] for s in r["trace"]] == ["find_slide", "extract_chart", "verify_chart"]


def test_wrong_chart_value_is_contradicted(tmp_path):
    r = make_service(tmp_path, value=9000).ask(CHART_Q)
    assert r["verdict"] == "contradicted" and r["confidence"] == "high"
    assert r["claims"][0]["filing_value"] == 7685000000.0


def test_chart_question_without_a_matching_slide_is_an_error_payload(tmp_path):
    r = make_service(tmp_path, hit=False).ask(CHART_Q)
    assert r["error_type"] == "RetrievalError" and r["verdict"] is None and r["request_id"]


def test_chart_pipeline_not_configured(tmp_path):
    r = make_service(tmp_path, chart=False).ask(CHART_Q)
    assert r["error_type"] == "ConfigError"


def test_uploaded_chart_is_saved_and_verified(tmp_path):
    svc = make_service(tmp_path)
    r = svc.ask_chart_image(b"\x89PNG fake", "amd", "shot.PNG")
    assert r["verdict"] == "supported" and r["chart"]["slide"]["ticker"] == "AMD"
    saved = list((tmp_path / "uploads").glob("*.png"))
    assert len(saved) == 1 and svc.extractor.paths == [str(saved[0])]


def test_upload_validation(tmp_path):
    svc = make_service(tmp_path)
    assert svc.ask_chart_image(b"x", "ZZZZ", "a.png")["error_type"] == "BadRequestError"
    assert svc.ask_chart_image(b"x", "AMD", "a.exe")["error_type"] == "BadRequestError"
    assert svc.ask_chart_image(b"", "AMD", "a.png")["error_type"] == "BadRequestError"


def test_retrieval_failure_inside_the_agent_does_not_raise(tmp_path):
    class Down:
        def search(self, *a, **k):
            raise RetrievalError("qdrant down")

    svc = make_service(tmp_path)
    svc.agent.text = Down()
    r = svc.ask("Why did NVIDIA R&D grow?")
    assert r["error"] is None and r["evidence"] == []   # the agent logs the failure; the answer is empty


# ---- API --------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "slide.png").write_bytes(b"png")
    (data / "notes.txt").write_text("secret")
    (tmp_path / "outside.png").write_bytes(b"png")
    svc = make_service(data)
    return TestClient(create_app(service=svc, data_dir=data)), svc, data


def test_health_and_ask(client):
    c, _, _ = client
    assert c.get("/health").json() == {"status": "ok", "service_ready": True}
    r = c.post("/ask", json={"question": "Why did NVIDIA R&D grow?"})
    assert r.status_code == 200 and r.json()["confidence"] == "high"


def test_ask_validation(client):
    c, _, _ = client
    assert c.post("/ask", json={"question": "hi"}).status_code == 422
    assert c.post("/ask", json={}).status_code == 422


def test_error_payloads_map_to_http_status(client):
    c, svc, _ = client
    svc.page_search = Pages(hit=False)
    r = c.post("/ask", json={"question": CHART_Q})
    assert r.status_code == 503 and r.json()["error_type"] == "RetrievalError" and r.json()["request_id"]


def test_trace_endpoint(client):
    c, _, _ = client
    rid = c.post("/ask", json={"question": "Why did NVIDIA R&D grow?"}).json()["request_id"]
    assert c.get(f"/traces/{rid}").json()["request_id"] == rid
    assert c.get("/traces/20260101-000000-abcdef").status_code == 404
    assert c.get("/traces/..%2F..%2F.env").status_code in (404, 422)


def test_verify_chart_endpoint(client):
    c, _, _ = client
    body = {"image_b64": base64.b64encode(b"\x89PNG fake").decode(), "ticker": "AMD", "filename": "c.png"}
    r = c.post("/verify_chart", json=body)
    assert r.status_code == 200 and r.json()["mode"] == "chart"
    assert c.post("/verify_chart", json={**body, "ticker": "ZZZ"}).status_code == 400
    assert c.post("/verify_chart", json={**body, "image_b64": "!!notbase64!!"}).status_code == 400


def test_slide_endpoint_only_serves_images_under_the_data_folder(client):
    c, _, data = client
    assert c.get("/slide", params={"path": str(data / "slide.png")}).status_code == 200
    assert c.get("/slide", params={"path": str(data / "notes.txt")}).status_code == 404       # not an image
    assert c.get("/slide", params={"path": str(data.parent / "outside.png")}).status_code == 404
    assert c.get("/slide", params={"path": str(data / ".." / "outside.png")}).status_code == 404


def test_service_not_ready_returns_503():
    c = TestClient(create_app(service=None))      # lifespan not started: still loading
    assert c.get("/health").json()["service_ready"] is False
    assert c.post("/ask", json={"question": "Why did NVIDIA grow?"}).status_code == 503
