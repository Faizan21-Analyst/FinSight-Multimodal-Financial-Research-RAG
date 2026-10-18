"""Module 6 tests: page rendering, chart extraction, visual indexing/retrieval plumbing."""
from decimal import Decimal
from pathlib import Path

import pytest

from finsight.core.exceptions import IngestionError
from finsight.core.schemas import Modality, make_id
from finsight.retrieval.text_retriever import SearchFilter
from finsight.retrieval.visual_retriever import PageTextRetriever
from finsight.vision.chart_extractor import (
    ChartExtractionError,
    ChartExtractor,
    chart_evidence,
    parse_chart_json,
)
from finsight.vision.page_renderer import (
    PageRecord,
    discover_decks,
    parse_deck_filename,
    read_manifest,
    render_pdf,
    write_manifest,
)


# ---------- file names and manifest ----------

def test_parse_deck_filename():
    m = parse_deck_filename("data/raw/decks/NVDA/NVDA_FY2025_Q4.pdf")
    assert (m.ticker, m.fiscal_year, m.fiscal_period) == ("NVDA", 2025, "Q4")
    assert m.doc_id == "nvda-fy2025-q4-deck"


def test_parse_deck_filename_optional_parts():
    m = parse_deck_filename("amd_FY2024_cfo-commentary.pdf")
    assert (m.ticker, m.fiscal_year, m.fiscal_period) == ("AMD", 2024, None)
    m2 = parse_deck_filename("AAPL_FY2025_Q1_supplement.pdf")
    assert m2.fiscal_period == "Q1"


def test_parse_deck_filename_rejects_bad_names():
    with pytest.raises(IngestionError):
        parse_deck_filename("investor presentation.pdf")


def _page(doc, ticker, n, text, fy=2025):
    return PageRecord(
        page_id=make_id("page", doc, n), doc_id=doc, ticker=ticker, doc_type="presentation",
        fiscal_year=fy, fiscal_period="Q4", page=n, image_path=f"{doc}/p{n:03d}.png", text=text,
    )


PAGES = [
    _page("nvda-deck", "NVDA", 1, "Data Center revenue record quarter billion growth"),
    _page("nvda-deck", "NVDA", 2, "Gaming revenue and automotive revenue segment"),
    _page("amd-deck", "AMD", 1, "Data Center segment revenue EPYC Instinct"),
]


def test_page_id_is_hex_and_stable():
    p = PAGES[0]
    assert int(p.page_id, 16) >= 0
    assert p.page_id == make_id("page", "nvda-deck", 1)


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "m" / "pages.jsonl"
    assert write_manifest(PAGES, path) == 3
    assert read_manifest(path) == PAGES
    assert [p.ticker for p in read_manifest(path, ["amd"])] == ["AMD"]


def test_manifest_missing_file(tmp_path):
    with pytest.raises(IngestionError):
        read_manifest(tmp_path / "nope.jsonl")


# ---------- rendering ----------

def test_render_pdf_end_to_end(tmp_path):
    fitz = pytest.importorskip("fitz")
    decks = tmp_path / "decks"
    decks.mkdir()
    pdf = decks / "NVDA_FY2025_Q4.pdf"
    doc = fitz.open()
    for text in ("Data Center revenue", "Gaming revenue"):
        page = doc.new_page(width=720, height=405)
        page.insert_text((50, 100), text, fontsize=24)
    doc.save(str(pdf))
    doc.close()
    (decks / "notes.pdf").write_bytes(b"%PDF-1.4")  # bad name: must be skipped, not crash

    found = discover_decks(decks)
    assert [d.ticker for d in found] == ["NVDA"]

    out = tmp_path / "pages"
    recs = render_pdf(found[0], out, dpi=50)
    assert [r.page for r in recs] == [1, 2]
    assert all(Path(r.image_path).exists() for r in recs)
    assert "Data Center" in recs[0].text
    first_img = Path(recs[0].image_path)
    mtime = first_img.stat().st_mtime_ns

    again = render_pdf(found[0], out, dpi=50)  # idempotent: no redraw
    assert first_img.stat().st_mtime_ns == mtime
    assert [r.page_id for r in again] == [r.page_id for r in recs]

    assert len(render_pdf(found[0], out, dpi=50, max_pages=1)) == 1


# ---------- OCR/text baseline retriever ----------

def test_page_text_retriever_ranks_and_filters():
    r = PageTextRetriever(PAGES)
    hits = r.search("data center revenue", limit=3)
    assert hits[0].source.page == 1 and hits[0].modality == Modality.PAGE_IMAGE
    assert (hits[-1].source.doc_id, hits[-1].source.page) == ("nvda-deck", 2)  # only matches 'revenue'
    only_amd = r.search("data center revenue", limit=3, filt=SearchFilter(tickers=("AMD",)))
    assert [h.source.ticker for h in only_amd] == ["AMD"]
    assert r.search("zzzz nonsense", limit=3) == []


# ---------- visual index + retriever plumbing (in-memory Qdrant, hashing embedder) ----------

def test_visual_index_and_search_in_memory():
    qdrant = pytest.importorskip("qdrant_client")
    from finsight.indexing.visual_embedders import HashingVisualEmbedder
    from finsight.indexing.visual_index import VisualIndexer
    from finsight.retrieval.visual_retriever import VisualRetriever

    client = qdrant.QdrantClient(":memory:")
    emb = HashingVisualEmbedder(64)
    idx = VisualIndexer(client, emb, "pages_test", batch_size=2)
    idx.ensure_collection()
    stats = idx.index(PAGES)
    assert (stats.indexed, stats.skipped) == (3, 0)
    assert idx.index(PAGES).skipped == 3  # idempotent

    ret = VisualRetriever(client, emb, "pages_test")
    hits = ret.search("data center revenue", limit=3, filt=SearchFilter(tickers=("NVDA",)))
    assert hits and hits[0].source.ticker == "NVDA" and hits[0].source.page == 1
    assert all(h.source.ticker == "NVDA" for h in hits)
    assert hits[0].image_path == "nvda-deck/p001.png"


# ---------- chart extraction ----------

GOOD = """```json
{"has_chart": true, "chart_type": "bar", "title": "Data Center Revenue",
 "x_axis": "Quarter", "y_axis": "Revenue", "unit_scale": "billion", "unit_label": "USD",
 "period": "Q1-Q2 FY25", "values_are_estimates": false,
 "series": [{"name": "Data Center", "points": [
    {"label": "Q1 FY25", "value": 22.6}, {"label": "Q2 FY25", "value": 26.3}, {"label": "Q3 FY25", "value": null}]}],
 "caveats": []}
```"""


class FakeVLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, image_path, prompt, system):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_parse_chart_json_and_units():
    chart = parse_chart_json(GOOD)
    assert chart.title == "Data Center Revenue" and chart.unit_scale == "billion"
    vals = chart.absolute_values()
    assert vals[0] == ("Data Center", "Q1 FY25", Decimal("22600000000.0"))
    assert len(vals) == 2  # the null point is dropped, not guessed
    md = chart.to_markdown()
    assert "| Q2 FY25 | 26.3 |" in md and "n/a" in md


def test_percent_charts_are_not_scaled():
    chart = parse_chart_json(
        '{"unit_scale": "billion", "unit_label": "%", '
        '"series": [{"name": "GM", "points": [{"label": "Q1", "value": 74.6}]}]}'
    )
    assert chart.absolute_values()[0][2] == Decimal("74.6")


def test_parse_rejects_garbage():
    for bad in ("no json here", "{not json}", '{"unit_scale": "gazillion"}',
                '{"series": [{"points": [{"label": "Q1", "value": "$26B"}]}]}'):
        with pytest.raises(ChartExtractionError):
            parse_chart_json(bad)


def test_extractor_retries_once_with_the_error():
    vlm = FakeVLM("sorry, I can't", GOOD)
    chart = ChartExtractor(vlm).extract("p.png")
    assert chart.series and len(vlm.prompts) == 2
    assert "previous reply was rejected" in vlm.prompts[1]


def test_extractor_gives_up_after_retries():
    with pytest.raises(ChartExtractionError):
        ChartExtractor(FakeVLM("x", "y")).extract("p.png")


def test_no_chart_slide():
    chart = ChartExtractor(FakeVLM('{"has_chart": false, "series": []}')).extract("p.png")
    assert chart.has_chart is False and chart.to_markdown() == "(no chart data)"


def test_chart_evidence():
    ev = chart_evidence(parse_chart_json(GOOD), PAGES[0])
    assert ev.modality == Modality.CHART and ev.retriever == "vlm"
    assert ev.source.page == 1 and ev.image_path == PAGES[0].image_path
    assert "Data Center Revenue" in ev.content and "billion USD" in ev.content
    assert ev.evidence_id == make_id("chart", PAGES[0].page_id)


# ---------- Groq VLM client (no network: httpx.MockTransport) ----------

def _img(tmp_path):
    p = tmp_path / "p001.png"
    p.write_bytes(b"\x89PNG fake image bytes")
    return str(p)


def _ok(content=GOOD):
    import httpx

    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_groq_request_shape_and_disk_cache(tmp_path):
    import json

    import httpx

    from finsight.core.caching import DiskCache
    from finsight.vision.chart_extractor import GroqVLM

    calls = []

    def handler(req):
        calls.append(req)
        return _ok()

    vlm = GroqVLM("vision-model", "secret", cache=DiskCache(tmp_path),
                  client=httpx.Client(transport=httpx.MockTransport(handler)))
    img = _img(tmp_path)
    assert vlm.complete(img, "read it", "sys") == GOOD
    req = calls[0]
    assert req.headers["authorization"] == "Bearer secret"
    body = json.loads(req.content)
    assert body["model"] == "vision-model" and body["temperature"] == 0
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    parts = body["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "read it"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    assert vlm.complete(img, "read it", "sys") == GOOD  # served from the cache
    assert len(calls) == 1


def test_groq_retries_429_then_succeeds(tmp_path):
    import httpx

    from finsight.vision.chart_extractor import GroqVLM

    replies = [httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(503), _ok()]
    slept = []
    vlm = GroqVLM("m", "k", sleep=slept.append,
                  client=httpx.Client(transport=httpx.MockTransport(lambda r: replies.pop(0))))
    assert vlm.complete(_img(tmp_path), "p", "s") == GOOD
    assert slept[0] == 3.0 and len(slept) == 2  # honoured Retry-After, then backed off


def test_groq_gives_up_and_reports_errors(tmp_path):
    import httpx

    from finsight.core.exceptions import ConfigError
    from finsight.vision.chart_extractor import GroqVLM

    def make(status, **kw):
        return GroqVLM("m", "k", sleep=lambda s: None, **kw,
                       client=httpx.Client(transport=httpx.MockTransport(
                           lambda r: httpx.Response(status, text="nope"))))

    with pytest.raises(ChartExtractionError, match="500"):
        make(500, retries=2).complete(_img(tmp_path), "p", "s")
    with pytest.raises(ConfigError, match="401"):
        make(401).complete(_img(tmp_path), "p", "s")
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        GroqVLM("m", "").complete(_img(tmp_path), "p", "s")


def test_groq_downscales_oversized_images(tmp_path):
    import json

    import httpx

    PIL = pytest.importorskip("PIL.Image")
    from finsight.vision.chart_extractor import GroqVLM

    path = tmp_path / "big.png"
    PIL.new("RGB", (400, 300), "white").save(path)
    seen = []
    vlm = GroqVLM("m", "k", client=httpx.Client(transport=httpx.MockTransport(
        lambda r: (seen.append(json.loads(r.content)), _ok())[1])))
    vlm.MAX_B64_BYTES = 100  # pretend the PNG is over the limit (instance only: the JPEG fits the real one)
    vlm.complete(str(path), "p", "s")
    assert seen[0]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_custom_base_url_needs_no_api_key(tmp_path):
    import httpx

    from finsight.vision.chart_extractor import GroqVLM

    urls, auth = [], []

    def handler(req):
        urls.append(str(req.url))
        auth.append("authorization" in req.headers)
        return _ok()

    vlm = GroqVLM("qwen2.5vl:7b", "", base_url="http://localhost:11434/v1/",
                  client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert vlm.complete(_img(tmp_path), "p", "s") == GOOD
    assert urls == ["http://localhost:11434/v1/chat/completions"] and auth == [False]


def test_extra_body_is_merged_and_part_of_the_cache_key(tmp_path):
    import json

    import httpx

    from finsight.core.caching import DiskCache
    from finsight.vision.chart_extractor import GroqVLM

    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return _ok()

    def make(extra):
        return GroqVLM("qwen/qwen3.8-27b", "k", cache=DiskCache(tmp_path), extra_body=extra,
                       client=httpx.Client(transport=httpx.MockTransport(handler)))

    img = _img(tmp_path)
    extra = {"reasoning_effort": "none", "response_format": {"type": "json_object"}, "temperature": 0.2}
    make(extra).complete(img, "p", "s")
    assert bodies[0]["reasoning_effort"] == "none"
    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert bodies[0]["temperature"] == 0.2  # overrides the default 0
    make(extra).complete(img, "p", "s")      # same params: cached
    assert len(bodies) == 1
    make({"reasoning_effort": "high"}).complete(img, "p", "s")  # different params: new call
    assert len(bodies) == 2
