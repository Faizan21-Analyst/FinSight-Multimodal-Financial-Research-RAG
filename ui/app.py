"""FinSight UI (Streamlit). Talks to the API; start that first:

    uvicorn finsight.app.api:app --port 8000
    streamlit run ui/app.py

Two tabs: ask a question (text pipeline, or the chart pipeline when the question is about a deck chart),
or upload a chart image and check it against a company's filings.
"""
import base64
from urllib.parse import quote

import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FinSight", layout="wide")
st.title("FinSight")
st.caption("Multimodal financial research. Every number is checked against the filings before you see it.")

api = st.sidebar.text_input("API URL", "http://localhost:8000").rstrip("/")
TICKERS = ["NVDA", "AMD", "AAPL", "TSLA"]
VERDICT_ICON = {"supported": "✅", "partially_supported": "🟡", "contradicted": "❌", "insufficient_evidence": "⚪"}
CONF_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}

try:
    ready = httpx.get(f"{api}/health", timeout=3).json().get("service_ready")
    st.sidebar.success("API ready" if ready else "API is still loading models")
except httpx.HTTPError:
    st.sidebar.error("API not reachable. Start it with: uvicorn finsight.app.api:app --port 8000")


def call(path: str, body: dict) -> dict | None:
    try:
        r = httpx.post(f"{api}{path}", json=body, timeout=600)
    except httpx.HTTPError as e:
        st.error(f"Could not reach the API: {e}")
        return None
    if r.status_code == 422:
        st.error("The API rejected the input (is the question long enough?).")
        return None
    try:
        return r.json()
    except ValueError:
        st.error(f"API error {r.status_code}: {r.text[:300]}")
        return None


def slide_url(path: str) -> str:
    return f"{api}/slide?path={quote(path)}"


def render(res: dict) -> None:
    if res.get("error"):
        st.error(f"{res['error_type']}: {res['error']}")
        st.caption(f"request {res.get('request_id')} (trace saved)")
        return

    verdict, conf = res.get("verdict"), res.get("confidence")
    c1, c2, c3 = st.columns(3)
    c1.metric("Verdict", f"{VERDICT_ICON.get(verdict, '')} {verdict or 'n/a'}")
    c2.metric("Confidence", f"{CONF_ICON.get(conf, '')} {conf or 'n/a'}")
    c3.metric("Time", f"{res.get('total_seconds', 0):.1f}s")
    with st.expander("Why this confidence?", expanded=conf in ("low", None)):
        for r in res.get("confidence_reasons", []):
            st.write(f"- {r}")

    st.subheader("Answer")
    st.markdown(res.get("answer") or "_no answer was generated_")

    chart = res.get("chart")
    if chart:
        st.subheader("Chart as read from the slide")
        left, right = st.columns([1, 1])
        left.image(slide_url(chart["slide"]["image_path"]),
                   caption=f"{chart['slide']['ticker']} p.{chart['slide']['page']}")
        right.markdown(chart["markdown"])
        if chart.get("caveats"):
            right.caption("Caveats: " + "; ".join(chart["caveats"]))

    claims = res.get("claims", [])
    if claims:
        st.subheader("Claim by claim")
        df = pd.DataFrame(claims)
        df.insert(0, "", df["verdict"].map(lambda v: VERDICT_ICON.get(v, "")))
        cols = [c for c in ("", "text", "verdict", "judged_by", "reason", "chart_value", "filing_value", "flags")
                if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

    ev = res.get("evidence", [])
    if ev:
        st.subheader(f"Evidence ({len(ev)})")
        for e in ev:
            head = f"[{e['label']}] {e['ticker']} {e['doc_type']} FY{e['fiscal_year']} · {e['modality']}"
            with st.expander(head):
                st.write(e["content"])
                if e.get("image_path"):
                    st.image(slide_url(e["image_path"]))

    with st.expander("Trace and usage"):
        st.json({"request_id": res.get("request_id"), "mode": res.get("mode"), "intent": res.get("intent"),
                 "usage": res.get("usage"), "step_seconds": res.get("step_seconds"), "trace": res.get("trace")})
        st.caption(f"Saved as data/traces/{res.get('request_id')}.json")


ask_tab, chart_tab = st.tabs(["Ask a question", "Verify a chart"])

with ask_tab:
    st.write("Try: *Compare NVIDIA and AMD research and development expense in fiscal 2024*, or "
             "*Here's a chart from the AMD investor deck showing Data Center revenue. Is it supported by the 10-Q?*")
    q = st.text_area("Question", height=90, key="q")
    if st.button("Ask", type="primary", disabled=len(q.strip()) < 3):
        with st.spinner("Retrieving, answering and verifying every claim…"):
            out = call("/ask", {"question": q})
        if out:
            render(out)

with chart_tab:
    st.write("Upload a chart or table screenshot from an investor presentation. FinSight reads its numbers and "
             "checks each against the company's XBRL facts and 10-K/10-Q text.")
    up = st.file_uploader("Chart image", type=["png", "jpg", "jpeg", "webp"])
    ticker = st.selectbox("Company", TICKERS)
    if up is not None:
        st.image(up.getvalue(), width=520)
    if st.button("Verify against filings", type="primary", disabled=up is None):
        with st.spinner("Reading the chart and checking every point…"):
            out = call("/verify_chart", {"image_b64": base64.b64encode(up.getvalue()).decode(),
                                         "ticker": ticker, "filename": up.name})
        if out:
            render(out)
