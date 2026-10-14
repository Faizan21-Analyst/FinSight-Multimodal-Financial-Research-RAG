"""Module 3 pipeline: manifest -> parse -> sections -> chunks -> data/processed/chunks/*.jsonl"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from finsight.core.logging import get_logger
from finsight.indexing.xbrl_parse import fiscal_position, infer_fye_month
from finsight.ingestion.chunking.chunker import ChunkConfig, chunk_document
from finsight.ingestion.parsing.sections import FRONT, missing_sections

log = get_logger(__name__)


@dataclass
class DocSummary:
    doc_id: str
    chunks: int
    text_chunks: int
    table_chunks: int
    sections: int
    missing: list[str]
    skipped: bool = False


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fye_months(records: list[dict]) -> dict[str, int]:
    """Per-ticker fiscal-year-end month, inferred from that company's 10-K period-end dates."""
    ends: dict[str, list[date]] = defaultdict(list)
    for r in records:
        if r["form"] == "10-K":
            ends[r["ticker"]].append(date.fromisoformat(r["report_date"]))
    return {t: infer_fye_month(v) for t, v in ends.items()}


def fiscal_labels(record: dict, fye: int) -> tuple[int, str]:
    fy, q = fiscal_position(date.fromisoformat(record["report_date"]), fye)
    return fy, ("FY" if record["form"] == "10-K" else f"Q{q}")


def run(
    manifest_path: Path,
    out_dir: Path,
    cfg: ChunkConfig = ChunkConfig(),
    tickers: list[str] | None = None,
    force: bool = False,
) -> list[DocSummary]:
    records = read_manifest(manifest_path)
    fye = fye_months(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[DocSummary] = []

    for r in records:
        if tickers and r["ticker"] not in {t.upper() for t in tickers}:
            continue
        if not r.get("path") or not Path(r["path"]).exists():
            log.warning("%s: file missing, skipped", r["doc_id"])
            continue
        out = out_dir / f"{r['doc_id']}.jsonl"
        if out.exists() and out.stat().st_size > 0 and not force:
            log.info("skip   %s (already chunked)", r["doc_id"])
            continue

        fy, fp = fiscal_labels(r, fye[r["ticker"]]) if r["ticker"] in fye else (None, None)
        html = Path(r["path"]).read_text(encoding="utf-8", errors="replace")
        chunks, labels = chunk_document(html, r["doc_id"], r["ticker"], r["form"], fy, fp, cfg)

        out.write_text(
            "\n".join(c.model_dump_json() for c in chunks) + ("\n" if chunks else ""),
            encoding="utf-8",
        )
        kinds = Counter(c.modality.value for c in chunks)
        found = {k for k, _ in labels if k != FRONT}
        miss = missing_sections(labels, r["form"])
        s = DocSummary(r["doc_id"], len(chunks), kinds["text"], kinds["table"], len(found), miss)
        summaries.append(s)
        level = log.warning if (miss or not chunks) else log.info
        level("%s: %d chunks (%d text, %d table), %d sections%s", s.doc_id, s.chunks,
              s.text_chunks, s.table_chunks, s.sections,
              f", MISSING {miss}" if miss else "")
    return summaries