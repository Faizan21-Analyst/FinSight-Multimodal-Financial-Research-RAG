"""Turn parsed blocks into Chunk objects.

* Text is packed paragraph-by-paragraph into chunks of <= max_tokens, never crossing a
  section boundary. Overlap carries whole trailing paragraphs (<= overlap_tokens) forward.
* Each data table becomes its own chunk(s) with its caption (the short paragraph just before
  it). Oversized tables are split by rows and the header row is repeated in every part.
* Chunks store clean text only. Module 4 adds the "TICKER | form | FY | section | page" prefix
  at embedding time, so the stored text stays faithful to the filing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from finsight.core.schemas import Chunk, DocType, Modality, SourceRef, make_id
from finsight.ingestion.parsing.html_parser import Block, parse_html
from finsight.ingestion.parsing.sections import FRONT, assign_sections
from finsight.ingestion.parsing.tables import approx_tokens, split_table, to_markdown

CAPTION_MAX_CHARS = 200
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64
    max_table_tokens: int = 1200
    keep_front_matter_tables: bool = False


def split_long(text: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries; hard-split overlong sentences."""
    pieces: list[str] = []
    cur = ""
    for sent in _SENTENCE_END.split(text):
        while len(sent) > max_chars:  # no sentence boundary in range: cut on whitespace
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if cur and len(cur) + 1 + len(sent) > max_chars:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p]


def pack_paragraphs(
    paras: list[tuple[str, int]], max_tokens: int, overlap_tokens: int
) -> list[tuple[str, int]]:
    """paras: (text, page). Returns (chunk_text, start_page)."""
    chunks: list[tuple[str, int]] = []
    cur: list[tuple[str, int]] = []
    cur_tok = 0

    def flush() -> None:
        if cur:
            chunks.append(("\n\n".join(t for t, _ in cur), cur[0][1]))

    for text, page in paras:
        pieces = split_long(text, max_tokens * 4) if approx_tokens(text) > max_tokens else [text]
        for piece in pieces:
            t = approx_tokens(piece)
            if cur and cur_tok + t > max_tokens:
                flush()
                carry: list[tuple[str, int]] = []
                ct = 0
                for p in reversed(cur):
                    pt = approx_tokens(p[0])
                    if ct + pt > overlap_tokens:
                        break
                    carry.insert(0, p)
                    ct += pt
                if ct + t > max_tokens:  # overlap would overflow the next chunk: skip it
                    carry, ct = [], 0
                cur, cur_tok = carry, ct
            cur.append((piece, page))
            cur_tok += t
    flush()
    return chunks


def build_chunks(
    blocks: list[Block],
    sections: list[tuple[str, str]],
    doc_id: str,
    ticker: str,
    doc_type: DocType,
    fiscal_year: int | None,
    fiscal_period: str | None,
    cfg: ChunkConfig = ChunkConfig(),
) -> list[Chunk]:
    def source(page: int, section: str) -> SourceRef:
        return SourceRef(
            doc_id=doc_id, ticker=ticker, doc_type=doc_type, fiscal_year=fiscal_year,
            fiscal_period=fiscal_period, page=page, section=section,
        )

    # group by section, preserving order
    order: list[str] = []
    titles: dict[str, str] = {}
    paras: dict[str, list[tuple[str, int]]] = {}
    tables: dict[str, list[tuple[Block, str]]] = {}
    last_text: tuple[str, str] | None = None  # (section, text) of the previous paragraph

    for b, (sec, title) in zip(blocks, sections):
        if sec not in paras:
            order.append(sec)
            titles[sec] = title
            paras[sec], tables[sec] = [], []
        if b.kind == "text":
            paras[sec].append((b.text, b.page))
            last_text = (sec, b.text)
        else:
            caption = ""
            if last_text and last_text[0] == sec and len(last_text[1]) <= CAPTION_MAX_CHARS:
                caption = last_text[1]
            tables[sec].append((b, caption))

    chunks: list[Chunk] = []
    n = 0
    for sec in order:
        meta_base = {"section_title": titles[sec]}
        for text, page in pack_paragraphs(paras[sec], cfg.max_tokens, cfg.overlap_tokens):
            chunks.append(Chunk(
                chunk_id=make_id(doc_id, "text", n), text=text, modality=Modality.TEXT,
                source=source(page, sec), metadata=dict(meta_base),
            ))
            n += 1
        if sec == FRONT and not cfg.keep_front_matter_tables:
            continue  # cover page and table of contents tables are noise
        for t_idx, (blk, caption) in enumerate(tables[sec]):
            parts = split_table(blk.rows, cfg.max_table_tokens)
            for p_idx, part_rows in enumerate(parts):
                md = to_markdown(part_rows)
                chunks.append(Chunk(
                    chunk_id=make_id(doc_id, "table", n),
                    text=f"{caption}\n{md}" if caption else md,
                    modality=Modality.TABLE,
                    source=source(blk.page, sec),
                    metadata={**meta_base, "caption": caption, "table_index": t_idx,
                              "part": p_idx + 1, "parts": len(parts), "rows": len(part_rows)},
                ))
                n += 1
    return chunks


def chunk_document(
    html: str,
    doc_id: str,
    ticker: str,
    form: str,
    fiscal_year: int | None = None,
    fiscal_period: str | None = None,
    cfg: ChunkConfig = ChunkConfig(),
) -> tuple[list[Chunk], list[tuple[str, str]]]:
    """HTML -> (chunks, per-block section labels). Labels are returned for coverage checks."""
    blocks = parse_html(html)
    sections = assign_sections(blocks, form)
    doc_type = DocType.TEN_K if form == "10-K" else DocType.TEN_Q
    chunks = build_chunks(blocks, sections, doc_id, ticker, doc_type, fiscal_year, fiscal_period, cfg)
    return chunks, sections