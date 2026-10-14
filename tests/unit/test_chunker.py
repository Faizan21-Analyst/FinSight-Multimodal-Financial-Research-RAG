from finsight.core.schemas import Modality
from finsight.ingestion.chunking.chunker import ChunkConfig, chunk_document, pack_paragraphs, split_long
from finsight.ingestion.parsing.tables import approx_tokens


def test_split_long_respects_limit_and_loses_nothing():
    text = " ".join(f"Sentence {i} is here." for i in range(100))
    pieces = split_long(text, 200)
    assert all(len(p) <= 200 for p in pieces)
    assert " ".join(pieces).split() == text.split()
    word = "x" * 500                                     # no spaces at all: hard cut
    assert all(len(p) <= 200 for p in split_long(word, 200))


def test_pack_respects_max_and_carries_overlap():
    paras = [(f"Paragraph {i} " + "word " * 30, 1) for i in range(12)]   # ~40 tokens each
    chunks = pack_paragraphs(paras, max_tokens=120, overlap_tokens=50)
    assert len(chunks) > 3
    assert all(approx_tokens(t) <= 120 for t, _ in chunks)
    first_tail = chunks[0][0].split("\n\n")[-1]
    assert chunks[1][0].startswith(first_tail)           # overlap = last paragraph repeated


def test_no_overlap_when_disabled():
    paras = [(f"P{i} " + "w " * 40, 1) for i in range(6)]
    chunks = pack_paragraphs(paras, max_tokens=60, overlap_tokens=0)
    joined = "\n\n".join(t for t, _ in chunks)
    assert all(joined.count(f"P{i} ") == 1 for i in range(6))


def test_document_chunks_have_sources_sections_and_tables(tenk_html):
    chunks, _ = chunk_document(tenk_html, "acme-10-k-2024-01-28", "ACME", "10-K", 2024, "FY")
    assert all(c.source.ticker == "ACME" and c.source.fiscal_year == 2024 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    secs = {c.source.section for c in chunks}
    assert {"Item 1", "Item 1A", "Item 7"} <= secs
    tables = [c for c in chunks if c.modality == Modality.TABLE]
    assert len(tables) == 1                                # front-matter TOC table dropped
    t = tables[0]
    assert t.source.section == "Item 7" and t.source.page is not None
    assert t.metadata["caption"] == "Revenue by segment is shown below."
    assert t.text.startswith("Revenue by segment is shown below.\n| ")
    assert "$47,525" in t.text


def test_sections_never_mixed_in_a_text_chunk(tenk_html):
    chunks, _ = chunk_document(tenk_html, "d", "ACME", "10-K", cfg=ChunkConfig(max_tokens=4000))
    for c in chunks:
        if c.source.section == "Item 1A":
            assert "Revenue by segment" not in c.text


def test_chunk_ids_deterministic(tenk_html):
    a, _ = chunk_document(tenk_html, "d", "ACME", "10-K")
    b, _ = chunk_document(tenk_html, "d", "ACME", "10-K")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]