from finsight.core.schemas import (
    Answer, Chunk, Confidence, DocType, Modality, SourceRef, make_id,
)


def _src():
    return SourceRef(doc_id="nvda-10k-2024", ticker="NVDA", doc_type=DocType.TEN_K,
                     fiscal_year=2024, page=41, section="Item 7")


def test_make_id_is_deterministic():
    assert make_id("a", 1) == make_id("a", 1)
    assert make_id("a", 1) != make_id("a", 2)


def test_chunk_defaults_to_text():
    c = Chunk(chunk_id="x", text="hello", source=_src())
    assert c.modality == Modality.TEXT
    assert c.source.ticker == "NVDA"


def test_answer_defaults_are_conservative():
    a = Answer(question="q", answer="a")
    assert a.confidence == Confidence.LOW
    assert a.claims == [] and a.verdict is None