from finsight.ingestion.parsing.html_parser import parse_html
from finsight.ingestion.parsing.sections import FRONT, assign_sections, missing_sections


def by_section(blocks, labels):
    out = {}
    for b, (k, _) in zip(blocks, labels):
        out.setdefault(k, []).append(b)
    return out


def test_toc_lines_lose_to_real_headings(tenk_html):
    blocks = parse_html(tenk_html)
    labels = assign_sections(blocks, "10-K")
    sec = by_section(blocks, labels)
    # the real Item 1 body (long text) is in Item 1, not the one-line TOC entry
    assert any("Sentence number 10" in b.text for b in sec["Item 1"])
    assert any("Risk paragraph two" in b.text for b in sec["Item 1A"])
    assert any(b.kind == "table" for b in sec["Item 7"])
    # TOC-as-text lines and cover page fall into front matter
    assert any(b.text == "ACME Corporation" for b in sec[FRONT])


def test_cross_reference_inside_paragraph_is_not_a_heading(tenk_html):
    blocks = parse_html(tenk_html)
    labels = assign_sections(blocks, "10-K")
    i = next(i for i, b in enumerate(blocks) if b.text.startswith("We face risks"))
    assert labels[i][0] == "Item 1A"


def test_title_captured(tenk_html):
    blocks = parse_html(tenk_html)
    labels = assign_sections(blocks, "10-K")
    assert ("Item 7", "Management Discussion and Analysis") in labels


def test_10q_parts_disambiguate_repeated_item_numbers(tenq_html):
    blocks = parse_html(tenq_html)
    keys = {k for k, _ in assign_sections(blocks, "10-Q")}
    assert {"Part I Item 1", "Part I Item 2", "Part II Item 1", "Part II Item 1A"} <= keys


def test_missing_sections_reported(tenk_html):
    blocks = parse_html(tenk_html)
    labels = assign_sections(blocks, "10-K")
    miss = missing_sections(labels, "10-K")
    assert "Item 7A" in miss and "Item 1" not in miss