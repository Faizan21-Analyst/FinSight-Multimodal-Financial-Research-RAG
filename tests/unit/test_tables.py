from finsight.ingestion.parsing.tables import (
    approx_tokens, clean_row, clean_table, is_data_table, split_table, to_markdown,
)


def test_clean_row_merges_dollar_and_parens():
    assert clean_row(["Revenue", "$", "60,922", "", "(1,234", ")"]) == ["Revenue", "$60,922", "(1,234)"]
    assert clean_row(["Margin", "74.0", "%"]) == ["Margin", "74.0%"]
    assert clean_row(["", "  ", "\xa0"]) == []


def test_data_table_detection():
    assert is_data_table([["a", "b"], ["c", "d"]])
    assert not is_data_table([["only one cell"]])
    assert not is_data_table([["a"], ["b"], ["c"]])          # single column = layout
    assert not is_data_table([["a", "b"]])                    # single row = layout


def test_markdown_and_pipe_escape():
    md = to_markdown([["Item", "FY24"], ["R|D", "$8,675"]])
    assert md.splitlines()[1] == "| --- | --- |"
    assert "R\\|D" in md


def test_split_table_repeats_header():
    rows = [["Metric", "Value"]] + [[f"row {i} " + "x" * 80, str(i)] for i in range(40)]
    parts = split_table(rows, max_tokens=200)
    assert len(parts) > 1
    assert all(p[0] == ["Metric", "Value"] for p in parts)
    assert sum(len(p) - 1 for p in parts) == 40                # no row lost or duplicated
    assert all(approx_tokens(to_markdown(p)) <= 260 for p in parts)


def test_small_table_not_split():
    rows = [["a", "b"], ["c", "d"], ["e", "f"]]
    assert split_table(rows, 500) == [rows]


def test_markdown_pads_header_row_to_align_columns():
    rows = [["Jan 28, 2024", "Jan 29, 2023"], ["Data Center", "$47,525", "$15,005"]]
    lines = to_markdown(rows).splitlines()
    assert lines[0] == "|  | Jan 28, 2024 | Jan 29, 2023 |"          # blank label column restored
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| Data Center | $47,525 | $15,005 |"
    # a label row with a missing value is right-padded instead
    assert to_markdown([["h1", "h2", "h3"], ["Label", "5"]]).splitlines()[2] == "| Label | 5 |  |"

def test_numeric_cells_are_tidied_but_text_is_not():
    assert clean_row(["( 45 )"]) == ["(45)"]
    assert clean_row(["$ 1,234"]) == ["$1,234"]
    assert clean_row(["74.0 %"]) == ["74.0%"]
    assert clean_row(["$ ( 1,234 )"]) == ["$(1,234)"]
    assert clean_row(["Net income (loss) attributable"]) == ["Net income (loss) attributable"]
    assert clean_row(["Fiscal 2024 ( restated )"]) == ["Fiscal 2024 ( restated )"]