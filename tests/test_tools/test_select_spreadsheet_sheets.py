"""Sheet selection in read_spreadsheet.

Regression coverage for #1569: ``_select_spreadsheet_sheets`` tested the
positional-index reading of the argument before looking for a sheet with that
exact name, so a workbook holding a sheet literally named "1" could never be
reached -- ``sheet="1"`` always resolved to the first sheet instead.
"""

from __future__ import annotations

import pytest

from agentos.tools.builtin.filesystem import _select_spreadsheet_sheets
from agentos.tools.types import ToolError

# (name, rows, total_rows); rows is the 1-indexed sparse map that
# _read_xlsx_sheets / _read_delimited_rows produce.
SHEETS: list[tuple[str, dict[int, list[str]], int]] = [
    ("Summary", {1: ["a"]}, 1),
    ("1", {1: ["b"]}, 1),
    ("0", {1: ["c"]}, 1),
]


def test_exact_numeric_sheet_name_wins_over_positional_index() -> None:
    assert _select_spreadsheet_sheets(SHEETS, "1") == [("1", {1: ["b"]}, 1)]


def test_sheet_named_zero_is_reachable() -> None:
    # "0" is never a valid 1-based index, so it could only ever have fallen
    # through to the name match -- but it must resolve to the sheet, not to
    # the "Sheet not found" error.
    assert _select_spreadsheet_sheets(SHEETS, "0") == [("0", {1: ["c"]}, 1)]


def test_positional_index_still_used_when_no_exact_name_matches() -> None:
    assert _select_spreadsheet_sheets(SHEETS, "2") == [("1", {1: ["b"]}, 1)]


def test_out_of_range_numeric_name_still_matches_by_name() -> None:
    sheets: list[tuple[str, dict[int, list[str]], int]] = [("2024", {1: ["x"]}, 1)]

    assert _select_spreadsheet_sheets(sheets, "2024") == [("2024", {1: ["x"]}, 1)]


def test_int_request_resolves_the_same_way_as_the_string_form() -> None:
    # The tool declares sheet as a string, but the helper accepts int too;
    # the same argument must not mean two different sheets depending on how
    # the caller typed it.
    assert _select_spreadsheet_sheets(SHEETS, 1) == _select_spreadsheet_sheets(SHEETS, "1")


def test_int_request_falls_back_to_position_when_no_sheet_has_that_name() -> None:
    assert _select_spreadsheet_sheets(SHEETS, 3) == [("0", {1: ["c"]}, 1)]


def test_selection_preserves_the_total_rows_element() -> None:
    # The selected entry has to keep its (name, rows, total_rows) shape --
    # _format_spreadsheet unpacks all three downstream.
    sheets: list[tuple[str, dict[int, list[str]], int]] = [("Data", {1: ["x"], 9: ["y"]}, 9)]

    (selected,) = _select_spreadsheet_sheets(sheets, "Data")

    assert selected == ("Data", {1: ["x"], 9: ["y"]}, 9)


def test_none_and_empty_string_return_all_sheets() -> None:
    assert _select_spreadsheet_sheets(SHEETS, None) == SHEETS
    assert _select_spreadsheet_sheets(SHEETS, "") == SHEETS


def test_exact_name_match_still_wins_for_non_numeric_names() -> None:
    assert _select_spreadsheet_sheets(SHEETS, "Summary") == [("Summary", {1: ["a"]}, 1)]


def test_case_insensitive_fallback_still_applies() -> None:
    assert _select_spreadsheet_sheets(SHEETS, "summary") == [("Summary", {1: ["a"]}, 1)]


def test_unmatched_sheet_name_raises_tool_error_listing_available_sheets() -> None:
    with pytest.raises(ToolError, match="Sheet not found: missing"):
        _select_spreadsheet_sheets(SHEETS, "missing")
