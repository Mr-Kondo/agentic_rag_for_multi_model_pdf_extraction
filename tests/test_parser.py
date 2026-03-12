"""Tests for parser heuristics that separate tables from figures."""

from __future__ import annotations

from src.core.parser import PDFParser


def _word(text: str, x0: float, top: float, x1: float, bottom: float) -> dict[str, float | str]:
    return {
        "text": text,
        "x0": x0,
        "top": top,
        "x1": x1,
        "bottom": bottom,
    }


def test_table_candidate_rejected_when_overlapping_figure() -> None:
    parser = PDFParser()
    rows = [["Year", "Value"], ["2020", "10"], ["2021", "12"]]

    is_table = parser._is_probable_table(
        rows=rows,
        bbox=(10.0, 20.0, 180.0, 140.0),
        page_words=[],
        figure_bboxes=[(0.0, 0.0, 200.0, 200.0)],
    )

    assert is_table is False


def test_table_candidate_rejected_by_figure_caption() -> None:
    parser = PDFParser()
    rows = [["Q1", "Q2", "Q3"], ["10", "20", "30"]]
    page_words = [
        _word("Figure", 10.0, 8.0, 48.0, 18.0),
        _word("2.", 50.0, 8.0, 60.0, 18.0),
        _word("Sales", 64.0, 8.0, 92.0, 18.0),
        _word("trend", 96.0, 8.0, 126.0, 18.0),
    ]

    is_table = parser._is_probable_table(
        rows=rows,
        bbox=(10.0, 20.0, 180.0, 140.0),
        page_words=page_words,
        figure_bboxes=[],
    )

    assert is_table is False


def test_regular_table_candidate_is_accepted() -> None:
    parser = PDFParser()
    rows = [["Metric", "2023", "2024"], ["Revenue", "10", "12"], ["Cost", "5", "6"]]
    page_words = [
        _word("Table", 10.0, 8.0, 42.0, 18.0),
        _word("1.", 44.0, 8.0, 52.0, 18.0),
        _word("Quarterly", 56.0, 8.0, 102.0, 18.0),
        _word("results", 106.0, 8.0, 138.0, 18.0),
    ]

    is_table = parser._is_probable_table(
        rows=rows,
        bbox=(10.0, 20.0, 180.0, 140.0),
        page_words=page_words,
        figure_bboxes=[],
    )

    assert is_table is True
