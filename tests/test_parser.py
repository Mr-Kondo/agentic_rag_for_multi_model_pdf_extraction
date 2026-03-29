"""Tests for parser heuristics that separate tables from figures."""

from __future__ import annotations

import sys
from types import SimpleNamespace

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


def test_fallback_candidate_rejected_for_prose_like_cells_without_table_cue() -> None:
    parser = PDFParser()
    rows = [
        ["This section describes the methodology in detail", "The approach was validated with multiple runs"],
        ["The authors discuss assumptions and limitations extensively", "Findings are contextualized with prior work"],
    ]

    rejection = parser._table_rejection_reason(
        rows=rows,
        bbox=(10.0, 20.0, 560.0, 300.0),
        page_words=[],
        figure_bboxes=[],
        is_fallback=True,
    )

    assert rejection == "fallback_low_table_signal"


def test_fallback_candidate_with_table_cue_can_be_accepted() -> None:
    parser = PDFParser()
    rows = [["Category", "Status"], ["Alpha", "Open"], ["Beta", "Closed"]]
    page_words = [
        _word("Table", 10.0, 8.0, 42.0, 18.0),
        _word("2.", 44.0, 8.0, 52.0, 18.0),
        _word("Lookup", 56.0, 8.0, 98.0, 18.0),
    ]

    rejection = parser._table_rejection_reason(
        rows=rows,
        bbox=(10.0, 20.0, 180.0, 140.0),
        page_words=page_words,
        figure_bboxes=[],
        is_fallback=True,
    )

    assert rejection is None


def test_default_candidate_rejected_for_prose_like_cells_without_table_cue() -> None:
    parser = PDFParser()
    rows = [
        ["This section describes the methodology in detail", "The approach was validated with multiple runs"],
        ["The authors discuss assumptions and limitations extensively", "Findings are contextualized with prior work"],
        ["Additional narrative paragraphs explain implications", "The content is mostly prose rather than metrics"],
    ]

    rejection = parser._table_rejection_reason(
        rows=rows,
        bbox=(10.0, 20.0, 560.0, 320.0),
        page_words=[],
        figure_bboxes=[],
        is_fallback=False,
    )

    assert rejection == "default_prose_like_without_table_cue"


def test_build_table_metrics_aggregates_candidate_sources_and_reasons() -> None:
    parser = PDFParser()
    table_candidates = [
        (object(), False),
        (object(), False),
        (object(), True),
    ]
    metrics = parser._build_table_metrics(
        table_candidates=table_candidates,
        accepted_tables=1,
        rejected_reasons={"fallback_low_table_signal": 1, "too_sparse": 1},
    )

    assert metrics["total_candidates"] == 3
    assert metrics["default_candidates"] == 2
    assert metrics["fallback_candidates"] == 1
    assert metrics["accepted_candidates"] == 1
    assert metrics["rejected_candidates"] == 2
    assert metrics["rejected_reasons"] == {"fallback_low_table_signal": 1, "too_sparse": 1}


class _DummyCandidate:
    def __init__(self, bbox: tuple[float, float, float, float]):
        self.bbox = bbox


class _DummyPage:
    def __init__(self, default_candidates: list[object], fallback_candidates: list[object]):
        self._default_candidates = default_candidates
        self._fallback_candidates = fallback_candidates
        self.calls: list[str] = []

    def find_tables(self, table_settings=None):
        if table_settings is None:
            self.calls.append("default")
            return self._default_candidates
        self.calls.append("fallback")
        return self._fallback_candidates


def test_find_table_candidates_skips_fallback_when_default_exists_and_flag_off() -> None:
    parser = PDFParser(enable_figure_aware_fallback=False)
    page = _DummyPage(
        default_candidates=[_DummyCandidate((0.0, 0.0, 100.0, 100.0))],
        fallback_candidates=[_DummyCandidate((10.0, 10.0, 110.0, 110.0))],
    )

    result = parser._find_table_candidates(page, has_figures=True)

    assert page.calls == ["default"]
    assert len(result) == 1
    assert result[0][1] is False


def test_find_table_candidates_uses_fallback_when_flag_on_and_figures_present() -> None:
    parser = PDFParser(enable_figure_aware_fallback=True)
    page = _DummyPage(
        default_candidates=[_DummyCandidate((0.0, 0.0, 100.0, 100.0))],
        fallback_candidates=[_DummyCandidate((10.0, 10.0, 110.0, 110.0))],
    )

    result = parser._find_table_candidates(page, has_figures=True)

    assert page.calls == ["default", "fallback"]
    assert len(result) == 2
    assert result[0][1] is False
    assert result[1][1] is True


def test_find_table_candidates_uses_fallback_when_no_default_candidates() -> None:
    parser = PDFParser(enable_figure_aware_fallback=False)
    page = _DummyPage(
        default_candidates=[],
        fallback_candidates=[_DummyCandidate((10.0, 10.0, 110.0, 110.0))],
    )

    result = parser._find_table_candidates(page, has_figures=False)

    assert page.calls == ["default", "fallback"]
    assert len(result) == 1
    assert result[0][1] is True


def test_select_text_words_prefers_source_with_fewer_cid_tokens() -> None:
    parser = PDFParser()
    plumb_words = [
        _word("(cid:12345)", 10.0, 10.0, 40.0, 20.0),
        _word("(cid:67890)", 42.0, 10.0, 70.0, 20.0),
    ]
    fitz_words = [
        _word("日本語", 10.0, 10.0, 30.0, 20.0),
        _word("テキスト", 32.0, 10.0, 62.0, 20.0),
    ]

    selected = parser._select_text_words(plumb_words, fitz_words)

    assert selected == fitz_words


def test_select_text_words_falls_back_when_fitz_empty() -> None:
    parser = PDFParser()
    plumb_words = [_word("valid", 10.0, 10.0, 30.0, 20.0)]

    selected = parser._select_text_words(plumb_words, [])

    assert selected == plumb_words


def test_word_quality_score_penalizes_cid_artifacts() -> None:
    parser = PDFParser()
    cid_heavy = [_word("(cid:100)", 0.0, 0.0, 20.0, 10.0), _word("(cid:101)", 22.0, 0.0, 42.0, 10.0)]
    readable = [_word("Revenue", 0.0, 0.0, 30.0, 10.0), _word("2025", 32.0, 0.0, 50.0, 10.0)]

    assert parser._word_quality_score(readable) > parser._word_quality_score(cid_heavy)


def test_should_try_ocr_for_low_readability_text() -> None:
    parser = PDFParser()
    garbled = "ᣑᙇ,62㧗ឤᗘ ࣓࢝ࣛࢆ ⏝ ࠸ࡓ 67,9⏬ീゎᯒ"

    assert parser._should_try_ocr(garbled) is True


def test_should_not_try_ocr_for_clean_text() -> None:
    parser = PDFParser()
    clean = "河川流量観測の手法を比較し、STIVとDIEXの特徴を評価した。"

    assert parser._should_try_ocr(clean) is False


def test_choose_better_text_prefers_ocr_candidate() -> None:
    parser = PDFParser()
    original = "ᣑᙇ,62㧗ឤᗘ ࣓࢝ࣛࢆ ⏝ ࠸ࡓ"
    ocr = "拡張ISO高感度カメラを用いた"

    assert parser._choose_better_text(original, ocr) == ocr


def test_choose_better_text_keeps_original_when_ocr_worse() -> None:
    parser = PDFParser()
    original = "RIVER DISCHARGE OBSERVATION BY STIV IMAGE ANALYSIS"
    ocr = "R1VER D1SCHARGE 0BSERVAT10N"

    assert parser._choose_better_text(original, ocr) == original


def test_ocr_language_sequence_removes_duplicates() -> None:
    parser = PDFParser()
    parser.OCR_DEFAULT_LANG = "jpn+eng"
    parser.OCR_JAPANESE_LANG = "jpn"
    parser.OCR_FALLBACK_LANG = "jpn+eng"

    assert parser._ocr_language_sequence() == ("jpn+eng", "jpn")


def test_ocr_language_sequence_filters_unavailable_languages() -> None:
    parser = PDFParser()
    parser.OCR_DEFAULT_LANG = "jpn+eng"
    parser.OCR_JAPANESE_LANG = "jpn"
    parser.OCR_FALLBACK_LANG = "eng"

    parser._get_available_tesseract_languages = lambda: {"eng", "osd"}  # type: ignore[assignment]

    assert parser._ocr_language_sequence() == ("eng",)


def test_clean_extracted_text_removes_cid_and_normalizes_whitespace() -> None:
    parser = PDFParser()
    raw_text = "  (cid:120)流量\u3000観測   \n  (cid:33)  DIEX\t法  "

    cleaned = parser._clean_extracted_text(raw_text)

    assert cleaned == "流量 観測\nDIEX 法"


def test_ocr_text_from_bbox_tries_japanese_before_english(monkeypatch) -> None:
    parser = PDFParser()
    parser.OCR_DEFAULT_LANG = "jpn+eng"
    parser.OCR_JAPANESE_LANG = "jpn"
    parser.OCR_FALLBACK_LANG = "eng"

    class _DummyPix:
        width = 2
        height = 2
        samples = bytes([255, 255, 255] * 4)

    class _DummyPage:
        def get_pixmap(self, matrix, clip, alpha):
            return _DummyPix()

    calls: list[str] = []

    def _fake_ocr(_image, lang, config):
        calls.append(lang)
        if lang in {"jpn+eng", "jpn"}:
            raise RuntimeError("ocr failed")
        return "fallback text"

    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(image_to_string=_fake_ocr),
    )

    text = parser._ocr_text_from_bbox(_DummyPage(), (0.0, 0.0, 10.0, 10.0))

    assert text == "fallback text"
    assert calls == ["jpn+eng", "jpn", "eng"]
