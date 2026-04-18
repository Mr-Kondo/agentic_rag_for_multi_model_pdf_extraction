"""Tests for audit artifact generation and provenance serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from src.core.models import ChunkType, ProcessedChunk, RawChunk
from src.utils.audit import save_chunk_audit
from src.utils.serialization import serialize_chunk


def test_serialize_chunk_includes_provenance_fields() -> None:
    """serialize_chunk should retain bbox, page size, and cross-link metadata."""
    chunk = ProcessedChunk(
        chunk_type=ChunkType.TEXT,
        page_num=2,
        source_file="paper.pdf",
        bbox=(10.0, 20.0, 110.0, 180.0),
        page_width=612.0,
        page_height=792.0,
        artifact_path="audit/figures/example.png",
        source_preview="Original source preview",
        structured_text="Structured content",
        intuition_summary="Summary",
        key_concepts=["alpha", "beta"],
        confidence=0.88,
    )

    data = serialize_chunk(chunk)

    assert data["bbox"] == {"x0": 10.0, "y0": 20.0, "x1": 110.0, "y1": 180.0}
    assert data["page_size"] == {"width": 612.0, "height": 792.0}
    assert data["artifact_path"] == "audit/figures/example.png"
    assert data["source_preview"] == "Original source preview"


def test_save_chunk_audit_generates_json_and_html(tmp_path: Path) -> None:
    """save_chunk_audit should emit page previews plus JSON and HTML reports."""
    pdf_path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=300)
    page.insert_text((24, 48), "Audit me")
    document.save(pdf_path)
    document.close()

    raw = RawChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file=pdf_path.name,
        raw_content="Audit me",
        bbox=(20.0, 30.0, 120.0, 70.0),
        page_width=200.0,
        page_height=300.0,
        source_preview="Audit me",
    )
    processed = ProcessedChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file=pdf_path.name,
        bbox=raw.bbox,
        page_width=raw.page_width,
        page_height=raw.page_height,
        source_preview=raw.source_preview,
        structured_text="Audit me",
        intuition_summary="Short text block",
        confidence=0.95,
    )

    paths = save_chunk_audit(
        pdf_path=pdf_path,
        extracted=[(raw, processed)],
        accepted=[processed],
        output_dir=tmp_path / "out",
    )

    assert paths["json"].exists()
    assert paths["html"].exists()
    assert (tmp_path / "out" / "sample_audit" / "pages" / "page_001.png").exists()

    audit_data = json.loads(paths["json"].read_text(encoding="utf-8"))
    html = paths["html"].read_text(encoding="utf-8")

    assert audit_data["pdf_file"] == "sample.pdf"
    assert audit_data["pdf_path"] == "../sample.pdf"
    assert audit_data["chunks"][0]["status"] == "accepted"
    assert audit_data["chunks"][0]["raw"]["bbox"]["x0"] == 20.0
    assert processed.chunk_id in html
    assert "Open source PDF" in html
    assert 'split("\\n")' in html
    assert 'join("\\n- ")' in html


def test_save_chunk_audit_preserves_japanese_text_in_html(tmp_path: Path) -> None:
    """HTML report should keep Japanese text without ASCII escaping."""
    pdf_path = tmp_path / "jp_sample.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=300)
    page.insert_text((24, 48), "dummy")
    document.save(pdf_path)
    document.close()

    raw = RawChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file=pdf_path.name,
        raw_content="日本語の本文です",
        bbox=(20.0, 30.0, 160.0, 90.0),
        page_width=200.0,
        page_height=300.0,
        source_preview="日本語のプレビュー",
    )
    processed = ProcessedChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file=pdf_path.name,
        bbox=raw.bbox,
        page_width=raw.page_width,
        page_height=raw.page_height,
        source_preview=raw.source_preview,
        structured_text="これは日本語テキストです",
        intuition_summary="概要: 日本語テスト",
        confidence=0.95,
    )

    paths = save_chunk_audit(
        pdf_path=pdf_path,
        extracted=[(raw, processed)],
        accepted=[processed],
        output_dir=tmp_path / "out",
    )

    html = paths["html"].read_text(encoding="utf-8")
    assert "日本語" in html
    assert "\\u65e5\\u672c\\u8a9e" not in html


def test_save_chunk_audit_skips_page_preview_rendering_when_disabled(tmp_path: Path) -> None:
    """Audit generation should skip page preview rendering work when disabled."""
    pdf_path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=300)
    page.insert_text((24, 48), "Audit me")
    document.save(pdf_path)
    document.close()

    raw = RawChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file=pdf_path.name,
        raw_content="Audit me",
        bbox=(20.0, 30.0, 120.0, 70.0),
        page_width=200.0,
        page_height=300.0,
        source_preview="Audit me",
    )
    processed = ProcessedChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file=pdf_path.name,
        bbox=raw.bbox,
        page_width=raw.page_width,
        page_height=raw.page_height,
        source_preview=raw.source_preview,
        structured_text="Audit me",
        intuition_summary="Short text block",
        confidence=0.95,
    )

    paths = save_chunk_audit(
        pdf_path=pdf_path,
        extracted=[(raw, processed)],
        accepted=[processed],
        output_dir=tmp_path / "out",
        render_page_previews=False,
    )

    assert paths["json"].exists()
    assert paths["html"].exists()
    assert not (tmp_path / "out" / "sample_audit" / "pages").exists()

    audit_data = json.loads(paths["json"].read_text(encoding="utf-8"))
    html = paths["html"].read_text(encoding="utf-8")

    assert audit_data["pages"] == []
    assert audit_data["pdf_path"] == "../sample.pdf"
    assert "ページプレビューは生成されていません。" in html


def test_save_chunk_audit_persists_parser_table_metrics(tmp_path: Path) -> None:
    """Audit report should retain parser table diagnostics for later comparisons."""
    pdf_path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=300)
    page.insert_text((24, 48), "Audit me")
    document.save(pdf_path)
    document.close()

    raw = RawChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file=pdf_path.name,
        raw_content="Audit me",
        bbox=(20.0, 30.0, 120.0, 70.0),
        page_width=200.0,
        page_height=300.0,
        source_preview="Audit me",
    )
    processed = ProcessedChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file=pdf_path.name,
        bbox=raw.bbox,
        page_width=raw.page_width,
        page_height=raw.page_height,
        source_preview=raw.source_preview,
        structured_text="Audit me",
        intuition_summary="Short text block",
        confidence=0.95,
    )
    parser_metrics = [
        {
            "page_num": 1,
            "total_candidates": 4,
            "default_candidates": 2,
            "fallback_candidates": 2,
            "accepted_candidates": 1,
            "rejected_candidates": 3,
            "rejected_reasons": {"too_sparse": 2, "overlaps_figure": 1},
        }
    ]

    paths = save_chunk_audit(
        pdf_path=pdf_path,
        extracted=[(raw, processed)],
        accepted=[processed],
        output_dir=tmp_path / "out",
        parser_table_metrics=parser_metrics,
    )

    audit_data = json.loads(paths["json"].read_text(encoding="utf-8"))
    html = paths["html"].read_text(encoding="utf-8")

    assert audit_data["parser_table_metrics"] == parser_metrics
    assert "Parser table metrics" in html
    assert '"parser_table_metrics"' in html
    assert '"too_sparse": 2' in html
