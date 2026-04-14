"""Tests for embedding input trimming safeguards in ChunkStore."""

import logging

from src.core.models import ChunkType, ProcessedChunk
from src.core.store import ChunkStore


def test_build_embedding_inputs_trims_and_logs_chunk_id(caplog):
    """Oversized chunks should be trimmed with chunk-level warning logs."""
    store = object.__new__(ChunkStore)
    store._max_embed_input_chars = 20

    chunk = ProcessedChunk(
        chunk_type=ChunkType.TEXT,
        page_num=3,
        source_file="sample.pdf",
        structured_text="A" * 50,
        intuition_summary="summary",
    )

    with caplog.at_level(logging.WARNING):
        texts = store._build_embedding_inputs([chunk])

    assert len(texts) == 1
    assert len(texts[0]) == 20
    assert f"chunk_id={chunk.chunk_id}" in caplog.text
    assert "Embedding trim summary: 1/1 chunks trimmed" in caplog.text


def test_build_embedding_inputs_keeps_short_chunks():
    """Short chunks should pass through unchanged."""
    store = object.__new__(ChunkStore)
    store._max_embed_input_chars = 200

    chunk = ProcessedChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file="sample.pdf",
        structured_text="short",
        intuition_summary="ok",
    )

    texts = store._build_embedding_inputs([chunk])

    assert texts == ["short\n\nok"]
