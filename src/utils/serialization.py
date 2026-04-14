"""
Output serialization utilities for saving chunks and answers to JSON.

Handles conversion of dataclass objects to JSON-serializable formats.
"""

import json
import logging
from pathlib import Path

from src.core.models import ProcessedChunk, RAGAnswer

log = logging.getLogger(__name__)

# Output directory for generated files
OUTPUT_DIR = Path("./out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _serialize_bbox(bbox: tuple[float, float, float, float] | None) -> dict | None:
    """Convert bbox tuple into a named dict for JSON serialization."""
    if bbox is None:
        return None
    return {
        "x0": bbox[0],
        "y0": bbox[1],
        "x1": bbox[2],
        "y1": bbox[3],
    }


def serialize_chunk(chunk: ProcessedChunk) -> dict:
    """
    Convert ProcessedChunk to JSON-serializable dict.

    Truncates long text fields to keep file size manageable.

    Args:
        chunk: ProcessedChunk to serialize

    Returns:
        Dictionary with chunk data
    """
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type.value,
        "page_num": chunk.page_num,
        "source_file": chunk.source_file,
        "bbox": _serialize_bbox(chunk.bbox),
        "page_size": {
            "width": chunk.page_width,
            "height": chunk.page_height,
        },
        "artifact_path": chunk.artifact_path or None,
        "source_preview": chunk.source_preview,
        "structured_text": chunk.structured_text[:500] + "..." if len(chunk.structured_text) > 500 else chunk.structured_text,
        "intuition_summary": chunk.intuition_summary,
        "key_concepts": chunk.key_concepts,
        "confidence": chunk.confidence,
        "agent_notes": chunk.agent_notes,
        "validation": {
            "is_valid": chunk.validation.is_valid,
            "verdict_score": chunk.validation.verdict_score,
            "issues": chunk.validation.issues,
            "validator_notes": chunk.validation.validator_notes,
            "corrected_text": chunk.validation.corrected.structured_text[:500] + "..."
            if chunk.validation.corrected and len(chunk.validation.corrected.structured_text or "") > 500
            else (chunk.validation.corrected.structured_text if chunk.validation.corrected else None),
            "corrected_chunk_id": chunk.validation.corrected.chunk_id if chunk.validation.corrected else None,
        }
        if chunk.validation
        else None,
    }


def save_chunks(chunks: list[ProcessedChunk], pdf_name: str | Path, output_dir: str | Path | None = None) -> None:
    """
    Save processed chunks to JSON file.

    Args:
        chunks: List of ProcessedChunk objects
        pdf_name: Source PDF filename (used for output filename)
        output_dir: Optional output directory override
    """
    chunks_data = [serialize_chunk(c) for c in chunks]
    base_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_dir / f"{Path(pdf_name).stem}_chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    log.info(f"✓ Saved {len(chunks)} chunks to {output_path}")


def save_answer(
    result: RAGAnswer,
    pdf_name: str | Path,
    question: str,
    output_dir: str | Path | None = None,
) -> None:
    """
    Save RAG answer to JSON file.

    Args:
        result: RAGAnswer object with answer and metadata
        pdf_name: Source PDF filename (str or Path)
        question: Original question
        output_dir: Optional output directory override
    """
    # Convert Path to string for JSON serialization
    pdf_name_str = str(pdf_name) if isinstance(pdf_name, Path) else pdf_name

    answer_data = {
        "pdf_file": pdf_name_str,
        "question": question,
        "answer": result.answer,
        "reasoning_trace": result.reasoning_trace[:1000] + "..."
        if len(result.reasoning_trace) > 1000
        else result.reasoning_trace,
        "source_chunks": result.source_chunks,
        "trace_id": result.trace_id,
        "validation": {
            "answer_is_grounded": result.validation_summary.answer_is_grounded,
            "hallucinations": result.validation_summary.hallucinations,
            "answer_verdict_score": result.validation_summary.answer_verdict_score,
            "validator_notes": result.validation_summary.validator_notes,
            "answer_was_revised": result.validation_summary.answer_was_revised,
        }
        if result.validation_summary
        else None,
    }

    base_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_dir / f"{Path(pdf_name).stem}_answer.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answer_data, f, ensure_ascii=False, indent=2)

    log.info(f"✓ Saved answer to {output_path}")
