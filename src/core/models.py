"""
Core data structures for the Agentic RAG pipeline.

This module contains all fundamental data models with no internal dependencies,
breaking circular imports between modules.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ═══════════════════════════════════════════════════════════
# CHUNK TYPES AND RAW DATA
# ═══════════════════════════════════════════════════════════


class ChunkType(str, Enum):
    """Type of content chunk from PDF."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


@dataclass
class RawChunk:
    """
    Raw content extracted from PDF before LLM processing.

    Attributes:
        chunk_type: Type of content (text/table/figure)
        page_num: Page number in source PDF (1-indexed)
        raw_content: Raw content (str for text/table, PIL.Image for figure)
        bbox: Bounding box coordinates (x0, y0, x1, y1) or None
        source_file: Source PDF filename
    """

    chunk_type: ChunkType
    page_num: int
    raw_content: Any
    bbox: tuple | None = None
    source_file: str = ""


# ═══════════════════════════════════════════════════════════
# VALIDATION RESULTS
# ═══════════════════════════════════════════════════════════


@dataclass
class ChunkValidationResult:
    """
    Result of Checkpoint A: chunk extraction quality audit.

    Used by ChunkValidatorAgent to assess and correct extraction quality
    before storing in vector database.

    Attributes:
        is_valid: Whether chunk passes validation
        issues: List of detected problems
        corrected: Corrected version of chunk (if invalid and fixable)
        verdict_score: Quality score 0-1
        validator_notes: Human-readable validation notes
    """

    is_valid: bool
    issues: list[str] = field(default_factory=list)
    corrected: "ProcessedChunk | None" = None
    verdict_score: float = 1.0
    validator_notes: str = ""


@dataclass
class AnswerValidationResult:
    """
    Result of Checkpoint B: hallucination/grounding check.

    Used by AnswerValidatorAgent to detect unsupported claims in RAG answers.

    Attributes:
        is_grounded: Whether answer is grounded in source chunks
        hallucinations: List of ungrounded claims detected
        revised_answer: Corrected answer (if ungrounded and fixable)
        verdict_score: Grounding score 0-1
        validator_notes: Human-readable validation notes
    """

    is_grounded: bool
    hallucinations: list[str] = field(default_factory=list)
    revised_answer: str | None = None
    verdict_score: float = 1.0
    validator_notes: str = ""


# ═══════════════════════════════════════════════════════════
# PROCESSED CHUNKS
# ═══════════════════════════════════════════════════════════


@dataclass
class ProcessedChunk:
    """
    Structured content after LLM extraction and validation.

    Stored in vector database and retrieved for RAG queries.

    Attributes:
        chunk_id: Unique identifier (UUID)
        chunk_type: Type of content
        page_num: Source page number
        source_file: Source PDF filename
        structured_text: LLM-extracted structured representation
        intuition_summary: High-level summary for semantic search
        key_concepts: Extracted key concepts/entities
        confidence: Agent confidence score 0-1
        agent_notes: Agent's internal notes/reasoning
        embedding: Vector embedding (populated by ChunkStore)
        validation: Validation result from ChunkValidatorAgent
    """

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_type: ChunkType = ChunkType.TEXT
    page_num: int = 0
    source_file: str = ""
    structured_text: str = ""
    intuition_summary: str = ""
    key_concepts: list[str] = field(default_factory=list)
    confidence: float = 1.0
    agent_notes: str = ""
    embedding: list[float] = field(default_factory=list)
    validation: ChunkValidationResult | None = None


# ═══════════════════════════════════════════════════════════
# RAG ANSWER STRUCTURES
# ═══════════════════════════════════════════════════════════


@dataclass
class ValidationSummary:
    """
    Summary of answer validation results.

    Attached to RAGAnswer to show validation status to user.

    Attributes:
        answer_is_grounded: Whether answer is grounded in sources
        hallucinations: List of detected hallucinations
        answer_verdict_score: Overall grounding score 0-1
        validator_notes: Human-readable validation notes
        answer_was_revised: Whether answer was corrected by validator
    """

    answer_is_grounded: bool
    hallucinations: list[str]
    answer_verdict_score: float
    validator_notes: str
    answer_was_revised: bool


@dataclass
class RAGAnswer:
    """
    Final RAG answer with metadata and validation.

    Returned by AgenticRAGPipeline.query() and displayed to user.

    Attributes:
        question: User's original question
        answer: Generated answer text
        reasoning_trace: LLM's reasoning process (from <think> tags)
        source_chunks: Retrieved chunks used for answer
        trace_id: Langfuse trace ID for observability
        validation_summary: Validation results from AnswerValidatorAgent
    """

    question: str
    answer: str
    reasoning_trace: str
    source_chunks: list[dict] = field(default_factory=list)
    trace_id: str = ""
    validation_summary: ValidationSummary | None = None
