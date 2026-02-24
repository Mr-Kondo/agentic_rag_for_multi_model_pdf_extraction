"""
State schemas for LangGraph workflow execution.

This module defines TypedDict schemas that represent the state passed
between nodes in LangGraph-based pipelines. These schemas provide:
- Type safety via static type checking
- Clear documentation of data flow
- IDE autocompletion support
- Runtime validation boundaries

Design Principles:
- Immutable data flow: Each node returns a new state dict
- Explicit phases: Track current workflow stage
- Error accumulation: Collect errors without halting pipeline
- Trace integration: Maintain Langfuse observability
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from src.core.models import RAGAnswer, RawChunk, ProcessedChunk
from src.integrations.langfuse import TraceHandle

__all__ = [
    "QueryState",
    "IngestState",
]

logger = logging.getLogger(__name__)


# ============================================================================
# Query Pipeline State
# ============================================================================


class QueryState(TypedDict, total=False):
    """
    State schema for the query (RAG) pipeline graph.

    Workflow phases:
    1. retrieve: Search vector store for relevant chunks
    2. check_quality: Validate retrieval results (quality gate)
    3. generate: Produce answer using orchestrator agent
    4. validate: Check answer grounding (optional)
    5. revise: Fix hallucinations if detected (optional)
    6. finalize: Prepare final output

    Attributes:
        # Input (required at graph entry)
        question: User's natural language question
        validates: Whether to run answer validation (CHECKPOINT B)
        session_id: Optional session ID for context continuity

        # Intermediate results
        retrieved_hits: List of chunks from vector store retrieval
        raw_answer: Initial answer from orchestrator agent
        validated_answer: Answer after validation/revision (if validates=True)
        final_answer: Output answer (may be raw or validated)

        # Metadata
        trace: Langfuse trace handle for observability
        errors: List of error messages encountered during execution
        warnings: List of non-fatal warnings
        stats: Statistics dictionary (timings, counts, etc.)

        # Control flow
        current_phase: Current workflow stage name
        needs_revision: Flag indicating answer needs revision due to hallucination
        skip_validation: Internal flag to bypass validation
        insufficient_context: Flag indicating retrieval returned no hits
    """

    # ──────── Input ────────
    question: str
    validates: bool
    session_id: Optional[str]

    # ──────── Intermediate Results ────────
    retrieved_hits: List[Dict[str, Any]]
    raw_answer: Optional[RAGAnswer]
    validated_answer: Optional[RAGAnswer]
    final_answer: Optional[RAGAnswer]

    # ──────── Metadata ────────
    trace: Optional[TraceHandle]
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

    # ──────── Control Flow ────────
    current_phase: str  # "retrieve" | "check_quality" | "generate" | "validate" | "revise" | "finalize"
    needs_revision: bool
    skip_validation: bool
    insufficient_context: bool


# ============================================================================
# Ingest Pipeline State (Future Implementation)
# ============================================================================


class IngestState(TypedDict, total=False):
    """
    State schema for the ingest (PDF processing) pipeline graph.

    This is designed for future Phase 2 implementation when migrating
    the ingest pipeline to LangGraph for parallelization benefits.

    Workflow phases:
    1. parse: Extract raw chunks from PDF
    2. extract_text: Process TEXT chunks (parallel)
    3. extract_table: Process TABLE chunks (parallel)
    4. extract_figure: Process FIGURE chunks (parallel)
    5. merge: Combine extraction results
    6. validate: Run chunk validation (CHECKPOINT A, optional)
    7. store: Upsert validated chunks to vector store

    Attributes:
        # Input (required at graph entry)
        pdf_path: Path to input PDF file
        validates: Whether to run chunk validation (CHECKPOINT A)
        storage_dir: Optional custom storage directory for ChromaDB

        # Intermediate results
        raw_chunks: Raw chunks from PDF parsing
        extracted_text_chunks: Processed TEXT chunks
        extracted_table_chunks: Processed TABLE chunks
        extracted_figure_chunks: Processed FIGURE chunks
        all_extracted: All extracted chunks merged
        validated_chunks: Chunks after validation (if validates=True)
        accepted_chunks: Final chunks to be stored

        # Metadata
        trace: Langfuse trace handle for observability
        errors: List of error messages
        warnings: List of non-fatal warnings
        stats: Statistics (chunk counts, timings, correction rates)

        # Control flow
        current_phase: Current workflow stage
        retry_count: Number of retries attempted (for error recovery)
        skip_validation: Internal flag to bypass validation
    """

    # ──────── Input ────────
    pdf_path: str
    validates: bool
    storage_dir: Optional[str]

    # ──────── Intermediate Results ────────
    raw_chunks: List[RawChunk]
    extracted_text_chunks: List[ProcessedChunk]
    extracted_table_chunks: List[ProcessedChunk]
    extracted_figure_chunks: List[ProcessedChunk]
    all_extracted: List[ProcessedChunk]
    validated_chunks: List[ProcessedChunk]
    accepted_chunks: List[ProcessedChunk]

    # ──────── Metadata ────────
    trace: Optional[TraceHandle]
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

    # ──────── Control Flow ────────
    current_phase: str  # "parse" | "extract_*" | "merge" | "validate" | "store"
    retry_count: int
    skip_validation: bool


# ============================================================================
# State Helper Functions
# ============================================================================


def init_query_state(
    question: str,
    validates: bool = False,
    session_id: Optional[str] = None,
    trace: Optional[TraceHandle] = None,
) -> QueryState:
    """
    Initialize a QueryState with sensible defaults.

    Args:
        question: User's question
        validates: Whether to enable answer validation
        session_id: Optional session ID for tracking
        trace: Optional Langfuse trace handle

    Returns:
        QueryState: Initialized state dictionary

    Example:
        >>> state = init_query_state("What is the main topic?", validates=True)
        >>> state["question"]
        'What is the main topic?'
        >>> state["current_phase"]
        'retrieve'
    """
    return QueryState(
        # Input
        question=question,
        validates=validates,
        session_id=session_id,
        # Intermediate
        retrieved_hits=[],
        raw_answer=None,
        validated_answer=None,
        final_answer=None,
        # Metadata
        trace=trace,
        errors=[],
        warnings=[],
        stats={},
        # Control
        current_phase="retrieve",
        needs_revision=False,
        skip_validation=False,
        insufficient_context=False,
    )


def init_ingest_state(
    pdf_path: str,
    validates: bool = False,
    storage_dir: Optional[str] = None,
    trace: Optional[TraceHandle] = None,
) -> IngestState:
    """
    Initialize an IngestState with sensible defaults.

    Args:
        pdf_path: Path to PDF file
        validates: Whether to enable chunk validation
        storage_dir: Optional custom storage directory
        trace: Optional Langfuse trace handle

    Returns:
        IngestState: Initialized state dictionary

    Example:
        >>> state = init_ingest_state("paper.pdf", validates=True)
        >>> state["pdf_path"]
        'paper.pdf'
        >>> state["current_phase"]
        'parse'
    """
    return IngestState(
        # Input
        pdf_path=pdf_path,
        validates=validates,
        storage_dir=storage_dir,
        # Intermediate
        raw_chunks=[],
        extracted_text_chunks=[],
        extracted_table_chunks=[],
        extracted_figure_chunks=[],
        all_extracted=[],
        validated_chunks=[],
        accepted_chunks=[],
        # Metadata
        trace=trace,
        errors=[],
        warnings=[],
        stats={},
        # Control
        current_phase="parse",
        retry_count=0,
        skip_validation=False,
    )
