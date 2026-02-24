"""
agentic_rag_flow.py - DEPRECATED BACKWARD COMPATIBILITY WRAPPER

⚠️  DEPRECATION WARNING ⚠️

This file is maintained for backward compatibility only.
New code should use the modular src/ package structure:

  Old (deprecated):
    from agentic_rag_flow import AgenticRAGPipeline, ProcessedChunk, RAGAnswer

  New (recommended):
    from src.core.pipeline import AgenticRAGPipeline
    from src.core.models import ProcessedChunk, RAGAnswer
    from src.utils.serialization import save_chunks, save_answer

For CLI usage, use the new app.py entry point:
    python app.py --help

This compatibility wrapper will be removed in v1.0.0.
See MIGRATION.md for migration guide.

═══════════════════════════════════════════════════════════════════════════
"""

import logging
import sys
import warnings
from pathlib import Path

# Show deprecation warning on import
warnings.warn(
    "\n\n"
    "═" * 70 + "\n"
    "⚠️  DEPRECATION WARNING\n"
    "═" * 70 + "\n"
    "agentic_rag_flow.py is deprecated and will be removed in v1.0.0.\n"
    "\n"
    "Please migrate to the new modular structure:\n"
    "  • from src.core.pipeline import AgenticRAGPipeline\n"
    "  • from src.core.models import ProcessedChunk, RAGAnswer\n"
    "  • python app.py --help  (for CLI usage)\n"
    "\n"
    "See MIGRATION.md for details.\n"
    "═" * 70 + "\n",
    DeprecationWarning,
    stacklevel=2,
)

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY EXPORTS
# ═══════════════════════════════════════════════════════════

# Re-export core data structures (previously defined here)
from src.core.models import (
    AnswerValidationResult,
    ChunkType,
    ChunkValidationResult,
    ProcessedChunk,
    RAGAnswer,
    RawChunk,
    ValidationSummary,
)

# Re-export core classes (previously defined here)
from src.core.cache import ModelCache, _model_cache
from src.core.parser import PDFParser
from src.core.pipeline import AgenticRAGPipeline
from src.core.store import ChunkStore

# Re-export agents (previously defined here)
from src.agents.base import BaseAgent, BaseLoadableModel
from src.agents.extraction import TableAgent, TextAgent, VisionAgent
from src.agents.orchestrator import ReasoningOrchestratorAgent
from src.agents.router import AgentRouter
from src.agents.validation import AnswerValidatorAgent, ChunkValidatorAgent

# Re-export utilities
from src.utils.serialization import save_answer, save_chunks

# Re-export integrations
from src.integrations.langfuse import LangfuseTracer, TraceHandle

# ═══════════════════════════════════════════════════════════
# LEGACY __main__ DEMO (preserved for backward compatibility)
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    warnings.warn(
        "\n\n"
        "⚠️  Running deprecated demo script.\n"
        "Use the new CLI instead:\n"
        "  python app.py pipeline <pdf_path> <question>\n"
        "\n"
        "Example:\n"
        "  python app.py pipeline ./input/paper.pdf 'Summarize the main findings'\n",
        DeprecationWarning,
        stacklevel=1,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python agentic_rag_flow.py <pdf_path> [question]")
        print()
        print("⚠️  This script is deprecated. Use the new CLI:")
        print("  python app.py --help")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "Summarise the main findings. Describe any key figures or tables."

    rag = AgenticRAGPipeline.build(
        orchestrator_model="mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        chunk_validator_model="mlx-community/SmolVLM-256M-Instruct-4bit",
        answer_validator_model="mlx-community/Qwen3-8B-4bit",
    )

    print(f"\n[INGEST] {pdf_path}")
    chunks = rag.ingest(pdf_path, validates=True)
    stats = {ct.value: sum(1 for c in chunks if c.chunk_type == ct) for ct in ChunkType}
    print(f"[CHUNK STATS] {stats}")

    # Save chunks to output directory
    save_chunks(chunks, Path(pdf_path))

    print(f"\n[QUERY] {question}")
    result = rag.query(question, validates=True)

    print("\n=== ANSWER ===")
    print(result.answer)

    if result.validation_summary:
        v = result.validation_summary
        print("\n=== VALIDATION SUMMARY ===")
        print(f"  Grounded       : {v.answer_is_grounded}")
        print(f"  Verdict score  : {v.answer_verdict_score:.2f}")
        print(f"  Was revised    : {v.answer_was_revised}")
        if v.hallucinations:
            print(f"  Hallucinations : {v.hallucinations}")

    print(f"\n[Langfuse trace ID] {result.trace_id}")

    # Save answer to output directory
    save_answer(result, Path(pdf_path), question)

    # Clean up unused models from cache
    log.info("🧹 Cleaning up unused models...")
    _model_cache.cleanup_unused_models()

    print("\n⚠️  Migration reminder: Use 'python app.py --help' for the new CLI")
