#!/usr/bin/env python3
"""
Agentic RAG CLI - MLX-powered multi-modal document intelligence.

Command-line interface for the Agentic RAG pipeline, supporting:
  - PDF ingestion with quality validation
  - Natural language querying with hallucination detection
  - End-to-end pipeline execution

Example usage:
    # Ingest a PDF
    python app.py ingest paper.pdf --validate

    # Query the vector store
    python app.py query "What are the main findings?"

    # Full pipeline (ingest + query)
    python app.py pipeline paper.pdf "Summarize the methodology"

    # Custom models
    python app.py ingest paper.pdf \\
        --text-model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --orchestrator mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit
"""

import argparse
import logging
import sys
from pathlib import Path

from src.core.cache import _model_cache
from src.core.models import ChunkType
from src.core.pipeline import AgenticRAGPipeline
from src.utils.serialization import save_answer, save_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# DEFAULT MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════

DEFAULT_MODELS = {
    "text": "mlx-community/Phi-3.5-mini-Instruct-4bit",
    "table": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "vision": "mlx-community/SmolVLM-256M-Instruct-4bit",
    "orchestrator": "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
    "chunk_validator": "mlx-community/Qwen2-VL-7B-Instruct-4bit",
    "answer_validator": "mlx-community/Qwen3-8B-4bit",
}


# ═══════════════════════════════════════════════════════════
# COMMAND IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════


def cmd_ingest(args: argparse.Namespace) -> int:
    """
    Ingest a PDF document into the vector store.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        log.error(f"❌ PDF file not found: {pdf_path}")
        return 1

    log.info("Building RAG pipeline...")
    pipeline = AgenticRAGPipeline.build(
        text_model=args.text_model,
        table_model=args.table_model,
        vision_model=args.vision_model,
        orchestrator_model=args.orchestrator_model,
        chunk_validator_model=args.chunk_validator_model,
        answer_validator_model=args.answer_validator_model,
        persist_dir=args.storage_dir,
        lazy_agents=args.lazy_agents,
    )

    log.info(f"\n{'=' * 70}")
    log.info(f"📂 Ingesting: {pdf_path.name}")
    log.info(f"{'=' * 70}\n")

    chunks = pipeline.ingest(pdf_path, validates=args.validate)

    # Print statistics
    stats = {ct.value: sum(1 for c in chunks if c.chunk_type == ct) for ct in ChunkType}
    log.info("\n📊 Chunk Statistics:")
    for chunk_type, count in stats.items():
        log.info(f"  {chunk_type:6s}: {count:3d}")
    log.info(f"  {'TOTAL':6s}: {len(chunks):3d}\n")

    # Save chunks to output
    if args.output:
        save_chunks(chunks, pdf_path)
        log.info(f"💾 Saved chunks to {args.output}/")

    # Clean up unused models
    log.info("🧹 Cleaning up unused models...")
    _model_cache.cleanup_unused_models()

    log.info("✅ Ingestion complete!\n")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """
    Query the RAG system with a natural language question.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    question = args.question

    log.info("Building RAG pipeline...")
    pipeline = AgenticRAGPipeline.build(
        text_model=args.text_model,
        table_model=args.table_model,
        vision_model=args.vision_model,
        orchestrator_model=args.orchestrator_model,
        chunk_validator_model=args.chunk_validator_model,
        answer_validator_model=args.answer_validator_model,
        persist_dir=args.storage_dir,
        lazy_agents=args.lazy_agents,
    )

    log.info(f"\n{'=' * 70}")
    log.info(f"🔍 Query: {question}")
    log.info(f"{'=' * 70}\n")

    result = pipeline.query(question, session_id=args.session_id, validates=args.validate)

    # Display answer
    print("\n" + "=" * 70)
    print("📝 ANSWER")
    print("=" * 70)
    print(result.answer)
    print()

    # Display validation summary
    if result.validation_summary:
        v = result.validation_summary
        print("=" * 70)
        print("✅ VALIDATION SUMMARY")
        print("=" * 70)
        print(f"  Grounded       : {v.answer_is_grounded}")
        print(f"  Verdict score  : {v.answer_verdict_score:.2f}")
        print(f"  Was revised    : {v.answer_was_revised}")
        if v.hallucinations:
            print(f"  Hallucinations : {', '.join(v.hallucinations)}")
        print()

    # Display trace ID
    if result.trace_id:
        print("=" * 70)
        print(f"🔗 Langfuse Trace: {result.trace_id}")
        print("=" * 70)
        print()

    # Save answer to output
    if args.output:
        save_answer(result, Path("query.pdf"), question)
        log.info(f"💾 Saved answer to {args.output}/")

    # Clean up unused models
    log.info("🧹 Cleaning up unused models...")
    _model_cache.cleanup_unused_models()

    log.info("✅ Query complete!\n")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """
    Run full pipeline: ingest PDF then query it.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        log.error(f"❌ PDF file not found: {pdf_path}")
        return 1

    question = args.question

    log.info("Building RAG pipeline...")
    pipeline = AgenticRAGPipeline.build(
        text_model=args.text_model,
        table_model=args.table_model,
        vision_model=args.vision_model,
        orchestrator_model=args.orchestrator_model,
        chunk_validator_model=args.chunk_validator_model,
        answer_validator_model=args.answer_validator_model,
        persist_dir=args.storage_dir,
        lazy_agents=args.lazy_agents,
    )

    # ── PHASE 1: INGEST ──
    log.info(f"\n{'=' * 70}")
    log.info(f"📂 PHASE 1: INGESTING {pdf_path.name}")
    log.info(f"{'=' * 70}\n")

    chunks = pipeline.ingest(pdf_path, validates=args.validate)

    # Print statistics
    stats = {ct.value: sum(1 for c in chunks if c.chunk_type == ct) for ct in ChunkType}
    log.info("\n📊 Chunk Statistics:")
    for chunk_type, count in stats.items():
        log.info(f"  {chunk_type:6s}: {count:3d}")
    log.info(f"  {'TOTAL':6s}: {len(chunks):3d}\n")

    # Save chunks
    if args.output:
        save_chunks(chunks, pdf_path)

    # ── PHASE 2: QUERY ──
    log.info(f"\n{'=' * 70}")
    log.info(f"🔍 PHASE 2: QUERYING")
    log.info(f"{'=' * 70}\n")

    result = pipeline.query(question, session_id=args.session_id, validates=args.validate)

    # Display answer
    print("\n" + "=" * 70)
    print("📝 ANSWER")
    print("=" * 70)
    print(result.answer)
    print()

    # Display validation summary
    if result.validation_summary:
        v = result.validation_summary
        print("=" * 70)
        print("✅ VALIDATION SUMMARY")
        print("=" * 70)
        print(f"  Grounded       : {v.answer_is_grounded}")
        print(f"  Verdict score  : {v.answer_verdict_score:.2f}")
        print(f"  Was revised    : {v.answer_was_revised}")
        if v.hallucinations:
            print(f"  Hallucinations : {', '.join(v.hallucinations)}")
        print()

    # Display trace ID
    if result.trace_id:
        print("=" * 70)
        print(f"🔗 Langfuse Trace: {result.trace_id}")
        print("=" * 70)
        print()

    # Save answer
    if args.output:
        save_answer(result, pdf_path, question)
        log.info(f"💾 Saved outputs to {args.output}/")

    # Clean up
    log.info("🧹 Cleaning up unused models...")
    _model_cache.cleanup_unused_models()

    log.info("\n✅ Pipeline complete!\n")
    return 0


# ═══════════════════════════════════════════════════════════
# ARGUMENT PARSER SETUP
# ═══════════════════════════════════════════════════════════


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all subcommands and options.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Agentic RAG - MLX-powered multi-modal document intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a PDF with validation
  %(prog)s ingest paper.pdf --validate

  # Query without validation (faster)
  %(prog)s query "What are the main findings?" --no-validate

  # Full pipeline with custom orchestrator
  %(prog)s pipeline paper.pdf "Summarize methodology" \\
      --orchestrator mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit

  # Use different vector store
  %(prog)s ingest paper.pdf --storage-dir ./custom_db

For more information, see: https://github.com/yourusername/agentic-rag
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Agentic RAG v0.3.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── INGEST SUBCOMMAND ──
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a PDF document into the vector store",
        description="Parse PDF, extract chunks, validate quality, and store in vector database.",
    )
    ingest_parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to PDF file to ingest",
    )
    ingest_parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Enable chunk quality validation (CHECKPOINT A) [default: enabled]",
    )
    ingest_parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip chunk quality validation (faster, less reliable)",
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    # ── QUERY SUBCOMMAND ──
    query_parser = subparsers.add_parser(
        "query",
        help="Query the RAG system with a natural language question",
        description="Retrieve relevant chunks and generate a grounded answer.",
    )
    query_parser.add_argument(
        "question",
        type=str,
        help="Natural language question to ask",
    )
    query_parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for grouping related queries in Langfuse",
    )
    query_parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Enable hallucination detection (CHECKPOINT B) [default: enabled]",
    )
    query_parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip hallucination detection (faster, less reliable)",
    )
    query_parser.set_defaults(func=cmd_query)

    # ── PIPELINE SUBCOMMAND ──
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline: ingest PDF then query it",
        description="End-to-end workflow: parse → extract → validate → store → query → answer.",
    )
    pipeline_parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to PDF file to ingest",
    )
    pipeline_parser.add_argument(
        "question",
        type=str,
        help="Natural language question to ask",
    )
    pipeline_parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for grouping in Langfuse",
    )
    pipeline_parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Enable validation at both checkpoints [default: enabled]",
    )
    pipeline_parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip all validation (fastest, least reliable)",
    )
    pipeline_parser.set_defaults(func=cmd_pipeline)

    # ── SHARED OPTIONS ──
    for subparser in [ingest_parser, query_parser, pipeline_parser]:
        model_group = subparser.add_argument_group("Model Configuration")
        model_group.add_argument(
            "--text-model",
            type=str,
            default=DEFAULT_MODELS["text"],
            help=f"Text extraction model [default: {DEFAULT_MODELS['text']}]",
        )
        model_group.add_argument(
            "--table-model",
            type=str,
            default=DEFAULT_MODELS["table"],
            help=f"Table extraction model [default: {DEFAULT_MODELS['table']}]",
        )
        model_group.add_argument(
            "--vision-model",
            type=str,
            default=DEFAULT_MODELS["vision"],
            help=f"Vision extraction model [default: {DEFAULT_MODELS['vision']}]",
        )
        model_group.add_argument(
            "--orchestrator-model",
            type=str,
            default=DEFAULT_MODELS["orchestrator"],
            help=f"Reasoning orchestrator model [default: {DEFAULT_MODELS['orchestrator']}]",
        )
        model_group.add_argument(
            "--chunk-validator-model",
            type=str,
            default=DEFAULT_MODELS["chunk_validator"],
            help=f"Chunk quality validator model [default: {DEFAULT_MODELS['chunk_validator']}]",
        )
        model_group.add_argument(
            "--answer-validator-model",
            type=str,
            default=DEFAULT_MODELS["answer_validator"],
            help=f"Answer grounding validator model [default: {DEFAULT_MODELS['answer_validator']}]",
        )

        storage_group = subparser.add_argument_group("Storage Configuration")
        storage_group.add_argument(
            "--storage-dir",
            type=str,
            default="./chroma_db",
            help="Directory for vector store persistence [default: ./chroma_db]",
        )
        storage_group.add_argument(
            "--output",
            type=str,
            default="./output",
            help="Directory for saving output files [default: ./output]",
        )

        perf_group = subparser.add_argument_group("Performance Options")
        perf_group.add_argument(
            "--lazy-agents",
            action="store_true",
            default=False,
            help="Load/unload extraction agents per chunk (saves VRAM, slower)",
        )

    return parser


# ═══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════


def main() -> int:
    """
    Main entry point for the CLI application.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.warning("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        log.error(f"\n\n❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
