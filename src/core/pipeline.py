"""
Core RAG pipeline orchestrating all components.

AgenticRAGPipeline coordinates the entire workflow:
1. Ingestion: Parse PDF → Extract chunks → Validate quality → Store vectors
2. Query: Retrieve chunks → Generate answer → Validate grounding

Memory management strategy (v3):
    - Small extraction agents (2-4B): Always loaded during ingestion
    - Large validators (7-16B): Explicit load/unload per phase with context managers
    - Peak VRAM: max(extraction+chunk_validator, orchestrator, answer_validator)
    - Typical: ~22GB for ingestion, ~16GB for query (never simultaneous)

For 16GB GPUs: Set lazy_agents=True to also load/unload extraction agents per chunk.
"""

import logging
from pathlib import Path
from typing import Any

from src.agents.extraction import TableAgent, TextAgent, VisionAgent
from src.agents.orchestrator import ReasoningOrchestratorAgent
from src.agents.router import AgentRouter
from src.agents.validation import AnswerValidatorAgent, ChunkValidatorAgent
from src.core.models import (
    ChunkType,
    ProcessedChunk,
    RAGAnswer,
    RawChunk,
    ValidationSummary,
)
from src.core.parser import PDFParser
from src.core.store import ChunkStore
from src.integrations.langfuse import LangfuseTracer

log = logging.getLogger(__name__)


class AgenticRAGPipeline:
    """
    Main RAG pipeline with sequential load/unload for memory efficiency.

    Model memory timeline:

    ingest():
      Phase 1 — Extraction  (small SLMs, always loaded):
        TextAgent(3-4B) + TableAgent(3B) + VisionAgent(2B) run concurrently in VRAM.
        Total: ~8-9 GB. Acceptable for 16 GB cards.

      Phase 2 — Chunk Validation  (ChunkValidatorAgent, Qwen2-VL-7B):
        [LOAD]   ChunkValidatorAgent  (+14 GB)
        run validate_chunk() for all chunks
        [UNLOAD] ChunkValidatorAgent  (-14 GB + CUDA cache clear)

    query():
      Phase 1 — Retrieval  (embedding model only, ~120 MB):
        retrieve() — no LLM needed

      Phase 2 — Generation  (OrchestratorAgent ~16 GB):
        [LOAD]   OrchestratorAgent
        generate()
        [UNLOAD] OrchestratorAgent  (-16 GB + CUDA cache clear)

      Phase 3 — Answer Validation  (AnswerValidatorAgent ~16 GB):
        [LOAD]   AnswerValidatorAgent
        validate_answer()
        [UNLOAD] AnswerValidatorAgent  (-16 GB + CUDA cache clear)

    Peak VRAM requirement: max(small_SLMs + chunk_validator, orchestrator, answer_validator)
      ≈ max(~22 GB, ~16 GB, ~16 GB)
      → 24 GB GPU (e.g. RTX 4090, A10G) sufficient with 4-bit quant on small SLMs.
      → 16 GB GPU: enable lazy_agents=True (small SLMs also load/unload per chunk).

    Usage:
        >>> pipeline = AgenticRAGPipeline.build()
        >>> chunks = pipeline.ingest("paper.pdf", validates=True)
        >>> answer = pipeline.query("What are the main findings?", validates=True)
    """

    @classmethod
    def build(
        cls,
        text_model: str = "mlx-community/Phi-3.5-mini-Instruct-4bit",
        table_model: str = "mlx-community/Qwen2.5-3B-Instruct-4bit",
        vision_model: str = "mlx-community/SmolVLM-256M-Instruct-4bit",
        orchestrator_model: str = "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        chunk_validator_model: str = "mlx-community/Qwen2-VL-7B-Instruct-4bit",  # ← Checkpoint A
        answer_validator_model: str = "mlx-community/Qwen3-8B-4bit",  # ← Checkpoint B
        persist_dir: str = "./chroma_db",
        lazy_agents: bool = False,  # True → small SLMs also load/unload per chunk
    ) -> "AgenticRAGPipeline":
        """
        Initialize the RAG pipeline with all required components.

        Args:
            text_model: Model ID for text extraction agent (~3-4B)
            table_model: Model ID for table extraction agent (~3B)
            vision_model: Model ID for vision extraction agent (~256M-2B)
            orchestrator_model: Model ID for reasoning orchestrator (~8-10B)
            chunk_validator_model: Model ID for chunk quality validation (~7B VLM)
            answer_validator_model: Model ID for answer hallucination detection (~8-10B)
            persist_dir: Directory for ChromaDB vector store persistence
            lazy_agents: If True, load/unload extraction agents per chunk (saves VRAM)

        Returns:
            Configured AgenticRAGPipeline instance ready for use
        """
        log.info("=" * 70)
        log.info("🚀 Initializing Agentic RAG Pipeline")
        log.info("=" * 70)

        obj = cls()
        obj.lazy_agents = lazy_agents

        log.info("📂 Setting up vector store: %s", persist_dir)
        obj.parser = PDFParser()
        obj.store = ChunkStore(persist_dir)
        log.info("✓ Vector store initialized")

        log.info("📡 Setting up Langfuse tracer...")
        obj.tracer = LangfuseTracer()
        log.info("✓ Tracer initialized")

        # Small SLMs — load immediately (stay loaded throughout ingest)
        log.info("\n📦 Loading extraction agents (small SLMs - stay loaded):")

        log.info("  🚀 Text agent: %s", text_model)
        text_agent = TextAgent(text_model)
        log.info("  ✓ Text agent ready")

        log.info("  🚀 Table agent: %s", table_model)
        table_agent = TableAgent(table_model)
        log.info("  ✓ Table agent ready")

        log.info("  🚀 Vision agent: %s", vision_model)
        vision_agent = VisionAgent(vision_model)
        log.info("  ✓ Vision agent ready")

        obj.router = AgentRouter(text_agent, table_agent, vision_agent)
        log.info("\n✅ Extraction agents loaded")

        # Heavy models — instantiate WITHOUT loading; load/unload per phase
        log.info("\n📋 Initializing validator agents (lazy-loaded on demand):")

        log.info("  📋 Orchestrator: %s", orchestrator_model)
        obj.orchestrator = ReasoningOrchestratorAgent(orchestrator_model)
        log.info("  ✓ Orchestrator initialized")

        log.info("  📋 Chunk validator: %s", chunk_validator_model)
        obj.chunk_validator = ChunkValidatorAgent(chunk_validator_model)
        log.info("  ✓ Chunk validator initialized")

        log.info("  📋 Answer validator: %s", answer_validator_model)
        obj.answer_validator = AnswerValidatorAgent(answer_validator_model, use_dspy=True)
        log.info("  ✓ Answer validator initialized (DSPy-enhanced)")

        log.info("\n" + "=" * 70)
        log.info("✅ Pipeline ready for ingestion and querying")
        log.info("=" * 70 + "\n")

        return obj

    # ── Ingestion ──────────────────────────────────────────

    def ingest(
        self,
        pdf_path: str | Path,
        validates: bool = True,
    ) -> list[ProcessedChunk]:
        """
        Ingest a PDF document into the vector store.

        Pipeline phases:
        1. Parse PDF into raw chunks (text/table/figure)
        2. Extract structured data from each chunk with specialized agents
        3. Validate extraction quality (CHECKPOINT A) - optional
        4. Upsert validated chunks into vector database

        Args:
            pdf_path: Path to PDF file to ingest
            validates: If True, run chunk quality validation (CHECKPOINT A)

        Returns:
            List of accepted ProcessedChunk objects stored in vector DB
        """
        pdf_path = Path(pdf_path)

        log.info("=" * 70)
        log.info("📂 INGEST PHASE: %s", pdf_path.name)
        log.info("=" * 70)

        with self.tracer.trace(
            "ingest_pdf",
            input={"file": pdf_path.name, "validates": validates},
            metadata={"pipeline": "agentic_rag_v3"},
        ) as trace:
            # ── Phase 1: Parse ─────────────────────────────
            log.info("📄 Parsing PDF...")
            with trace.span("parse_pdf") as s:
                raw_chunks = self.parser.parse(pdf_path)
                log.info("✓ Parsed %d raw chunks (text/table/figure)", len(raw_chunks))
                s.update(output={"n_raw": len(raw_chunks)})

            # ── Phase 2: Extract (small SLMs always loaded) ─
            log.info("🔄 Extracting chunks with agents...")
            extracted: list[tuple[RawChunk, ProcessedChunk]] = []
            for raw in raw_chunks:
                processed = self.router.route(raw, trace=trace)
                extracted.append((raw, processed))

            # ── Phase 3: Chunk Validation (load → run → unload) ─
            accepted: list[ProcessedChunk] = []
            corrected_count = 0
            discarded_count = 0

            if validates:
                log.info("✅ CHECKPOINT A: Starting chunk validation...")
                try:
                    with self.chunk_validator:  # ← load on enter, unload on exit
                        log.info("  [LOAD] ChunkValidatorAgent loaded")
                        for idx, (raw, processed) in enumerate(extracted, 1):
                            val = self.chunk_validator.validate_chunk(raw=raw, processed=processed, trace=trace)
                            processed.validation = val

                            self.tracer.score(
                                trace_id=trace.trace_id,
                                name="chunk_quality",
                                value=val.verdict_score,
                                comment=f"p.{processed.page_num} {processed.chunk_type.value} | " + "; ".join(val.issues),
                            )

                            if not val.is_valid:
                                if val.corrected is not None:
                                    val.corrected.validation = val
                                    accepted.append(val.corrected)
                                    corrected_count += 1
                                    log.debug(
                                        "  ↻ p.%d %s — corrected by validator",
                                        processed.page_num,
                                        processed.chunk_type.value,
                                    )
                                else:
                                    discarded_count += 1
                                    log.debug(
                                        "  ✗ p.%d %s — discarded",
                                        processed.page_num,
                                        processed.chunk_type.value,
                                    )
                            elif processed.confidence >= 0.25:
                                accepted.append(processed)
                            else:
                                discarded_count += 1
                    # ← ChunkValidatorAgent.unload() called here automatically
                    log.info("  [UNLOAD] ChunkValidatorAgent unloaded")
                    log.info("✓ Chunk validation complete: %d corrected, %d discarded", corrected_count, discarded_count)
                
                except (TypeError, Exception) as e:
                    # Vision model loading failed - fallback to confidence-based filtering
                    log.error(
                        f"❌ Chunk validation failed (vision model error): {e}\n"
                        f"   Falling back to confidence-based filtering (>= 0.25)"
                    )
                    accepted = [p for (_, p) in extracted if p.confidence >= 0.25]
                    log.warning(f"⚠️  Accepted {len(accepted)} chunks without validation")

            else:
                # Skip validation — accept all chunks above confidence floor
                accepted = [p for (_, p) in extracted if p.confidence >= 0.25]
                log.info("⊘ Validation skipped")

            log.info(
                "Ingestion result: accepted=%d corrected=%d discarded=%d",
                len(accepted),
                corrected_count,
                discarded_count,
            )

            # ── Phase 4: Upsert ────────────────────────────
            log.info("💾 Upserting %d chunks into vector store...", len(accepted))
            with trace.span("upsert_store", input={"n": len(accepted)}) as s:
                self.store.upsert(accepted)
                s.update(output={"upserted": len(accepted)})
            log.info("✓ Chunks stored")

            log.info("=" * 70 + "\n")

        return accepted

    # ── Query ──────────────────────────────────────────────

    def query(
        self,
        question: str,
        session_id: str | None = None,
        validates: bool = True,
    ) -> RAGAnswer:
        """
        Query the RAG system with a natural language question.

        Pipeline phases:
        1. Retrieve relevant chunks from vector store
        2. Generate answer with reasoning orchestrator
        3. Validate answer grounding (CHECKPOINT B) - optional

        Args:
            question: Natural language query
            session_id: Optional session ID for grouping related queries
            validates: If True, run hallucination detection (CHECKPOINT B)

        Returns:
            RAGAnswer with answer text, sources, reasoning, and validation results
        """
        log.info("=" * 70)
        log.info("🔍 QUERY PHASE: %s", question[:80])
        log.info("=" * 70)

        with self.tracer.trace(
            "rag_query",
            input={"question": question, "validates": validates},
            session_id=session_id,
        ) as trace:
            # ── Phase 1: Retrieve (embedding model only, no LLM) ─
            log.info("📚 Retrieving relevant chunks...")
            hits = self.orchestrator.retrieve(question, self.store, trace=trace)
            log.info("✓ Retrieved %d chunks", len(hits))

            # ── Phase 2: Generate (load orchestrator → generate → unload) ─
            log.info("🤖 Generating answer with orchestrator...")
            with self.orchestrator:  # ← load on enter, unload on exit
                log.info("  [LOAD] OrchestratorAgent loaded")
                result = self.orchestrator.generate(question, hits, trace=trace)
                log.info("  ✓ Answer generated")
            # ← OrchestratorAgent.unload() called here — VRAM freed
            log.info("  [UNLOAD] OrchestratorAgent unloaded")

            result.trace_id = trace.trace_id

            if validates:
                # ── Phase 3: Validate answer (load answer_validator → validate → unload) ─
                log.info("✅ CHECKPOINT B: Starting answer validation...")
                source_texts = [sc["text"] for sc in result.source_chunks]

                with self.answer_validator:  # ← load on enter, unload on exit
                    log.info("  [LOAD] AnswerValidatorAgent loaded")
                    ans_val = self.answer_validator.validate_answer(
                        question=question,
                        answer=result,
                        source_texts=source_texts,
                        trace=trace,
                    )
                    log.info("  ✓ Answer validation complete")
                # ← AnswerValidatorAgent.unload() called here — VRAM freed
                log.info("  [UNLOAD] AnswerValidatorAgent unloaded")

                self.tracer.score(
                    trace_id=trace.trace_id,
                    name="answer_grounding",
                    value=ans_val.verdict_score,
                    comment=f"grounded={ans_val.is_grounded} | " + "; ".join(ans_val.hallucinations),
                )

                was_revised = False
                if not ans_val.is_grounded:
                    if ans_val.revised_answer:
                        log.warning(
                            "⚠️  Hallucinations detected — substituting revised answer\n  Hallucinations: %s",
                            ans_val.hallucinations,
                        )
                        result.answer = ans_val.revised_answer
                        was_revised = True
                    else:
                        log.warning(
                            "⚠️  Hallucinations detected, no revision available\n  Hallucinations: %s",
                            ans_val.hallucinations,
                        )
                        result.answer = "[VALIDATION WARNING: claims may not be grounded]\n\n" + result.answer

                result.validation_summary = ValidationSummary(
                    answer_is_grounded=ans_val.is_grounded,
                    hallucinations=ans_val.hallucinations,
                    answer_verdict_score=ans_val.verdict_score,
                    validator_notes=ans_val.validator_notes,
                    answer_was_revised=was_revised,
                )
            else:
                log.info("⊘ Answer validation skipped")

        log.info("=" * 70)
        log.info("✅ Query complete - Trace ID: %s", result.trace_id)
        log.info("=" * 70 + "\n")

        return result
