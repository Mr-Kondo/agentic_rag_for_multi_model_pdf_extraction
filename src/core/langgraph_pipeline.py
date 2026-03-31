"""
LangGraph-based RAG pipeline for improved readability and performance.

This module implements both the query pipeline and the ingest pipeline using
LangGraph's state graph architecture, providing:
- Visual workflow representation (nodes and edges)
- Conditional branching (quality gates)
- Better error handling and recovery
- Easier debugging with intermediate state snapshots

Pipelines
---------
LangGraphQueryPipeline:
    8-node StateGraph for answer generation with optional validation.
    Use via --use-langgraph on the query and pipeline subcommands.

LangGraphIngestPipeline:
    4-node StateGraph for PDF extraction with optional chunk validation.
    Use via --use-langgraph on the ingest and pipeline subcommands.

Usage:
    >>> from src.core.langgraph_pipeline import LangGraphQueryPipeline
    >>> pipeline = LangGraphQueryPipeline.build()
    >>> answer = pipeline.query("What are the main findings?", validates=True)

    >>> from src.core.langgraph_pipeline import LangGraphIngestPipeline
    >>> ingest_pipeline = LangGraphIngestPipeline.build()
    >>> chunks = ingest_pipeline.ingest("paper.pdf", validates=True)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from langgraph.graph import END, StateGraph

from src.agents.extraction import TableAgent, TextAgent, VisionAgent
from src.agents.orchestrator import ReasoningOrchestratorAgent
from src.agents.router import AgentRouter
from src.agents.validation import AnswerValidatorAgent, ChunkValidatorAgent
from src.core.config import config
from src.core.graph_state import IngestState, QueryState, init_ingest_state, init_query_state
from src.core.models import ProcessedChunk, RAGAnswer, RawChunk, ValidationSummary
from src.core.parser import PDFParser
from src.core.store import ChunkStore
from src.integrations.langfuse import LangfuseTracer
from src.utils.audit import save_chunk_audit

__all__ = ["LangGraphQueryPipeline", "LangGraphIngestPipeline"]

log = logging.getLogger(__name__)


# ============================================================================
# Node Functions (Pure State Transformations)
# ============================================================================


def retrieve_node(state: QueryState) -> QueryState:
    """
    Node 1: Retrieve relevant chunks from vector store.

    Uses embedding model only (no LLM), so this is fast (~500ms).

    Args:
        state: Query state with 'question' field

    Returns:
        Updated state with 'retrieved_hits' populated

    Phase: retrieve → check_quality
    """
    log.info("📚 [retrieve_node] Retrieving chunks for question: %s", state["question"][:60])

    # Extract dependencies from state
    orchestrator: ReasoningOrchestratorAgent = state.get("_orchestrator")  # type: ignore
    store: ChunkStore = state.get("_store")  # type: ignore
    trace = state.get("trace")

    # Perform retrieval (embedding search, no model loading)
    hits = orchestrator.retrieve(state["question"], store, trace=trace)

    log.info("✓ [retrieve_node] Retrieved %d chunks", len(hits))

    # Update state
    state["retrieved_hits"] = hits
    state["current_phase"] = "check_quality"
    state["stats"]["retrieved_count"] = len(hits)

    return state


def check_retrieval_quality_node(state: QueryState) -> QueryState:
    """
    Node 2: Quality gate - check if retrieval returned sufficient context.

    This is a critical optimization: if no chunks found, we can skip
    expensive generation and validation phases entirely.

    Args:
        state: Query state with 'retrieved_hits' populated

    Returns:
        Updated state with 'insufficient_context' flag

    Phase: check_quality → (generate if hits > 0, else finalize)
    """
    hits = state["retrieved_hits"]
    hit_count = len(hits)

    log.info("🔍 [check_quality] Evaluating retrieval quality: %d hits", hit_count)

    if hit_count == 0:
        log.warning("⚠️  [check_quality] No relevant chunks found - skipping generation")
        state["insufficient_context"] = True
        state["warnings"].append("No relevant context found for question")
        state["current_phase"] = "finalize"
    else:
        log.info("✓ [check_quality] Sufficient context available")
        state["insufficient_context"] = False
        state["current_phase"] = "generate"

    return state


def generate_answer_node(state: QueryState) -> QueryState:
    """
    Node 3: Generate answer using orchestrator agent.

    Loads large orchestrator model (~16GB), generates answer, then unloads.
    Uses context manager pattern for automatic resource management.

    Args:
        state: Query state with 'retrieved_hits' and 'question'

    Returns:
        Updated state with 'raw_answer' populated

    Phase: generate → decide_validate
    """
    log.info("🤖 [generate_answer] Generating answer with orchestrator...")

    orchestrator: ReasoningOrchestratorAgent = state.get("_orchestrator")  # type: ignore
    trace = state.get("trace")

    # Load orchestrator model, generate answer, unload
    with orchestrator:  # ← Load on entry, unload on exit
        log.info("  [LOAD] OrchestratorAgent loaded (~16GB)")

        result = orchestrator.generate(
            query=state["question"],
            context_chunks=state["retrieved_hits"],
            trace=trace,
        )

        log.info("  ✓ Answer generated (%d chars)", len(result.answer))

    # ← Orchestrator unloaded here, VRAM freed
    log.info("  [UNLOAD] OrchestratorAgent unloaded")

    # Update trace ID if available
    if trace:
        result.trace_id = trace.trace_id

    state["raw_answer"] = result
    state["current_phase"] = "decide_validate"
    state["stats"]["answer_length"] = len(result.answer)

    return state


def decide_validate_node(state: QueryState) -> QueryState:
    """
    Node 4: Decision node - determine if validation is needed.

    Checks the 'validates' flag to decide next step.

    Args:
        state: Query state with 'validates' flag

    Returns:
        Updated state with routing decision

    Phase: decide_validate → (validate if True, else finalize)
    """
    if state["validates"]:
        log.info("✅ [decide_validate] Validation enabled - proceeding to validation")
        state["current_phase"] = "validate"
    else:
        log.info("⊘ [decide_validate] Validation disabled - skipping to finalize")
        state["skip_validation"] = True
        state["current_phase"] = "finalize"

    return state


def validate_answer_node(state: QueryState) -> QueryState:
    """
    Node 5: Validate answer grounding (CHECKPOINT B).

    Loads answer validator model (~16GB), checks for hallucinations,
    then unloads. Uses DSPy-enhanced validation for precise detection.

    Args:
        state: Query state with 'raw_answer'

    Returns:
        Updated state with 'validated_answer' and 'needs_revision' flag

    Phase: validate → check_grounding
    """
    log.info("✅ [validate_answer] CHECKPOINT B: Starting answer validation...")

    answer_validator: AnswerValidatorAgent = state.get("_answer_validator")  # type: ignore
    trace = state.get("trace")
    result = state["raw_answer"]

    # Extract source texts for validation
    source_texts = [sc["text"] for sc in result.source_chunks]

    # Load answer validator, validate, unload
    with answer_validator:  # ← Load on entry, unload on exit
        log.info("  [LOAD] AnswerValidatorAgent loaded (~16GB)")

        ans_val = answer_validator.validate_answer(
            question=state["question"],
            answer=result,
            source_texts=source_texts,
            trace=trace,
        )

        log.info("  ✓ Validation complete - Grounded: %s", ans_val.is_grounded)

    # ← Answer validator unloaded here, VRAM freed
    log.info("  [UNLOAD] AnswerValidatorAgent unloaded")

    # Record validation score in trace
    if trace:
        tracer: LangfuseTracer = state.get("_tracer")  # type: ignore
        tracer.score(
            trace_id=trace.trace_id,
            name="answer_grounding",
            value=ans_val.verdict_score,
            comment=f"grounded={ans_val.is_grounded} | " + "; ".join(ans_val.hallucinations),
        )

    # Store validation result in state
    state["_validation_result"] = ans_val  # type: ignore
    state["needs_revision"] = not ans_val.is_grounded and ans_val.revised_answer is not None
    state["current_phase"] = "check_grounding"
    state["stats"]["is_grounded"] = ans_val.is_grounded
    state["stats"]["hallucination_count"] = len(ans_val.hallucinations)

    return state


def check_grounding_node(state: QueryState) -> QueryState:
    """
    Node 6: Decision node - check if answer needs revision.

    Based on validation results, decide whether to revise or finalize.

    Args:
        state: Query state with validation results

    Returns:
        Updated state with routing decision

    Phase: check_grounding → (revise if needs_revision, else finalize)
    """
    if state["needs_revision"]:
        log.warning("⚠️  [check_grounding] Hallucinations detected - needs revision")
        state["current_phase"] = "revise"
    else:
        log.info("✓ [check_grounding] Answer is grounded - proceeding to finalize")
        state["current_phase"] = "finalize"

    return state


def revise_answer_node(state: QueryState) -> QueryState:
    """
    Node 7: Revise answer to fix hallucinations.

    Uses the revised answer provided by the validator.

    Args:
        state: Query state with validation results

    Returns:
        Updated state with revised answer

    Phase: revise → finalize
    """
    log.info("🔧 [revise_answer] Applying revised answer to fix hallucinations")

    ans_val = state.get("_validation_result")  # type: ignore
    result = state["raw_answer"]

    if ans_val and ans_val.revised_answer:
        log.info("  ✓ Substituting revised answer")
        result.answer = ans_val.revised_answer
        state["stats"]["was_revised"] = True
    else:
        log.warning("  ⚠️  No revised answer available - adding warning prefix")
        result.answer = "[VALIDATION WARNING: claims may not be grounded]\n\n" + result.answer
        state["warnings"].append("Hallucinations detected but no revision available")
        state["stats"]["was_revised"] = False

    state["validated_answer"] = result
    state["current_phase"] = "finalize"

    return state


def finalize_node(state: QueryState) -> QueryState:
    """
    Node 8: Finalize answer and prepare output.

    Handles all terminal cases:
    - Insufficient context (no hits)
    - Unvalidated answer (validation skipped)
    - Validated answer (grounded)
    - Revised answer (hallucinations fixed)

    Args:
        state: Query state at final stage

    Returns:
        Updated state with 'final_answer' populated

    Phase: finalize → END
    """
    log.info("🎯 [finalize] Finalizing answer...")

    if state.get("insufficient_context"):
        # Case 1: No context found
        log.info("  → Insufficient context case")
        final = RAGAnswer(
            question=state["question"],
            answer="[No relevant context found] I couldn't find relevant information in the document to answer your question.",
            reasoning_trace="No chunks retrieved from vector store",
            source_chunks=[],
            trace_id=state.get("trace").trace_id if state.get("trace") else "",
        )

    elif state.get("validated_answer"):
        # Case 2: Validated and possibly revised answer
        log.info("  → Validated answer case")
        final = state["validated_answer"]
        ans_val = state.get("_validation_result")  # type: ignore

        # Add validation summary
        final.validation_summary = ValidationSummary(
            answer_is_grounded=ans_val.is_grounded if ans_val else True,
            hallucinations=ans_val.hallucinations if ans_val else [],
            answer_verdict_score=ans_val.verdict_score if ans_val else 1.0,
            validator_notes=ans_val.validator_notes if ans_val else "",
            answer_was_revised=bool(state["stats"].get("was_revised", False)),
        )

    elif state.get("raw_answer"):
        # Case 3: Unvalidated answer (validation skipped)
        log.info("  → Unvalidated answer case (validation skipped)")
        final = state["raw_answer"]

    else:
        # Case 4: Unexpected - no answer generated
        log.error("  ⚠️  No answer available - this should not happen")
        state["errors"].append("No answer generated")
        final = RAGAnswer(
            question=state["question"],
            answer="[Error] Failed to generate answer",
            reasoning_trace="Pipeline error",
            source_chunks=[],
            trace_id="",
        )

    state["final_answer"] = final
    state["current_phase"] = "done"

    log.info("✅ [finalize] Answer finalized")
    log.info("  - Answer length: %d chars", len(final.answer))
    log.info("  - Source chunks: %d", len(final.source_chunks))
    log.info("  - Trace ID: %s", final.trace_id)

    return state


# ============================================================================
# Conditional Edge Functions (Routing Logic)
# ============================================================================


def route_after_quality_check(state: QueryState) -> Literal["generate", "finalize"]:
    """Route based on retrieval quality."""
    if state.get("insufficient_context"):
        return "finalize"
    return "generate"


def route_after_decide_validate(state: QueryState) -> Literal["validate", "finalize"]:
    """Route based on validation flag."""
    if state["validates"]:
        return "validate"
    return "finalize"


def route_after_grounding_check(state: QueryState) -> Literal["revise", "finalize"]:
    """Route based on hallucination detection."""
    if state.get("needs_revision"):
        return "revise"
    return "finalize"


# ============================================================================
# Pipeline Class (Graph Builder)
# ============================================================================


class LangGraphQueryPipeline:
    """
    LangGraph-based query pipeline for improved workflow visibility.

    This pipeline uses LangGraph's state graph architecture to implement
    the RAG query workflow with explicit phases, conditional branching,
    and quality gates.

    Workflow:
        START
          ↓
        retrieve (fast, embedding only)
          ↓
        check_quality ─┐
          ↓            └──→ finalize (if no hits)
        generate (load orchestrator)
          ↓
        decide_validate ─┐
          ↓              └──→ finalize (if validates=False)
        validate (load answer_validator)
          ↓
        check_grounding ─┐
          ↓              └──→ finalize (if grounded)
        revise (fix hallucinations)
          ↓
        finalize
          ↓
        END

    Advantages:
    - Visual workflow representation
    - Quality gates prevent wasted computation
    - Conditional branching for validation and revision
    - Clear phase boundaries for debugging
    - Foundation for future parallelization

    Usage:
        >>> pipeline = LangGraphQueryPipeline.build()
        >>> answer = pipeline.query("What is the main topic?", validates=True)
    """

    def __init__(
        self,
        orchestrator: ReasoningOrchestratorAgent,
        answer_validator: AnswerValidatorAgent,
        store: ChunkStore,
        tracer: LangfuseTracer,
    ):
        """
        Initialize pipeline with required components.

        Args:
            orchestrator: Agent for answer generation
            answer_validator: Agent for answer validation
            store: Vector store for chunk retrieval
            tracer: Langfuse tracer for observability
        """
        self.orchestrator = orchestrator
        self.answer_validator = answer_validator
        self.store = store
        self.tracer = tracer

        # Build the state graph
        self.graph = self._build_graph()

        log.info("✓ LangGraphQueryPipeline initialized")

    @classmethod
    def build(
        cls,
        orchestrator_model: str | None = None,
        answer_validator_model: str | None = None,
        persist_dir: str = "./chroma_db",
    ) -> "LangGraphQueryPipeline":
        """Build pipeline with models from settings.json."""
        # Load defaults from configuration
        orchestrator_model = orchestrator_model or config.get_model("orchestrator")
        answer_validator_model = answer_validator_model or config.get_model("answer_validator")
        """
        Build pipeline with default models.

        Args:
            orchestrator_model: Model for answer generation (~8B)
            answer_validator_model: Model for answer validation (~8B)
            persist_dir: ChromaDB persistence directory

        Returns:
            Initialized LangGraphQueryPipeline
        """
        log.info("Building LangGraphQueryPipeline...")

        # Initialize components
        orchestrator = ReasoningOrchestratorAgent(orchestrator_model)

        answer_validator = AnswerValidatorAgent(
            answer_validator_model,
            use_dspy=True,
        )

        store = ChunkStore(persist_dir)
        tracer = LangfuseTracer()

        return cls(
            orchestrator=orchestrator,
            answer_validator=answer_validator,
            store=store,
            tracer=tracer,
        )

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph for query pipeline.

        Returns:
            Compiled state graph ready for execution
        """
        log.info("Building LangGraph query workflow...")

        # Define node functions with dependency injection via closure
        # This allows nodes to access self.orchestrator, self.store, etc.

        def _retrieve(state: QueryState) -> QueryState:
            """Retrieve chunks using orchestrator and store from closure."""
            orchestrator = self.orchestrator
            store = self.store
            trace = state.get("trace")

            log.info("📚 [retrieve_node] Retrieving chunks for question: %s", state["question"][:60])
            hits = orchestrator.retrieve(state["question"], store, trace=trace)
            log.info("✓ [retrieve_node] Retrieved %d chunks", len(hits))

            state["retrieved_hits"] = hits
            state["current_phase"] = "check_quality"
            state["stats"]["retrieved_count"] = len(hits)
            return state

        def _generate(state: QueryState) -> QueryState:
            """Generate answer using orchestrator from closure."""
            orchestrator = self.orchestrator
            trace = state.get("trace")

            log.info("🤖 [generate_answer] Generating answer with orchestrator...")
            with orchestrator:
                log.info("  [LOAD] OrchestratorAgent loaded (~16GB)")
                result = orchestrator.generate(
                    question=state["question"],
                    hits=state["retrieved_hits"],
                    trace=trace,
                )
                log.info("  ✓ Answer generated (%d chars)", len(result.answer))
            log.info("  [UNLOAD] OrchestratorAgent unloaded")

            if trace:
                result.trace_id = trace.trace_id

            state["raw_answer"] = result
            state["current_phase"] = "decide_validate"
            state["stats"]["answer_length"] = len(result.answer)
            return state

        def _validate(state: QueryState) -> QueryState:
            """Validate answer using answer_validator from closure."""
            answer_validator = self.answer_validator
            tracer = self.tracer
            trace = state.get("trace")
            result = state["raw_answer"]

            log.info("✅ [validate_answer] CHECKPOINT B: Starting answer validation...")
            source_texts = [sc["text"] for sc in result.source_chunks]

            with answer_validator:
                log.info("  [LOAD] AnswerValidatorAgent loaded (~16GB)")
                ans_val = answer_validator.validate_answer(
                    question=state["question"],
                    answer=result,
                    source_texts=source_texts,
                    trace=trace,
                )
                log.info("  ✓ Validation complete - Grounded: %s", ans_val.is_grounded)
            log.info("  [UNLOAD] AnswerValidatorAgent unloaded")

            if trace:
                tracer.score(
                    trace_id=trace.trace_id,
                    name="answer_grounding",
                    value=ans_val.verdict_score,
                    comment=f"grounded={ans_val.is_grounded} | " + "; ".join(ans_val.hallucinations),
                )

            state["_validation_result"] = ans_val  # type: ignore
            state["needs_revision"] = not ans_val.is_grounded and ans_val.revised_answer is not None
            state["current_phase"] = "check_grounding"
            state["stats"]["is_grounded"] = ans_val.is_grounded
            state["stats"]["hallucination_count"] = len(ans_val.hallucinations)
            return state

        # Initialize graph with QueryState schema
        builder = StateGraph(QueryState)

        # ──────── Add Nodes ────────
        builder.add_node("retrieve", _retrieve)
        builder.add_node("check_quality", check_retrieval_quality_node)
        builder.add_node("generate", _generate)
        builder.add_node("decide_validate", decide_validate_node)
        builder.add_node("validate", _validate)
        builder.add_node("check_grounding", check_grounding_node)
        builder.add_node("revise", revise_answer_node)
        builder.add_node("finalize", finalize_node)

        # ──────── Add Edges ────────
        # Linear flow
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "check_quality")
        builder.add_edge("generate", "decide_validate")
        builder.add_edge("validate", "check_grounding")
        builder.add_edge("revise", "finalize")
        builder.add_edge("finalize", END)

        # Conditional edges (quality gates)
        builder.add_conditional_edges(
            "check_quality",
            route_after_quality_check,
            {
                "generate": "generate",
                "finalize": "finalize",
            },
        )

        builder.add_conditional_edges(
            "decide_validate",
            route_after_decide_validate,
            {
                "validate": "validate",
                "finalize": "finalize",
            },
        )

        builder.add_conditional_edges(
            "check_grounding",
            route_after_grounding_check,
            {
                "revise": "revise",
                "finalize": "finalize",
            },
        )

        # Compile graph
        graph = builder.compile()

        log.info("✓ LangGraph workflow compiled successfully")
        return graph

    def query(
        self,
        question: str,
        validates: bool = False,
        session_id: str | None = None,
    ) -> RAGAnswer:
        """
        Execute query pipeline using LangGraph workflow.

        Args:
            question: User's natural language question
            validates: Whether to run answer validation (CHECKPOINT B)
            session_id: Optional session ID for trace tracking

        Returns:
            RAGAnswer with generated answer and metadata

        Example:
            >>> pipeline = LangGraphQueryPipeline.build()
            >>> answer = pipeline.query("What are the main findings?", validates=True)
            >>> print(answer.answer)
            >>> print(f"Confidence: {answer.confidence}")
        """
        log.info("=" * 70)
        log.info("🔍 LANGGRAPH QUERY PHASE: %s", question[:80])
        log.info("=" * 70)

        # Start Langfuse trace
        with self.tracer.trace(
            "langgraph_query",
            input={"question": question, "validates": validates},
            session_id=session_id,
        ) as trace:
            # Initialize state
            state = init_query_state(
                question=question,
                validates=validates,
                session_id=session_id,
                trace=trace,
            )

            # Execute graph (dependencies injected via closure in _build_graph)
            log.info("▶️  Executing LangGraph workflow...")
            final_state = self.graph.invoke(state)

            # Extract final answer
            result = final_state["final_answer"]

            if result is None:
                raise RuntimeError("Graph execution failed: No final answer produced")

            # Log statistics
            stats = final_state.get("stats", {})
            log.info("📊 Pipeline Statistics:")
            log.info("  - Retrieved chunks: %d", stats.get("retrieved_count", 0))
            log.info("  - Answer length: %d", stats.get("answer_length", 0))
            log.info("  - Grounded: %s", stats.get("is_grounded", "N/A"))
            log.info("  - Hallucinations: %d", stats.get("hallucination_count", 0))
            log.info("  - Revised: %s", stats.get("was_revised", False))

            # Log warnings and errors
            if final_state.get("warnings"):
                for warning in final_state["warnings"]:
                    log.warning("⚠️  %s", warning)

            if final_state.get("errors"):
                for error in final_state["errors"]:
                    log.error("❌ %s", error)

        log.info("=" * 70)
        log.info("✅ LangGraph Query complete - Trace ID: %s", result.trace_id)
        log.info("=" * 70 + "\n")

        return result


# ============================================================================
# Ingest Routing Functions
# ============================================================================


def route_after_extract(state: IngestState) -> Literal["validate_chunks", "store"]:
    """Route based on whether chunk validation is enabled."""
    if state.get("validates"):
        return "validate_chunks"
    return "store"


def _safe_extracted_pairs(state: IngestState, phase: str) -> List[Tuple[RawChunk, ProcessedChunk]]:
    """Return extracted raw/processed pairs with defensive fallback for invalid state."""
    pairs = state.get("_extracted_pairs")
    if not isinstance(pairs, list):
        warning_msg = (
            f"[{phase}] Missing or invalid _extracted_pairs in state (got {type(pairs).__name__}); falling back to empty list"
        )
        log.warning(warning_msg)
        state["warnings"].append(warning_msg)
        return []
    return pairs


# ============================================================================
# LangGraphIngestPipeline
# ============================================================================


class LangGraphIngestPipeline:
    """
    LangGraph-based ingest pipeline for sequential PDF document processing.

    Uses LangGraph's state graph to implement the PDF ingest workflow with
    explicit phases, conditional chunk validation, and audit trail support.

    Workflow:
        START
          |
        parse  (PDFParser, no LLM)
          |
        extract  (TextAgent / TableAgent / VisionAgent)
          |
        validate_chunks  (optional, CHECKPOINT A — ChunkValidatorAgent)
          |
        store  (ChromaDB upsert + optional audit)
          |
        END

    Notes:
        - Extraction agents (small SLMs) are loaded at build time and remain
          loaded for the duration of the ingest call.
        - ChunkValidatorAgent is loaded/unloaded with its context manager to
          free VRAM promptly after validation completes.
        - Audit artifacts (HTML/JSON) are written from the store node when
          audit_output_dir is provided to ingest().

    Usage:
        >>> pipeline = LangGraphIngestPipeline.build()
        >>> chunks = pipeline.ingest("paper.pdf", validates=True)
        >>> print(f"Ingested {len(chunks)} chunks")
    """

    def __init__(
        self,
        text_agent: TextAgent,
        table_agent: TableAgent,
        vision_agent: VisionAgent,
        chunk_validator: ChunkValidatorAgent,
        store: ChunkStore,
        tracer: LangfuseTracer,
        enable_figure_aware_fallback: bool = False,
    ):
        """
        Initialize ingest pipeline with extraction and validation components.

        Args:
            text_agent: Agent for processing text chunks
            table_agent: Agent for processing table chunks
            vision_agent: Agent for processing figure/image chunks
            chunk_validator: Agent for chunk quality validation (CHECKPOINT A)
            store: Vector store for persisting extracted chunks
            tracer: Langfuse tracer for observability
            enable_figure_aware_fallback: Enable parser fallback table search on figure pages
        """
        self.parser = PDFParser(enable_figure_aware_fallback=enable_figure_aware_fallback)
        self.router = AgentRouter(text_agent, table_agent, vision_agent)
        self.chunk_validator = chunk_validator
        self.store = store
        self.tracer = tracer
        self.audit_render_page_previews = bool(config.get("audit.render_page_previews", True))
        self.graph = self._build_graph()
        log.info("LangGraphIngestPipeline initialized")

    @classmethod
    def build(
        cls,
        text_model: str | None = None,
        table_model: str | None = None,
        vision_model: str | None = None,
        chunk_validator_model: str | None = None,
        persist_dir: str = "./chroma_db",
        enable_figure_aware_fallback: bool = False,
    ) -> "LangGraphIngestPipeline":
        """
        Build ingest pipeline with models resolved from settings.json.

        Args:
            text_model: Model ID for text extraction (~3-4B)
            table_model: Model ID for table extraction (~3B)
            vision_model: Model ID for vision extraction (~256M-2B)
            chunk_validator_model: Model ID for chunk validation (~7B VLM)
            persist_dir: ChromaDB persistence directory
            enable_figure_aware_fallback: Enable parser fallback table search on figure pages

        Returns:
            Initialized LangGraphIngestPipeline
        """
        text_model = text_model or config.get_model("text_extraction")
        table_model = table_model or config.get_model("table_extraction")
        vision_model = vision_model or config.get_model("vision_extraction")
        chunk_validator_model = chunk_validator_model or config.get_model("chunk_validator")

        log.info("Building LangGraphIngestPipeline...")
        log.info("  text_model       : %s", text_model)
        log.info("  table_model      : %s", table_model)
        log.info("  vision_model     : %s", vision_model)
        log.info("  chunk_validator  : %s", chunk_validator_model)

        text_agent = TextAgent(text_model)
        table_agent = TableAgent(table_model)
        vision_agent = VisionAgent(vision_model)
        chunk_validator = ChunkValidatorAgent(chunk_validator_model)
        store = ChunkStore(persist_dir)
        tracer = LangfuseTracer()

        return cls(
            text_agent=text_agent,
            table_agent=table_agent,
            vision_agent=vision_agent,
            chunk_validator=chunk_validator,
            store=store,
            tracer=tracer,
            enable_figure_aware_fallback=enable_figure_aware_fallback,
        )

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph for the ingest pipeline.

        Nodes are implemented as closures to access pipeline components
        (router, chunk_validator, store, tracer) without global state.

        Returns:
            Compiled state graph ready for execution
        """
        log.info("Building LangGraph ingest workflow...")

        router = self.router
        parser = self.parser
        chunk_validator = self.chunk_validator
        store = self.store
        tracer = self.tracer

        def _parse(state: IngestState) -> IngestState:
            """Node 1: Parse PDF into raw chunks (PDFParser, no LLM)."""
            trace = state.get("trace")
            pdf_path = Path(state["pdf_path"])

            log.info("[parse] Parsing PDF: %s", pdf_path.name)

            if trace:
                with trace.span("parse_pdf") as s:
                    raw_chunks = self.parser.parse(pdf_path)
                    s.update(output={"n_raw": len(raw_chunks)})
            else:
                raw_chunks = self.parser.parse(pdf_path)

            log.info("[parse] Parsed %d raw chunks", len(raw_chunks))

            state["raw_chunks"] = raw_chunks
            state["current_phase"] = "extract"
            state["stats"]["raw_count"] = len(raw_chunks)
            return state

        def _extract(state: IngestState) -> IngestState:
            """Node 2: Route each raw chunk to its specialized extraction agent."""
            trace = state.get("trace")
            raw_chunks = state["raw_chunks"]

            log.info("[extract] Extracting %d chunks with specialized agents...", len(raw_chunks))

            pairs: List[Tuple[RawChunk, ProcessedChunk]] = []
            for raw in raw_chunks:
                region_policy = parser._get_region_policy(raw.chunk_type)
                processed = router.route_with_policy(raw, region_policy, trace=trace)
                pairs.append((raw, processed))

            all_processed = [processed for _, processed in pairs]
            log.info("[extract] Extracted %d chunks", len(all_processed))

            state["all_extracted"] = all_processed
            # Store raw-processed pairs as hidden state for validation and audit use.
            state["_extracted_pairs"] = pairs
            # Default acceptance uses confidence floor; overwritten if validation runs.
            state["accepted_chunks"] = [p for p in all_processed if p.confidence >= 0.25]
            state["current_phase"] = "decide_validate"
            state["stats"]["extracted_count"] = len(all_processed)
            return state

        def _validate_chunks(state: IngestState) -> IngestState:
            """Node 3: CHECKPOINT A — validate each chunk against its source."""
            trace = state.get("trace")
            pairs = _safe_extracted_pairs(state, phase="validate_chunks")

            log.info("[validate_chunks] CHECKPOINT A: validating %d chunks...", len(pairs))

            accepted: List[ProcessedChunk] = []
            corrected_count = 0
            discarded_count = 0

            try:
                with chunk_validator:
                    log.info("  [LOAD] ChunkValidatorAgent loaded")
                    for raw, processed in pairs:
                        val = chunk_validator.validate_chunk(raw=raw, processed=processed, trace=trace)
                        processed.validation = val

                        if trace:
                            tracer.score(
                                trace_id=trace.trace_id,
                                name="chunk_quality",
                                value=val.verdict_score,
                                comment=(f"p.{processed.page_num} {processed.chunk_type.value} | " + "; ".join(val.issues)),
                            )

                        if not val.is_valid:
                            if val.corrected is not None:
                                val.corrected.validation = val
                                accepted.append(val.corrected)
                                corrected_count += 1
                            else:
                                discarded_count += 1
                        elif processed.confidence >= 0.25:
                            accepted.append(processed)
                        else:
                            discarded_count += 1

                log.info("  [UNLOAD] ChunkValidatorAgent unloaded")

            except Exception as e:
                log.error(
                    "[validate_chunks] Validation failed (%s) — falling back to confidence filter",
                    e,
                )
                accepted = [p for _, p in pairs if p.confidence >= 0.25]
                state["warnings"].append(f"Chunk validation failed: {e}")

            log.info(
                "[validate_chunks] Done: accepted=%d corrected=%d discarded=%d",
                len(accepted),
                corrected_count,
                discarded_count,
            )

            state["accepted_chunks"] = accepted
            state["current_phase"] = "store"
            state["stats"]["corrected_count"] = corrected_count
            state["stats"]["discarded_count"] = discarded_count
            return state

        def _store(state: IngestState) -> IngestState:
            """Node 4: Upsert accepted chunks into ChromaDB, optionally write audit."""
            trace = state.get("trace")
            accepted = state["accepted_chunks"]
            pdf_path = Path(state["pdf_path"])
            audit_dir = state.get("_audit_output_dir")  # type: ignore

            log.info("[store] Upserting %d chunks into vector store...", len(accepted))

            if trace:
                with trace.span("upsert_store", input={"n": len(accepted)}) as s:
                    store.upsert(accepted)
                    s.update(output={"upserted": len(accepted)})
            else:
                store.upsert(accepted)

            log.info("[store] Upserted successfully")

            if audit_dir is not None:
                pairs = _safe_extracted_pairs(state, phase="store")
                save_chunk_audit(
                    pdf_path=pdf_path,
                    extracted=pairs,
                    accepted=accepted,
                    output_dir=audit_dir,
                    render_page_previews=self.audit_render_page_previews,
                )
                log.info("[store] Audit written to %s", audit_dir)

            state["current_phase"] = "done"
            state["stats"]["accepted_count"] = len(accepted)
            return state

        # ──────── Build Graph ────────
        builder = StateGraph(IngestState)

        builder.add_node("parse", _parse)
        builder.add_node("extract", _extract)
        builder.add_node("validate_chunks", _validate_chunks)
        builder.add_node("store", _store)

        builder.set_entry_point("parse")
        builder.add_edge("parse", "extract")
        builder.add_edge("validate_chunks", "store")
        builder.add_edge("store", END)

        builder.add_conditional_edges(
            "extract",
            route_after_extract,
            {
                "validate_chunks": "validate_chunks",
                "store": "store",
            },
        )

        graph = builder.compile()
        log.info("LangGraph ingest workflow compiled successfully")
        return graph

    def ingest(
        self,
        pdf_path: str | Path,
        validates: bool = True,
        audit_output_dir: str | Path | None = None,
    ) -> List[ProcessedChunk]:
        """
        Execute ingest pipeline using LangGraph workflow.

        Args:
            pdf_path: Path to PDF file to ingest
            validates: Whether to run chunk quality validation (CHECKPOINT A)
            audit_output_dir: Optional directory for HTML/JSON audit artifacts

        Returns:
            List of accepted ProcessedChunk objects stored in vector DB

        Example:
            >>> pipeline = LangGraphIngestPipeline.build()
            >>> chunks = pipeline.ingest("paper.pdf", validates=True)
            >>> print(f"Ingested {len(chunks)} chunks")
        """
        pdf_path = Path(pdf_path)

        log.info("=" * 70)
        log.info("LANGGRAPH INGEST PHASE: %s", pdf_path.name)
        log.info("=" * 70)

        with self.tracer.trace(
            "langgraph_ingest",
            input={"file": pdf_path.name, "validates": validates},
            metadata={"pipeline": "langgraph_ingest_v1"},
        ) as trace:
            state = init_ingest_state(
                pdf_path=str(pdf_path),
                validates=validates,
                trace=trace,
            )
            # Inject non-schema keys for audit path and tracer access.
            state["_audit_output_dir"] = (  # type: ignore
                str(audit_output_dir) if audit_output_dir else None
            )

            log.info("Executing LangGraph ingest workflow...")
            final_state = self.graph.invoke(state)

            accepted = final_state.get("accepted_chunks", [])
            stats = final_state.get("stats", {})

            log.info("Pipeline statistics:")
            log.info("  - Raw chunks parsed  : %d", stats.get("raw_count", 0))
            log.info("  - Extracted chunks   : %d", stats.get("extracted_count", 0))
            log.info("  - Accepted chunks    : %d", stats.get("accepted_count", len(accepted)))
            log.info("  - Corrected chunks   : %d", stats.get("corrected_count", 0))
            log.info("  - Discarded chunks   : %d", stats.get("discarded_count", 0))

            for warning in final_state.get("warnings", []):
                log.warning("  %s", warning)
            for error in final_state.get("errors", []):
                log.error("  %s", error)

        log.info("=" * 70)
        log.info("LangGraph ingest complete")
        log.info("=" * 70 + "\n")

        return accepted
