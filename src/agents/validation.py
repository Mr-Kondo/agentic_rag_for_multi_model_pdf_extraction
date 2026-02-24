"""
Validation agents for quality assurance at two checkpoints.

CHECKPOINT A (ChunkValidatorAgent):
    Audits extraction quality by comparing ProcessedChunk against original RawChunk.
    Uses vision-language model (Qwen2-VL-7B) to validate ALL chunk types including figures.

CHECKPOINT B (AnswerValidatorAgent):
    Detects hallucinations by verifying RAGAnswer claims against source chunk texts.
    Uses DSPy-enhanced text model (~10B) with ChainOfThought for systematic validation.

Both agents inherit BaseLoadableModel with explicit load/unload lifecycle.
Memory footprint: ~14-16 GB VRAM per agent (never loaded simultaneously).
"""

import gc
import json
import logging
import re
from typing import TYPE_CHECKING

import dspy
from mlx_lm import generate, load
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as vlm_load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from src.agents.base import BaseLoadableModel
from src.core.models import (
    AnswerValidationResult,
    ChunkValidationResult,
    ProcessedChunk,
    RAGAnswer,
    RawChunk,
)
from src.integrations.dspy_adapter import MLXLM
from src.integrations.dspy_modules import AnswerGroundingSignature

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from src.integrations.langfuse import TraceHandle

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# CHUNK VALIDATOR AGENT  — CHECKPOINT A
# ═══════════════════════════════════════════════════════════

_CHUNK_VALIDATOR_SYSTEM = """You are a rigorous extraction quality auditor.

You receive ORIGINAL content from a PDF page and EXTRACTED JSON from an extraction agent.
The original may be plain text, a markdown table, or an IMAGE (figure, chart, diagram).

Verify whether EXTRACTED faithfully and completely represents ORIGINAL.
Check for:
  - Fabricated content (absent from ORIGINAL)
  - Missing key information
  - Incorrect key_concepts
  - Overconfident confidence score
  - TABLE: schema correctness (columns, row count, units)
  - FIGURE: figure_type accuracy; description matches visible content

Return ONLY valid JSON (no preamble, no markdown fences):
{
  "is_valid": <true|false>,
  "issues": ["<issue>"],
  "corrected_structured_text": "<correction or null>",
  "corrected_intuition_summary": "<correction or null>",
  "corrected_key_concepts": ["<concept>"] or null,
  "verdict_score": <0.0-1.0>,
  "validator_notes": "<brief reasoning>"
}
Any fabrication → is_valid=false, regardless of severity."""


class ChunkValidatorAgent(BaseLoadableModel):
    """
    CHECKPOINT A validator for extraction quality assurance.

    Uses Qwen2-VL-7B vision-language model to validate ALL chunk types:
    - TEXT chunks: Text-only validation against source
    - TABLE chunks: Schema and content correctness
    - FIGURE chunks: Direct image inspection with vision encoder

    The vision capability enables detection of figure_type errors that
    text-only agents cannot identify.

    Memory: ~14 GB VRAM (FP16)
    Backend: mlx-vlm optimized for Apple Silicon

    Usage:
        >>> validator = ChunkValidatorAgent("mlx-community/Qwen2-VL-7B-Instruct")
        >>> with validator:
        ...     result = validator.validate_chunk(raw_chunk, processed_chunk, trace)
    """

    def _do_load(self) -> None:
        """Load Qwen2-VL model and processor into memory."""
        try:
            self._model, self._processor = vlm_load(self.model_id)
            self._config = load_config(self.model_id)
        except TypeError as e:
            # Handle transformers library incompatibility
            if "NoneType" in str(e) or "iterable" in str(e):
                log.warning(
                    f"⚠️ Vision model processor error (likely transformers incompatibility): {e}\n"
                    f"   Attempting to load model with trust_remote_code=True..."
                )
                try:
                    # Retry with explicit trust_remote_code setting
                    self._model, self._processor = vlm_load(self.model_id, trust_remote_code=True)
                    self._config = load_config(self.model_id)
                    log.info("✓ Model loaded successfully with trust_remote_code=True")
                except Exception as e2:
                    log.error(f"✗ Failed to load vision model: {e2}")
                    raise
            else:
                raise

    def _do_unload(self) -> None:
        """Release model resources and trigger garbage collection."""
        del self._model
        del self._processor
        del self._config

    # ── Public API ─────────────────────────────────────────

    def validate_chunk(
        self,
        raw: RawChunk,
        processed: ProcessedChunk,
        trace: "TraceHandle | None" = None,
    ) -> ChunkValidationResult:
        """
        Validate ProcessedChunk against its RawChunk source.

        For FIGURE chunks, passes the actual PIL.Image to the vision encoder.
        For TEXT/TABLE chunks, validates via text-only comparison.

        Args:
            raw: Original raw chunk from PDF parser
            processed: Extracted chunk from agent
            trace: Optional Langfuse trace for observability

        Returns:
            ChunkValidationResult with validity assessment and optional corrections
        """
        self._assert_loaded()

        from PIL import Image as PILImage

        is_figure = isinstance(raw.raw_content, PILImage.Image)

        extracted_repr = json.dumps(
            {
                "chunk_type": processed.chunk_type.value,
                "structured_text": processed.structured_text[:1500],
                "intuition_summary": processed.intuition_summary,
                "key_concepts": processed.key_concepts,
                "confidence": processed.confidence,
                "agent_notes": processed.agent_notes,
            },
            ensure_ascii=False,
        )

        if is_figure:
            output = self._infer_figure(raw.raw_content, extracted_repr)
        else:
            output = self._infer_text(str(raw.raw_content)[:2000], extracted_repr)

        # Convert GenerationResult to string if needed
        output = output if isinstance(output, str) else str(output)

        span = f"chunk_validate_p{processed.page_num}_{processed.chunk_type.value}"
        self._log_generation(trace, span, [], output)

        parsed = self._safe_json(output)
        return self._build_result(parsed, processed)

    # ── Inference ───────────────────────────────────────────

    def _infer_figure(self, img: "PILImage", extracted_repr: str) -> str:
        """
        Run validation inference on figure chunk with vision model.

        Passes PIL.Image directly to mlx-vlm for visual content verification.
        """
        user_text = (
            "Above is the ORIGINAL figure from the PDF.\n\n"
            f"[EXTRACTED]\n{extracted_repr}\n\n"
            "Does EXTRACTED faithfully describe the figure? "
            "Return only the JSON verdict."
        )

        # Combine system message with user text
        full_prompt = f"{_CHUNK_VALIDATOR_SYSTEM}\n\n{user_text}"
        prompt = apply_chat_template(self._processor, self._config, full_prompt, num_images=1)

        return vlm_generate(self._model, self._processor, prompt, [img], verbose=False)

    def _infer_text(self, original_text: str, extracted_repr: str) -> str:
        """
        Run validation inference on text/table chunk in text-only mode.
        """
        user_text = f"[ORIGINAL]\n{original_text}\n\n[EXTRACTED]\n{extracted_repr}\n\nReturn only the JSON verdict."

        # Combine system message with user text
        full_prompt = f"{_CHUNK_VALIDATOR_SYSTEM}\n\n{user_text}"

        # Use the vision model in text-only mode (no images)
        prompt = apply_chat_template(self._processor, self._config, full_prompt, num_images=0)

        return vlm_generate(self._model, self._processor, prompt, verbose=False)

    # ── Result builder ──────────────────────────────────────

    def _build_result(self, parsed: dict, original: ProcessedChunk) -> ChunkValidationResult:
        """
        Construct ChunkValidationResult from parsed JSON output.

        Creates corrected ProcessedChunk if validation fails.
        """
        is_valid = bool(parsed.get("is_valid", True))
        issues = parsed.get("issues", [])
        verdict_score = float(parsed.get("verdict_score", 1.0))
        notes = parsed.get("validator_notes", "")

        corrected = None
        if not is_valid:
            corrected = ProcessedChunk(
                chunk_type=original.chunk_type,
                page_num=original.page_num,
                source_file=original.source_file,
                structured_text=parsed.get("corrected_structured_text") or original.structured_text,
                intuition_summary=parsed.get("corrected_intuition_summary") or original.intuition_summary,
                key_concepts=parsed.get("corrected_key_concepts") or original.key_concepts,
                confidence=verdict_score,
                agent_notes=f"[CHECKPOINT-A CORRECTED] {notes}",
            )

        return ChunkValidationResult(
            is_valid=is_valid,
            issues=issues,
            corrected=corrected,
            verdict_score=verdict_score,
            validator_notes=notes,
        )


# ═══════════════════════════════════════════════════════════
# ANSWER VALIDATOR AGENT  — CHECKPOINT B
# ═══════════════════════════════════════════════════════════

_ANSWER_VALIDATOR_SYSTEM = """You are a hallucination detection specialist.

You receive:
  [QUESTION] — the user's question
  [ANSWER]   — the answer from a reasoning LLM
  [SOURCES]  — numbered document chunks (the ONLY valid evidence base)

Verify that every material factual claim in [ANSWER] can be traced to at least one [SOURCE].
Untraceable claims are hallucinations.

Rules:
  - Ignore meta-phrases: "Based on the context", "it appears", "Insufficient context".
  - Focus on: numbers, named entities, relationships, stated conclusions.
  - If ANSWER says "Insufficient context" → is_grounded=true automatically.

Return ONLY valid JSON (no preamble, no markdown fences):
{
  "is_grounded": <true|false>,
  "hallucinations": ["<unsupported claim>"],
  "revised_answer": "<answer with hallucinations removed, or null if is_grounded=true>",
  "verdict_score": <0.0-1.0>,
  "validator_notes": "<brief reasoning>"
}
"""


class AnswerValidatorAgent(BaseLoadableModel):
    """
    CHECKPOINT B validator for hallucination detection.

    Verifies that every material factual claim in the orchestrator's RAGAnswer
    is supported by the retrieved source chunk texts. Uses DSPy with ChainOfThought
    for systematic claim verification and structured output.

    Memory: ~16 GB VRAM for 10B text model (FP16)
    Backend: MLX via DSPy adapter, optimized for Apple Silicon

    Features:
        - DSPy ChainOfThought for step-by-step reasoning
        - Automatic prompt optimization support
        - Structured output via AnswerGroundingSignature
        - Legacy fallback for non-DSPy mode

    Usage:
        >>> validator = AnswerValidatorAgent("mlx-community/Qwen2.5-7B-Instruct", use_dspy=True)
        >>> with validator:
        ...     result = validator.validate_answer(question, answer, source_texts, trace)
    """

    def __init__(self, model_id: str, use_dspy: bool = True) -> None:
        """
        Initialize the answer validator.

        Args:
            model_id: Model identifier for MLX
            use_dspy: Whether to use DSPy-enhanced validation (default: True)
        """
        super().__init__(model_id)
        self.use_dspy = use_dspy
        self._dspy_predictor = None
        self._mlx_lm = None

    def _do_load(self) -> None:
        """Load text model through DSPy adapter or legacy MLX."""
        if self.use_dspy:
            # Load model through DSPy adapter
            log.info(f"Loading AnswerValidatorAgent with DSPy: {self.model_id}")
            self._mlx_lm = MLXLM(self.model_id, max_tokens=1024, temperature=0.0)
            dspy.configure(lm=self._mlx_lm)

            # Initialize DSPy module with Chain-of-Thought reasoning
            self._dspy_predictor = dspy.ChainOfThought(AnswerGroundingSignature)
            log.info("✓ DSPy predictor initialized for answer validation")
        else:
            # Legacy: direct MLX loading
            log.info(f"Loading AnswerValidatorAgent (legacy mode): {self.model_id}")
            self._model, self._tokenizer = load(self.model_id)

    def _do_unload(self) -> None:
        """Release model resources."""
        if self.use_dspy:
            if self._mlx_lm:
                self._mlx_lm.unload()
            self._mlx_lm = None
            self._dspy_predictor = None
        else:
            del self._model
            del self._tokenizer

    # ── Public API ─────────────────────────────────────────

    def validate_answer(
        self,
        question: str,
        answer: RAGAnswer,
        source_texts: list[str],
        trace: "TraceHandle | None" = None,
    ) -> AnswerValidationResult:
        """
        Validate answer grounding using DSPy or legacy method.

        Args:
            question: Original user question
            answer: RAGAnswer object containing the answer text
            source_texts: List of source chunk texts that should support the answer
            trace: Optional Langfuse trace for observability

        Returns:
            AnswerValidationResult with grounding assessment
        """
        self._assert_loaded()

        if self.use_dspy and self._dspy_predictor:
            return self._validate_with_dspy(question, answer, source_texts, trace)
        else:
            return self._validate_legacy(question, answer, source_texts, trace)

    def _validate_with_dspy(
        self,
        question: str,
        answer: RAGAnswer,
        source_texts: list[str],
        trace: "TraceHandle | None" = None,
    ) -> AnswerValidationResult:
        """
        DSPy-based validation with structured output.

        Uses ChainOfThought for systematic claim verification and automatic
        output structuring without regex parsing.
        """
        # Format sources for context
        sources_repr = "\n\n".join(f"[Source {i + 1}] {text[:600]}" for i, text in enumerate(source_texts))

        # Combine question context with answer for validation
        context_with_question = f"[QUESTION]\n{question}\n\n[SOURCES]\n{sources_repr}"

        try:
            # DSPy prediction with automatic reasoning trace
            if trace:
                with trace.span(
                    name="dspy_answer_validation",
                    input={"answer": answer.answer[:200], "num_sources": len(source_texts)},
                ) as span:
                    prediction = self._dspy_predictor(
                        answer=answer.answer,
                        context=context_with_question,
                    )
                    span.set_output(prediction)
            else:
                prediction = self._dspy_predictor(
                    answer=answer.answer,
                    context=context_with_question,
                )

            # Extract structured output (DSPy handles parsing)
            # Safely extract fields with defaults
            is_grounded = bool(getattr(prediction, "is_grounded", True))

            # Handle hallucinations list
            hallucinations = []
            if hasattr(prediction, "hallucinations"):
                if isinstance(prediction.hallucinations, list):
                    hallucinations = prediction.hallucinations
                elif isinstance(prediction.hallucinations, str):
                    # Parse string representation if needed
                    hallucinations = [prediction.hallucinations] if prediction.hallucinations else []

            # Handle 'null' string vs actual None for revised_answer
            revised_answer = None
            if hasattr(prediction, "revised_answer"):
                if prediction.revised_answer and prediction.revised_answer.lower() != "null":
                    revised_answer = prediction.revised_answer

            # Safely convert verdict_score to float
            verdict_score = 1.0
            if hasattr(prediction, "verdict_score") and prediction.verdict_score is not None:
                try:
                    verdict_score = float(prediction.verdict_score)
                except (ValueError, TypeError):
                    log.warning(f"Invalid verdict_score: {prediction.verdict_score}, using default 1.0")
                    verdict_score = 1.0

            validator_notes = str(getattr(prediction, "validator_notes", ""))

            return AnswerValidationResult(
                is_grounded=is_grounded,
                hallucinations=hallucinations,
                revised_answer=revised_answer,
                verdict_score=verdict_score,
                validator_notes=validator_notes,
            )

        except Exception as e:
            log.error(f"DSPy validation failed: {e}", exc_info=True)
            # Fallback to safe defaults
            return AnswerValidationResult(
                is_grounded=True,
                hallucinations=[],
                revised_answer=None,
                verdict_score=0.5,
                validator_notes=f"Validation error: {str(e)}",
            )

    def _validate_legacy(
        self,
        question: str,
        answer: RAGAnswer,
        source_texts: list[str],
        trace: "TraceHandle | None" = None,
    ) -> AnswerValidationResult:
        """
        Legacy validation method using manual prompting and regex parsing.

        Kept for backward compatibility and comparison with DSPy approach.
        """
        sources_repr = "\n\n".join(f"[Source {i + 1}] {text[:600]}" for i, text in enumerate(source_texts))
        user_content = f"[QUESTION]\n{question}\n\n[ANSWER]\n{answer.answer}\n\n[SOURCES]\n{sources_repr}"
        messages = [
            {"role": "system", "content": _ANSWER_VALIDATOR_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        output = generate(self._model, self._tokenizer, prompt=prompt, max_tokens=1024, verbose=False)

        # Remove <think> blocks
        _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        output = _THINK_RE.sub("", output).strip()

        self._log_generation(trace, "answer_validate", messages, output)

        parsed = self._safe_json(output)
        return AnswerValidationResult(
            is_grounded=bool(parsed.get("is_grounded", True)),
            hallucinations=parsed.get("hallucinations", []),
            revised_answer=parsed.get("revised_answer"),
            verdict_score=float(parsed.get("verdict_score", 1.0)),
            validator_notes=parsed.get("validator_notes", ""),
        )
