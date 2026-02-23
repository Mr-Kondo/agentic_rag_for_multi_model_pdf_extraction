"""
validator_agent.py  (v2)
========================
Changes from v1:
  - ValidatorAgent split into two dedicated classes:

      ChunkValidatorAgent   — CHECKPOINT A
        Model : Qwen/Qwen2-VL-7B-Instruct  (vision-language, 7B)
        Scope : validates text, table, AND figure chunks
                figures are passed as actual PIL.Image — no text-only fallback
        Load/Unload: explicit, called by pipeline before/after bulk chunk validation

      AnswerValidatorAgent  — CHECKPOINT B
        Model : any ~10B text LLM (same class as orchestrator)
        Scope : hallucination check on final RAGAnswer vs source chunk texts
        Load/Unload: explicit, called by pipeline around the answer validation step

  - Both classes share BaseLoadableModel mixin:
      .load()        → initialise model + processor/tokenizer into GPU/CPU memory
      .unload()      → delete model objects + torch.cuda.empty_cache()
      .is_loaded     → bool property
      Context manager support: `with validator: ...` → auto load/unload

  - Qwen2-VL interface differs from standard text pipelines:
      Uses AutoProcessor + Qwen2VLForConditionalGeneration
      Images passed via processor(images=[img]) as pixel_values
      Text + image interleaved in message content list

Memory profile (FP16):
  ChunkValidatorAgent  (Qwen2-VL-7B)   : ~14 GB VRAM
  AnswerValidatorAgent (Qwen3-8B)      : ~16 GB VRAM
  OrchestratorAgent    (DeepSeek-R1-8B): ~16 GB VRAM
  → Sequential load/unload: never more than one ~16 GB model in memory at a time.
"""

from __future__ import annotations

import gc
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mlx_lm import generate, load
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as vlm_load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from agentic_rag_flow import ProcessedChunk, RAGAnswer, RawChunk
    from langfuse_tracer import _TraceHandle

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. RESULT DATACLASSES  (unchanged from v1)
# ═══════════════════════════════════════════════════════════


@dataclass
class ChunkValidationResult:
    """
    Result of Checkpoint A: chunk extraction quality audit.
    is_valid=False triggers correction or discard in the pipeline.
    """

    is_valid: bool
    issues: list[str] = field(default_factory=list)
    corrected: "ProcessedChunk | None" = None
    verdict_score: float = 1.0
    validator_notes: str = ""


@dataclass
class AnswerValidationResult:
    """
    Result of Checkpoint B: hallucination / grounding check.
    is_grounded=False triggers revised_answer substitution or warning prefix.
    """

    is_grounded: bool
    hallucinations: list[str] = field(default_factory=list)
    revised_answer: str | None = None
    verdict_score: float = 1.0
    validator_notes: str = ""


# ═══════════════════════════════════════════════════════════
# 2. BASE LOADABLE MODEL  (load / unload lifecycle mixin)
# ═══════════════════════════════════════════════════════════


class BaseLoadableModel:
    """
    Mixin that provides explicit load/unload lifecycle for heavy LLMs.

    Subclasses must implement:
        _do_load()    → initialise self._model and self._processor / self._tokenizer
        _do_unload()  → delete those references (do NOT call gc here)

    Usage patterns:

        # 1. Explicit load/unload
        agent.load()
        result = agent.run(...)
        agent.unload()

        # 2. Context manager  (preferred — guarantees unload even on exception)
        with agent:
            result = agent.run(...)

        # 3. Pipeline helper  (see AgenticRAGPipeline._run_with_model)
        with agent:
            results = [agent.run(x) for x in items]
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._loaded = False

    # ── Lifecycle ──────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            log.debug("%s already loaded — skipping.", self.__class__.__name__)
            return
        log.info("[LOAD]   %s  model=%s", self.__class__.__name__, self.model_id)
        self._do_load()
        self._loaded = True
        log.info("[READY]  %s", self.__class__.__name__)

    def unload(self) -> None:
        if not self._loaded:
            return
        log.info("[UNLOAD] %s  model=%s", self.__class__.__name__, self.model_id)
        self._do_unload()
        self._loaded = False
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Context manager ────────────────────────────────────

    def __enter__(self) -> "BaseLoadableModel":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    # ── Subclass interface ─────────────────────────────────

    def _do_load(self) -> None:
        raise NotImplementedError

    def _do_unload(self) -> None:
        raise NotImplementedError

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} is not loaded. Call .load() or use 'with agent:' context manager before inference."
            )

    # ── Shared utilities ───────────────────────────────────

    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    _JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

    @classmethod
    def _safe_json(cls, text: str) -> dict:
        """Strip <think> blocks then extract first JSON object."""
        text = cls._THINK_RE.sub("", text).strip()
        match = cls._JSON_RE.search(text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _log_generation(
        self,
        trace: "_TraceHandle | None",
        span_name: str,
        messages: list[dict],
        output: str,
    ) -> None:
        """
        Post a Langfuse Generation node if a trace handle is active.
        Uses tokenizer for accurate token counting if available.
        """
        if trace is None:
            return

        # Try to use tokenizer for accurate token counting
        input_tokens = None
        output_tokens = None

        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            try:
                # Accurate token counting using tokenizer
                # apply_chat_template returns token IDs array
                prompt_tokens = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                input_tokens = (
                    len(prompt_tokens) if isinstance(prompt_tokens, list) else len(self._tokenizer.encode(prompt_tokens))
                )
                # Rough estimate for output: ~1 token per word
                output_tokens = len(output.split())
            except Exception:
                # Fallback to word count if tokenizer fails
                prompt_flat = " ".join(m["content"] if isinstance(m["content"], str) else str(m["content"]) for m in messages)
                input_tokens = len(prompt_flat.split())
                output_tokens = len(output.split())
        else:
            # Fallback to word count for VLM models without direct tokenizer access
            prompt_flat = " ".join(m["content"] if isinstance(m["content"], str) else str(m["content"]) for m in messages)
            input_tokens = len(prompt_flat.split())
            output_tokens = len(output.split())

        with trace.generation(
            name=span_name,
            model=self.model_id,
            input={"messages": [{**m, "content": str(m["content"])[:300]} for m in messages]},
            model_params={"do_sample": False},
            metadata={"role": self.__class__.__name__},
        ) as g:
            g.set_output(output, input_tokens=input_tokens, output_tokens=output_tokens)


# ═══════════════════════════════════════════════════════════
# 3. CHUNK VALIDATOR AGENT  — CHECKPOINT A
#    Model : Qwen/Qwen2-VL-7B-Instruct  (vision-language)
#    Key   : passes actual PIL.Image for figure chunks
#            → can detect figure_type errors the text agents cannot
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
    CHECKPOINT A validator.

    Qwen2-VL-7B can inspect figure images directly with its vision encoder.
    Text / table chunks are validated via text-only messages.

    Uses mlx-vlm for optimized Apple Silicon inference.
    """

    def _do_load(self) -> None:
        try:
            self._model, self._processor = vlm_load(self.model_id)
            self._config = load_config(self.model_id)
        except TypeError as e:
            # Handle transformers library incompatibility
            if "NoneType" in str(e) or "iterable" in str(e):
                import logging

                log = logging.getLogger(__name__)
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
        del self._model
        del self._processor
        del self._config

    # ── Public API ─────────────────────────────────────────

    def validate_chunk(
        self,
        raw: "RawChunk",
        processed: "ProcessedChunk",
        trace: "_TraceHandle | None" = None,
    ) -> ChunkValidationResult:
        """
        Validate ProcessedChunk against its RawChunk source.
        - FIGURE chunks  → image passed directly to vision encoder
        - TEXT/TABLE chunks → text-only message
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
        """Infer on figure using mlx-vlm."""
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
        """Infer on text using vision model in text-only mode."""
        user_text = f"[ORIGINAL]\n{original_text}\n\n[EXTRACTED]\n{extracted_repr}\n\nReturn only the JSON verdict."

        # Combine system message with user text
        full_prompt = f"{_CHUNK_VALIDATOR_SYSTEM}\n\n{user_text}"

        # Use the vision model in text-only mode (no images)
        prompt = apply_chat_template(self._processor, self._config, full_prompt, num_images=0)

        return vlm_generate(self._model, self._processor, prompt, verbose=False)

    # ── Result builder ──────────────────────────────────────

    def _build_result(self, parsed: dict, original: "ProcessedChunk") -> ChunkValidationResult:
        from agentic_rag_flow import ProcessedChunk as PC

        is_valid = bool(parsed.get("is_valid", True))
        issues = parsed.get("issues", [])
        verdict_score = float(parsed.get("verdict_score", 1.0))
        notes = parsed.get("validator_notes", "")

        corrected = None
        if not is_valid:
            corrected = PC(
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
# 4. ANSWER VALIDATOR AGENT  — CHECKPOINT B
#    Model : ~10B text LLM  (Qwen3-8B recommended)
#    Key   : hallucination detection on RAGAnswer vs source chunks
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
    CHECKPOINT B validator.
    Text-only 10B model: verifies every claim in the orchestrator's answer
    is supported by retrieved source chunk texts.

    Uses mlx-lm for optimized Apple Silicon inference.
    """

    def _do_load(self) -> None:
        self._model, self._tokenizer = load(self.model_id)

    def _do_unload(self) -> None:
        del self._model
        del self._tokenizer

    # ── Public API ─────────────────────────────────────────

    def validate_answer(
        self,
        question: str,
        answer: "RAGAnswer",
        source_texts: list[str],
        trace: "_TraceHandle | None" = None,
    ) -> AnswerValidationResult:
        self._assert_loaded()

        sources_repr = "\n\n".join(f"[Source {i + 1}] {text[:600]}" for i, text in enumerate(source_texts))
        user_content = f"[QUESTION]\n{question}\n\n[ANSWER]\n{answer.answer}\n\n[SOURCES]\n{sources_repr}"
        messages = [
            {"role": "system", "content": _ANSWER_VALIDATOR_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        output = generate(self._model, self._tokenizer, prompt=prompt, max_tokens=1024, verbose=False)
        output = self._THINK_RE.sub("", output).strip()

        self._log_generation(trace, "answer_validate", messages, output)

        parsed = self._safe_json(output)
        return AnswerValidationResult(
            is_grounded=bool(parsed.get("is_grounded", True)),
            hallucinations=parsed.get("hallucinations", []),
            revised_answer=parsed.get("revised_answer"),
            verdict_score=float(parsed.get("verdict_score", 1.0)),
            validator_notes=parsed.get("validator_notes", ""),
        )
