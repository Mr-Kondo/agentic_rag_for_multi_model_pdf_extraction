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

import torch

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from agentic_rag_flow_v3 import ProcessedChunk, RAGAnswer, RawChunk
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
    is_valid        : bool
    issues          : list[str]               = field(default_factory=list)
    corrected       : "ProcessedChunk | None" = None
    verdict_score   : float                   = 1.0
    validator_notes : str                     = ""


@dataclass
class AnswerValidationResult:
    """
    Result of Checkpoint B: hallucination / grounding check.
    is_grounded=False triggers revised_answer substitution or warning prefix.
    """
    is_grounded     : bool
    hallucinations  : list[str]  = field(default_factory=list)
    revised_answer  : str | None = None
    verdict_score   : float      = 1.0
    validator_notes : str        = ""


# ═══════════════════════════════════════════════════════════
# 2. BASE LOADABLE MODEL  (load / unload lifecycle mixin)
# ═══════════════════════════════════════════════════════════

class BaseLoadableModel:
    """
    Mixin that provides explicit load/unload lifecycle for heavy LLMs.

    Subclasses must implement:
        _do_load()    → initialise self._model and self._processor / self._pipe
        _do_unload()  → delete those references (do NOT call gc or cuda.empty_cache here)

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

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device   = device
        self._loaded  = False

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.debug("CUDA cache cleared after %s unload.", self.__class__.__name__)

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
                f"{self.__class__.__name__} is not loaded. "
                "Call .load() or use 'with agent:' context manager before inference."
            )

    # ── Shared utilities ───────────────────────────────────

    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    _JSON_RE  = re.compile(r"\{.*\}", re.DOTALL)

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
        trace    : "_TraceHandle | None",
        span_name: str,
        messages : list[dict],
        output   : str,
    ) -> None:
        """Post a Langfuse Generation node if a trace handle is active."""
        if trace is None:
            return
        prompt_flat   = " ".join(
            m["content"] if isinstance(m["content"], str) else str(m["content"])
            for m in messages
        )
        with trace.generation(
            name         = span_name,
            model        = self.model_id,
            input        = {"messages": [
                {**m, "content": str(m["content"])[:300]} for m in messages
            ]},
            model_params = {"do_sample": False},
            metadata     = {"role": self.__class__.__name__},
        ) as g:
            g.set_output(
                output,
                input_tokens  = len(prompt_flat.split()),
                output_tokens = len(output.split()),
            )


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

    Model loading uses Qwen2VLForConditionalGeneration (NOT the generic pipeline),
    which is required for correct image token injection.
    """

    def _do_load(self) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype      = torch.float16 if self.device != "cpu" else torch.float32,
            device_map       = self.device,
            trust_remote_code= True,
        )
        self._model.eval()

    def _do_unload(self) -> None:
        del self._model
        del self._processor

    # ── Public API ─────────────────────────────────────────

    def validate_chunk(
        self,
        raw      : "RawChunk",
        processed: "ProcessedChunk",
        trace    : "_TraceHandle | None" = None,
    ) -> ChunkValidationResult:
        """
        Validate ProcessedChunk against its RawChunk source.
        - FIGURE chunks  → image passed directly to vision encoder
        - TEXT/TABLE chunks → text-only message
        """
        self._assert_loaded()

        from PIL import Image as PILImage
        is_figure = isinstance(raw.raw_content, PILImage.Image)

        extracted_repr = json.dumps({
            "chunk_type"       : processed.chunk_type.value,
            "structured_text"  : processed.structured_text[:1500],
            "intuition_summary": processed.intuition_summary,
            "key_concepts"     : processed.key_concepts,
            "confidence"       : processed.confidence,
            "agent_notes"      : processed.agent_notes,
        }, ensure_ascii=False)

        if is_figure:
            messages, images = self._figure_messages(raw.raw_content, extracted_repr)
        else:
            messages = self._text_messages(str(raw.raw_content)[:2000], extracted_repr)
            images   = None

        output = self._infer(messages, images)
        span   = f"chunk_validate_p{processed.page_num}_{processed.chunk_type.value}"
        self._log_generation(trace, span, messages, output)

        parsed = self._safe_json(output)
        return self._build_result(parsed, processed)

    # ── Message builders ────────────────────────────────────

    @staticmethod
    def _figure_messages(
        img           : "PILImage",
        extracted_repr: str,
    ) -> tuple[list[dict], list]:
        """
        Qwen2-VL multimodal message: image token + extracted JSON side by side.
        The model sees the actual figure pixels when auditing the agent's description.
        """
        messages = [
            {"role": "system", "content": _CHUNK_VALIDATOR_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {
                        "type": "text",
                        "text": (
                            "Above is the ORIGINAL figure from the PDF.\n\n"
                            f"[EXTRACTED]\n{extracted_repr}\n\n"
                            "Does EXTRACTED faithfully describe the figure? "
                            "Return only the JSON verdict."
                        ),
                    },
                ],
            },
        ]
        return messages, [img]

    @staticmethod
    def _text_messages(original_text: str, extracted_repr: str) -> list[dict]:
        return [
            {"role": "system", "content": _CHUNK_VALIDATOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"[ORIGINAL]\n{original_text}\n\n"
                    f"[EXTRACTED]\n{extracted_repr}\n\n"
                    "Return only the JSON verdict."
                ),
            },
        ]

    # ── Inference ───────────────────────────────────────────

    def _infer(self, messages: list[dict], images: list | None) -> str:
        """
        Qwen2-VL generation.
        apply_chat_template inserts the <|image_pad|> tokens at the correct position.
        Prompt tokens are stripped from the output before returning.
        """
        text_prompt = self._processor.apply_chat_template(
            messages,
            tokenize             = False,
            add_generation_prompt= True,
        )
        if images:
            inputs = self._processor(
                text=text_prompt, images=images, return_tensors="pt", padding=True
            )
        else:
            inputs = self._processor(
                text=text_prompt, return_tensors="pt", padding=True
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=1024, do_sample=False
            )
        # Strip input prompt tokens from output
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens           = True,
            clean_up_tokenization_spaces  = False,
        )[0]

    # ── Result builder ──────────────────────────────────────

    def _build_result(self, parsed: dict, original: "ProcessedChunk") -> ChunkValidationResult:
        from agentic_rag_flow_v3 import ProcessedChunk as PC

        is_valid      = bool(parsed.get("is_valid", True))
        issues        = parsed.get("issues", [])
        verdict_score = float(parsed.get("verdict_score", 1.0))
        notes         = parsed.get("validator_notes", "")

        corrected = None
        if not is_valid:
            corrected = PC(
                chunk_type       = original.chunk_type,
                page_num         = original.page_num,
                source_file      = original.source_file,
                structured_text  = parsed.get("corrected_structured_text") or original.structured_text,
                intuition_summary= parsed.get("corrected_intuition_summary") or original.intuition_summary,
                key_concepts     = parsed.get("corrected_key_concepts") or original.key_concepts,
                confidence       = verdict_score,
                agent_notes      = f"[CHECKPOINT-A CORRECTED] {notes}",
            )

        return ChunkValidationResult(
            is_valid        = is_valid,
            issues          = issues,
            corrected       = corrected,
            verdict_score   = verdict_score,
            validator_notes = notes,
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
    """

    def _do_load(self) -> None:
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline(
            "text-generation",
            model          = self.model_id,
            device         = self.device,
            max_new_tokens = 1024,
            do_sample      = False,
        )

    def _do_unload(self) -> None:
        del self._pipe

    # ── Public API ─────────────────────────────────────────

    def validate_answer(
        self,
        question    : str,
        answer      : "RAGAnswer",
        source_texts: list[str],
        trace       : "_TraceHandle | None" = None,
    ) -> AnswerValidationResult:
        self._assert_loaded()

        sources_repr = "\n\n".join(
            f"[Source {i+1}] {text[:600]}"
            for i, text in enumerate(source_texts)
        )
        user_content = (
            f"[QUESTION]\n{question}\n\n"
            f"[ANSWER]\n{answer.answer}\n\n"
            f"[SOURCES]\n{sources_repr}"
        )
        messages = [
            {"role": "system", "content": _ANSWER_VALIDATOR_SYSTEM},
            {"role": "user",   "content": user_content},
        ]

        raw    = self._pipe(messages)[0]["generated_text"]
        output = raw[-1]["content"] if isinstance(raw, list) else str(raw)
        output = self._THINK_RE.sub("", output).strip()

        self._log_generation(trace, "answer_validate", messages, output)

        parsed = self._safe_json(output)
        return AnswerValidationResult(
            is_grounded     = bool(parsed.get("is_grounded", True)),
            hallucinations  = parsed.get("hallucinations", []),
            revised_answer  = parsed.get("revised_answer"),
            verdict_score   = float(parsed.get("verdict_score", 1.0)),
            validator_notes = parsed.get("validator_notes", ""),
        )
