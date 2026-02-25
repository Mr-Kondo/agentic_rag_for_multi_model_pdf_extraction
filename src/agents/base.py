"""
Base classes for agent implementations.

Provides BaseAgent for small always-loaded agents and BaseLoadableModel
for large models with explicit load/unload lifecycle management.
"""

from __future__ import annotations

import gc
import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.models import ProcessedChunk, RawChunk
    from src.integrations.langfuse import TraceHandle

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# BASE AGENT (Small SLMs - always loaded)
# ═══════════════════════════════════════════════════════════


class BaseAgent:
    """
    Base class for small always-loaded extraction agents.

    Provides retry logic with self-reflection when confidence is low.
    Used by TextAgent, TableAgent, VisionAgent - these 2-4B models stay
    loaded throughout ingestion.

    Attributes:
        CONFIDENCE_THRESHOLD: Minimum confidence to skip retry (0-1)
        RETRY_SUFFIX: Text appended to prompt on retry attempt
        model_id: HuggingFace model identifier
    """

    CONFIDENCE_THRESHOLD = 0.5
    RETRY_SUFFIX = "\n[RETRY] Low confidence. Be conservative; flag unknowns explicitly."

    def __init__(self, model_id: str):
        """
        Initialize agent and load model.

        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id
        self._load_model()

    def _load_model(self):
        """Load model into memory. Must be implemented by subclass."""
        raise NotImplementedError

    def process(self, chunk: "RawChunk", trace: "TraceHandle | None" = None) -> "ProcessedChunk":
        """
        Process a raw chunk with retry logic.

        Args:
            chunk: Raw content to process
            trace: Optional Langfuse trace handle for observability

        Returns:
            Processed chunk with structured data
        """
        if trace:
            with trace.span(
                f"agent_{chunk.chunk_type.value}",
                input={"page": chunk.page_num},
            ) as s:
                result = self._run_with_retry(chunk, trace)
                s.update(output={"confidence": result.confidence})
        else:
            result = self._run_with_retry(chunk, None)
        return result

    def _run_with_retry(self, chunk: "RawChunk", trace: "TraceHandle | None") -> "ProcessedChunk":
        """
        Run extraction with automatic retry if confidence is low.

        Args:
            chunk: Raw content to process
            trace: Optional Langfuse trace handle

        Returns:
            Processed chunk (may be retry result if initial confidence low)
        """
        result = self._run(chunk, retry=False, trace=trace)
        if result.confidence < self.CONFIDENCE_THRESHOLD:
            log.warning("%s: retrying p.%d (conf=%.2f)", self.__class__.__name__, chunk.page_num, result.confidence)
            result = self._run(chunk, retry=True, trace=trace)
        return result

    def _run(self, chunk: "RawChunk", retry: bool = False, trace: "TraceHandle | None" = None) -> "ProcessedChunk":
        """
        Execute model inference. Must be implemented by subclass.

        Args:
            chunk: Raw content to process
            retry: Whether this is a retry attempt
            trace: Optional Langfuse trace handle

        Returns:
            Processed chunk with structured data
        """
        raise NotImplementedError

    @staticmethod
    def _safe_json(text: str) -> dict:
        """
        Extract JSON object from text, handling malformed responses.

        Args:
            text: Text containing JSON (possibly with surrounding text)

        Returns:
            Parsed dict, or empty dict if no valid JSON found
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _last_content(output: Any) -> str:
        """
        Extract content string from various output formats.

        Args:
            output: Model output (string, list, or dict)

        Returns:
            Extracted text content
        """
        if isinstance(output, list) and output:
            last = output[-1]
            if isinstance(last, dict):
                return last.get("content", str(last))
        return str(output)


# ═══════════════════════════════════════════════════════════
# BASE LOADABLE MODEL (Large models - explicit lifecycle)
# ═══════════════════════════════════════════════════════════


class BaseLoadableModel:
    """
    Mixin that provides explicit load/unload lifecycle for heavy LLMs.

    Used by large models (8B+) that cannot stay in memory simultaneously.
    Enables sequential loading to keep VRAM usage under control.

    Subclasses must implement:
        _do_load(): Initialize self._model and self._processor/self._tokenizer
        _do_unload(): Delete model references (gc is called automatically)

    Usage patterns:
        # 1. Explicit load/unload
        agent.load()
        result = agent.run(...)
        agent.unload()

        # 2. Context manager (preferred - guarantees cleanup)
        with agent:
            result = agent.run(...)

    Attributes:
        model_id: HuggingFace model identifier
        _loaded: Whether model is currently loaded
    """

    def __init__(self, model_id: str):
        """
        Initialize loadable model (does not load until .load() called).

        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id
        self._loaded = False

    # ── Lifecycle ──────────────────────────────────────────

    def load(self) -> None:
        """Load model into memory."""
        if self._loaded:
            log.debug("%s already loaded — skipping.", self.__class__.__name__)
            return
        log.info("[LOAD]   %s  model=%s", self.__class__.__name__, self.model_id)
        self._do_load()
        self._loaded = True
        log.info("[READY]  %s", self.__class__.__name__)

    def unload(self) -> None:
        """Unload model from memory and trigger garbage collection."""
        if not self._loaded:
            return
        log.info("[UNLOAD] %s  model=%s", self.__class__.__name__, self.model_id)
        self._do_unload()
        self._loaded = False
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._loaded

    # ── Context manager ────────────────────────────────────

    def __enter__(self) -> "BaseLoadableModel":
        """Enter context manager - loads model."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - unloads model."""
        self.unload()

    # ── Subclass interface ─────────────────────────────────

    def _do_load(self) -> None:
        """Load model implementation. Must be overridden by subclass."""
        raise NotImplementedError

    def _do_unload(self) -> None:
        """Unload model implementation. Must be overridden by subclass."""
        raise NotImplementedError

    def _assert_loaded(self) -> None:
        """
        Assert that model is loaded before inference.

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} is not loaded. Call .load() or use 'with agent:' context manager before inference."
            )

    # ── Shared utilities ───────────────────────────────────

    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    _JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

    @classmethod
    def _safe_json(cls, text: str) -> dict:
        """
        Strip <think> blocks then extract first JSON object.

        Args:
            text: Text potentially containing JSON and reasoning

        Returns:
            Parsed dict, or empty dict if no valid JSON found
        """
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
        trace: "TraceHandle | None",
        span_name: str,
        messages: list[dict],
        output: str,
    ) -> None:
        """
        Post a Langfuse Generation node if a trace handle is active.

        Uses tokenizer for accurate token counting if available.

        Args:
            trace: Optional Langfuse trace handle
            span_name: Name for the generation span
            messages: Input messages
            output: Generated output text
        """
        if trace is None:
            return

        # Try to use tokenizer for accurate token counting
        input_tokens = None
        output_tokens = None

        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            try:
                # Accurate token counting using tokenizer
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
            # No tokenizer - use word count estimation
            prompt_flat = " ".join(m["content"] if isinstance(m["content"], str) else str(m["content"]) for m in messages)
            input_tokens = len(prompt_flat.split())
            output_tokens = len(output.split())

        # Log to Langfuse
        with trace.generation(
            name=span_name,
            model=self.model_id,
            input={"messages": messages},
            model_params={},
        ) as g:
            g.set_output(output, input_tokens=input_tokens, output_tokens=output_tokens)
