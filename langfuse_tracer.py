"""
langfuse_tracer.py
==================
Thin wrapper around the Langfuse Python SDK.
Every major operation in the pipeline is wrapped as a Span or Generation.

Trace hierarchy produced per PDF ingestion:
  Trace: ingest_pdf
    ├── Span: parse_pdf
    ├── Span: agent_text     (× N)
    ├── Span: agent_table    (× M)
    ├── Span: agent_vision   (× K)
    └── Span: upsert_store

Trace hierarchy produced per query:
  Trace: rag_query
    ├── Span: retrieve_chunks
    ├── Span: retrieve_figures   (conditional)
    └── Generation: orchestrator_reasoning
          input : prompt
          output: raw + cleaned answer
          model : <model_id>
          usage : token counts (if available)

Set environment variables (or pass explicitly):
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.model import ModelUsage

log = logging.getLogger(__name__)

_ALLOWED_SCORE_DATA_TYPES = {
    "NUMERIC",
    "BOOLEAN",
    "CATEGORICAL",
    "CORRECTION",
}
_SCORE_DATA_TYPE_ALIASES = {
    "numeric": "NUMERIC",
    "boolean": "BOOLEAN",
    "categorical": "CATEGORICAL",
    "correction": "CORRECTION",
}


def _normalize_score_data_type(value: Any) -> str:
    """
    Normalize score data types to Langfuse SDK enum strings.

    Args:
        value: Raw data type value.

    Returns:
        Normalized data type string.
    """
    if value is None:
        return "NUMERIC"

    if hasattr(value, "value"):
        value = value.value

    if isinstance(value, str):
        stripped = value.strip()
        upper_value = stripped.upper()
        if upper_value in _ALLOWED_SCORE_DATA_TYPES:
            return upper_value
        alias_value = _SCORE_DATA_TYPE_ALIASES.get(stripped.lower())
        if alias_value:
            return alias_value
    else:
        string_value = str(value).strip()
        upper_value = string_value.upper()
        if upper_value in _ALLOWED_SCORE_DATA_TYPES:
            return upper_value

    return "NUMERIC"


# ──────────────────────────────────────────────
# Singleton Langfuse client
# ──────────────────────────────────────────────


def _make_client() -> Langfuse | None:
    """
    Create Langfuse client if credentials are available.
    Loads environment variables from .env file first.
    Returns None if credentials are not configured.
    """
    # Load .env file (if exists)
    load_dotenv()

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        log.warning(
            "⚠️  Langfuse credentials not found. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env file to enable tracing."
        )
        return None

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


class LangfuseTracer:
    """
    Central tracing facade for the Agentic RAG pipeline.

    Usage pattern:
        tracer  = LangfuseTracer()

        # Context-managed trace (auto-finalises on exit)
        with tracer.trace("rag_query", input={"q": question}) as t:
            with t.span("retrieve") as s:
                hits = store.query(question)
                s.end(output={"n_hits": len(hits)})

            with t.generation("orchestrator", model="Qwen3-8B", input=prompt) as g:
                answer = llm(prompt)
                g.end(output=answer)

    The underlying Langfuse objects are accessible via .raw if you need
    advanced features (scores, datasets, etc.).
    """

    def __init__(self):
        self._client: Langfuse | None = _make_client()

    # ── Trace ────────────────────────────────
    @contextmanager
    def trace(
        self,
        name: str,
        input: dict | None = None,
        metadata: dict | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Generator[_TraceHandle, None, None]:
        """
        Create a trace context using Langfuse SDK v3.14.4 API.

        Uses start_as_current_span() to ensure OpenTelemetry context propagation
        is properly set up, allowing child spans and generations to discover this
        trace as their parent.

        Args:
            name: Trace name (e.g., "ingest_pdf", "rag_query")
            input: Input data for this trace
            metadata: Additional metadata
            user_id: User ID for tracking
            session_id: Session ID for grouping traces

        Returns:
            Generator yielding a _TraceHandle for context management
        """
        if self._client is None:
            yield _TraceHandle(None, None)
            return

        # ✅ Use start_as_current_span to ensure OpenTelemetry context is set
        # This allows child spans/generations to discover this trace as parent
        with self._client.start_as_current_span(
            name=name,
            input=input or {},
            metadata=metadata or {},
        ) as span:
            # Retrieve the trace ID from current context
            trace_id = self._client.get_current_trace_id()
            log.debug(f"✓ Trace started: {name} (trace_id={trace_id})")

            handle = _TraceHandle(span, trace_id)
            try:
                yield handle
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                log.error(f"Trace error in '{name}': {e}")
                raise

    # ── Scoring ──────────────────────────────
    def score(
        self,
        trace_id: str,
        name: str,
        value: float | int,
        comment: str | None = None,
        data_type: str = "numeric",
    ) -> None:
        """
        Score a trace using Langfuse's scoring API.

        Args:
            trace_id: ID of the trace to score
            name: Name of the score (e.g., "chunk_quality", "answer_grounding")
            value: Numeric score (0.0-1.0)
            comment: Optional human-readable comment
            data_type: Type of score ("NUMERIC", "BOOLEAN", etc.)
        """
        if self._client is None:
            log.debug(f"⊘ Score skipped (no Langfuse client): {name}={value}")
            return

        normalized_data_type = _normalize_score_data_type(data_type)
        if normalized_data_type != data_type:
            log.warning(
                "Normalizing score data_type from '%s' to '%s' for score '%s'.",
                data_type,
                normalized_data_type,
                name,
            )

        try:
            self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
                data_type=normalized_data_type,
            )
            log.debug(f"✓ Score posted: {name}={value}")
        except Exception as e:
            log.warning(f"Failed to post score '{name}': {e}")


# ──────────────────────────────────────────────
# Handle objects (thin proxies around Langfuse types)
# ──────────────────────────────────────────────


class _TraceHandle:
    def __init__(self, raw, trace_id: str | None = None):
        self.raw = raw
        self.trace_id: str = trace_id or (raw.id if raw else "no-op")
        self._spans: list = []

    def _finalise(self):
        # nothing extra needed; Langfuse auto-closes on flush
        pass

    @contextmanager
    def span(
        self,
        name: str,
        input: dict | None = None,
        metadata: dict | None = None,
    ) -> Generator[_SpanHandle, None, None]:
        """
        Create a child span with proper OpenTelemetry context propagation.

        Uses standard Python 'with' statement to ensure OTel context variables
        are properly set, allowing child operations to discover this span as parent.
        This fixes the "No active span in current context" warning.
        """
        if self.raw is None:
            yield _SpanHandle(None)  # No-op span
            return

        # ✅ Use standard 'with' statement for proper OTel context management
        # Manual __enter__/__exit__() calls skip context.attach(), which breaks
        # OpenTelemetry's context propagation to child spans/generations
        with self.raw.start_as_current_span(
            name=name,
            input=input or {},
            metadata=metadata or {},
        ) as s:
            handle = _SpanHandle(s)
            t0 = time.perf_counter()
            try:
                yield handle
            except Exception as exc:
                s.update(level="ERROR", status_message=str(exc))
                raise
            finally:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                handle._elapsed_ms = elapsed_ms

    @contextmanager
    def generation(
        self,
        name: str,
        model: str,
        input: Any = None,
        model_params: dict | None = None,
        metadata: dict | None = None,
    ) -> Generator[_GenerationHandle, None, None]:
        """
        Create a child generation with proper OpenTelemetry context propagation.

        Uses standard Python 'with' statement to ensure OTel context variables
        are properly set, allowing this generation to be discovered by parent spans.
        This fixes the "No active span in current context" warning.
        """
        if self.raw is None:
            yield _GenerationHandle(None)  # No-op generation
            return

        # ✅ Use standard 'with' statement for proper OTel context management
        # Manual __enter__/__exit__() calls skip context.attach(), which breaks
        # OpenTelemetry's context propagation to parent spans
        with self.raw.start_as_current_generation(
            name=name,
            model=model,
            input=input,
            model_parameters=model_params or {},
            metadata=metadata or {},
        ) as g:
            handle = _GenerationHandle(g)
            try:
                yield handle
            except Exception as exc:
                g.update(level="ERROR", status_message=str(exc))
                raise
            finally:
                # Update with output/tokens if set
                if handle.output or handle.input_tokens:
                    update_kwargs = {}
                    if handle.output:
                        update_kwargs["output"] = handle.output
                    if handle.input_tokens and handle.output_tokens:
                        update_kwargs["usage_details"] = {
                            "input": handle.input_tokens,
                            "output": handle.output_tokens,
                        }
                    if update_kwargs:
                        g.update(**update_kwargs)


class _SpanHandle:
    def __init__(self, raw):
        self.raw = raw
        self._elapsed_ms: int = 0

    def update(self, output: dict | None = None, **kwargs):
        if self.raw is None:
            return  # No-op update
        self.raw.update(output=output or {}, **kwargs)


class _GenerationHandle:
    def __init__(self, raw):
        self.raw = raw
        self.output: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def set_output(self, text: str, input_tokens: int = None, output_tokens: int = None):
        self.output = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.output_tokens = output_tokens


# ──────────────────────────────────────────────
# Decorator helpers (for simpler annotation-based tracing)
# ──────────────────────────────────────────────


def traced_span(tracer_attr: str, span_name: str):
    """
    Method decorator. Wraps a method in a Langfuse span.
    Assumes the class has a `_trace` attribute set to an active _TraceHandle.

    Example:
        class TextAgent:
            @traced_span("_tracer", "agent_text")
            def process(self, chunk): ...
    """

    def decorator(fn: Callable) -> Callable:
        import functools

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            trace: _TraceHandle | None = getattr(self, "_trace", None)
            if trace is None:
                return fn(self, *args, **kwargs)
            with trace.span(span_name, input={"args": str(args)[:200]}) as s:
                result = fn(self, *args, **kwargs)
                s.update(output={"result_type": type(result).__name__})
                return result

        return wrapper

    return decorator
