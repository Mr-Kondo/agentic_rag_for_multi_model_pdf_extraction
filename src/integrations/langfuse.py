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
  LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from dotenv import load_dotenv
from langfuse import Langfuse

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

_CURRENT_TRACE: contextvars.ContextVar["TraceHandle | None"] = contextvars.ContextVar(
    "langfuse_current_trace",
    default=None,
)


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


def _build_usage_details(input_tokens: int | None, output_tokens: int | None) -> dict[str, int] | None:
    """
    Build Langfuse generation usage_details payload when both token counts are known.

    Args:
        input_tokens: Prompt/input token count.
        output_tokens: Completion/output token count.

    Returns:
        Langfuse-compatible usage_details payload, or None when either side is missing.
    """
    if input_tokens is None or output_tokens is None:
        return None

    input_token_count = int(input_tokens)
    output_token_count = int(output_tokens)
    return {
        "input": input_token_count,
        "output": output_token_count,
        "total": input_token_count + output_token_count,
    }


# ──────────────────────────────────────────────
# Singleton Langfuse client
# ──────────────────────────────────────────────

_client_instance: Langfuse | None = None
_client_initialized: bool = False


def _get_client() -> Langfuse | None:
    """
    Get or create singleton Langfuse client if credentials are available.
    Loads environment variables from .env file on first call only.
    Returns None if credentials are not configured.

    This singleton pattern prevents duplicate initialization warnings from
    OpenTelemetry when multiple LangfuseTracer instances are created.
    """
    global _client_instance, _client_initialized

    if _client_initialized:
        return _client_instance

    # Mark as initialized to prevent re-entry
    _client_initialized = True

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

    _client_instance = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    return _client_instance


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
        self._client: Langfuse | None = _get_client()

    # ── Trace ────────────────────────────────
    @contextmanager
    def trace(
        self,
        name: str,
        input: dict | None = None,
        metadata: dict | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Generator[TraceHandle, None, None]:
        """
        Create a trace context using Langfuse SDK.

        Creates a span context that allows child spans and generations to
        discover the parent trace.

        Args:
            name: Trace name (e.g., "ingest_pdf", "rag_query")
            input: Input data for this trace
            metadata: Additional metadata
            user_id: User ID for tracking
            session_id: Session ID for grouping traces

        Returns:
            Generator yielding a TraceHandle for context management
        """
        if self._client is None:
            handle = TraceHandle(None, None)
            token = _CURRENT_TRACE.set(handle)
            try:
                yield handle
            finally:
                _CURRENT_TRACE.reset(token)
            return

        # Fallback: If start_as_current_span is not available, use basic tracing
        try:
            # Try new API if available
            span = self._client.start_as_current_span(
                name=name,
                input=input or {},
                metadata=metadata or {},
            )
        except AttributeError:
            # Fallback to disabled tracing (method not available in this SDK version)
            # This allows the pipeline to run without observability
            log.warning(f"Langfuse trace unavailable (SDK version incompatible). Trace '{name}' disabled.")
            span = None

        if span is None:
            # Run without tracing
            handle = TraceHandle(None, None)
            token = _CURRENT_TRACE.set(handle)
            try:
                yield handle
            finally:
                _CURRENT_TRACE.reset(token)
            return

        trace_id = None
        try:
            trace_id = self._client.get_current_trace_id()
            log.debug(f"✓ Trace started: {name} (trace_id={trace_id})")
        except Exception:
            pass

        handle = TraceHandle(span, trace_id)
        token = _CURRENT_TRACE.set(handle)
        try:
            yield handle
        except Exception as e:
            if hasattr(span, "update"):
                try:
                    span.update(level="ERROR", status_message=str(e))
                except Exception:
                    pass
            log.error(f"Trace error in '{name}': {e}")
            raise
        finally:
            _CURRENT_TRACE.reset(token)

    # ── Scoring ──────────────────────────────
    def score(
        self,
        trace_id: str,
        name: str,
        value: float | int,
        comment: str | None = None,
        data_type: str = "NUMERIC",
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


class TraceHandle:
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

        # Use the Langfuse v3 generation API shipped in this repository.
        #
        # IMPORTANT: The SDK ends the OTel span as soon as the 'with' block exits
        # (end_on_exit=True is the SDK default).  Any g.update() call made in a
        # 'finally' clause—after 'yield' returns—would execute *after* the span is
        # already ended, meaning is_recording() == False and the update is silently
        # dropped.  Token counts and output must therefore be written while the span
        # is still alive, i.e. inside _GenerationHandle.set_output().
        with self.raw.start_as_current_generation(
            name=name,
            model=model,
            input=input,
            model_parameters=model_params or {},
            metadata=metadata or {},
        ) as g:
            handle = _GenerationHandle(g, model=model)
            try:
                yield handle
            except Exception as exc:
                g.update(level="ERROR", status_message=str(exc))
                raise


class _SpanHandle:
    def __init__(self, raw):
        self.raw = raw
        self._elapsed_ms: int = 0

    def update(self, output: dict | None = None, **kwargs):
        if self.raw is None:
            return  # No-op update
        self.raw.update(output=output or {}, **kwargs)


class _GenerationHandle:
    def __init__(self, raw, model: str | None = None):
        self.raw = raw
        self.model = model
        self.output: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def set_output(self, text: str, input_tokens: int | None = None, output_tokens: int | None = None):
        """Record output and token counts, writing them to the Langfuse span immediately.

        This must be called while the generation context manager is still active (i.e.
        before the 'with trace.generation(...)' block exits), because the SDK ends the
        underlying OTel span on context-manager exit and subsequent update() calls are
        silently dropped.
        """
        self.output = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

        if self.raw is None:
            return

        update_kwargs: dict = {"output": text}
        if self.model is not None:
            update_kwargs["model"] = self.model
        usage_details = _build_usage_details(input_tokens, output_tokens)
        if usage_details is not None:
            update_kwargs["usage_details"] = usage_details
        try:
            self.raw.update(**update_kwargs)
        except Exception as e:
            log.warning("Failed to update generation with output/usage: %s", e)


# ──────────────────────────────────────────────
# Decorator helpers (for simpler annotation-based tracing)
# ──────────────────────────────────────────────


def traced_span(tracer_attr: str, span_name: str):
    """
    Method decorator. Wraps a method in a Langfuse span.
    Assumes the class has a `_trace` attribute set to an active TraceHandle.

    Example:
        class TextAgent:
            @traced_span("_tracer", "agent_text")
            def process(self, chunk): ...
    """

    def decorator(fn: Callable) -> Callable:
        import functools

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            trace: TraceHandle | None = getattr(self, "_trace", None)
            if trace is None:
                return fn(self, *args, **kwargs)
            with trace.span(span_name, input={"args": str(args)[:200]}) as s:
                result = fn(self, *args, **kwargs)
                s.update(output={"result_type": type(result).__name__})
                return result

        return wrapper

    return decorator


def get_trace() -> "TraceHandle | None":
    """
    Return the active Langfuse trace handle for the current context.

    Returns:
        TraceHandle if a trace is active, otherwise None.
    """
    return _CURRENT_TRACE.get()
