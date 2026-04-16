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
import importlib.metadata
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
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

_TRACE_START_API_CANDIDATES = ("start_as_current_span", "start_span")
_CHILD_SPAN_API_CANDIDATES = ("start_as_current_span", "start_span")
_GENERATION_API_CANDIDATES = ("start_as_current_generation", "start_generation")
_TRACE_ID_API_CANDIDATES = ("get_current_trace_id",)
_SCORE_API_CANDIDATES = ("create_score", "score")
_COMPATIBILITY_LOGGED = False


@dataclass(slots=True)
class LangfuseDiagnostics:
    """
    Diagnostic payload describing Langfuse compatibility state.

    Attributes:
        sdk_version: Installed Langfuse SDK version, if known.
        client_class: Runtime class name of the Langfuse client.
        raw_class: Runtime class name of the active trace/span handle.
        resolved_apis: Mapping of operation group to resolved API name.
        reason: Short machine-readable reason for degraded tracing.
        error: Optional exception string captured while probing/starting.
        trace_name: Trace name associated with this diagnostic event.
    """

    sdk_version: str = "unknown"
    client_class: str | None = None
    raw_class: str | None = None
    resolved_apis: dict[str, str | None] = field(default_factory=dict)
    reason: str | None = None
    error: str | None = None
    trace_name: str | None = None

    def format(self) -> str:
        """Render a compact diagnostic string for logs."""
        parts: list[str] = [f"sdk_version={self.sdk_version}"]
        if self.trace_name:
            parts.append(f"trace={self.trace_name}")
        if self.reason:
            parts.append(f"reason={self.reason}")
        if self.client_class:
            parts.append(f"client_class={self.client_class}")
        if self.raw_class:
            parts.append(f"raw_class={self.raw_class}")
        for key, value in self.resolved_apis.items():
            parts.append(f"{key}={value or 'missing'}")
        if self.error:
            parts.append(f"error={self.error}")
        return " ".join(parts)


def _get_langfuse_sdk_version() -> str:
    """Return the installed Langfuse SDK version when discoverable."""
    try:
        return importlib.metadata.version("langfuse")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _resolve_api(target: Any, candidates: tuple[str, ...]) -> tuple[str | None, Callable | None]:
    """Resolve the first available callable API name from a candidate list."""
    for name in candidates:
        method = getattr(target, name, None)
        if callable(method):
            return name, method
    return None, None


def _build_diagnostics(
    *,
    client: Any = None,
    raw: Any = None,
    trace_name: str | None = None,
    reason: str | None = None,
    error: Exception | str | None = None,
) -> LangfuseDiagnostics:
    """Build a diagnostic snapshot for client/raw Langfuse compatibility."""
    resolved_apis: dict[str, str | None] = {}
    if client is not None:
        resolved_apis["trace_api"] = _resolve_api(client, _TRACE_START_API_CANDIDATES)[0]
        resolved_apis["trace_id_api"] = _resolve_api(client, _TRACE_ID_API_CANDIDATES)[0]
        resolved_apis["score_api"] = _resolve_api(client, _SCORE_API_CANDIDATES)[0]
    if raw is not None:
        resolved_apis["child_span_api"] = _resolve_api(raw, _CHILD_SPAN_API_CANDIDATES)[0]
        resolved_apis["generation_api"] = _resolve_api(raw, _GENERATION_API_CANDIDATES)[0]

    return LangfuseDiagnostics(
        sdk_version=_get_langfuse_sdk_version(),
        client_class=type(client).__name__ if client is not None else None,
        raw_class=type(raw).__name__ if raw is not None else None,
        resolved_apis=resolved_apis,
        reason=reason,
        error=str(error) if error is not None else None,
        trace_name=trace_name,
    )


def _log_diagnostics(level: int, message: str, diagnostics: LangfuseDiagnostics) -> None:
    """Log a Langfuse diagnostic line with structured details."""
    log.log(level, "%s %s", message, diagnostics.format())


@contextmanager
def _wrap_context_object(context: Any) -> Generator[Any, None, None]:
    """Use an object as a context manager when supported, or yield it directly."""
    if hasattr(context, "__enter__") and hasattr(context, "__exit__"):
        with context as entered:
            yield entered
        return

    yield context


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
        self._client_diagnostics = _build_diagnostics(client=self._client, reason="client_unavailable")
        self._log_compatibility_once()

    def _log_compatibility_once(self) -> None:
        """Emit a one-time compatibility summary for the current Langfuse client."""
        global _COMPATIBILITY_LOGGED

        if _COMPATIBILITY_LOGGED or self._client is None:
            return

        diagnostics = _build_diagnostics(client=self._client)
        trace_api = diagnostics.resolved_apis.get("trace_api")
        generation_api = diagnostics.resolved_apis.get("generation_api")
        if trace_api is None:
            diagnostics.reason = "missing_client_trace_api"
            _log_diagnostics(logging.ERROR, "Langfuse compatibility check failed.", diagnostics)
        elif diagnostics.resolved_apis.get("trace_id_api") is None or diagnostics.resolved_apis.get("score_api") is None:
            diagnostics.reason = "partial_client_api_support"
            _log_diagnostics(logging.WARNING, "Langfuse compatibility check detected limited API support.", diagnostics)
        elif generation_api is not None:
            log.debug("Langfuse compatibility check passed. %s", diagnostics.format())

        _COMPATIBILITY_LOGGED = True

    @staticmethod
    def _disabled_handle(trace_name: str, reason: str, error: Exception | str | None = None) -> "TraceHandle":
        """Create a no-op trace handle with diagnostics attached."""
        diagnostics = _build_diagnostics(trace_name=trace_name, reason=reason, error=error)
        return TraceHandle.noop(reason=reason, diagnostics=diagnostics)

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
            handle = self._disabled_handle(name, "client_unavailable")
            token = _CURRENT_TRACE.set(handle)
            try:
                yield handle
            finally:
                _CURRENT_TRACE.reset(token)
            return

        trace_api_name, trace_api = _resolve_api(self._client, _TRACE_START_API_CANDIDATES)
        if trace_api is None:
            diagnostics = _build_diagnostics(
                client=self._client,
                trace_name=name,
                reason="missing_client_trace_api",
            )
            _log_diagnostics(logging.WARNING, "Langfuse trace unavailable.", diagnostics)
            handle = TraceHandle.noop(reason="missing_client_trace_api", diagnostics=diagnostics)
            token = _CURRENT_TRACE.set(handle)
            try:
                yield handle
            finally:
                _CURRENT_TRACE.reset(token)
            return

        try:
            span = trace_api(
                name=name,
                input=input or {},
                metadata=metadata or {},
            )
        except Exception as exc:
            diagnostics = _build_diagnostics(
                client=self._client,
                trace_name=name,
                reason="trace_start_failed",
                error=exc,
            )
            diagnostics.resolved_apis["trace_api"] = trace_api_name
            _log_diagnostics(logging.WARNING, "Langfuse trace unavailable.", diagnostics)
            handle = TraceHandle.noop(reason="trace_start_failed", diagnostics=diagnostics)
            token = _CURRENT_TRACE.set(handle)
            try:
                yield handle
            finally:
                _CURRENT_TRACE.reset(token)
            return

        if span is None:
            diagnostics = _build_diagnostics(
                client=self._client,
                trace_name=name,
                reason="trace_start_returned_none",
            )
            diagnostics.resolved_apis["trace_api"] = trace_api_name
            _log_diagnostics(logging.WARNING, "Langfuse trace unavailable.", diagnostics)
            span = None

        if span is None:
            handle = TraceHandle.noop(reason="trace_start_returned_none", diagnostics=diagnostics)
            token = _CURRENT_TRACE.set(handle)
            try:
                yield handle
            finally:
                _CURRENT_TRACE.reset(token)
            return

        trace_id = None
        trace_id_api_name, trace_id_api = _resolve_api(self._client, _TRACE_ID_API_CANDIDATES)
        if trace_id_api is not None:
            try:
                trace_id = trace_id_api()
                log.debug("✓ Trace started: %s (trace_id=%s)", name, trace_id)
            except Exception as exc:
                diagnostics = _build_diagnostics(
                    client=self._client,
                    raw=span,
                    trace_name=name,
                    reason="trace_id_lookup_failed",
                    error=exc,
                )
                diagnostics.resolved_apis["trace_api"] = trace_api_name
                diagnostics.resolved_apis["trace_id_api"] = trace_id_api_name
                _log_diagnostics(logging.WARNING, "Langfuse trace started without trace id lookup.", diagnostics)

        handle_diagnostics = _build_diagnostics(client=self._client, raw=span, trace_name=name)
        handle_diagnostics.resolved_apis["trace_api"] = trace_api_name
        handle_diagnostics.resolved_apis["trace_id_api"] = trace_id_api_name
        if (
            handle_diagnostics.resolved_apis.get("child_span_api") is None
            or handle_diagnostics.resolved_apis.get("generation_api") is None
        ):
            handle_diagnostics.reason = "partial_trace_handle_api_support"
            _log_diagnostics(logging.WARNING, "Langfuse trace started with limited child API support.", handle_diagnostics)

        handle = TraceHandle(span, trace_id, diagnostics=handle_diagnostics)
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

        if trace_id == "no-op":
            log.debug("⊘ Score skipped (trace disabled): %s=%s", name, value)
            return

        normalized_data_type = _normalize_score_data_type(data_type)
        if normalized_data_type != data_type:
            log.warning(
                "Normalizing score data_type from '%s' to '%s' for score '%s'.",
                data_type,
                normalized_data_type,
                name,
            )

        score_api_name, score_api = _resolve_api(self._client, _SCORE_API_CANDIDATES)
        if score_api is None:
            diagnostics = _build_diagnostics(
                client=self._client,
                reason="missing_score_api",
            )
            _log_diagnostics(logging.WARNING, f"Langfuse score unavailable for '{name}'.", diagnostics)
            return

        try:
            score_api(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
                data_type=normalized_data_type,
            )
            log.debug("✓ Score posted via %s: %s=%s", score_api_name, name, value)
        except Exception as e:
            diagnostics = _build_diagnostics(
                client=self._client,
                reason="score_post_failed",
                error=e,
            )
            diagnostics.resolved_apis["score_api"] = score_api_name
            _log_diagnostics(logging.WARNING, f"Failed to post score '{name}'.", diagnostics)


# ──────────────────────────────────────────────
# Handle objects (thin proxies around Langfuse types)
# ──────────────────────────────────────────────


class TraceHandle:
    def __init__(
        self,
        raw,
        trace_id: str | None = None,
        diagnostics: LangfuseDiagnostics | None = None,
        disable_reason: str | None = None,
    ):
        self.raw = raw
        self.trace_id: str = trace_id or getattr(raw, "id", None) or "no-op"
        self._spans: list = []
        self.diagnostics = diagnostics
        self.disable_reason = disable_reason or (diagnostics.reason if diagnostics else None)

    @classmethod
    def noop(cls, reason: str, diagnostics: LangfuseDiagnostics | None = None) -> "TraceHandle":
        """Create a disabled trace handle with diagnostic context."""
        return cls(None, None, diagnostics=diagnostics, disable_reason=reason)

    def diagnostic_summary(self) -> str:
        """Return a compact diagnostic summary for disabled tracing."""
        if self.diagnostics is None:
            return self.disable_reason or "trace_disabled"
        return self.diagnostics.format()

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

        span_api_name, span_api = _resolve_api(self.raw, _CHILD_SPAN_API_CANDIDATES)
        if span_api is None:
            diagnostics = _build_diagnostics(
                raw=self.raw,
                trace_name=self.diagnostics.trace_name if self.diagnostics else None,
                reason="missing_child_span_api",
            )
            _log_diagnostics(logging.WARNING, "Langfuse child span unavailable.", diagnostics)
            yield _SpanHandle(None)
            return

        try:
            context = span_api(
                name=name,
                input=input or {},
                metadata=metadata or {},
            )
            with _wrap_context_object(context) as s:
                handle = _SpanHandle(s)
                t0 = time.perf_counter()
                try:
                    yield handle
                except Exception as exc:
                    if hasattr(s, "update"):
                        s.update(level="ERROR", status_message=str(exc))
                    raise
                finally:
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    handle._elapsed_ms = elapsed_ms
        except Exception as exc:
            diagnostics = _build_diagnostics(
                raw=self.raw,
                trace_name=self.diagnostics.trace_name if self.diagnostics else None,
                reason="child_span_failed",
                error=exc,
            )
            diagnostics.resolved_apis["child_span_api"] = span_api_name
            _log_diagnostics(logging.WARNING, "Langfuse child span unavailable.", diagnostics)
            yield _SpanHandle(None)

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

        generation_api_name, generation_api = _resolve_api(self.raw, _GENERATION_API_CANDIDATES)
        if generation_api is None:
            diagnostics = _build_diagnostics(
                raw=self.raw,
                trace_name=self.diagnostics.trace_name if self.diagnostics else None,
                reason="missing_generation_api",
            )
            _log_diagnostics(logging.WARNING, "Langfuse generation unavailable.", diagnostics)
            yield _GenerationHandle(None)
            return

        try:
            context = generation_api(
                name=name,
                model=model,
                input=input,
                model_parameters=model_params or {},
                metadata=metadata or {},
            )
            with _wrap_context_object(context) as g:
                handle = _GenerationHandle(g, model=model)
                try:
                    yield handle
                except Exception as exc:
                    if hasattr(g, "update"):
                        g.update(level="ERROR", status_message=str(exc))
                    raise
        except Exception as exc:
            diagnostics = _build_diagnostics(
                raw=self.raw,
                trace_name=self.diagnostics.trace_name if self.diagnostics else None,
                reason="generation_start_failed",
                error=exc,
            )
            diagnostics.resolved_apis["generation_api"] = generation_api_name
            _log_diagnostics(logging.WARNING, "Langfuse generation unavailable.", diagnostics)
            yield _GenerationHandle(None)


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
