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

import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from langfuse import Langfuse
from langfuse.model import ModelUsage


# ──────────────────────────────────────────────
# Singleton Langfuse client
# ──────────────────────────────────────────────

def _make_client() -> Langfuse:
    return Langfuse(
        public_key = os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key = os.environ["LANGFUSE_SECRET_KEY"],
        host       = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
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
        self._client: Langfuse = _make_client()

    # ── Trace ────────────────────────────────
    @contextmanager
    def trace(
        self,
        name      : str,
        input     : dict | None = None,
        metadata  : dict | None = None,
        user_id   : str | None  = None,
        session_id: str | None  = None,
    ) -> Generator[_TraceHandle, None, None]:
        t = self._client.trace(
            name       = name,
            input      = input or {},
            metadata   = metadata or {},
            user_id    = user_id,
            session_id = session_id,
        )
        handle = _TraceHandle(t)
        try:
            yield handle
        finally:
            handle._finalise()
            self._client.flush()

    # ── Direct score posting (outside context) ─
    def score(self, trace_id: str, name: str, value: float, comment: str = ""):
        self._client.score(
            trace_id = trace_id,
            name     = name,
            value    = value,
            comment  = comment,
        )


# ──────────────────────────────────────────────
# Handle objects (thin proxies around Langfuse types)
# ──────────────────────────────────────────────

class _TraceHandle:
    def __init__(self, raw):
        self.raw      = raw
        self.trace_id : str = raw.id
        self._spans   : list = []

    def _finalise(self):
        # nothing extra needed; Langfuse auto-closes on flush
        pass

    @contextmanager
    def span(
        self,
        name    : str,
        input   : dict | None = None,
        metadata: dict | None = None,
    ) -> Generator[_SpanHandle, None, None]:
        s = self.raw.span(name=name, input=input or {}, metadata=metadata or {})
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
            s.end()

    @contextmanager
    def generation(
        self,
        name          : str,
        model         : str,
        input         : Any    = None,
        model_params  : dict   | None = None,
        metadata      : dict   | None = None,
    ) -> Generator[_GenerationHandle, None, None]:
        g = self.raw.generation(
            name         = name,
            model        = model,
            input        = input,
            model_params = model_params or {},
            metadata     = metadata or {},
        )
        handle = _GenerationHandle(g)
        try:
            yield handle
        except Exception as exc:
            g.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            g.end(
                output = handle.output,
                usage  = ModelUsage(
                    input  = handle.input_tokens,
                    output = handle.output_tokens,
                ) if handle.input_tokens else None,
            )


class _SpanHandle:
    def __init__(self, raw):
        self.raw        = raw
        self._elapsed_ms: int = 0

    def update(self, output: dict | None = None, **kwargs):
        self.raw.update(output=output or {}, **kwargs)


class _GenerationHandle:
    def __init__(self, raw):
        self.raw          = raw
        self.output       : str | None = None
        self.input_tokens : int | None = None
        self.output_tokens: int | None = None

    def set_output(self, text: str, input_tokens: int = None, output_tokens: int = None):
        self.output        = text
        self.input_tokens  = input_tokens
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
