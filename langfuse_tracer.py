"""
langfuse_tracer.py - DEPRECATED BACKWARD COMPATIBILITY WRAPPER

⚠️  DEPRECATION WARNING ⚠️

This file is maintained for backward compatibility only.
New code should use: from src.integrations.langfuse import LangfuseTracer, TraceHandle

Note: _TraceHandle has been renamed to TraceHandle (public API).

This compatibility wrapper will be removed in v1.0.0.
"""

import warnings

warnings.warn(
    "langfuse_tracer.py is deprecated. Use: from src.integrations.langfuse import LangfuseTracer, TraceHandle",
    DeprecationWarning,
    stacklevel=2,
)

from src.integrations.langfuse import LangfuseTracer, TraceHandle

# Backward compatibility: old code might import _TraceHandle
_TraceHandle = TraceHandle

__all__ = ["LangfuseTracer", "TraceHandle", "_TraceHandle"]
