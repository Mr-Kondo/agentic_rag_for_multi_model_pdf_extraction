"""
dspy_mlx_adapter.py - DEPRECATED BACKWARD COMPATIBILITY WRAPPER

⚠️  DEPRECATION WARNING ⚠️

This file is maintained for backward compatibility only.
New code should use: from src.integrations.dspy_adapter import MLXLM

This compatibility wrapper will be removed in v1.0.0.
"""

import warnings

warnings.warn(
    "dspy_mlx_adapter.py is deprecated. Use: from src.integrations.dspy_adapter import MLXLM",
    DeprecationWarning,
    stacklevel=2,
)

from src.integrations.dspy_adapter import MLXLM

__all__ = ["MLXLM"]
