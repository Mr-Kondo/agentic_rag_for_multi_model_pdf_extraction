"""
validator_agent.py - DEPRECATED BACKWARD COMPATIBILITY WRAPPER

⚠️  DEPRECATION WARNING ⚠️

This file is maintained for backward compatibility only.
New code should use the modular src/ package structure:

  Old (deprecated):
    from validator_agent import ChunkValidatorAgent, AnswerValidatorAgent
    from validator_agent import ChunkValidationResult, AnswerValidationResult

  New (recommended):
    from src.agents.validation import ChunkValidatorAgent, AnswerValidatorAgent
    from src.core.models import ChunkValidationResult, AnswerValidationResult
    from src.integrations.dspy_modules import AnswerGroundingSignature

This compatibility wrapper will be removed in v1.0.0.
See MIGRATION.md for migration guide.

═══════════════════════════════════════════════════════════════════════════
"""

import warnings

# Show deprecation warning on import
warnings.warn(
    "\n\n"
    "═" * 70 + "\n"
    "⚠️  DEPRECATION WARNING\n"
    "═" * 70 + "\n"
    "validator_agent.py is deprecated and will be removed in v1.0.0.\n"
    "\n"
    "Please migrate to the new modular structure:\n"
    "  • from src.agents.validation import ChunkValidatorAgent, AnswerValidatorAgent\n"
    "  • from src.core.models import ChunkValidationResult, AnswerValidationResult\n"
    "  • from src.integrations.dspy_modules import AnswerGroundingSignature\n"
    "\n"
    "See MIGRATION.md for details.\n"
    "═" * 70 + "\n",
    DeprecationWarning,
    stacklevel=2,
)

# ═══════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY EXPORTS
# ═══════════════════════════════════════════════════════════

# Re-export validation result dataclasses
from src.core.models import AnswerValidationResult, ChunkValidationResult

# Re-export base classes
from src.agents.base import BaseLoadableModel

# Re-export validation agents
from src.agents.validation import AnswerValidatorAgent, ChunkValidatorAgent

# Re-export DSPy modules
from src.integrations.dspy_modules import (
    AnswerGroundingOutput,
    AnswerGroundingSignature,
    ChunkQualityOutput,
    ChunkQualitySignature,
)

# For complete backward compatibility, ensure all previously exported names are available
__all__ = [
    "ChunkValidationResult",
    "AnswerValidationResult",
    "ChunkValidatorAgent",
    "AnswerValidatorAgent",
    "BaseLoadableModel",
    "AnswerGroundingOutput",
    "AnswerGroundingSignature",
    "ChunkQualityOutput",
    "ChunkQualitySignature",
]
