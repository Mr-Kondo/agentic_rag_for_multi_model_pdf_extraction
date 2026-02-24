# Migration Guide

Guide for migrating from the old flat file structure to the new modular `src/` package structure (v0.3.0+).

## Overview

The repository has been restructured from a flat 5-file structure into a modular package organization. All functionality remains the same, but imports and CLI usage have changed.

**Good news**: Backward compatibility wrappers are in place, so existing code will continue to work with deprecation warnings until v1.0.0.

## Key Changes

### 1. File Organization

**Old Structure (v0.2.x):**
```
agentic_rag_flow.py          (1,230 lines - everything)
validator_agent.py           (772 lines - validation)
dspy_mlx_adapter.py          (300 lines - DSPy integration)
langfuse_tracer.py           (402 lines - tracing)
```

**New Structure (v0.3.0+):**
```
app.py                       (New CLI entry point)
src/
├── core/                    (Core functionality)
│   ├── models.py              - Data structures
│   ├── pipeline.py            - Main orchestrator
│   ├── parser.py              - PDF parsing
│   ├── cache.py               - Model caching
│   └── store.py               - Vector storage
├── agents/                  (AI agents)
│   ├── base.py                - Base classes
│   ├── extraction.py          - Text/Table/Vision agents
│   ├── orchestrator.py        - Reasoning agent
│   ├── router.py              - Agent dispatcher
│   └── validation.py          - Quality validators
├── integrations/            (External integrations)
│   ├── dspy_adapter.py        - DSPy ↔ MLX bridge
│   ├── dspy_modules.py        - DSPy signatures
│   └── langfuse.py            - Observability
└── utils/                   (Utilities)
    └── serialization.py       - JSON output
tests/                       (Test suite)
```

### 2. Import Changes

#### Data Structures

**Old:**
```python
from agentic_rag_flow import (
    ChunkType,
    RawChunk,
    ProcessedChunk,
    RAGAnswer,
    ValidationSummary,
)
```

**New:**
```python
from src.core.models import (
    ChunkType,
    RawChunk,
    ProcessedChunk,
    RAGAnswer,
    ValidationSummary,
    ChunkValidationResult,
    AnswerValidationResult,
)
```

#### Pipeline

**Old:**
```python
from agentic_rag_flow import AgenticRAGPipeline

pipeline = AgenticRAGPipeline.build()
chunks = pipeline.ingest("paper.pdf")
answer = pipeline.query("What are the findings?")
```

**New:**
```python
from src.core.pipeline import AgenticRAGPipeline

pipeline = AgenticRAGPipeline.build()
chunks = pipeline.ingest("paper.pdf", validates=True)
answer = pipeline.query("What are the findings?", validates=True)
```

#### Validation Agents

**Old:**
```python
from validator_agent import (
    ChunkValidatorAgent,
    AnswerValidatorAgent,
    ChunkValidationResult,
    AnswerValidationResult,
)
```

**New:**
```python
from src.agents.validation import (
    ChunkValidatorAgent,
    AnswerValidatorAgent,
)
from src.core.models import (
    ChunkValidationResult,
    AnswerValidationResult,
)
```

#### DSPy Integration

**Old:**
```python
from dspy_mlx_adapter import MLXLM
from validator_agent import AnswerGroundingSignature
```

**New:**
```python
from src.integrations.dspy_adapter import MLXLM
from src.integrations.dspy_modules import (
    AnswerGroundingSignature,
    AnswerGroundingOutput,
    ChunkQualitySignature,
    ChunkQualityOutput,
)
```

#### Langfuse Tracing

**Old:**
```python
from langfuse_tracer import LangfuseTracer, _TraceHandle
```

**New:**
```python
from src.integrations.langfuse import LangfuseTracer, TraceHandle

# Note: _TraceHandle renamed to TraceHandle (public API)
# Old code using _TraceHandle will still work via compatibility wrapper
```

#### Utilities

**Old:**
```python
from agentic_rag_flow import save_chunks, save_answer
```

**New:**
```python
from src.utils.serialization import save_chunks, save_answer
```

### 3. CLI Changes

#### Old CLI (deprecated)

```bash
# Old way - still works with warnings
python agentic_rag_flow.py paper.pdf
python agentic_rag_flow.py paper.pdf "What are the findings?"
```

#### New CLI (recommended)

```bash
# New CLI with subcommands
python app.py --help

# Ingest a PDF
python app.py ingest paper.pdf --validate

# Query the vector store
python app.py query "What are the main findings?"

# Full pipeline (ingest + query)
python app.py pipeline paper.pdf "Summarize the methodology"

# Custom models
python app.py ingest paper.pdf \
    --text-model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --orchestrator mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit

# Skip validation (faster, less reliable)
python app.py query "Question?" --no-validate

# Custom storage location
python app.py ingest paper.pdf --storage-dir ./custom_db
```

## Migration Strategies

### Strategy 1: Keep Using Old Imports (Easiest)

**Timeline**: Immediate, works until v1.0.0

Your existing code will continue to work with deprecation warnings:

```python
# This still works, but shows warnings
from agentic_rag_flow import AgenticRAGPipeline
from validator_agent import AnswerValidatorAgent

# Run your code as before
pipeline = AgenticRAGPipeline.build()
# ... rest of your code unchanged
```

**Pros**:
- Zero code changes required
- Immediate compatibility

**Cons**:
- Deprecation warnings in logs
- Will break in v1.0.0

### Strategy 2: Gradual Migration (Recommended)

**Timeline**: 1-2 hours per module

Migrate one module at a time:

```python
# Step 1: Update imports
# from agentic_rag_flow import AgenticRAGPipeline
from src.core.pipeline import AgenticRAGPipeline

# Step 2: Update data structure imports
# from agentic_rag_flow import ProcessedChunk, RAGAnswer
from src.core.models import ProcessedChunk, RAGAnswer

# Step 3: Update validation imports
# from validator_agent import AnswerValidatorAgent
from src.agents.validation import AnswerValidatorAgent
from src.core.models import AnswerValidationResult

# Step 4: Update utility imports
# from agentic_rag_flow import save_chunks
from src.utils.serialization import save_chunks

# Rest of your code unchanged
```

**Pros**:
- Low risk, incremental changes
- Can test after each step
- Forward compatible

**Cons**:
- Takes some time

### Strategy 3: Switch to New CLI (Fastest for scripts)

**Timeline**: 5-10 minutes

Replace script execution with new CLI:

```bash
# Old script
# python my_script.py

# New approach - use CLI directly
python app.py pipeline ./input/paper.pdf "My question?"
```

**Pros**:
- No code changes
- Immediate benefits (better logging, options)

**Cons**:
- Only works for simple workflows
- Can't customize programmatically

## Breaking Changes (v1.0.0 - Future)

These changes will happen in v1.0.0 (estimated: 6+ months):

1. **Old files removed**:
   - `agentic_rag_flow.py` → Use `src.core.pipeline`
   - `validator_agent.py` → Use `src.agents.validation`
   - `dspy_mlx_adapter.py` → Use `src.integrations.dspy_adapter`
   - `langfuse_tracer.py` → Use `src.integrations.langfuse`

2. **`_TraceHandle` removed**:
   - Already renamed to `TraceHandle` in new code
   - Backward compatible wrapper via `langfuse_tracer.py`

3. **Old import paths removed**:
   - Must use `src.*` imports

## Testing Your Migration

### 1. Check for deprecation warnings

```bash
# Run your code and look for warnings
python -W default your_script.py 2>&1 | grep "DeprecationWarning"
```

### 2. Run the test suite

```bash
# Install test dependencies
pip install -e .[dev]

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 3. Verify imports work

```python
# Test new imports
from src.core.pipeline import AgenticRAGPipeline
from src.core.models import ChunkType, ProcessedChunk, RAGAnswer
from src.agents.validation import AnswerValidatorAgent
from src.integrations.langfuse import LangfuseTracer

print("✅ All imports successful!")
```

## Common Migration Issues

### Issue 1: Import errors from src

**Problem:**
```python
ModuleNotFoundError: No module named 'src'
```

**Solution:**
Make sure you're running from the project root, or install the package:
```bash
# Option 1: Run from project root
cd /path/to/agentic_rag_for_multi_model_pdf_extraction
python your_script.py

# Option 2: Install package in editable mode
pip install -e .
```

### Issue 2: Old imports still working but warnings everywhere

**Problem:**
Log spam with deprecation warnings

**Solution:**
Either:
1. Migrate imports to new structure (recommended)
2. Suppress warnings (not recommended):
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

### Issue 3: TraceHandle vs _TraceHandle

**Problem:**
Code uses `_TraceHandle` but new code has `TraceHandle`

**Solution:**
Update type hints:
```python
# Old
from langfuse_tracer import _TraceHandle
def my_func(trace: _TraceHandle | None = None): ...

# New
from src.integrations.langfuse import TraceHandle
def my_func(trace: TraceHandle | None = None): ...
```

## Benefits of Migration

### For Development

1. **Faster imports**: Only load what you need
2. **Better IDE support**: Clear module boundaries
3. **Easier testing**: Mock specific modules
4. **Type safety**: Better type hints and checking

### For Production

1. **Memory efficiency**: Lazy loading of heavy modules
2. **Maintainability**: Clear separation of concerns
3. **Extensibility**: Easy to add new agents/features
4. **Professional**: Standard Python package structure

## Need Help?

1. **Check the examples**:
   - `tests/` directory has many examples
   - `src/` docstrings are comprehensive

2. **Run the test suite**:
   ```bash
   pytest tests/ -v
   ```

3. **Try the new CLI**:
   ```bash
   python app.py --help
   python app.py ingest --help
   python app.py query --help
   python app.py pipeline --help
   ```

4. **Use backward compatibility**:
   - Old imports still work
   - Take your time migrating

## Version Support Timeline

| Version | Old Imports | New Imports | CLI | Status |
|---------|-------------|-------------|-----|--------|
| v0.2.x  | ✅ Only     | ❌          | Old | Legacy |
| v0.3.x  | ✅ (warns)  | ✅          | Both| Current|
| v1.0.0  | ❌          | ✅ Only     | New | Future |

**Current version**: v0.3.0  
**Deprecation period**: At least 6 months  
**Hard removal**: v1.0.0 (TBD)
