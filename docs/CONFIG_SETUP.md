# Model Configuration System - Implementation Guide

## Overview

The repository now uses a **settings.json**-based configuration system for managing model IDs across the entire pipeline. This eliminates hardcoded model references and allows easy customization without code changes.

### 🔑 Important: External API Keys Not Required

**The system runs completely locally using MLX models.** No external API keys (OpenAI, HuggingFace, etc.) are required for operation. All models are:
- Quantized 4-bit MLX compatible models
- Cached locally after first download
- Optimized for Apple Silicon
- Fully self-contained within the pipeline

You can safely omit `OPENAI_API_KEY` from your `.env` file.

## Files Created

### 1. `src/core/config.py` (155 lines)
Configuration loader with built-in defaults and fallback mechanism.

**Key Features:**
- Reads from `settings.json` in project root
- Falls back to defaults if file missing or incomplete
- Supports dot-notation for nested config access
- Deep merge of user config with defaults

**Usage:**
```python
from src.core.config import config

# Get a specific model ID
text_model = config.get_model("text_extraction")

# Get nested config with dot notation
confidence_threshold = config.get("validation.confidence_threshold")

# Get all models
all_models = config.config["models"]
```

### 2. `settings.json` (22 lines)
Default configuration file with all model IDs and pipeline settings.

**Structure:**
```json
{
  "models": {
    "text_extraction": "...",
    "table_extraction": "...",
    "vision_extraction": "...",
    "orchestrator": "...",
    "answer_validator": "...",
    "dspy_lm": "...",
    "embedder": "..."
  },
  "pipeline": {...},
  "cache": {...},
  "validation": {...}
}
```

### 3. `settings.example.json` (25 lines)
Template file for users to copy and customize.

### 4. `.gitignore` Update
Added entry to prevent committing user-customized `settings.json`:
```
# Configuration (user-customized)
settings.json
```

## Files Modified

### 1. `src/core/pipeline.py`
- Added import: `from src.core.config import config`
- Updated `AgenticRAGPipeline.build()` method signature:
  - Changed integer defaults to `Optional[str]` with `None` defaults
  - Load models from config if not provided as arguments
  ```python
  text_model = text_model or config.get_model("text_extraction")
  table_model = table_model or config.get_model("table_extraction")
  # ... etc
  ```

### 2. `src/core/langgraph_pipeline.py`
- Added import: `from src.core.config import config`
- Updated `LangGraphQueryPipeline.build()` method:
  - Load orchestrator and answer_validator models from config
  - Maintains override capability via method parameters

### 3. `src/integrations/crew_mlx_tools.py`
- Added import: `from src.core.config import config`
- Updated all tool classes:
  - `MLXTextExtractionTool`: Load text extraction model from config
  - `MLXTableExtractionTool`: Load table extraction model from config
  - `MLXVisionExtractionTool`: Load vision extraction model from config
  - `CrewMLXToolkit`: Initialize all tools with config-based models

### 4. `src/integrations/dspy_adapter.py`
- Added import: `from src.core.config import config`
- Updated `configure_mlx_lm()` function:
  - Load dspy_lm model ID from config by default
  - Maintains backward compatibility with explicit model_id parameter

## How It Works

### Configuration Loading Flow

```
1. Application Start
   └─→ import src.core.config.ConfigLoader
   
2. ConfigLoader Initialization
   ├─→ Load defaults from _DEFAULTS dict
   ├─→ Check for settings.json in project root
   ├─→ If found: Deep merge with defaults
   ├─→ If not found: Use defaults only + warn in logs
   └─→ config singleton instance ready

3. Module Import
   └─→ from src.core.config import config
       (Reuses singleton instance)

4. Model ID Access
   ├─→ Explicit override: TextAgent(model_id="custom/model")
   ├─→ Config default: TextAgent(model_id=config.get_model("text_extraction"))
   └─→ Built-in default: config.get_model("text_extraction") → "mlx-community/Phi-3.5-mini-Instruct-4bit"
```

### Customization Example

To use different models, edit `settings.json`:

```json
{
  "models": {
    "text_extraction": "my-custom-text-model-4bit",
    "table_extraction": "my-custom-table-model-4bit",
    "orchestrator": "my-custom-reasoning-8b",
    "_comment": "All models are MLX-compatible"
  }
}
```

Next application start will automatically use these custom models.

## Fallback Behavior

If `settings.json` is missing or contains invalid JSON:

1. **Load phase**: Log warning "settings.json not found..."
2. **Fallback**: Use hardcoded defaults from `ConfigLoader._DEFAULTS`
3. **Application**: Continues normally with defaults
4. **User action**: Copy `settings.example.json` to `settings.json` to customize

## Key Features

✅ **No Hardcoded Model IDs**: All references moved to `settings.json`  
✅ **Centralized Configuration**: Single source of truth for all models  
✅ **Easy Customization**: Edit JSON, no code changes needed  
✅ **Version Control Safe**: `settings.json` in `.gitignore`  
✅ **Backward Compatible**: Existing code works unchanged  
✅ **Fallback Defaults**: Works without `settings.json` file  
✅ **Type Hints**: Proper Python typing throughout  
✅ **Logging**: Info and warning messages for debugging  

## Configuration Keys Reference

### Models
- `text_extraction`: TextAgent model (default: Phi-3.5-mini-Instruct-4bit)
- `table_extraction`: TableAgent model (default: Qwen2.5-3B-Instruct-4bit)
- `vision_extraction`: VisionAgent model (default: SmolVLM-256M-Instruct-4bit)
- `chunk_validator`: ChunkValidatorAgent (default: SmolVLM-256M-Instruct-4bit)
- `orchestrator`: ReasoningOrchestratorAgent (default: DeepSeek-R1-Distill-Llama-8B-4bit)
- `answer_validator`: AnswerValidatorAgent (default: Qwen3-8B-4bit)
- `dspy_lm`: DSPy language model (default: Qwen2.5-7B-Instruct-4bit)
- `embedder`: Embedding model (default: intfloat/multilingual-e5-small)

### Pipeline
- `max_context_chunks`: Number of chunks retrieved (default: 8)
- `embedder_batch_size`: Batch size for embeddings (default: 32)
- `chunk_size`: Character size of chunks (default: 800)

### Validation
- `confidence_threshold`: Self-reflection threshold (default: 0.5)
- `enable_checkpoint_a`: Enable chunk validation (default: true)
- `enable_checkpoint_b`: Enable answer validation (default: true)

## Testing Configuration

Verify the configuration system:

```bash
# Check configuration loads
python -c "from src.core.config import config; print(config.get_model('orchestrator'))"

# Expected output:
# mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit
```

## Future Enhancements

Potential improvements:
- Environment variable overrides (`.env` support)
- Profile support (dev/test/prod)
- Schema validation with Pydantic
- Model aliasing for easier switching
- Remote configuration server support
