# Tests

Test suite for the Agentic RAG pipeline.

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_models.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_dspy_validator.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run a specific test function
```bash
pytest tests/test_models.py::TestChunkType::test_chunk_type_values -v
```

## Test Structure

- **conftest.py** - Pytest configuration and shared fixtures
- **test_models.py** - Unit tests for data structures
- **test_pipeline.py** - Integration tests for pipeline workflow
- **test_dspy_validator.py** - DSPy integration tests for validation agents

## Test Fixtures

Common fixtures available in all tests (defined in conftest.py):

- `sample_question` - Example question string
- `sample_raw_chunk` - RawChunk instance for testing
- `sample_processed_chunk` - ProcessedChunk instance for testing
- `sample_rag_answer` - RAGAnswer instance for testing
- `sample_source_texts` - List of source text strings
- `temp_output_dir` - Temporary output directory
- `temp_storage_dir` - Temporary storage directory
- `test_model_config` - Small/fast model configuration

## Writing New Tests

### Unit Test Example
```python
from src.core.models import ChunkType

def test_chunk_type():
    assert ChunkType.TEXT.value == "TEXT"
```

### Integration Test Example
```python
from src.core.pipeline import AgenticRAGPipeline

def test_pipeline_build(test_model_config):
    pipeline = AgenticRAGPipeline.build(**test_model_config)
    assert pipeline is not None
```

### Using Fixtures
```python
def test_with_fixture(sample_processed_chunk):
    assert sample_processed_chunk.confidence > 0
```

## Testing Guidelines

1. **Fast tests first**: Unit tests should run in milliseconds
2. **Mock heavy dependencies**: Use `unittest.mock` for model loading
3. **Isolated tests**: Each test should be independent
4. **Clear assertions**: Use descriptive assertion messages
5. **Fixtures for common data**: Reuse test data via conftest.py

## CI/CD Integration

Tests are run automatically on:
- Pull requests
- Commits to main/feature branches
- Release tags

Required checks:
- All tests must pass
- Code coverage >80%
- No linting errors
