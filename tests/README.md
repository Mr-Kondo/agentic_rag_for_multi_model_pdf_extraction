# Tests

Test suite for the active Sequential + Ollama pipeline.

## Running tests

### Run all tests
```bash
pytest tests/ -v
```

### Run focused tests
```bash
pytest tests/test_models.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_parser.py -v
pytest tests/test_embedder.py -v
pytest tests/test_dspy_adapter.py -v
pytest tests/test_dspy_validator.py -v
pytest tests/test_langfuse.py -v
pytest tests/test_audit.py -v
pytest tests/test_table_extraction.py -v
pytest tests/test_token_counter.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test structure

- `conftest.py` - shared fixtures and sample payloads
- `test_models.py` - unit tests for data structures
- `test_pipeline.py` - pipeline construction and workflow structure checks
- `test_parser.py` - PDF parser unit tests
- `test_embedder.py` - embedder backend unit tests
- `test_dspy_adapter.py` - DSPy adapter configuration tests
- `test_dspy_validator.py` - mocked unit tests for `AnswerValidatorAgent`
- `test_langfuse.py` - Langfuse tracer unit tests
- `test_audit.py` - audit output generation tests
- `test_table_extraction.py` - table extraction agent unit tests
- `test_token_counter.py` - token counter utility tests

## Fixtures

Shared fixtures from `conftest.py`:

- `sample_question`
- `sample_raw_chunk`
- `sample_processed_chunk`
- `sample_rag_answer`
- `sample_source_texts`
- `temp_output_dir`
- `temp_storage_dir`
- `test_model_config`

`test_model_config` uses the same Ollama model names as the application defaults. Tests should still mock model loading and network calls unless a file is explicitly marked as integration coverage.

## Guidelines

1. Keep default tests hermetic and fast.
2. Mock Ollama, DSPy predictors, ChromaDB, and parser/model loading when verifying control flow.
3. Use fixtures for repeated domain objects instead of rebuilding them inline.
4. Reserve live-model checks for explicitly separated integration tests.
