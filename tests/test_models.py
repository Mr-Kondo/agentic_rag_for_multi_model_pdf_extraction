"""
test_models.py
==============
Unit tests for data model classes.

Tests:
- ChunkType enum
- Dataclass instantiation
- Field validation
- Serialization compatibility
"""

import pytest

from src.core.models import (
    AnswerValidationResult,
    ChunkType,
    ChunkValidationResult,
    ProcessedChunk,
    RAGAnswer,
    RawChunk,
    ValidationSummary,
)


class TestChunkType:
    """Test ChunkType enum."""

    def test_chunk_type_values(self):
        """Test ChunkType has expected values."""
        assert ChunkType.TEXT.value == "TEXT"
        assert ChunkType.TABLE.value == "TABLE"
        assert ChunkType.FIGURE.value == "FIGURE"

    def test_chunk_type_iteration(self):
        """Test iterating over ChunkType values."""
        types = list(ChunkType)
        assert len(types) == 3
        assert ChunkType.TEXT in types
        assert ChunkType.TABLE in types
        assert ChunkType.FIGURE in types


class TestRawChunk:
    """Test RawChunk dataclass."""

    def test_raw_chunk_creation(self):
        """Test creating a RawChunk instance."""
        chunk = RawChunk(
            chunk_type=ChunkType.TEXT,
            page_num=1,
            source_file="test.pdf",
            raw_content="Sample text content",
        )
        
        assert chunk.chunk_type == ChunkType.TEXT
        assert chunk.page_num == 1
        assert chunk.source_file == "test.pdf"
        assert chunk.raw_content == "Sample text content"

    def test_raw_chunk_with_image(self):
        """Test RawChunk can hold image content."""
        from PIL import Image
        
        img = Image.new("RGB", (100, 100), color="red")
        
        chunk = RawChunk(
            chunk_type=ChunkType.FIGURE,
            page_num=2,
            source_file="test.pdf",
            raw_content=img,
        )
        
        assert chunk.chunk_type == ChunkType.FIGURE
        assert isinstance(chunk.raw_content, Image.Image)


class TestProcessedChunk:
    """Test ProcessedChunk dataclass."""

    def test_processed_chunk_creation(self, sample_processed_chunk):
        """Test creating a ProcessedChunk instance."""
        assert sample_processed_chunk.chunk_type == ChunkType.TEXT
        assert sample_processed_chunk.page_num == 1
        assert sample_processed_chunk.confidence == 0.9
        assert len(sample_processed_chunk.key_concepts) == 3

    def test_processed_chunk_defaults(self):
        """Test ProcessedChunk default values."""
        chunk = ProcessedChunk(
            chunk_type=ChunkType.TEXT,
            page_num=1,
            source_file="test.pdf",
            structured_text="Content",
            intuition_summary="Summary",
            key_concepts=["concept"],
            confidence=0.8,
        )
        
        # Check default values
        assert chunk.agent_notes == ""
        assert chunk.validation is None


class TestRAGAnswer:
    """Test RAGAnswer dataclass."""

    def test_rag_answer_creation(self, sample_rag_answer):
        """Test creating a RAGAnswer instance."""
        assert sample_rag_answer.question == "What is the capital of France?"
        assert "Paris" in sample_rag_answer.answer
        assert len(sample_rag_answer.source_chunks) == 1

    def test_rag_answer_defaults(self):
        """Test RAGAnswer default values."""
        answer = RAGAnswer(
            question="Test question?",
            answer="Test answer.",
            reasoning_trace="",
            source_chunks=[],
        )
        
        assert answer.trace_id is None
        assert answer.validation_summary is None


class TestValidationResults:
    """Test validation result dataclasses."""

    def test_chunk_validation_result(self):
        """Test ChunkValidationResult creation."""
        result = ChunkValidationResult(
            is_valid=False,
            issues=["Missing key information"],
            corrected=None,
            verdict_score=0.6,
            validator_notes="Needs improvement",
        )
        
        assert result.is_valid is False
        assert len(result.issues) == 1
        assert result.verdict_score == 0.6

    def test_answer_validation_result(self):
        """Test AnswerValidationResult creation."""
        result = AnswerValidationResult(
            is_grounded=True,
            hallucinations=[],
            revised_answer=None,
            verdict_score=0.95,
            validator_notes="All claims supported",
        )
        
        assert result.is_grounded is True
        assert len(result.hallucinations) == 0
        assert result.verdict_score == 0.95

    def test_validation_summary(self):
        """Test ValidationSummary creation."""
        summary = ValidationSummary(
            answer_is_grounded=True,
            hallucinations=[],
            answer_verdict_score=0.9,
            validator_notes="Clean",
            answer_was_revised=False,
        )
        
        assert summary.answer_is_grounded is True
        assert summary.answer_was_revised is False


def test_all_models_import():
    """Test that all models can be imported."""
    from src.core.models import (
        AnswerValidationResult,
        ChunkType,
        ChunkValidationResult,
        ProcessedChunk,
        RAGAnswer,
        RawChunk,
        ValidationSummary,
    )
    
    # Verify all imports are available
    assert ChunkType is not None
    assert RawChunk is not None
    assert ProcessedChunk is not None
    assert RAGAnswer is not None
    assert ValidationSummary is not None
    assert ChunkValidationResult is not None
    assert AnswerValidationResult is not None
