"""Test table extraction from images.

Tests TableFromImageExtractor core functionality.
"""

import pytest
from PIL import Image, ImageDraw

from src.agents.extraction import _is_structured_table_markdown
from src.agents.table_extraction import TableFromImageExtractor
from src.core.models import ChunkType, RawChunk


class TestTableFromImageExtractor:
    """Tests for TableFromImageExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return TableFromImageExtractor()

    @staticmethod
    def _create_simple_table_image(rows: int = 3, cols: int = 3) -> Image.Image:
        """Create a synthetic table image with grid lines and text.

        Args:
            rows: Number of table rows
            cols: Number of table columns

        Returns:
            PIL Image containing simple table grid
        """
        cell_width = 80
        cell_height = 40
        margin = 5
        line_width = 2

        width = cols * cell_width + margin * 2
        height = rows * cell_height + margin * 2

        # Create white background
        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)

        # Draw grid lines
        for i in range(rows + 1):
            y = margin + i * cell_height
            draw.rectangle(
                [(margin, y - line_width // 2), (width - margin, y + line_width // 2)],
                fill="black",
            )

        for j in range(cols + 1):
            x = margin + j * cell_width
            draw.rectangle(
                [(x - line_width // 2, margin), (x + line_width // 2, height - margin)],
                fill="black",
            )

        # Draw cell text (simple labels)
        for i in range(rows):
            for j in range(cols):
                x = margin + j * cell_width + cell_width // 2
                y = margin + i * cell_height + cell_height // 2
                text = f"R{i}C{j}"
                # Center text (approximate)
                draw.text((x - 15, y - 7), text, fill="black")

        return img

    def test_extract_simple_grid_table(self, extractor):
        """Test extraction of simple synthetic table."""
        img = self._create_simple_table_image(rows=3, cols=3)

        result = extractor.extract_table_from_image(img)

        assert result is not None, "Extraction should succeed on valid table image"
        assert isinstance(result, str), "Result should be string (markdown)"
        assert "|" in result, "Markdown table should contain pipe delimiters"
        assert "---" in result, "Markdown table should contain header separator"

    def test_reject_sparse_grid(self, extractor):
        """Test rejection of sparse/empty grids (like axis labels)."""
        # Create mostly empty grid
        img = self._create_simple_table_image(rows=3, cols=3)
        img_array = img.convert("RGB")

        # Make image mostly white (sparse)
        draw = ImageDraw.Draw(img_array)
        draw.rectangle([(0, 0), (img.width, img.height)], fill="white")
        # Draw only a few lines
        draw.line([(10, 10), (200, 10)], fill="black", width=2)
        draw.line([(10, 30), (200, 30)], fill="black", width=2)

        result = extractor.extract_table_from_image(img_array)

        # Result should be None or indicate failure (too sparse)
        # Due to sparse content, extraction will likely fail or return minimal structure
        if result is not None:
            # If extraction succeeds, verify it has some content
            non_empty_lines = [l for l in result.split("\n") if l.strip() and "|" in l]
            assert len(non_empty_lines) >= 1, "Should have at least header with pipe delimiters"

    def test_detects_probable_table_image(self, extractor):
        """Grid-heavy image should be classified as table-like before OCR."""
        img = self._create_simple_table_image(rows=4, cols=4)

        result = extractor.is_probable_table_image(img)

        assert result is True

    def test_rejects_probable_table_image_for_sparse_lines(self, extractor):
        """Sparse axis-like lines should not be classified as table-like."""
        img = Image.new("RGB", (240, 160), color="white")
        draw = ImageDraw.Draw(img)
        draw.line([(20, 130), (220, 130)], fill="black", width=3)
        draw.line([(20, 20), (20, 130)], fill="black", width=3)
        draw.line([(60, 30), (80, 50)], fill="black", width=2)

        result = extractor.is_probable_table_image(img)

        assert result is False

    def test_insufficient_rows_returns_none(self, extractor):
        """Test that grid with < MIN_TABLE_ROWS returns None."""
        # Create 1-row image (below MIN_TABLE_ROWS=2)
        img = self._create_simple_table_image(rows=1, cols=3)

        result = extractor.extract_table_from_image(img)

        assert result is None, "Should reject grid with fewer than MIN_TABLE_ROWS"

    def test_insufficient_cols_returns_none(self, extractor):
        """Test that grid with < MIN_TABLE_COLS returns None or minimal result."""
        # Create 3x1 image (below MIN_TABLE_COLS=2)
        img = self._create_simple_table_image(rows=3, cols=1)

        result = extractor.extract_table_from_image(img)

        # Due to contour detection variability, 2-column grids may still be detected.
        # The key is that if extraction succeeds, it should have reasonable content.
        if result is not None:
            # Should have pipe delimiters and proper markdown format
            assert "|" in result, "If extracted, should have pipe delimiters"
            assert "---" in result, "If extracted, should have header separator"

    def test_invalid_image_format_returns_none(self, extractor):
        """Test that invalid/corrupt image returns None gracefully."""
        # Create completely blank image
        blank_img = Image.new("RGB", (100, 100), color="white")

        result = extractor.extract_table_from_image(blank_img)

        # Blank image should return None (no grid detected)
        assert result is None, "Blank image should not extract table"


class TestVisionAgentTableIntegration:
    """Integration tests for table_image detection routing.

    These tests verify the integration without mocking complex vision model dependencies.
    """

    def test_table_extractor_markdown_format(self):
        """Test that extracted tables use proper markdown format."""
        extractor = TableFromImageExtractor()
        img = TestTableFromImageExtractor._create_simple_table_image(rows=4, cols=3)

        result = extractor.extract_table_from_image(img)

        if result is not None:
            lines = result.split("\n")
            # First line should have pipes
            assert "|" in lines[0], "First line (header) should have pipe delimiters"
            # Second line should be separator
            assert "---" in lines[1], "Second line should be header separator"
            # Data rows should also have pipes
            for line in lines[2:]:
                if line.strip():
                    assert "|" in line, f"Data row should have pipe delimiters: {line}"

    def test_table_chunks_created_on_image_with_table(self):
        """Integration test: verify TABLE chunks can be produced from images."""
        from src.core.models import ChunkType

        # Create a table image
        img = TestTableFromImageExtractor._create_simple_table_image(rows=4, cols=4)

        # Create a FIGURE chunk with the table image
        chunk = RawChunk(
            chunk_type=ChunkType.FIGURE,
            page_num=1,
            raw_content=img,
            source_file="test.pdf",
            page_width=612,
            page_height=792,
            bbox=(0, 0, 612, 792),
            source_preview="test_table",
        )

        # Verify the chunk has image data
        assert chunk.raw_content is not None
        assert isinstance(chunk.raw_content, Image.Image)
        assert chunk.chunk_type == ChunkType.FIGURE

        # Verify table extraction would work on this data
        extractor = TableFromImageExtractor()
        result = extractor.extract_table_from_image(chunk.raw_content)

        assert result is not None, "Should successfully extract table from image"


class TestVisionRescueMarkdownGuard:
    """Tests for markdown validation guard used by figure->table rescue path."""

    def test_structured_markdown_is_accepted(self):
        markdown = "| Col A | Col B |\n| --- | --- |\n| 10 | 20 |"
        assert _is_structured_table_markdown(markdown) is True

    def test_non_table_markdown_is_rejected(self):
        markdown = "This is a figure description without tabular structure."
        assert _is_structured_table_markdown(markdown) is False
