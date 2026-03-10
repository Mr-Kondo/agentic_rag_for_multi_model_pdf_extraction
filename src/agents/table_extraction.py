"""Table extraction from images using OCR and contour detection.

Provides TableFromImageExtractor for converting image-embedded tables to
structured markdown format using pytesseract (OCR) and OpenCV (structure detection).
"""

import logging
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

log = logging.getLogger(__name__)


class TableFromImageExtractor:
    """
    Extract structured tables from image data using OCR and grid detection.

    Uses pytesseract for optical character recognition and OpenCV contour
    detection to identify cell boundaries. Converts detected cells to markdown
    table format.

    Attributes:
        MIN_TABLE_ROWS: Minimum rows to consider valid table (default: 3)
        MIN_TABLE_COLS: Minimum columns to consider valid table (default: 3)
        MIN_CELL_HEIGHT: Minimum pixel height for a row (default: 15)
        MIN_CELL_WIDTH: Minimum pixel width for a column (default: 20)
        TESSERACT_CONFIG: Configuration string for pytesseract
    """

    MIN_TABLE_ROWS = 3
    MIN_TABLE_COLS = 3
    MIN_CELL_HEIGHT = 15
    MIN_CELL_WIDTH = 20
    TESSERACT_CONFIG = r"--oem 3 --psm 6"

    def __init__(self):
        """Initialize extractor and verify pytesseract is available."""
        try:
            pytesseract.pytesseract.get_tesseract_version()
            log.debug("TableFromImageExtractor: tesseract-ocr available")
        except pytesseract.TesseractNotFoundError as e:
            log.error("TableFromImageExtractor: tesseract-ocr not found. Install with: brew install tesseract")
            raise RuntimeError("System dependency 'tesseract-ocr' not found. Please install it first.") from e

    def extract_table_from_image(self, image: Image.Image) -> Optional[str]:
        """
        Extract table from image as markdown-formatted string.

        Attempts to:
        1. Detect table grid structure via contour analysis
        2. Extract cell text using OCR
        3. Format cells into markdown table with borders

        Args:
            image: PIL Image object potentially containing a table

        Returns:
            Markdown-formatted table string (with |/--- separators) or None if
            extraction fails (grid too sparse, no cells detected, or too few rows/columns)

        Raises:
            RuntimeError: If tesseract-ocr system dependency is not installed
        """
        try:
            # Convert PIL Image to numpy array (OpenCV format)
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Detect table grid structure
            cells = self._detect_table_structure(img_cv)
            if cells is None or len(cells) < self.MIN_TABLE_ROWS:
                log.debug(
                    "TableFromImageExtractor: insufficient rows (got %d, need %d)",
                    len(cells) if cells else 0,
                    self.MIN_TABLE_ROWS,
                )
                return None

            if len(cells[0]) < self.MIN_TABLE_COLS:
                log.debug(
                    "TableFromImageExtractor: insufficient columns (got %d, need %d)",
                    len(cells[0]),
                    self.MIN_TABLE_COLS,
                )
                return None

            # Extract text from detected cells
            cells_with_text = self._extract_cell_text(img_cv, cells)

            # Check content sparsity
            non_empty_cells = sum(1 for row in cells_with_text for cell in row if cell.strip())
            total_cells = sum(len(row) for row in cells_with_text)
            empty_ratio = 1.0 - (non_empty_cells / total_cells) if total_cells > 0 else 1.0

            if empty_ratio > 0.8:
                log.debug("TableFromImageExtractor: too sparse (empty_ratio=%.2f > 0.80)", empty_ratio)
                return None

            # Convert cells to markdown table
            markdown = self._cells_to_markdown(cells_with_text)
            log.debug(
                "TableFromImageExtractor: extracted table (%d rows x %d cols, empty=%.2f)",
                len(cells_with_text),
                len(cells_with_text[0]),
                empty_ratio,
            )
            return markdown

        except Exception as e:
            log.warning("TableFromImageExtractor.extract_table_from_image() failed: %s", e)
            return None

    def _detect_table_structure(self, img: np.ndarray) -> Optional[list[tuple[int, int, int, int]]]:
        """
        Detect table grid structure using contour analysis.

        Converts image to grayscale, applies thresholding, detects horizontal and
        vertical lines, and extracts cell bounding boxes from grid intersections.

        Args:
            img: OpenCV image array (BGR format)

        Returns:
            List of 2D cell bounding boxes [(x, y, w, h), ...] organized by row-column,
            or None if grid detection fails

        Structure:
            cells[row][col] = (x, y, w, h) bounding box
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Threshold to binary for line detection
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

            # Combine lines
            grid = cv2.add(horizontal_lines, vertical_lines)

            # Find contours to detect cells
            contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                log.debug("TableFromImageExtractor._detect_table_structure: no contours found")
                return None

            # Extract bounding boxes
            bboxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w >= self.MIN_CELL_WIDTH and h >= self.MIN_CELL_HEIGHT:
                    bboxes.append((x, y, w, h))

            if not bboxes:
                log.debug("TableFromImageExtractor._detect_table_structure: no valid cells detected")
                return None

            # Sort by position: top-to-bottom, left-to-right
            bboxes = sorted(bboxes, key=lambda b: (b[1], b[0]))

            # Group into rows
            rows = []
            current_row = []
            current_y = bboxes[0][1]

            for bbox in bboxes:
                x, y, w, h = bbox
                # Check if bbox is in current row (within vertical tolerance)
                if abs(y - current_y) <= 10:
                    current_row.append(bbox)
                else:
                    # New row
                    if current_row:
                        # Sort current row by x position
                        current_row = sorted(current_row, key=lambda b: b[0])
                        rows.append(current_row)
                    current_row = [bbox]
                    current_y = y

            # Add last row
            if current_row:
                current_row = sorted(current_row, key=lambda b: b[0])
                rows.append(current_row)

            if not rows:
                return None

            return rows

        except Exception as e:
            log.warning("TableFromImageExtractor._detect_table_structure() failed: %s", e)
            return None

    def _extract_cell_text(self, img: np.ndarray, cells: list[list[tuple[int, int, int, int]]]) -> list[list[str]]:
        """
        Extract text from each detected cell using OCR.

        Args:
            img: OpenCV image array (BGR format)
            cells: 2D list of cell bounding boxes from _detect_table_structure()

        Returns:
            2D list of cell text strings: cells_with_text[row][col] = "cell text"
        """
        cells_with_text: list[list[str]] = []

        for row in cells:
            text_row: list[str] = []
            for x, y, w, h in row:
                # Extract cell region
                cell_region = img[y : y + h, x : x + w]

                # OCR on cell
                try:
                    text = pytesseract.image_to_string(cell_region, config=self.TESSERACT_CONFIG).strip()
                except Exception as e:
                    log.debug("TableFromImageExtractor: OCR failed for cell at (%d, %d): %s", x, y, e)
                    text = ""

                text_row.append(text)
            cells_with_text.append(text_row)

        return cells_with_text

    def _cells_to_markdown(self, cells: list[list[str]]) -> str:
        """
        Convert 2D cell array to markdown table format.

        Args:
            cells: 2D list of cell text strings

        Returns:
            Markdown table with pipe delimiters and header separator row
        """
        if not cells or not cells[0]:
            return ""

        lines: list[str] = []

        # First row as header
        if cells:
            header_row = cells[0]
            header_line = "| " + " | ".join(cell or "---" for cell in header_row) + " |"
            lines.append(header_line)

            # Separator row
            sep_line = "|" + "|".join(" --- " for _ in header_row) + "|"
            lines.append(sep_line)

            # Data rows
            for row in cells[1:]:
                data_line = "| " + " | ".join(cell or "" for cell in row) + " |"
                lines.append(data_line)

        return "\n".join(lines)
