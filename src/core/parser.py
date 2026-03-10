"""
PDF parsing module for extracting raw chunks from PDF files.

Extracts text, tables, and figures using pdfplumber and PyMuPDF.
"""

import logging
import math
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf
from PIL import Image

from src.core.models import BBox, ChunkType, RawChunk

log = logging.getLogger(__name__)


class PDFParser:
    """
    Extracts raw content chunks from PDF files.

    Uses pdfplumber for text/table extraction and PyMuPDF for images.
    No LLM required - pure PDF processing.

    Attributes:
        MIN_TABLE_ROWS: Minimum rows to consider valid table
        MIN_TEXT_LEN: Minimum text length to extract as text chunk
    """

    MIN_TABLE_ROWS = 2
    MIN_TEXT_LEN = 40
    LINE_MERGE_TOLERANCE = 3.0
    BLOCK_GAP_FACTOR = 1.6
    INDENT_TOLERANCE = 24.0
    COL_GAP_THRESHOLD = 72.0  # horizontal gap in pts that signals a column boundary
    BLOCK_INDENT_TOLERANCE = 80.0  # indent shift in pts that triggers a new block
    TABLE_FIGURE_OVERLAP_THRESHOLD = 0.5
    MAX_TABLE_EMPTY_RATIO = 0.6
    CAPTION_SEARCH_MARGIN = 28.0

    def parse(self, pdf_path: str | Path) -> list[RawChunk]:
        """
        Parse PDF file into raw chunks.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of RawChunk objects (text, table, or figure)
        """
        pdf_path = Path(pdf_path)
        chunks: list[RawChunk] = []
        doc_fitz = pymupdf.open(str(pdf_path))
        doc_plumb = pdfplumber.open(str(pdf_path))

        try:
            for page_idx in range(len(doc_fitz)):
                fitz_page = doc_fitz[page_idx]
                plumb_page = doc_plumb.pages[page_idx]
                page_width = float(plumb_page.width)
                page_height = float(plumb_page.height)
                page_words = plumb_page.extract_words(use_text_flow=True, keep_blank_chars=False) or []

                table_bboxes: list[BBox] = []
                figure_chunks: list[RawChunk] = []
                figure_bboxes: list[BBox] = []

                # Extract images and their placement rectangles first so table
                # filtering can reject figure-like regions.
                for img_info in fitz_page.get_images(full=True):
                    try:
                        xref = img_info[0]
                        rects = fitz_page.get_image_rects(xref) or [None]
                        image = self._extract_image(doc_fitz, xref, page_idx + 1)
                        if image is None:
                            continue

                        for rect in rects:
                            bbox = self._fitz_rect_to_bbox(rect) if rect is not None else None
                            if bbox is not None:
                                figure_bboxes.append(bbox)
                            figure_chunks.append(
                                RawChunk(
                                    chunk_type=ChunkType.FIGURE,
                                    page_num=page_idx + 1,
                                    raw_content=image.copy(),
                                    bbox=bbox,
                                    page_width=page_width,
                                    page_height=page_height,
                                    source_preview=f"image:{image.width}x{image.height}",
                                    source_file=pdf_path.name,
                                )
                            )
                    except Exception as e:
                        log.warning("Error extracting image from page %d: %s. Skipping.", page_idx + 1, e)
                        continue

                # Extract tables with bounding boxes.
                for table in plumb_page.find_tables():
                    rows = table.extract()
                    bbox = self._normalize_bbox(table.bbox)
                    if (
                        rows
                        and len(rows) >= self.MIN_TABLE_ROWS
                        and self._is_probable_table(
                            rows=rows,
                            bbox=bbox,
                            page_words=page_words,
                            figure_bboxes=figure_bboxes,
                        )
                    ):
                        table_bboxes.append(bbox)
                        markdown = self._to_markdown(rows)
                        chunks.append(
                            RawChunk(
                                chunk_type=ChunkType.TABLE,
                                page_num=page_idx + 1,
                                raw_content=markdown,
                                bbox=bbox,
                                page_width=page_width,
                                page_height=page_height,
                                source_preview=markdown[:280],
                                source_file=pdf_path.name,
                            )
                        )

                chunks.extend(figure_chunks)

                # Extract text blocks with bounding boxes.
                for text_chunk in self._extract_text_blocks(
                    words=page_words,
                    page_num=page_idx + 1,
                    page_width=page_width,
                    page_height=page_height,
                    source_file=pdf_path.name,
                    excluded_bboxes=table_bboxes,
                ):
                    chunks.append(text_chunk)
        finally:
            doc_fitz.close()
            doc_plumb.close()

        log.info("Parsed %d raw chunks from %s", len(chunks), pdf_path.name)
        return chunks

    def _extract_text_blocks(
        self,
        words: list[dict[str, Any]],
        page_num: int,
        page_width: float,
        page_height: float,
        source_file: str,
        excluded_bboxes: list[BBox],
    ) -> list[RawChunk]:
        """Extract text as block-level chunks with approximate layout boxes."""
        if not words:
            return []

        filtered_words = [
            word for word in words if not self._is_mostly_within_any_bbox(self._word_bbox(word), excluded_bboxes)
        ]
        lines = self._group_words_into_lines(filtered_words)
        blocks = self._group_lines_into_blocks(lines)

        text_chunks: list[RawChunk] = []
        for block in blocks:
            block_text = "\n".join(line["text"] for line in block).strip()
            if len(block_text) < self.MIN_TEXT_LEN:
                continue

            bbox = self._merge_bboxes([line["bbox"] for line in block])
            text_chunks.append(
                RawChunk(
                    chunk_type=ChunkType.TEXT,
                    page_num=page_num,
                    raw_content=block_text,
                    bbox=bbox,
                    page_width=page_width,
                    page_height=page_height,
                    source_preview=block_text[:280],
                    source_file=source_file,
                )
            )

        return text_chunks

    def _is_probable_table(
        self,
        rows: list[list[Any]],
        bbox: BBox,
        page_words: list[dict[str, Any]],
        figure_bboxes: list[BBox],
    ) -> bool:
        """Return True when a detected grid looks like a real table."""
        normalized_rows = [[str(cell or "").strip() for cell in row] for row in rows]
        if len(normalized_rows) < self.MIN_TABLE_ROWS:
            return False

        total_cells = sum(len(row) for row in normalized_rows)
        non_empty_cells = [cell for row in normalized_rows for cell in row if cell]
        if total_cells == 0 or len(non_empty_cells) < 4:
            return False

        max_cols = max((len(row) for row in normalized_rows), default=0)
        multi_cell_rows = sum(1 for row in normalized_rows if sum(1 for cell in row if cell) >= 2)
        if max_cols < 2 or multi_cell_rows < 2:
            return False

        empty_ratio = 1.0 - (len(non_empty_cells) / total_cells)
        if empty_ratio > self.MAX_TABLE_EMPTY_RATIO:
            return False

        if any(
            self._bbox_overlap_ratio(bbox, figure_bbox) >= self.TABLE_FIGURE_OVERLAP_THRESHOLD for figure_bbox in figure_bboxes
        ):
            return False

        caption_text = self._nearby_caption_text(page_words, bbox).lower()
        has_table_cue = any(token in caption_text for token in ("table", "表"))
        has_figure_cue = any(token in caption_text for token in ("figure", "fig.", "fig ", "図", "chart", "graph", "plot"))
        if has_figure_cue and not has_table_cue:
            return False

        return True

    def _nearby_caption_text(self, page_words: list[dict[str, Any]], bbox: BBox) -> str:
        """Collect nearby caption text around a candidate table bbox."""
        nearby: list[tuple[float, float, str]] = []
        search_top = bbox[1] - self.CAPTION_SEARCH_MARGIN
        search_bottom = bbox[3] + self.CAPTION_SEARCH_MARGIN

        for word in page_words:
            word_bbox = self._word_bbox(word)
            if word_bbox[3] < search_top or word_bbox[1] > search_bottom:
                continue
            if self._horizontal_overlap_ratio(word_bbox, bbox) <= 0.2:
                continue
            nearby.append((word_bbox[1], word_bbox[0], str(word.get("text", "")).strip()))

        return " ".join(text for _, _, text in sorted(nearby) if text)

    def _group_words_into_lines(self, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group pdfplumber words into visual lines."""
        if not words:
            return []

        sorted_words = sorted(words, key=lambda word: (round(float(word["top"]), 1), float(word["x0"])))
        lines: list[list[dict[str, Any]]] = []

        for word in sorted_words:
            if not lines:
                lines.append([word])
                continue

            current_line = lines[-1]
            current_top = float(current_line[0]["top"])
            if math.fabs(float(word["top"]) - current_top) <= self.LINE_MERGE_TOLERANCE:
                current_line.append(word)
            else:
                lines.append([word])

        grouped_lines: list[dict[str, Any]] = []
        for line_words in lines:
            ordered_words = sorted(line_words, key=lambda word: float(word["x0"]))

            # Split on large horizontal gaps to prevent cross-column word merging.
            sub_groups: list[list[dict[str, Any]]] = [[ordered_words[0]]]
            for word in ordered_words[1:]:
                gap = float(word["x0"]) - float(sub_groups[-1][-1]["x1"])
                if gap > self.COL_GAP_THRESHOLD:
                    sub_groups.append([word])
                else:
                    sub_groups[-1].append(word)

            for sub in sub_groups:
                text = " ".join(str(w["text"]).strip() for w in sub if str(w["text"]).strip())
                if not text:
                    continue
                bbox = self._merge_bboxes([self._word_bbox(w) for w in sub])
                grouped_lines.append(
                    {
                        "text": text,
                        "bbox": bbox,
                        "height": max(bbox[3] - bbox[1], 1.0),
                        "left": bbox[0],
                        "top": bbox[1],
                    }
                )

        return grouped_lines

    def _group_lines_into_blocks(self, lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Combine nearby lines into paragraph-like blocks."""
        if not lines:
            return []

        blocks: list[list[dict[str, Any]]] = [[lines[0]]]
        for line in lines[1:]:
            current_block = blocks[-1]
            prev_line = current_block[-1]
            gap = line["bbox"][1] - prev_line["bbox"][3]
            typical_height = max(prev_line["height"], line["height"])
            max_gap = max(typical_height * self.BLOCK_GAP_FACTOR, 10.0)
            indent_delta = abs(line["left"] - prev_line["left"])

            if gap > max_gap or indent_delta > self.BLOCK_INDENT_TOLERANCE:
                blocks.append([line])
            else:
                current_block.append(line)

        return blocks

    def _extract_image(self, doc_fitz: pymupdf.Document, xref: int, page_num: int) -> Image.Image | None:
        """Extract a PIL image from a PyMuPDF xref."""
        pix = pymupdf.Pixmap(doc_fitz, xref)

        if pix.n > 4 or pix.n == 4 or pix.n not in (3, 4):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

        if pix.width < 80 or pix.height < 80:
            return None

        try:
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except ValueError:
            try:
                import numpy as np

                img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                expected_size = pix.width * pix.height * 3
                if len(img_array) < expected_size:
                    log.warning(
                        "Image on page %d has insufficient data (%d bytes, expected %d). Skipping.",
                        page_num,
                        len(img_array),
                        expected_size,
                    )
                    return None
                img_array = img_array[:expected_size].reshape((pix.height, pix.width, 3))
                return Image.fromarray(img_array, "RGB")
            except ImportError:
                log.warning("Cannot extract image from page %d (numpy not available). Skipping.", page_num)
                return None

    @staticmethod
    def _to_markdown(table: list[list]) -> str:
        """
        Convert table to markdown format.

        Args:
            table: List of rows, each row is list of cells

        Returns:
            Markdown-formatted table string
        """
        if not table:
            return ""
        header = "| " + " | ".join(str(c or "") for c in table[0]) + " |"
        sep = "| " + " | ".join("---" for _ in table[0]) + " |"
        rows = ["| " + " | ".join(str(c or "") for c in row) + " |" for row in table[1:]]
        return "\n".join([header, sep] + rows)

    @staticmethod
    def _word_bbox(word: dict[str, Any]) -> BBox:
        """Convert a pdfplumber word record into bbox coordinates."""
        return (
            float(word["x0"]),
            float(word["top"]),
            float(word["x1"]),
            float(word["bottom"]),
        )

    @staticmethod
    def _normalize_bbox(bbox: tuple[float, float, float, float] | list[float]) -> BBox:
        """Convert bbox-like values into a normalized tuple."""
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    @staticmethod
    def _fitz_rect_to_bbox(rect: pymupdf.Rect) -> BBox:
        """Convert a PyMuPDF Rect into a bbox tuple."""
        return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))

    @staticmethod
    def _merge_bboxes(bboxes: list[BBox]) -> BBox:
        """Merge multiple bounding boxes into a single enclosing box."""
        return (
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        )

    def _is_mostly_within_any_bbox(self, bbox: BBox, excluded_bboxes: list[BBox]) -> bool:
        """Return True when the bbox overlaps excluded regions significantly."""
        return any(self._bbox_overlap_ratio(bbox, other) >= 0.6 for other in excluded_bboxes)

    @staticmethod
    def _horizontal_overlap_ratio(left: BBox, right: BBox) -> float:
        """Compute horizontal overlap ratio against the left bbox width."""
        inter_x0 = max(left[0], right[0])
        inter_x1 = min(left[2], right[2])
        if inter_x1 <= inter_x0:
            return 0.0

        intersection = inter_x1 - inter_x0
        left_width = max(left[2] - left[0], 1.0)
        return intersection / left_width

    @staticmethod
    def _bbox_overlap_ratio(left: BBox, right: BBox) -> float:
        """Compute overlap ratio against the left bbox area."""
        inter_x0 = max(left[0], right[0])
        inter_y0 = max(left[1], right[1])
        inter_x1 = min(left[2], right[2])
        inter_y1 = min(left[3], right[3])

        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0.0

        intersection = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        left_area = max((left[2] - left[0]) * (left[3] - left[1]), 1.0)
        return intersection / left_area
