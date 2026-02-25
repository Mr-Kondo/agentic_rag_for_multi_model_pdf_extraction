"""
PDF parsing module for extracting raw chunks from PDF files.

Extracts text, tables, and figures using pdfplumber and PyMuPDF.
"""

import logging
from pathlib import Path

import pdfplumber
import pymupdf
from PIL import Image

from src.core.models import ChunkType, RawChunk

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

        for page_idx in range(len(doc_fitz)):
            fitz_page = doc_fitz[page_idx]
            plumb_page = doc_plumb.pages[page_idx]

            # Extract tables
            for table in plumb_page.extract_tables():
                if table and len(table) >= self.MIN_TABLE_ROWS:
                    chunks.append(
                        RawChunk(
                            chunk_type=ChunkType.TABLE,
                            page_num=page_idx + 1,
                            raw_content=self._to_markdown(table),
                            source_file=pdf_path.name,
                        )
                    )

            # Extract images
            for img_info in fitz_page.get_images(full=True):
                try:
                    xref = img_info[0]
                    pix = pymupdf.Pixmap(doc_fitz, xref)

                    # Convert to RGB if not already
                    if pix.n > 4:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    elif pix.n == 4:
                        # RGBA - convert to RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    elif pix.n not in (3, 4):
                        # Other color spaces - convert to RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                    # Skip very small images
                    if pix.width < 80 or pix.height < 80:
                        continue

                    # Create PIL Image from pixmap data with fallback handling
                    try:
                        # First try with raw samples (works for properly formatted pixmaps)
                        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    except ValueError:
                        # Fallback: use numpy array conversion if available
                        try:
                            import numpy as np

                            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                            # Reshape to proper dimensions
                            expected_size = pix.width * pix.height * 3
                            if len(img_array) >= expected_size:
                                img_array = img_array[:expected_size].reshape((pix.height, pix.width, 3))
                                img = Image.fromarray(img_array, "RGB")
                            else:
                                log.warning(
                                    f"Image on page {page_idx + 1} has insufficient data "
                                    f"({len(img_array)} bytes, expected {expected_size}). Skipping."
                                )
                                continue
                        except ImportError:
                            # numpy not available - skip this image
                            log.warning(f"Cannot extract image from page {page_idx + 1} (numpy not available). Skipping.")
                            continue

                    chunks.append(
                        RawChunk(
                            chunk_type=ChunkType.FIGURE,
                            page_num=page_idx + 1,
                            raw_content=img,
                            source_file=pdf_path.name,
                        )
                    )
                except Exception as e:
                    log.warning(f"Error extracting image from page {page_idx + 1}: {e}. Skipping.")
                    continue

            # Extract text
            raw_text = plumb_page.extract_text() or ""
            if len(raw_text.strip()) >= self.MIN_TEXT_LEN:
                chunks.append(
                    RawChunk(
                        chunk_type=ChunkType.TEXT,
                        page_num=page_idx + 1,
                        raw_content=raw_text,
                        source_file=pdf_path.name,
                    )
                )

        doc_fitz.close()
        doc_plumb.close()
        log.info("Parsed %d raw chunks from %s", len(chunks), pdf_path.name)
        return chunks

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
