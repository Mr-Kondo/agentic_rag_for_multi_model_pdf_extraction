"""
PDF parsing module for extracting raw chunks from PDF files.

Extracts text, tables, and figures using pdfplumber and PyMuPDF.
"""

import logging
import math
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf
from PIL import Image

from src.core.config import config
from src.core.models import BBox, ChunkType, RawChunk, RegionOCRPolicy

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
    MIN_TEXT_LEN = 24
    LINE_MERGE_TOLERANCE = 2.5
    BLOCK_GAP_FACTOR = 1.45
    INDENT_TOLERANCE = 24.0
    COL_GAP_THRESHOLD = 72.0  # horizontal gap in pts that signals a column boundary
    BLOCK_INDENT_TOLERANCE = 80.0  # indent shift in pts that triggers a new block
    TABLE_FIGURE_OVERLAP_THRESHOLD = 0.65
    MAX_TABLE_EMPTY_RATIO = 0.6
    CAPTION_SEARCH_MARGIN = 28.0
    DEFAULT_MIN_NUMERIC_RATIO_WITHOUT_TABLE_CUE = 0.20
    DEFAULT_MAX_LONG_CELL_RATIO_WITHOUT_TABLE_CUE = 0.40
    FALLBACK_MIN_NUMERIC_RATIO = 0.35
    FALLBACK_MAX_LONG_CELL_RATIO = 0.45
    FALLBACK_LONG_CELL_CHAR_THRESHOLD = 40
    CID_PATTERN = re.compile(r"\(cid:\d+\)")
    PARSER_ENABLE_NATIVE_PDF_HEURISTIC = bool(config.get("parser.enable_native_pdf_heuristic", True))
    PARSER_NATIVE_PDF_MAX_IMAGES = int(config.get("parser.native_pdf_max_images", 2))
    PARSER_NATIVE_PDF_MIN_WORDS = int(config.get("parser.native_pdf_min_words", 50))
    OCR_ENGINE = str(config.get("ocr.engine", "easyocr"))
    OCR_CONFIG = str(config.get("ocr.config", "--oem 3 --psm 6"))
    OCR_DEFAULT_LANG = str(config.get("ocr.default_lang", "jpn+eng"))
    OCR_JAPANESE_LANG = str(config.get("ocr.japanese_lang", "jpn"))
    OCR_FALLBACK_LANG = str(config.get("ocr.fallback_lang", "eng"))
    OCR_RENDER_SCALE = float(config.get("ocr.render_scale", 1.5))
    OCR_TESSERACT_RENDER_SCALE = float(config.get("ocr.tesseract_render_scale", 1.0))
    OCR_PREWARM_EASYOCR = bool(config.get("ocr.prewarm_easyocr", True))
    OCR_MAX_RESCUE_BLOCKS_PER_PAGE = int(config.get("ocr.max_rescue_blocks_per_page", 6))
    OCR_MIN_REOCR_TEXT_LENGTH = int(config.get("ocr.min_reocr_text_length", 6))
    OCR_CACHE_TESSERACT_LANGUAGES = bool(config.get("ocr.cache_tesseract_languages", True))
    OCR_LINE_CONFIDENCE_THRESHOLD = float(config.get("ocr.line_confidence_threshold", 0.55))
    OCR_ENABLE_LINE_REOCR = bool(config.get("ocr.enable_line_reocr", True))
    OCR_MAX_LINE_REOCR_ATTEMPTS = int(config.get("ocr.max_line_reocr_attempts", 2))
    OCR_REGION_POLICIES = dict(config.get("ocr.region_policies", {}))
    OCR_POST_CORRECTION_ENABLED = bool(config.get("ocr.post_correction.enabled", False))
    OCR_POST_CORRECTION_PATHS = list(config.get("ocr.post_correction.dictionary_paths", []))
    OCR_POST_CORRECTION_APPLY_TO_OCR_ONLY = bool(config.get("ocr.post_correction.apply_to_ocr_only", True))
    OCR_READABILITY_THRESHOLD = 5
    OCR_MIN_CHARS = 24
    FALLBACK_TABLE_SETTINGS = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
    }
    ENABLE_FIGURE_AWARE_FALLBACK = False

    def __init__(self, enable_figure_aware_fallback: bool | None = None):
        """Initialize parser with optional feature flag overrides."""
        if enable_figure_aware_fallback is None:
            self.enable_figure_aware_fallback = self.ENABLE_FIGURE_AWARE_FALLBACK
        else:
            self.enable_figure_aware_fallback = enable_figure_aware_fallback
        self._ocr_lang_warning_emitted = False
        self._available_tesseract_languages: set[str] | None = None
        self._available_tesseract_languages_loaded = False
        self._easyocr_reader: Any | None = None
        self._last_ocr_metadata: dict[str, Any] | None = None
        if self.OCR_PREWARM_EASYOCR and self.OCR_ENGINE == "easyocr":
            self._get_easyocr_reader()

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
                page_images = fitz_page.get_images(full=True)
                page_words = plumb_page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
                if self._should_skip_fitz_word_extraction(page_words, image_count=len(page_images)):
                    fitz_words: list[dict[str, Any]] = []
                    text_words = page_words
                else:
                    fitz_words = self._extract_fitz_words(fitz_page)
                    text_words = self._select_text_words(page_words, fitz_words)

                table_bboxes: list[BBox] = []
                figure_chunks: list[RawChunk] = []
                figure_bboxes: list[BBox] = []

                # Extract images and their placement rectangles first so table
                # filtering can reject figure-like regions.
                for img_info in page_images:
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
                table_candidates = self._find_table_candidates(
                    plumb_page,
                    has_figures=bool(figure_bboxes),
                )
                accepted_tables = 0
                rejected_reasons: dict[str, int] = {}
                for table, is_fallback in table_candidates:
                    rows = table.extract()
                    bbox = self._normalize_bbox(table.bbox)
                    reject_reason = self._table_rejection_reason(
                        rows=rows,
                        bbox=bbox,
                        page_words=text_words,
                        figure_bboxes=figure_bboxes,
                        is_fallback=is_fallback,
                    )
                    if rows and len(rows) >= self.MIN_TABLE_ROWS and reject_reason is None:
                        table_bboxes.append(bbox)
                        accepted_tables += 1
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
                    elif reject_reason is not None:
                        rejected_reasons[reject_reason] = rejected_reasons.get(reject_reason, 0) + 1
                metrics = self._build_table_metrics(table_candidates, accepted_tables, rejected_reasons)
                log.info(
                    "Page %d: table metrics total=%d default=%d fallback=%d accepted=%d rejected=%d",
                    page_idx + 1,
                    metrics["total_candidates"],
                    metrics["default_candidates"],
                    metrics["fallback_candidates"],
                    metrics["accepted_candidates"],
                    metrics["rejected_candidates"],
                )
                if metrics["rejected_reasons"]:
                    log.debug("Page %d: table rejection reasons=%s", page_idx + 1, metrics["rejected_reasons"])

                chunks.extend(figure_chunks)

                # Extract text blocks with bounding boxes.
                for text_chunk in self._extract_text_blocks(
                    words=text_words,
                    page_num=page_idx + 1,
                    page_width=page_width,
                    page_height=page_height,
                    source_file=pdf_path.name,
                    excluded_bboxes=table_bboxes,
                    fitz_page=fitz_page,
                    max_ocr_rescues_per_page=self.OCR_MAX_RESCUE_BLOCKS_PER_PAGE,
                ):
                    chunks.append(text_chunk)
        finally:
            doc_fitz.close()
            doc_plumb.close()

        log.info("Parsed %d raw chunks from %s", len(chunks), pdf_path.name)
        return chunks

    def _should_skip_fitz_word_extraction(self, plumb_words: list[dict[str, Any]], image_count: int) -> bool:
        """Return True when a page is likely native PDF text and fitz word extraction is unnecessary."""
        if not self.PARSER_ENABLE_NATIVE_PDF_HEURISTIC:
            return False
        if image_count > self.PARSER_NATIVE_PDF_MAX_IMAGES:
            return False
        if len(plumb_words) < self.PARSER_NATIVE_PDF_MIN_WORDS:
            return False

        has_cid_artifacts = any(self.CID_PATTERN.search(str(word.get("text", ""))) for word in plumb_words)
        if has_cid_artifacts:
            return False

        log.debug(
            "Skipping PyMuPDF word extraction for native-like page (words=%d, images=%d)",
            len(plumb_words),
            image_count,
        )
        return True

    def _extract_fitz_words(self, fitz_page: pymupdf.Page) -> list[dict[str, Any]]:
        """Extract word boxes from PyMuPDF and convert to a pdfplumber-like shape."""
        try:
            raw_words = fitz_page.get_text("words")
        except Exception as e:
            log.debug("PyMuPDF word extraction failed: %s", e)
            return []

        converted: list[dict[str, Any]] = []
        for item in raw_words:
            if len(item) < 5:
                continue
            text = str(item[4]).strip()
            if not text:
                continue
            converted.append(
                {
                    "x0": float(item[0]),
                    "top": float(item[1]),
                    "x1": float(item[2]),
                    "bottom": float(item[3]),
                    "text": text,
                }
            )
        return converted

    def _select_text_words(
        self,
        plumb_words: list[dict[str, Any]],
        fitz_words: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Choose the word source with better readability and fewer CID artifacts."""
        if not fitz_words:
            return plumb_words
        if not plumb_words:
            return fitz_words
        if len(plumb_words) == len(fitz_words):
            plumb_has_cid = any(self.CID_PATTERN.search(str(word.get("text", ""))) for word in plumb_words)
            fitz_has_cid = any(self.CID_PATTERN.search(str(word.get("text", ""))) for word in fitz_words)
            if not plumb_has_cid or fitz_has_cid:
                return plumb_words

        plumb_score = self._word_quality_score(plumb_words)
        fitz_score = self._word_quality_score(fitz_words)

        if fitz_score > plumb_score:
            log.info(
                "Selected PyMuPDF words over pdfplumber words (fitz=%d, plumb=%d)",
                fitz_score,
                plumb_score,
            )
            return fitz_words
        return plumb_words

    def _get_region_policy(self, chunk_type: ChunkType) -> RegionOCRPolicy:
        """Resolve OCR policy for a specific chunk type with safe fallbacks."""
        key = chunk_type.value
        policy_map = self.OCR_REGION_POLICIES if isinstance(self.OCR_REGION_POLICIES, dict) else {}
        policy = policy_map.get(key, {}) if isinstance(policy_map.get(key, {}), dict) else {}
        return RegionOCRPolicy(
            engine=str(policy.get("engine", self.OCR_ENGINE)),
            line_confidence_threshold=float(policy.get("line_confidence_threshold", self.OCR_LINE_CONFIDENCE_THRESHOLD)),
            enable_reocr=bool(policy.get("enable_reocr", self.OCR_ENABLE_LINE_REOCR)),
            max_reocr_attempts=int(policy.get("max_reocr_attempts", self.OCR_MAX_LINE_REOCR_ATTEMPTS)),
            apply_post_correction=bool(policy.get("apply_post_correction", self.OCR_POST_CORRECTION_ENABLED)),
        )

    def _word_quality_score(self, words: list[dict[str, Any]]) -> int:
        """Return a deterministic score where higher means more readable text."""
        if not words:
            return -10_000

        total_words = len(words)
        cid_words = 0
        cid_tokens = 0
        readable_words = 0
        total_chars = 0

        for word in words:
            text = str(word.get("text", ""))
            found_cid_tokens = self.CID_PATTERN.findall(text)
            if found_cid_tokens:
                cid_words += 1
                cid_tokens += len(found_cid_tokens)
            text_without_cid = self.CID_PATTERN.sub("", text)
            alpha_numeric_chars = sum(ch.isalnum() for ch in text_without_cid)
            total_chars += len(text)
            if alpha_numeric_chars > 0:
                readable_words += 1

        # Weighting prioritizes removing CID artifacts first, then readability.
        return (readable_words * 8) + total_chars + total_words - (cid_words * 15) - (cid_tokens * 40)

    def _extract_text_blocks(
        self,
        words: list[dict[str, Any]],
        page_num: int,
        page_width: float,
        page_height: float,
        source_file: str,
        excluded_bboxes: list[BBox],
        fitz_page: pymupdf.Page | None = None,
        max_ocr_rescues_per_page: int | None = None,
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
        text_policy = self._get_region_policy(ChunkType.TEXT)
        ocr_rescues_used = 0
        rescue_budget = max_ocr_rescues_per_page if max_ocr_rescues_per_page is None else max(0, max_ocr_rescues_per_page)
        for block in blocks:
            block_text = "\n".join(line["text"] for line in block)
            block_text = self._clean_extracted_text(block_text)
            if len(block_text) < self.MIN_TEXT_LEN:
                continue

            bbox = self._merge_bboxes([line["bbox"] for line in block])
            ocr_metadata: dict[str, Any] | None = None
            should_attempt_rescue = self._should_try_ocr(block_text)
            if should_attempt_rescue and (rescue_budget is None or ocr_rescues_used < rescue_budget):
                block_text, ocr_metadata = self._rescue_text_with_ocr(
                    original_text=block_text,
                    bbox=bbox,
                    fitz_page=fitz_page,
                    policy=text_policy,
                )
                ocr_rescues_used += 1
            block_text = self._clean_extracted_text(block_text)
            if len(block_text) < self.MIN_TEXT_LEN:
                continue
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
                    ocr_metadata=ocr_metadata or {},
                )
            )

        return text_chunks

    def _rescue_text_with_ocr(
        self,
        original_text: str,
        bbox: BBox,
        fitz_page: pymupdf.Page | None,
        policy: RegionOCRPolicy | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Try OCR rescue for low-readability text and keep the better candidate."""
        original_text = self._normalize_text(original_text)
        if fitz_page is None:
            return original_text, None

        if not self._should_try_ocr(original_text):
            return original_text, None

        self._last_ocr_metadata = None
        active_policy = policy or self._get_region_policy(ChunkType.TEXT)

        ocr_text = self._ocr_text_from_bbox(fitz_page=fitz_page, bbox=bbox, policy=active_policy)
        if not ocr_text:
            return original_text, None

        ocr_text = self._normalize_text(ocr_text)

        best_text = self._choose_better_text(original_text, ocr_text)
        metadata = self._last_ocr_metadata.copy() if self._last_ocr_metadata else None
        if best_text is ocr_text:
            log.info("Applied OCR rescue for low-readability text block")

        should_apply_post_correction = (
            active_policy.apply_post_correction
            and self.OCR_POST_CORRECTION_ENABLED
            and (not self.OCR_POST_CORRECTION_APPLY_TO_OCR_ONLY or best_text is ocr_text)
        )

        return best_text, metadata

    def _ocr_text_from_bbox(
        self,
        fitz_page: pymupdf.Page,
        bbox: BBox,
        policy: RegionOCRPolicy | None = None,
    ) -> str | None:
        """
        Extract text via OCR from a clipped page region.

        Dispatches to EasyOCR (primary) or Tesseract (fallback / explicit setting)
        based on the configured ``ocr.engine`` value.
        """
        self._last_ocr_metadata = None
        active_policy = policy or self._get_region_policy(ChunkType.TEXT)
        if active_policy.engine == "easyocr":
            result = self._ocr_text_from_bbox_easyocr(fitz_page=fitz_page, bbox=bbox, policy=active_policy)
            if result:
                return result
            log.debug("EasyOCR returned no text; falling back to Tesseract")
        return self._ocr_text_from_bbox_tesseract(fitz_page=fitz_page, bbox=bbox)

    def _ocr_text_from_bbox_easyocr(
        self,
        fitz_page: pymupdf.Page,
        bbox: BBox,
        policy: RegionOCRPolicy | None = None,
    ) -> str | None:
        """
        Extract text via EasyOCR from a clipped page region.

        Renders a 3x-scale pixmap of the bounding box and feeds it to the
        cached EasyOCR Reader.  Results are sorted top-to-bottom,
        left-to-right and joined with newlines.

        Args:
            fitz_page: PyMuPDF page object for pixel rendering.
            bbox: Region of interest as (x0, y0, x1, y1).

        Returns:
            Extracted text string, or None when EasyOCR is unavailable or
            returns no results.
        """
        reader = self._get_easyocr_reader()
        if reader is None:
            return None

        try:
            import numpy as np

            rect = pymupdf.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            pix = fitz_page.get_pixmap(
                matrix=pymupdf.Matrix(self.OCR_RENDER_SCALE, self.OCR_RENDER_SCALE), clip=rect, alpha=False
            )
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        except Exception as e:
            log.debug("EasyOCR: failed to render clip: %s", e)
            return None

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)
                results = reader.readtext(img_array, detail=1, paragraph=False)
        except Exception as e:
            log.debug("EasyOCR: readtext failed: %s", e)
            return None

        if not results:
            return None

        active_policy = policy or self._get_region_policy(ChunkType.TEXT)
        line_entries = self._easyocr_results_to_line_entries(results)
        if active_policy.enable_reocr:
            line_entries, metadata = self._reocr_low_confidence_lines(
                fitz_page=fitz_page,
                parent_bbox=bbox,
                line_entries=line_entries,
                policy=active_policy,
            )
        else:
            metadata = {
                "engine": "easyocr",
                "line_count": len(line_entries),
                "low_confidence_line_count": sum(
                    1 for line in line_entries if line["confidence"] < active_policy.line_confidence_threshold
                ),
                "reocr_attempts": 0,
                "reocr_replaced_lines": 0,
            }

        self._last_ocr_metadata = metadata
        text = "\n".join(line["text"] for line in line_entries if line["text"].strip())
        return text if text.strip() else None

    @staticmethod
    def _easyocr_results_to_line_entries(results: list[Any]) -> list[dict[str, Any]]:
        """
        Convert EasyOCR readtext results into ordered text lines.

        Each result is a tuple of (bbox_quad, text, confidence).  Items are
        sorted by the vertical centre of their bounding quadrilateral, then
        grouped into lines when successive items share a similar y-centre
        (within one character height).  Within each line items are sorted
        left-to-right by x-centre.

        Args:
            results: Raw output from ``reader.readtext(detail=1)``.

        Returns:
            List of dicts containing ``text``, ``confidence`` and ``bbox_local``.
        """
        if not results:
            return []

        annotated: list[tuple[float, float, str, float, tuple[float, float, float, float], float]] = []
        for item in results:
            quad, text, conf = item[0], item[1], item[2]
            ys = [pt[1] for pt in quad]
            xs = [pt[0] for pt in quad]
            min_x = float(min(xs))
            min_y = float(min(ys))
            max_x = float(max(xs))
            max_y = float(max(ys))
            y_centre = (min(ys) + max(ys)) / 2.0
            x_centre = (min(xs) + max(xs)) / 2.0
            char_height = max(ys) - min(ys)
            confidence = float(conf) if conf is not None else 0.0
            annotated.append((y_centre, x_centre, text, char_height, (min_x, min_y, max_x, max_y), confidence))

        annotated.sort(key=lambda t: t[0])

        lines: list[list[tuple[float, float, str, tuple[float, float, float, float], float]]] = []
        for y_centre, x_centre, text, char_height, local_bbox, confidence in annotated:
            tolerance = max(float(char_height) * 0.6, 8.0)
            if lines and abs(y_centre - lines[-1][0][0]) <= tolerance:
                lines[-1].append((y_centre, x_centre, text, local_bbox, confidence))
            else:
                lines.append([(y_centre, x_centre, text, local_bbox, confidence)])

        output: list[dict[str, Any]] = []
        for line_items in lines:
            sorted_items = sorted(line_items, key=lambda t: t[1])
            line_text = " ".join(item[2] for item in sorted_items).strip()
            weighted_chars = sum(max(len(item[2].strip()), 1) for item in sorted_items)
            weighted_confidence = sum(item[4] * max(len(item[2].strip()), 1) for item in sorted_items) / weighted_chars
            min_x = min(item[3][0] for item in sorted_items)
            min_y = min(item[3][1] for item in sorted_items)
            max_x = max(item[3][2] for item in sorted_items)
            max_y = max(item[3][3] for item in sorted_items)
            output.append(
                {
                    "text": line_text,
                    "confidence": weighted_confidence,
                    "bbox_local": (min_x, min_y, max_x, max_y),
                }
            )
        return output

    def _reocr_low_confidence_lines(
        self,
        fitz_page: pymupdf.Page,
        parent_bbox: BBox,
        line_entries: list[dict[str, Any]],
        policy: RegionOCRPolicy | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retry OCR only for low-confidence lines using Tesseract."""
        active_policy = policy or self._get_region_policy(ChunkType.TEXT)
        threshold = active_policy.line_confidence_threshold
        max_attempts = active_policy.max_reocr_attempts
        attempts = 0
        replaced_lines = 0
        low_confidence_lines = 0

        for entry in line_entries:
            confidence = float(entry.get("confidence", 0.0))
            if confidence >= threshold:
                continue
            low_confidence_lines += 1
            if attempts >= max_attempts:
                continue

            current_text = str(entry.get("text", "")).strip()
            if len(current_text) < self.OCR_MIN_REOCR_TEXT_LENGTH:
                continue

            local_bbox = entry.get("bbox_local")
            if not isinstance(local_bbox, tuple) or len(local_bbox) != 4:
                continue

            attempts += 1
            page_bbox = self._local_bbox_to_page_bbox(local_bbox, parent_bbox, self.OCR_RENDER_SCALE)
            tesseract_candidate = self._ocr_text_from_bbox_tesseract(fitz_page=fitz_page, bbox=page_bbox)
            if not tesseract_candidate:
                continue

            candidate_text = self._normalize_text(tesseract_candidate).strip()
            if not candidate_text:
                continue

            better_text = self._choose_better_text(current_text, candidate_text)
            if better_text == candidate_text:
                entry["text"] = candidate_text
                entry["reocr_applied"] = True
                replaced_lines += 1

        metadata = {
            "engine": "easyocr",
            "line_count": len(line_entries),
            "low_confidence_line_count": low_confidence_lines,
            "reocr_attempts": attempts,
            "reocr_replaced_lines": replaced_lines,
        }
        return line_entries, metadata

    def _local_bbox_to_page_bbox(
        self,
        local_bbox: tuple[float, float, float, float],
        parent_bbox: BBox,
        render_scale: float,
    ) -> BBox:
        """Convert local OCR image coordinates back into page coordinates."""
        x0 = parent_bbox[0] + (local_bbox[0] / render_scale)
        y0 = parent_bbox[1] + (local_bbox[1] / render_scale)
        x1 = parent_bbox[0] + (local_bbox[2] / render_scale)
        y1 = parent_bbox[1] + (local_bbox[3] / render_scale)
        return (x0, y0, x1, y1)

    def _get_easyocr_reader(self) -> Any | None:
        """
        Return the cached EasyOCR Reader, initialising it on first call.

        Uses Metal Performance Shaders (MPS) on Apple Silicon when available,
        otherwise CUDA, otherwise CPU.  Import errors are caught so that
        environments without EasyOCR installed degrade gracefully.

        Returns:
            Initialised ``easyocr.Reader`` instance, or None on failure.
        """
        if self._easyocr_reader is not None:
            return self._easyocr_reader

        try:
            import easyocr
            import torch
        except ImportError as exc:
            log.warning("EasyOCR is not installed; OCR will fall back to Tesseract. (%s)", exc)
            return None

        gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
        try:
            # Suppress pin_memory warning on MPS: EasyOCR hardcodes
            # DataLoader(pin_memory=True) which is unsupported on MPS devices.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*pin_memory.*",
                    category=UserWarning,
                )
                self._easyocr_reader = easyocr.Reader(["ja", "en"], gpu=gpu)
            log.info("EasyOCR Reader initialised (gpu=%s)", gpu)
        except Exception as exc:
            log.warning("EasyOCR Reader initialisation failed: %s", exc)
            return None

        return self._easyocr_reader

    def _ocr_text_from_bbox_tesseract(self, fitz_page: pymupdf.Page, bbox: BBox) -> str | None:
        """
        Extract text via Tesseract from a clipped page region.

        Renders a 2x-scale pixmap and tries each configured language in order.

        Args:
            fitz_page: PyMuPDF page object for pixel rendering.
            bbox: Region of interest as (x0, y0, x1, y1).

        Returns:
            Extracted text string, or None when Tesseract is unavailable or
            returns no results.
        """
        try:
            import pytesseract
        except Exception as e:
            log.debug("OCR dependency unavailable: %s", e)
            return None

        try:
            rect = pymupdf.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            pix = fitz_page.get_pixmap(
                matrix=pymupdf.Matrix(self.OCR_TESSERACT_RENDER_SCALE, self.OCR_TESSERACT_RENDER_SCALE),
                clip=rect,
                alpha=False,
            )
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception as e:
            log.debug("Failed to render OCR clip: %s", e)
            return None

        trial_langs = self._ocr_language_sequence()
        for lang in trial_langs:
            try:
                ocr_text = pytesseract.image_to_string(
                    image,
                    lang=lang,
                    config=self.OCR_CONFIG,
                ).strip()
                if ocr_text:
                    return ocr_text
                log.debug("OCR returned empty text with lang=%s", lang)
            except Exception as e:
                log.debug("OCR with lang=%s failed: %s", lang, e)

        log.debug("OCR failed for all configured languages: %s", trial_langs)

        return None

    def _ocr_language_sequence(self) -> tuple[str, ...]:
        """Return OCR language trial order with duplicates removed."""
        ordered = [
            self.OCR_DEFAULT_LANG.strip(),
            self.OCR_JAPANESE_LANG.strip(),
            self.OCR_FALLBACK_LANG.strip(),
        ]
        unique_langs: list[str] = []
        for lang in ordered:
            if lang and lang not in unique_langs:
                unique_langs.append(lang)

        available_langs = self._get_available_tesseract_languages()
        if available_langs:
            unavailable = [lang for lang in unique_langs if not self._is_ocr_lang_available(lang, available_langs)]
            if unavailable and not self._ocr_lang_warning_emitted:
                log.warning(
                    "Configured OCR languages unavailable in local tesseract data: %s. "
                    "Install language data (e.g., jpn) or update settings.",
                    unavailable,
                )
                self._ocr_lang_warning_emitted = True

            filtered_langs = [lang for lang in unique_langs if self._is_ocr_lang_available(lang, available_langs)]
            if filtered_langs:
                return tuple(filtered_langs)

        return tuple(unique_langs)

    @staticmethod
    def _is_ocr_lang_available(lang: str, available_langs: set[str]) -> bool:
        """Return True when all language tokens in a tesseract lang expression are available."""
        return all(token in available_langs for token in lang.split("+") if token)

    def _get_available_tesseract_languages(self) -> set[str] | None:
        """Return installed tesseract language codes, or None when detection is unavailable."""
        if self.OCR_CACHE_TESSERACT_LANGUAGES and self._available_tesseract_languages_loaded:
            return self._available_tesseract_languages

        try:
            import pytesseract
        except Exception as e:
            log.debug("Cannot validate OCR languages because pytesseract is unavailable: %s", e)
            if self.OCR_CACHE_TESSERACT_LANGUAGES:
                self._available_tesseract_languages = None
                self._available_tesseract_languages_loaded = True
            return None

        try:
            langs = pytesseract.get_languages(config="")
            available = {str(lang).strip() for lang in langs if str(lang).strip()}
            if self.OCR_CACHE_TESSERACT_LANGUAGES:
                self._available_tesseract_languages = available
                self._available_tesseract_languages_loaded = True
            return available
        except Exception as e:
            log.debug("Cannot list tesseract languages: %s", e)
            if self.OCR_CACHE_TESSERACT_LANGUAGES:
                self._available_tesseract_languages = None
                self._available_tesseract_languages_loaded = True
            return None

    def _clean_extracted_text(self, text: str) -> str:
        """Normalize and sanitize extracted text to reduce CID-driven mojibake."""
        normalized = self._normalize_text(text)
        without_cid = self.CID_PATTERN.sub("", normalized)
        without_cid = without_cid.replace("\u3000", " ")
        # Collapse repeated spaces while preserving intentional line breaks.
        return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in without_cid.splitlines()).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Apply Unicode normalization to stabilize text comparison and scoring."""
        return unicodedata.normalize("NFC", text or "")

    def _should_try_ocr(self, text: str) -> bool:
        """Return True when text likely suffers from extraction corruption."""
        stripped = text.strip()
        if len(stripped) < self.OCR_MIN_CHARS:
            return False
        return self._text_readability_score(stripped) < self.OCR_READABILITY_THRESHOLD

    def _choose_better_text(self, original_text: str, ocr_text: str) -> str:
        """Choose the more readable text candidate using deterministic scoring."""
        original_score = self._text_readability_score(original_text)
        ocr_score = self._text_readability_score(ocr_text)
        return ocr_text if ocr_score > original_score else original_text

    def _text_readability_score(self, text: str) -> int:
        """Score text readability for OCR fallback decisions."""
        if not text:
            return -10_000

        cid_tokens = len(self.CID_PATTERN.findall(text))
        ascii_alnum = 0
        cjk_chars = 0
        suspicious_chars = 0

        for ch in text:
            if ch.isascii() and ch.isalnum():
                ascii_alnum += 1
                continue
            code = ord(ch)
            is_cjk = (
                0x3000 <= code <= 0x303F  # CJK symbols and punctuation: 。、「」【】
                or 0x3040 <= code <= 0x30FF  # Hiragana / Katakana
                or 0x31F0 <= code <= 0x31FF  # Katakana phonetic extensions
                or 0x3400 <= code <= 0x4DBF  # CJK unified ideographs extension A
                or 0x4E00 <= code <= 0x9FFF  # CJK unified ideographs
                or 0xF900 <= code <= 0xFAFF  # CJK compatibility ideographs
                or 0xFF00 <= code <= 0xFFEF  # Fullwidth and halfwidth forms
            )
            if is_cjk:
                cjk_chars += 1
                continue
            if ch.isspace() or ch.isdigit() or ch in ".,;:!?()[]{}'\"+-*/=_#%&@~|":
                continue
            suspicious_chars += 1

        # Prefer ASCII/CJK readable text and penalize CID/suspicious glyph noise.
        return (ascii_alnum * 2) + (cjk_chars * 3) - (cid_tokens * 20) - (suspicious_chars * 4)

    def _find_table_candidates(
        self,
        plumb_page: pdfplumber.page.Page,
        has_figures: bool = False,
    ) -> list[tuple[Any, bool]]:
        """Find table candidates and mark whether they came from fallback strategy."""
        default_candidates: list[Any] = []
        fallback_candidates: list[Any] = []

        try:
            default_candidates = plumb_page.find_tables()
        except Exception as e:
            log.warning("find_tables() failed with default strategy: %s", e)

        # Precision-first policy by default: fallback text strategy is used only
        # when default candidates are absent. Phase 2 trial allows fallback on
        # figure-heavy pages when explicitly enabled by feature flag.
        should_try_fallback = (not default_candidates) or (self.enable_figure_aware_fallback and has_figures)
        if should_try_fallback:
            try:
                fallback_candidates = plumb_page.find_tables(table_settings=self.FALLBACK_TABLE_SETTINGS)
            except Exception as e:
                log.debug("find_tables() fallback strategy not available: %s", e)

        candidates_with_source: list[tuple[Any, bool]] = [(candidate, False) for candidate in default_candidates] + [
            (candidate, True) for candidate in fallback_candidates
        ]

        deduped: list[tuple[Any, bool]] = []
        seen_keys: set[tuple[float, float, float, float]] = set()
        for candidate, is_fallback in candidates_with_source:
            bbox = getattr(candidate, "bbox", None)
            if bbox is None:
                continue
            try:
                normalized = self._normalize_bbox(bbox)
            except Exception:
                continue
            key = tuple(round(v, 2) for v in normalized)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append((candidate, is_fallback))

        return deduped

    def _build_table_metrics(
        self,
        table_candidates: list[tuple[Any, bool]],
        accepted_tables: int,
        rejected_reasons: dict[str, int],
    ) -> dict[str, Any]:
        """Build page-level table detection metrics for diagnostics and baselines."""
        fallback_candidates = sum(1 for _, is_fallback in table_candidates if is_fallback)
        total_candidates = len(table_candidates)
        rejected_candidates = max(total_candidates - accepted_tables, 0)
        return {
            "total_candidates": total_candidates,
            "default_candidates": total_candidates - fallback_candidates,
            "fallback_candidates": fallback_candidates,
            "accepted_candidates": accepted_tables,
            "rejected_candidates": rejected_candidates,
            "rejected_reasons": dict(sorted(rejected_reasons.items())),
        }

    def _is_probable_table(
        self,
        rows: list[list[Any]],
        bbox: BBox,
        page_words: list[dict[str, Any]],
        figure_bboxes: list[BBox],
    ) -> bool:
        """Return True when a detected grid looks like a real table."""
        return self._table_rejection_reason(rows, bbox, page_words, figure_bboxes) is None

    def _table_rejection_reason(
        self,
        rows: list[list[Any]],
        bbox: BBox,
        page_words: list[dict[str, Any]],
        figure_bboxes: list[BBox],
        is_fallback: bool = False,
    ) -> str | None:
        """Return rejection reason for a table candidate, or None if accepted."""
        normalized_rows = [[str(cell or "").strip() for cell in row] for row in rows]
        if len(normalized_rows) < self.MIN_TABLE_ROWS:
            log.debug(
                "Rejecting table candidate: insufficient rows (%d < %d) at bbox %s",
                len(normalized_rows),
                self.MIN_TABLE_ROWS,
                bbox,
            )
            return "insufficient_rows"

        total_cells = sum(len(row) for row in normalized_rows)
        non_empty_cells = [cell for row in normalized_rows for cell in row if cell]
        if total_cells == 0 or len(non_empty_cells) < 4:
            log.debug(
                "Rejecting table candidate: insufficient content (total=%d, non_empty=%d) at bbox %s",
                total_cells,
                len(non_empty_cells),
                bbox,
            )
            return "insufficient_content"

        max_cols = max((len(row) for row in normalized_rows), default=0)
        multi_cell_rows = sum(1 for row in normalized_rows if sum(1 for cell in row if cell) >= 2)
        if max_cols < 2 or multi_cell_rows < 2:
            log.debug(
                "Rejecting table candidate: weak structure (cols=%d, multi_cell_rows=%d) at bbox %s",
                max_cols,
                multi_cell_rows,
                bbox,
            )
            return "weak_structure"

        empty_ratio = 1.0 - (len(non_empty_cells) / total_cells)
        if empty_ratio > self.MAX_TABLE_EMPTY_RATIO:
            log.debug(
                "Rejecting table candidate: too sparse (empty_ratio=%.2f > %.2f) at bbox %s",
                empty_ratio,
                self.MAX_TABLE_EMPTY_RATIO,
                bbox,
            )
            return "too_sparse"

        overlapping_figures = [
            (i, self._bbox_overlap_ratio(bbox, fb))
            for i, fb in enumerate(figure_bboxes)
            if self._bbox_overlap_ratio(bbox, fb) >= self.TABLE_FIGURE_OVERLAP_THRESHOLD
        ]
        if overlapping_figures:
            log.debug(
                "Rejecting table candidate: overlaps with figures %s (threshold=%.2f) at bbox %s",
                overlapping_figures,
                self.TABLE_FIGURE_OVERLAP_THRESHOLD,
                bbox,
            )
            return "overlaps_figure"

        caption_text = self._nearby_caption_text(page_words, bbox).lower()
        has_table_cue = any(token in caption_text for token in ("table", "表"))
        has_figure_cue = any(token in caption_text for token in ("figure", "fig.", "fig ", "図", "chart", "graph", "plot"))
        if has_figure_cue and not has_table_cue:
            log.debug("Rejecting table candidate: figure-like caption (caption='%s') at bbox %s", caption_text[:80], bbox)
            return "figure_like_caption"

        numeric_ratio = self._numeric_cell_ratio(non_empty_cells)
        long_cell_ratio = self._long_cell_ratio(non_empty_cells)

        if not is_fallback and not has_table_cue:
            if (
                numeric_ratio < self.DEFAULT_MIN_NUMERIC_RATIO_WITHOUT_TABLE_CUE
                and long_cell_ratio > self.DEFAULT_MAX_LONG_CELL_RATIO_WITHOUT_TABLE_CUE
            ):
                log.debug(
                    "Rejecting default table candidate: prose-like cells without table cue "
                    "(numeric_ratio=%.2f, long_cell_ratio=%.2f) at bbox %s",
                    numeric_ratio,
                    long_cell_ratio,
                    bbox,
                )
                return "default_prose_like_without_table_cue"

        if is_fallback:
            if not has_table_cue and numeric_ratio < self.FALLBACK_MIN_NUMERIC_RATIO:
                log.debug(
                    "Rejecting fallback table candidate: low numeric signal (ratio=%.2f < %.2f) at bbox %s",
                    numeric_ratio,
                    self.FALLBACK_MIN_NUMERIC_RATIO,
                    bbox,
                )
                return "fallback_low_table_signal"

            if long_cell_ratio > self.FALLBACK_MAX_LONG_CELL_RATIO:
                log.debug(
                    "Rejecting fallback table candidate: prose-like cells (ratio=%.2f > %.2f) at bbox %s",
                    long_cell_ratio,
                    self.FALLBACK_MAX_LONG_CELL_RATIO,
                    bbox,
                )
                return "fallback_prose_like_cells"

        return None

    def _numeric_cell_ratio(self, non_empty_cells: list[str]) -> float:
        """Estimate how table-like a candidate is based on numeric cell prevalence."""
        if not non_empty_cells:
            return 0.0
        numeric_cells = sum(1 for cell in non_empty_cells if self._looks_numeric_cell(cell))
        return numeric_cells / len(non_empty_cells)

    def _long_cell_ratio(self, non_empty_cells: list[str]) -> float:
        """Estimate prose-likeness by counting very long sentence-like cells."""
        if not non_empty_cells:
            return 0.0
        long_cells = sum(
            1 for cell in non_empty_cells if len(cell) >= self.FALLBACK_LONG_CELL_CHAR_THRESHOLD or cell.count(" ") >= 8
        )
        return long_cells / len(non_empty_cells)

    @staticmethod
    def _looks_numeric_cell(text: str) -> bool:
        """Return True when a cell resembles a numeric metric value."""
        normalized = text.strip().replace(",", "").replace("%", "").replace("$", "")
        if not normalized:
            return False

        digit_count = sum(ch.isdigit() for ch in normalized)
        alpha_count = sum(ch.isalpha() for ch in normalized)
        if digit_count == 0:
            return False
        return digit_count >= max(1, alpha_count)

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
