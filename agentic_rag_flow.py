"""
Agentic RAG Flow: 3 Specialized SLMs for Complex PDF Extraction
================================================================
Architecture:
  PDF → Parser → Router
        ├── Agent-1 (Text SLM)    : prose, headers, captions, footnotes
        ├── Agent-2 (Table SLM)   : tables, lists, structured data
        └── Agent-3 (Vision SLM)  : figures, graphs, flowcharts, diagrams
  → Unified Chunk Store (ChromaDB) with rich metadata
  → Orchestrator Agent (query routing + synthesis)

Model recommendations (swap freely):
  SLM-1 Text   : microsoft/Phi-3.5-mini-instruct    (3.8B)
  SLM-2 Table  : Qwen/Qwen2.5-3B-Instruct           (3B)
  SLM-3 Vision : HuggingFaceTB/SmolVLM-Instruct     (2B, multimodal)
  OR all three : microsoft/Phi-3.5-vision-instruct   (4.2B, unified)

Install:
  pip install unstructured[all-docs] pdfplumber pymupdf
  pip install transformers torch chromadb langchain sentence-transformers
  pip install pillow pytesseract camelot-py[cv]
"""

from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

import chromadb
import pdfplumber
import pymupdf  # fitz
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────

class ChunkType(str, Enum):
    TEXT   = "text"
    TABLE  = "table"
    FIGURE = "figure"   # image / graph / flowchart / chart


@dataclass
class RawChunk:
    """Output from PDF parser before any SLM processing."""
    chunk_type : ChunkType
    page_num   : int
    raw_content: Any          # str for text/table, PIL.Image for figure
    bbox       : tuple | None = None   # (x0, y0, x1, y1) on page
    source_file: str = ""


@dataclass
class ProcessedChunk:
    """Output after SLM agent processing — what goes into the vector store."""
    chunk_id      : str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_type    : ChunkType = ChunkType.TEXT
    page_num      : int = 0
    source_file   : str = ""

    # SLM-produced fields
    structured_text  : str = ""   # cleaned, normalised prose or markdown table
    intuition_summary: str = ""   # agent's 1-sentence semantic summary
    key_concepts     : list[str] = field(default_factory=list)
    confidence       : float = 1.0  # agent self-assessed 0–1
    agent_notes      : str = ""   # warnings, ambiguity flags

    # For retrieval
    embedding        : list[float] = field(default_factory=list)


# ─────────────────────────────────────────────
# 2. PDF PARSER  (modality-aware)
# ─────────────────────────────────────────────

class PDFParser:
    """
    Splits a PDF into typed RawChunks using pymupdf + pdfplumber.
    Strategy:
      - pymupdf  : fast page rendering, image extraction
      - pdfplumber: precise table detection
    """

    MIN_TABLE_ROWS = 2
    MIN_TEXT_LEN   = 40   # chars; shorter blocks treated as captions

    def parse(self, pdf_path: str | Path) -> list[RawChunk]:
        pdf_path = Path(pdf_path)
        chunks: list[RawChunk] = []

        doc_fitz = pymupdf.open(str(pdf_path))
        doc_plumber = pdfplumber.open(str(pdf_path))

        for page_idx in range(len(doc_fitz)):
            fitz_page    = doc_fitz[page_idx]
            plumber_page = doc_plumber.pages[page_idx]

            # --- Tables (pdfplumber has best table extraction) ---
            table_bboxes = []
            for table in plumber_page.extract_tables():
                if table and len(table) >= self.MIN_TABLE_ROWS:
                    md_table = self._table_to_markdown(table)
                    chunks.append(RawChunk(
                        chunk_type  = ChunkType.TABLE,
                        page_num    = page_idx + 1,
                        raw_content = md_table,
                        source_file = pdf_path.name,
                    ))
                    # Record bbox to avoid double-extracting this area as text
                    bbox = plumber_page.find_tables()[0].bbox if plumber_page.find_tables() else None
                    if bbox:
                        table_bboxes.append(bbox)

            # --- Images / figures ---
            for img_info in fitz_page.get_images(full=True):
                xref = img_info[0]
                pix  = pymupdf.Pixmap(doc_fitz, xref)
                if pix.n > 4:          # CMYK → RGB
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                if pix.width < 80 or pix.height < 80:
                    continue           # skip decorative tiny images
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                chunks.append(RawChunk(
                    chunk_type  = ChunkType.FIGURE,
                    page_num    = page_idx + 1,
                    raw_content = img,
                    source_file = pdf_path.name,
                ))

            # --- Text blocks (excluding table regions) ---
            raw_text = plumber_page.extract_text() or ""
            if len(raw_text.strip()) >= self.MIN_TEXT_LEN:
                chunks.append(RawChunk(
                    chunk_type  = ChunkType.TEXT,
                    page_num    = page_idx + 1,
                    raw_content = raw_text,
                    source_file = pdf_path.name,
                ))

        doc_fitz.close()
        doc_plumber.close()
        log.info("Parsed %d raw chunks from %s", len(chunks), pdf_path.name)
        return chunks

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """Convert pdfplumber table (list of rows) to Markdown."""
        if not table:
            return ""
        header = "| " + " | ".join(str(c or "") for c in table[0]) + " |"
        sep    = "| " + " | ".join("---" for _ in table[0]) + " |"
        rows   = [
            "| " + " | ".join(str(c or "") for c in row) + " |"
            for row in table[1:]
        ]
        return "\n".join([header, sep] + rows)


# ─────────────────────────────────────────────
# 3. AGENT BASE CLASS
# ─────────────────────────────────────────────

class BaseAgent:
    """
    Contract every SLM agent must fulfil:
      process(chunk) → ProcessedChunk
    Subclasses implement _build_prompt() and _parse_response().
    """

    MAX_RETRIES = 2   # retry if confidence < threshold

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device   = device
        self._load_model()

    def _load_model(self):
        raise NotImplementedError

    def process(self, chunk: RawChunk) -> ProcessedChunk:
        result = self._run(chunk)
        # Self-reflection loop: retry once if low confidence
        if result.confidence < 0.5:
            log.warning("Agent %s: low confidence (%.2f), retrying chunk p.%d",
                        self.__class__.__name__, result.confidence, chunk.page_num)
            result = self._run(chunk, retry=True)
        return result

    def _run(self, chunk: RawChunk, retry: bool = False) -> ProcessedChunk:
        raise NotImplementedError

    @staticmethod
    def _safe_json(text: str) -> dict:
        """Extract JSON from LLM output even if surrounded by markdown fences."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


# ─────────────────────────────────────────────
# 4. AGENT-1: TEXT AGENT
# ─────────────────────────────────────────────

TEXT_SYSTEM_PROMPT = """You are a precise academic document analyst.
Given a text passage from a PDF (academic paper or government report), return ONLY valid JSON:
{
  "structured_text": "<cleaned, de-hyphenated, paragraph-normalised passage>",
  "intuition_summary": "<1 sentence: what this passage establishes>",
  "key_concepts": ["<concept1>", "<concept2>", ...],
  "confidence": <0.0–1.0>,
  "agent_notes": "<any ambiguity, OCR noise, truncation>"
}
Rules:
- Remove page headers/footers/watermarks
- Preserve technical terms exactly
- confidence < 0.6 if passage is clearly incomplete or garbled
"""

RETRY_ADDENDUM = "\nNote: previous extraction had low confidence. Be more conservative; flag unknown parts."


class TextAgent(BaseAgent):

    def _load_model(self):
        log.info("Loading TextAgent: %s", self.model_id)
        self.pipe = pipeline(
            "text-generation",
            model    = self.model_id,
            device   = self.device,
            max_new_tokens = 512,
            do_sample      = False,
        )

    def _run(self, chunk: RawChunk, retry: bool = False) -> ProcessedChunk:
        user_msg = f"PASSAGE:\n{chunk.raw_content}"
        if retry:
            user_msg += RETRY_ADDENDUM

        messages = [
            {"role": "system", "content": TEXT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        output = self.pipe(messages)[0]["generated_text"]
        # Most chat models return the full conversation; extract assistant turn
        last = output[-1]["content"] if isinstance(output, list) else output
        parsed = self._safe_json(last)

        return ProcessedChunk(
            chunk_type       = ChunkType.TEXT,
            page_num         = chunk.page_num,
            source_file      = chunk.source_file,
            structured_text  = parsed.get("structured_text", str(chunk.raw_content)[:2000]),
            intuition_summary= parsed.get("intuition_summary", ""),
            key_concepts     = parsed.get("key_concepts", []),
            confidence       = float(parsed.get("confidence", 0.7)),
            agent_notes      = parsed.get("agent_notes", ""),
        )


# ─────────────────────────────────────────────
# 5. AGENT-2: TABLE AGENT
# ─────────────────────────────────────────────

TABLE_SYSTEM_PROMPT = """You are a structured-data extraction specialist.
Given a Markdown table from a PDF, return ONLY valid JSON:
{
  "structured_text": "<corrected, complete Markdown table>",
  "intuition_summary": "<1 sentence: what this table shows, including units if visible>",
  "key_concepts": ["<column header or metric>", ...],
  "schema": {"columns": ["col1","col2",...], "row_count": <int>, "units": {"col": "unit"}},
  "confidence": <0.0–1.0>,
  "agent_notes": "<merged cells, missing values, parsing artifacts>"
}
"""


class TableAgent(BaseAgent):

    def _load_model(self):
        log.info("Loading TableAgent: %s", self.model_id)
        self.pipe = pipeline(
            "text-generation",
            model    = self.model_id,
            device   = self.device,
            max_new_tokens = 768,
            do_sample      = False,
        )

    def _run(self, chunk: RawChunk, retry: bool = False) -> ProcessedChunk:
        user_msg = f"MARKDOWN TABLE:\n{chunk.raw_content}"
        if retry:
            user_msg += RETRY_ADDENDUM

        messages = [
            {"role": "system", "content": TABLE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        output = self.pipe(messages)[0]["generated_text"]
        last   = output[-1]["content"] if isinstance(output, list) else output
        parsed = self._safe_json(last)

        # Build augmented structured_text that includes schema context
        schema_str = json.dumps(parsed.get("schema", {}), ensure_ascii=False)
        combined   = parsed.get("structured_text", str(chunk.raw_content)) + \
                     f"\n<!-- schema: {schema_str} -->"

        return ProcessedChunk(
            chunk_type       = ChunkType.TABLE,
            page_num         = chunk.page_num,
            source_file      = chunk.source_file,
            structured_text  = combined,
            intuition_summary= parsed.get("intuition_summary", ""),
            key_concepts     = parsed.get("key_concepts", []),
            confidence       = float(parsed.get("confidence", 0.7)),
            agent_notes      = parsed.get("agent_notes", ""),
        )


# ─────────────────────────────────────────────
# 6. AGENT-3: VISION AGENT
# ─────────────────────────────────────────────

VISION_SYSTEM_PROMPT = """You are a scientific figure analyst.
Analyse the provided image (may be a graph, flowchart, diagram, photograph, equation screenshot, or map).
Return ONLY valid JSON:
{
  "figure_type": "<one of: bar_chart | line_chart | scatter_plot | flowchart | table_image |
                   map | photograph | equation | network_diagram | other>",
  "structured_text": "<full textual description including: axis labels, legend, values if readable,
                       flow relationships, or key visual elements>",
  "intuition_summary": "<1 sentence: what insight this figure provides>",
  "key_concepts": ["<axis/variable/node label>", ...],
  "confidence": <0.0–1.0>,
  "agent_notes": "<low resolution, overlapping labels, partially cropped, etc.>"
}
"""


class VisionAgent(BaseAgent):
    """
    Uses a vision-language SLM (SmolVLM, Phi-3.5-vision, Qwen2-VL-2B, etc.)
    Falls back to OCR-only description if vision model unavailable.
    """

    def _load_model(self):
        log.info("Loading VisionAgent: %s", self.model_id)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model     = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=self.device
            )
            self.use_vision = True
        except Exception as e:
            log.warning("VisionAgent: could not load vision model (%s). "
                        "Falling back to caption=filename only.", e)
            self.use_vision = False

    def _run(self, chunk: RawChunk, retry: bool = False) -> ProcessedChunk:
        img: Image.Image = chunk.raw_content

        if not self.use_vision:
            return self._fallback(chunk)

        # Encode image to base64 for models that accept it as text token,
        # OR pass as pixel_values depending on processor type.
        # Below targets SmolVLM / Phi-3.5-vision style interface.
        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user",   "content": [
                {"type": "image"},
                {"type": "text", "text":
                    "Describe this figure." + (RETRY_ADDENDUM if retry else "")
                },
            ]},
        ]

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        output = self.processor.decode(ids[0], skip_special_tokens=True)

        parsed = self._safe_json(output)

        return ProcessedChunk(
            chunk_type       = ChunkType.FIGURE,
            page_num         = chunk.page_num,
            source_file      = chunk.source_file,
            structured_text  = parsed.get("structured_text", output[:1000]),
            intuition_summary= parsed.get("intuition_summary", ""),
            key_concepts     = parsed.get("key_concepts", []),
            confidence       = float(parsed.get("confidence", 0.6)),
            agent_notes      = f"figure_type={parsed.get('figure_type','unknown')} | "
                               + parsed.get("agent_notes", ""),
        )

    def _fallback(self, chunk: RawChunk) -> ProcessedChunk:
        """OCR fallback via pytesseract when no vision model is available."""
        try:
            import pytesseract
            text = pytesseract.image_to_string(chunk.raw_content)
        except Exception:
            text = "[OCR unavailable]"
        return ProcessedChunk(
            chunk_type       = ChunkType.FIGURE,
            page_num         = chunk.page_num,
            source_file      = chunk.source_file,
            structured_text  = text,
            intuition_summary= "Figure content extracted via OCR fallback.",
            confidence       = 0.3,
            agent_notes      = "Vision model not loaded; used OCR.",
        )


# ─────────────────────────────────────────────
# 7. ROUTER  (dispatches RawChunk → correct Agent)
# ─────────────────────────────────────────────

class AgentRouter:

    def __init__(self, text_agent: TextAgent,
                 table_agent: TableAgent,
                 vision_agent: VisionAgent):
        self._map = {
            ChunkType.TEXT  : text_agent,
            ChunkType.TABLE : table_agent,
            ChunkType.FIGURE: vision_agent,
        }

    def route(self, chunk: RawChunk) -> ProcessedChunk:
        agent = self._map[chunk.chunk_type]
        log.info("Routing p.%d (%s) → %s",
                 chunk.page_num, chunk.chunk_type.value, agent.__class__.__name__)
        return agent.process(chunk)


# ─────────────────────────────────────────────
# 8. CHUNK STORE  (ChromaDB + SentenceTransformer)
# ─────────────────────────────────────────────

class ChunkStore:
    """
    Persists ProcessedChunks into ChromaDB.
    Embeds: structured_text + intuition_summary (concatenated).
    Metadata preserved for filtered retrieval.
    """

    EMBED_MODEL = "intfloat/multilingual-e5-small"  # ~120MB, supports Japanese

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embedder = SentenceTransformer(self.EMBED_MODEL)
        self.client   = chromadb.PersistentClient(path=persist_dir)
        self.col      = self.client.get_or_create_collection(
            "agentic_rag",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[ProcessedChunk]):
        texts = [
            f"{c.structured_text}\n\n{c.intuition_summary}"
            for c in chunks
        ]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True).tolist()

        self.col.upsert(
            ids        = [c.chunk_id for c in chunks],
            embeddings = embeddings,
            documents  = [c.structured_text for c in chunks],
            metadatas  = [{
                "chunk_type"      : c.chunk_type.value,
                "page_num"        : c.page_num,
                "source_file"     : c.source_file,
                "intuition_summary": c.intuition_summary,
                "key_concepts"    : json.dumps(c.key_concepts, ensure_ascii=False),
                "confidence"      : c.confidence,
                "agent_notes"     : c.agent_notes,
            } for c in chunks],
        )
        log.info("Upserted %d chunks into ChromaDB", len(chunks))

    def query(self, question: str,
              n_results: int = 6,
              chunk_type: ChunkType | None = None) -> list[dict]:
        vec = self.embedder.encode(
            [question], normalize_embeddings=True
        ).tolist()
        where = {"chunk_type": chunk_type.value} if chunk_type else None
        results = self.col.query(
            query_embeddings = vec,
            n_results        = n_results,
            where            = where,
            include          = ["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"text": doc, "meta": meta, "score": 1 - dist})
        return hits


# ─────────────────────────────────────────────
# 9. ORCHESTRATOR AGENT  (query → answer)
# ─────────────────────────────────────────────

ORCHESTRATOR_PROMPT = """You are a research assistant with access to indexed document chunks.

Retrieved context (ranked by relevance):
{context}

User question:
{question}

Instructions:
- Answer using ONLY the retrieved context.
- If a figure description is referenced, explicitly state it came from a figure.
- If data from a table is used, present it as a structured excerpt.
- State "Insufficient context" if the chunks do not support a reliable answer.
- Always cite source_file and page_num for every claim.
"""


class OrchestratorAgent:
    """
    Performs multi-modal retrieval and synthesis.
    Can be wired to any text-generation SLM or the Anthropic API.
    """

    def __init__(self, store: ChunkStore, llm_pipeline):
        self.store = store
        self.llm   = llm_pipeline

    def answer(self, question: str) -> dict:
        # Step 1: retrieve relevant chunks (all modalities)
        hits = self.store.query(question, n_results=8)

        # Step 2: if question hints at visual content, boost figure results
        visual_keywords = {"figure", "graph", "chart", "flow", "diagram",
                           "image", "plot", "map", "図", "グラフ", "フロー"}
        if any(kw in question.lower() for kw in visual_keywords):
            fig_hits = self.store.query(
                question, n_results=3, chunk_type=ChunkType.FIGURE
            )
            # Merge, deduplicate by chunk text
            seen = {h["text"] for h in hits}
            hits += [h for h in fig_hits if h["text"] not in seen]

        # Step 3: build context string with provenance
        context_parts = []
        for i, h in enumerate(hits, 1):
            meta = h["meta"]
            context_parts.append(
                f"[{i}] ({meta['chunk_type'].upper()} | "
                f"{meta['source_file']} p.{meta['page_num']} | "
                f"score={h['score']:.2f})\n"
                f"Summary: {meta['intuition_summary']}\n"
                f"Content: {h['text'][:800]}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        # Step 4: generate answer
        prompt = ORCHESTRATOR_PROMPT.format(
            context  = context_str,
            question = question,
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self.llm(messages)[0]["generated_text"]
        answer_text = raw[-1]["content"] if isinstance(raw, list) else raw

        return {
            "question"      : question,
            "answer"        : answer_text,
            "source_chunks" : [
                {
                    "type"   : h["meta"]["chunk_type"],
                    "file"   : h["meta"]["source_file"],
                    "page"   : h["meta"]["page_num"],
                    "score"  : round(h["score"], 3),
                    "summary": h["meta"]["intuition_summary"],
                }
                for h in hits
            ],
        }


# ─────────────────────────────────────────────
# 10. PIPELINE ORCHESTRATION  (main entry point)
# ─────────────────────────────────────────────

class AgenticRAGPipeline:
    """
    Wires together: Parser → Router(3 agents) → ChunkStore → OrchestratorAgent.

    Usage:
        pipeline = AgenticRAGPipeline.build(device="cuda")
        pipeline.ingest("path/to/paper.pdf")
        result = pipeline.query("What does Figure 3 show about compression ratio?")
    """

    @classmethod
    def build(
        cls,
        text_model  : str = "microsoft/Phi-3.5-mini-instruct",
        table_model : str = "Qwen/Qwen2.5-3B-Instruct",
        vision_model: str = "HuggingFaceTB/SmolVLM-Instruct",
        orchestrator_model: str = "microsoft/Phi-3.5-mini-instruct",
        device      : str = "cpu",
        persist_dir : str = "./chroma_db",
    ) -> "AgenticRAGPipeline":
        text_agent   = TextAgent(text_model, device)
        table_agent  = TableAgent(table_model, device)
        vision_agent = VisionAgent(vision_model, device)
        router       = AgentRouter(text_agent, table_agent, vision_agent)
        store        = ChunkStore(persist_dir)
        orch_pipe    = pipeline(
            "text-generation",
            model          = orchestrator_model,
            device         = device,
            max_new_tokens = 1024,
            do_sample      = False,
        )
        orchestrator = OrchestratorAgent(store, orch_pipe)

        obj = cls()
        obj.parser       = PDFParser()
        obj.router       = router
        obj.store        = store
        obj.orchestrator = orchestrator
        return obj

    def ingest(self, pdf_path: str | Path) -> list[ProcessedChunk]:
        raw_chunks       = self.parser.parse(pdf_path)
        processed_chunks = [self.router.route(c) for c in raw_chunks]
        # Filter out very low-confidence chunks (noisy / unreadable)
        good_chunks = [c for c in processed_chunks if c.confidence >= 0.25]
        discarded   = len(processed_chunks) - len(good_chunks)
        if discarded:
            log.warning("Discarded %d low-confidence chunks", discarded)
        self.store.upsert(good_chunks)
        return good_chunks

    def query(self, question: str) -> dict:
        return self.orchestrator.answer(question)


# ─────────────────────────────────────────────
# 11. DEMO (run as script)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agentic_rag_flow.py <path_to_pdf> [question]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else \
        "Summarise the main findings and describe any key figures or tables."

    # NOTE: For local testing without GPU, set device="cpu".
    # For production, use device="cuda" and load quantised models (4-bit via bitsandbytes).
    rag = AgenticRAGPipeline.build(device="cpu")

    print(f"\n[INGEST] {pdf_file}")
    chunks = rag.ingest(pdf_file)
    stats = {ct.value: sum(1 for c in chunks if c.chunk_type == ct) for ct in ChunkType}
    print(f"[STATS] {stats}")

    print(f"\n[QUERY] {question}")
    result = rag.query(question)
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for s in result["source_chunks"]:
        print(f"  [{s['type']}] {s['file']} p.{s['page']} (score={s['score']}) — {s['summary']}")
