"""
agentic_rag_flow_v2.py
======================
Changes from v1:
  1. OrchestratorAgent → ReasoningOrchestratorAgent
       - Uses a ~10B reasoning-capable SLM
       - Strips <think>...</think> internal chain-of-thought before returning answer
       - Separates reasoning_trace from final_answer in the return value
  2. Langfuse observability injected at every stage:
       - Ingestion: parse, per-agent calls, upsert
       - Query: retrieval spans, orchestrator generation with token usage
  3. Cleaner injection pattern: _trace handle is set on agents by the pipeline
     before each call — no global state, Copilot-friendly explicit flow.

Model recommendations for ReasoningOrchestratorAgent (~10B):
  - Qwen/Qwen3-8B                              (thinking mode via enable_thinking=True)
  - deepseek-ai/DeepSeek-R1-Distill-Llama-8B   (8B GGUF distil, good reasoning)
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   (14B if VRAM allows)
  All produce <think> blocks; _strip_reasoning() handles all three formats.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import pdfplumber
import pymupdf
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, pipeline

from langfuse_tracer import LangfuseTracer, _TraceHandle

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ═══════════════════════════════════════════════════════════
# 1. DATA STRUCTURES  (unchanged from v1)
# ═══════════════════════════════════════════════════════════

from enum import Enum


class ChunkType(str, Enum):
    TEXT   = "text"
    TABLE  = "table"
    FIGURE = "figure"


@dataclass
class RawChunk:
    chunk_type : ChunkType
    page_num   : int
    raw_content: Any
    bbox       : tuple | None = None
    source_file: str = ""


@dataclass
class ProcessedChunk:
    chunk_id         : str        = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_type       : ChunkType  = ChunkType.TEXT
    page_num         : int        = 0
    source_file      : str        = ""
    structured_text  : str        = ""
    intuition_summary: str        = ""
    key_concepts     : list[str]  = field(default_factory=list)
    confidence       : float      = 1.0
    agent_notes      : str        = ""
    embedding        : list[float]= field(default_factory=list)


@dataclass
class RAGAnswer:
    """Final output of ReasoningOrchestratorAgent."""
    question       : str
    answer         : str           # cleaned answer (think-block stripped)
    reasoning_trace: str           # raw <think> content for debugging / Langfuse
    source_chunks  : list[dict]    = field(default_factory=list)
    trace_id       : str           = ""   # Langfuse trace ID for drill-down


# ═══════════════════════════════════════════════════════════
# 2. PDF PARSER
# ═══════════════════════════════════════════════════════════

class PDFParser:
    MIN_TABLE_ROWS = 2
    MIN_TEXT_LEN   = 40

    def parse(self, pdf_path: str | Path) -> list[RawChunk]:
        pdf_path   = Path(pdf_path)
        chunks     : list[RawChunk] = []
        doc_fitz   = pymupdf.open(str(pdf_path))
        doc_plumb  = pdfplumber.open(str(pdf_path))

        for page_idx in range(len(doc_fitz)):
            fitz_page  = doc_fitz[page_idx]
            plumb_page = doc_plumb.pages[page_idx]

            # ── Tables ──
            for table in plumb_page.extract_tables():
                if table and len(table) >= self.MIN_TABLE_ROWS:
                    chunks.append(RawChunk(
                        chunk_type  = ChunkType.TABLE,
                        page_num    = page_idx + 1,
                        raw_content = self._table_to_markdown(table),
                        source_file = pdf_path.name,
                    ))

            # ── Figures ──
            for img_info in fitz_page.get_images(full=True):
                xref = img_info[0]
                pix  = pymupdf.Pixmap(doc_fitz, xref)
                if pix.n > 4:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                if pix.width < 80 or pix.height < 80:
                    continue
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                chunks.append(RawChunk(
                    chunk_type  = ChunkType.FIGURE,
                    page_num    = page_idx + 1,
                    raw_content = img,
                    source_file = pdf_path.name,
                ))

            # ── Text ──
            raw_text = plumb_page.extract_text() or ""
            if len(raw_text.strip()) >= self.MIN_TEXT_LEN:
                chunks.append(RawChunk(
                    chunk_type  = ChunkType.TEXT,
                    page_num    = page_idx + 1,
                    raw_content = raw_text,
                    source_file = pdf_path.name,
                ))

        doc_fitz.close()
        doc_plumb.close()
        log.info("Parsed %d raw chunks from %s", len(chunks), pdf_path.name)
        return chunks

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        if not table:
            return ""
        header = "| " + " | ".join(str(c or "") for c in table[0]) + " |"
        sep    = "| " + " | ".join("---" for _ in table[0]) + " |"
        rows   = ["| " + " | ".join(str(c or "") for c in row) + " |" for row in table[1:]]
        return "\n".join([header, sep] + rows)


# ═══════════════════════════════════════════════════════════
# 3. BASE AGENT
# ═══════════════════════════════════════════════════════════

class BaseAgent:
    """
    Contract: process(chunk, trace?) → ProcessedChunk
    Subclasses implement _build_messages() and _parse_response().
    _trace is injected by the pipeline per-call, not stored permanently.
    """

    CONFIDENCE_THRESHOLD = 0.5
    RETRY_SUFFIX = "\n[RETRY] Previous attempt had low confidence. Be conservative; mark unknowns."

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device   = device
        self._load_model()

    def _load_model(self):
        raise NotImplementedError

    def process(
        self,
        chunk: RawChunk,
        trace: _TraceHandle | None = None,
    ) -> ProcessedChunk:
        span_name = f"agent_{chunk.chunk_type.value}"
        if trace:
            with trace.span(span_name, input={"page": chunk.page_num, "type": chunk.chunk_type.value}) as s:
                result = self._run_with_retry(chunk)
                s.update(output={
                    "confidence": result.confidence,
                    "key_concepts": result.key_concepts[:5],
                    "notes": result.agent_notes,
                })
        else:
            result = self._run_with_retry(chunk)
        return result

    def _run_with_retry(self, chunk: RawChunk) -> ProcessedChunk:
        result = self._run(chunk, retry=False)
        if result.confidence < self.CONFIDENCE_THRESHOLD:
            log.warning("%s: low confidence (%.2f) on p.%d — retrying",
                        self.__class__.__name__, result.confidence, chunk.page_num)
            result = self._run(chunk, retry=True)
        return result

    def _run(self, chunk: RawChunk, retry: bool = False) -> ProcessedChunk:
        raise NotImplementedError

    @staticmethod
    def _safe_json(text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _last_content(output: Any) -> str:
        """Extract assistant text from various pipeline output shapes."""
        if isinstance(output, list) and output:
            last = output[-1]
            if isinstance(last, dict):
                return last.get("content", str(last))
        return str(output)


# ═══════════════════════════════════════════════════════════
# 4. TEXT AGENT  (SLM-1 ~3-4B)
# ═══════════════════════════════════════════════════════════

_TEXT_SYSTEM = """You are a precise academic document analyst.
Given a text passage from a PDF (academic paper or government report), return ONLY valid JSON:
{
  "structured_text": "<cleaned, de-hyphenated, paragraph-normalised passage>",
  "intuition_summary": "<1 sentence: what this passage establishes>",
  "key_concepts": ["<concept1>", "<concept2>"],
  "confidence": <0.0-1.0>,
  "agent_notes": "<ambiguity, OCR noise, truncation>"
}"""


class TextAgent(BaseAgent):

    def _load_model(self):
        log.info("Loading TextAgent: %s", self.model_id)
        self._pipe = pipeline(
            "text-generation",
            model          = self.model_id,
            device         = self.device,
            max_new_tokens = 512,
            do_sample      = False,
        )

    def _run(self, chunk: RawChunk, retry: bool = False) -> ProcessedChunk:
        user_content = str(chunk.raw_content)
        if retry:
            user_content += self.RETRY_SUFFIX
        messages = [
            {"role": "system", "content": _TEXT_SYSTEM},
            {"role": "user",   "content": f"PASSAGE:\n{user_content}"},
        ]
        raw    = self._pipe(messages)[0]["generated_text"]
        parsed = self._safe_json(self._last_content(raw))
        return ProcessedChunk(
            chunk_type       = ChunkType.TEXT,
            page_num         = chunk.page_num,
            source_file      = chunk.source_file,
            structured_text  = parsed.get("structured_text", user_content[:2000]),
            intuition_summary= parsed.get("intuition_summary", ""),
            key_concepts     = parsed.get("key_concepts", []),
            confidence       = float(parsed.get("confidence", 0.7)),
            agent_notes      = parsed.get("agent_notes", ""),
        )


# ═══════════════════════════════════════════════════════════
# 5. TABLE AGENT  (SLM-2 ~3B)
# ═══════════════════════════════════════════════════════════

_TABLE_SYSTEM = """You are a structured-data extraction specialist.
Given a Markdown table from a PDF, return ONLY valid JSON:
{
  "structured_text": "<corrected Markdown table>",
  "intuition_summary": "<1 sentence: what this table shows, including units>",
  "key_concepts": ["<column headers or metrics>"],
  "schema": {"columns": [], "row_count": 0, "units": {}},
  "confidence": <0.0-1.0>,
  "agent_notes": "<merged cells, missing values, parsing artifacts>"
}"""


class TableAgent(BaseAgent):

    def _load_model(self):
        log.info("Loading TableAgent: %s", self.model_id)
        self._pipe = pipeline(
            "text-generation",
            model          = self.model_id,
            device         = self.device,
            max_new_tokens = 768,
            do_sample      = False,
        )

    def _run(self, chunk: RawChunk, retry: bool = False) -> ProcessedChunk:
        user_content = str(chunk.raw_content)
        if retry:
            user_content += self.RETRY_SUFFIX
        messages = [
            {"role": "system", "content": _TABLE_SYSTEM},
            {"role": "user",   "content": f"MARKDOWN TABLE:\n{user_content}"},
        ]
        raw    = self._pipe(messages)[0]["generated_text"]
        parsed = self._safe_json(self._last_content(raw))
        schema_annotation = f"\n<!-- schema: {json.dumps(parsed.get('schema', {}), ensure_ascii=False)} -->"
        return ProcessedChunk(
            chunk_type       = ChunkType.TABLE,
            page_num         = chunk.page_num,
            source_file      = chunk.source_file,
            structured_text  = parsed.get("structured_text", user_content) + schema_annotation,
            intuition_summary= parsed.get("intuition_summary", ""),
            key_concepts     = parsed.get("key_concepts", []),
            confidence       = float(parsed.get("confidence", 0.7)),
            agent_notes      = parsed.get("agent_notes", ""),
        )


# ═══════════════════════════════════════════════════════════
# 6. VISION AGENT  (SLM-3 ~2B multimodal)
# ═══════════════════════════════════════════════════════════

_VISION_SYSTEM = """You are a scientific figure analyst.
Analyse the provided image. Return ONLY valid JSON:
{
  "figure_type": "<bar_chart|line_chart|scatter_plot|flowchart|table_image|map|photograph|equation|network_diagram|other>",
  "structured_text": "<full description: axes, legend, values, flow nodes, key elements>",
  "intuition_summary": "<1 sentence: what insight this figure provides>",
  "key_concepts": ["<axis/variable/node label>"],
  "confidence": <0.0-1.0>,
  "agent_notes": "<resolution issues, overlapping labels, cropping>"
}"""


class VisionAgent(BaseAgent):

    def _load_model(self):
        log.info("Loading VisionAgent: %s", self.model_id)
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model     = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=self.device
            )
            self._use_vision = True
        except Exception as e:
            log.warning("VisionAgent: vision model failed to load (%s). Using OCR fallback.", e)
            self._use_vision = False

    def _run(self, chunk: RawChunk, retry: bool = False) -> ProcessedChunk:
        if not self._use_vision:
            return self._ocr_fallback(chunk)

        img: Image.Image = chunk.raw_content
        extra = self.RETRY_SUFFIX if retry else ""
        messages = [
            {"role": "system", "content": _VISION_SYSTEM},
            {"role": "user",   "content": [
                {"type": "image"},
                {"type": "text", "text": f"Describe this figure.{extra}"},
            ]},
        ]
        prompt  = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs  = self._processor(text=prompt, images=[img], return_tensors="pt")
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
        output = self._processor.decode(ids[0], skip_special_tokens=True)
        parsed = self._safe_json(output)

        return ProcessedChunk(
            chunk_type       = ChunkType.FIGURE,
            page_num         = chunk.page_num,
            source_file      = chunk.source_file,
            structured_text  = parsed.get("structured_text", output[:1000]),
            intuition_summary= parsed.get("intuition_summary", ""),
            key_concepts     = parsed.get("key_concepts", []),
            confidence       = float(parsed.get("confidence", 0.6)),
            agent_notes      = f"figure_type={parsed.get('figure_type', 'unknown')} | {parsed.get('agent_notes', '')}",
        )

    def _ocr_fallback(self, chunk: RawChunk) -> ProcessedChunk:
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


# ═══════════════════════════════════════════════════════════
# 7. AGENT ROUTER
# ═══════════════════════════════════════════════════════════

class AgentRouter:

    def __init__(self, text: TextAgent, table: TableAgent, vision: VisionAgent):
        self._map = {
            ChunkType.TEXT  : text,
            ChunkType.TABLE : table,
            ChunkType.FIGURE: vision,
        }

    def route(self, chunk: RawChunk, trace: _TraceHandle | None = None) -> ProcessedChunk:
        agent = self._map[chunk.chunk_type]
        log.info("Routing p.%d (%s) → %s",
                 chunk.page_num, chunk.chunk_type.value, agent.__class__.__name__)
        return agent.process(chunk, trace=trace)


# ═══════════════════════════════════════════════════════════
# 8. CHUNK STORE  (ChromaDB + multilingual-e5)
# ═══════════════════════════════════════════════════════════

class ChunkStore:
    EMBED_MODEL = "intfloat/multilingual-e5-small"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._embedder = SentenceTransformer(self.EMBED_MODEL)
        self._client   = chromadb.PersistentClient(path=persist_dir)
        self._col      = self._client.get_or_create_collection(
            "agentic_rag",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[ProcessedChunk], trace: _TraceHandle | None = None):
        span_ctx = trace.span("upsert_store", input={"n": len(chunks)}) if trace else None
        texts = [f"{c.structured_text}\n\n{c.intuition_summary}" for c in chunks]
        embs  = self._embedder.encode(texts, normalize_embeddings=True).tolist()
        self._col.upsert(
            ids        = [c.chunk_id for c in chunks],
            embeddings = embs,
            documents  = [c.structured_text for c in chunks],
            metadatas  = [{
                "chunk_type"       : c.chunk_type.value,
                "page_num"         : c.page_num,
                "source_file"      : c.source_file,
                "intuition_summary": c.intuition_summary,
                "key_concepts"     : json.dumps(c.key_concepts, ensure_ascii=False),
                "confidence"       : c.confidence,
                "agent_notes"      : c.agent_notes,
            } for c in chunks],
        )
        log.info("Upserted %d chunks", len(chunks))
        if span_ctx:
            with span_ctx as s:
                s.update(output={"upserted": len(chunks)})

    def query(
        self,
        question  : str,
        n_results : int = 6,
        chunk_type: ChunkType | None = None,
    ) -> list[dict]:
        vec   = self._embedder.encode([question], normalize_embeddings=True).tolist()
        where = {"chunk_type": chunk_type.value} if chunk_type else None
        res   = self._col.query(
            query_embeddings = vec,
            n_results        = n_results,
            where            = where,
            include          = ["documents", "metadatas", "distances"],
        )
        return [
            {"text": doc, "meta": meta, "score": 1 - dist}
            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            )
        ]


# ═══════════════════════════════════════════════════════════
# 9. REASONING ORCHESTRATOR AGENT  (~10B SLM)
#    Key additions vs v1:
#    a) Strips <think>...</think> from raw output
#    b) Preserves reasoning_trace in RAGAnswer for Langfuse logging
#    c) Reports token usage to Langfuse generation span
# ═══════════════════════════════════════════════════════════

_ORCHESTRATOR_SYSTEM = """You are a research assistant with deep reasoning capability.
You have access to indexed document chunks that include text, tables, and figure descriptions.

Think step-by-step inside <think> tags before writing your final answer.
Your final answer must:
  - Be grounded ONLY in the retrieved context below.
  - Cite source_file and page_num for every claim.
  - If a figure description is relevant, explicitly note it is from a figure.
  - If context is insufficient, state "Insufficient context — cannot answer reliably."

Retrieved context:
{context}

Question:
{question}
"""

_VISUAL_KEYWORDS = {
    "figure", "graph", "chart", "flow", "diagram", "image", "plot", "map",
    "図", "グラフ", "フロー", "フローチャート", "チャート", "表",
}


class ReasoningOrchestratorAgent:
    """
    10B-class reasoning SLM orchestrator.

    Supported model formats:
      - Any HuggingFace chat model that wraps its CoT in <think>...</think>.
      - Confirmed: Qwen3-8B (thinking mode), DeepSeek-R1-Distill-Llama-8B,
                   DeepSeek-R1-Distill-Qwen-14B.
    """

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

    def __init__(self, store: ChunkStore, model_id: str, device: str = "cpu"):
        self.store    = store
        self.model_id = model_id
        log.info("Loading ReasoningOrchestratorAgent: %s", model_id)
        self._pipe = pipeline(
            "text-generation",
            model          = model_id,
            device         = device,
            max_new_tokens = 2048,   # reasoning needs room
            do_sample      = False,
        )

    def answer(
        self,
        question: str,
        trace   : _TraceHandle | None = None,
    ) -> RAGAnswer:
        # ── Retrieval ────────────────────────────────────────────
        hits = self._retrieve(question, trace=trace)

        # ── Build prompt ─────────────────────────────────────────
        context_str = self._build_context(hits)
        prompt      = _ORCHESTRATOR_SYSTEM.format(
            context  = context_str,
            question = question,
        )
        messages = [{"role": "user", "content": prompt}]

        # ── Generate with Langfuse generation span ───────────────
        raw_output = self._generate(messages, trace=trace)

        # ── Parse reasoning + answer ──────────────────────────────
        reasoning, answer = self._strip_reasoning(raw_output)

        return RAGAnswer(
            question        = question,
            answer          = answer,
            reasoning_trace = reasoning,
            source_chunks   = [
                {
                    "type"   : h["meta"]["chunk_type"],
                    "file"   : h["meta"]["source_file"],
                    "page"   : h["meta"]["page_num"],
                    "score"  : round(h["score"], 3),
                    "summary": h["meta"]["intuition_summary"],
                }
                for h in hits
            ],
            trace_id = trace.trace_id if trace else "",
        )

    # ── Private helpers ──────────────────────────────────────────

    def _retrieve(
        self,
        question: str,
        trace   : _TraceHandle | None = None,
    ) -> list[dict]:
        """
        Two-stage retrieval:
          1. General semantic search (all chunk types, top-8)
          2. If visual keywords detected → figure-boosted search (top-3, merged)
        """
        def _do_retrieve():
            hits = self.store.query(question, n_results=8)
            if any(kw in question.lower() for kw in _VISUAL_KEYWORDS):
                fig_hits = self.store.query(question, n_results=3, chunk_type=ChunkType.FIGURE)
                seen = {h["text"] for h in hits}
                hits += [h for h in fig_hits if h["text"] not in seen]
            return hits

        if trace:
            with trace.span("retrieve_chunks", input={"question": question[:200]}) as s:
                hits = _do_retrieve()
                s.update(output={"n_hits": len(hits)})
        else:
            hits = _do_retrieve()
        return hits

    def _generate(
        self,
        messages: list[dict],
        trace   : _TraceHandle | None = None,
    ) -> str:
        """Run the reasoning SLM; log as a Langfuse generation."""
        raw = self._pipe(messages)[0]["generated_text"]
        text = raw[-1]["content"] if isinstance(raw, list) else str(raw)

        if trace:
            # Estimate token counts (approximate — replace with real counts if available)
            prompt_text    = messages[-1]["content"]
            input_tokens   = len(prompt_text.split())      # rough word-count proxy
            output_tokens  = len(text.split())
            with trace.generation(
                name        = "orchestrator_reasoning",
                model       = self.model_id,
                input       = {"messages": messages},
                model_params= {"do_sample": False, "max_new_tokens": 2048},
            ) as g:
                g.set_output(text, input_tokens=input_tokens, output_tokens=output_tokens)

        return text

    def _strip_reasoning(self, raw: str) -> tuple[str, str]:
        """
        Separate <think>...</think> from final answer.
        Returns (reasoning_trace, final_answer).
        """
        match = self._THINK_RE.search(raw)
        if match:
            reasoning = match.group(1).strip()
            answer    = self._THINK_RE.sub("", raw).strip()
        else:
            reasoning = ""
            answer    = raw.strip()
        return reasoning, answer

    @staticmethod
    def _build_context(hits: list[dict]) -> str:
        parts = []
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            parts.append(
                f"[{i}] ({m['chunk_type'].upper()} | "
                f"{m['source_file']} p.{m['page_num']} | score={h['score']:.2f})\n"
                f"Summary: {m['intuition_summary']}\n"
                f"Content: {h['text'][:800]}"
            )
        return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════
# 10. PIPELINE  (main entry point)
# ═══════════════════════════════════════════════════════════

class AgenticRAGPipeline:
    """
    End-to-end pipeline with Langfuse observability.

    Quick start:
        import os
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."

        rag = AgenticRAGPipeline.build(device="cpu")
        rag.ingest("paper.pdf")
        result = rag.query("What does Figure 3 show?")
        print(result.answer)
        print("Langfuse trace:", result.trace_id)
    """

    @classmethod
    def build(
        cls,
        text_model         : str = "microsoft/Phi-3.5-mini-instruct",
        table_model        : str = "Qwen/Qwen2.5-3B-Instruct",
        vision_model       : str = "HuggingFaceTB/SmolVLM-Instruct",
        orchestrator_model : str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",   # ← ~10B reasoning
        device             : str = "cpu",
        persist_dir        : str = "./chroma_db",
    ) -> "AgenticRAGPipeline":
        obj              = cls()
        obj.parser       = PDFParser()
        obj.router       = AgentRouter(
            TextAgent(text_model, device),
            TableAgent(table_model, device),
            VisionAgent(vision_model, device),
        )
        obj.store        = ChunkStore(persist_dir)
        obj.orchestrator = ReasoningOrchestratorAgent(
            store    = obj.store,
            model_id = orchestrator_model,
            device   = device,
        )
        obj.tracer = LangfuseTracer()
        return obj

    # ── Ingestion ────────────────────────────────────────────────

    def ingest(self, pdf_path: str | Path) -> list[ProcessedChunk]:
        """
        Parse → route through 3 agents → upsert.
        Each stage is traced as a Langfuse span under one 'ingest_pdf' trace.
        """
        pdf_path = Path(pdf_path)

        with self.tracer.trace(
            "ingest_pdf",
            input   = {"file": pdf_path.name},
            metadata= {"pipeline": "agentic_rag_v2"},
        ) as trace:

            # 1. Parse
            with trace.span("parse_pdf", input={"path": str(pdf_path)}) as s:
                raw_chunks = self.parser.parse(pdf_path)
                s.update(output={"n_raw": len(raw_chunks)})

            # 2. Route through agents (trace is passed for per-agent spans)
            processed: list[ProcessedChunk] = []
            for chunk in raw_chunks:
                pc = self.router.route(chunk, trace=trace)
                processed.append(pc)

            # 3. Filter low-confidence
            good     = [c for c in processed if c.confidence >= 0.25]
            dropped  = len(processed) - len(good)
            if dropped:
                log.warning("Dropped %d low-confidence chunks", dropped)

            # 4. Upsert
            self.store.upsert(good, trace=trace)

            # 5. Log summary stats
            stats = {ct.value: sum(1 for c in good if c.chunk_type == ct) for ct in ChunkType}
            log.info("Ingestion complete: %s", stats)

        return good

    # ── Query ────────────────────────────────────────────────────

    def query(self, question: str, session_id: str | None = None) -> RAGAnswer:
        """
        Retrieve → reason → answer.
        Full pipeline traced under one 'rag_query' Langfuse trace.
        The trace_id is embedded in RAGAnswer for easy drill-down.
        """
        with self.tracer.trace(
            "rag_query",
            input      = {"question": question},
            session_id = session_id,
        ) as trace:
            result = self.orchestrator.answer(question, trace=trace)
            # Patch in the trace ID so caller can link to Langfuse dashboard
            result.trace_id = trace.trace_id

        return result


# ═══════════════════════════════════════════════════════════
# 11. DEMO
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, sys

    # --- Set Langfuse credentials before running ---
    # os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
    # os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."

    if len(sys.argv) < 2:
        print("Usage: python agentic_rag_flow_v2.py <pdf_path> [question]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else \
        "Summarise the main findings. Describe any key figures or tables."

    rag = AgenticRAGPipeline.build(
        orchestrator_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device             = "cpu",   # change to "cuda" for GPU
    )

    print(f"\n[INGEST] {pdf_path}")
    rag.ingest(pdf_path)

    print(f"\n[QUERY] {question}")
    result = rag.query(question)

    print("\n=== ANSWER ===")
    print(result.answer)

    if result.reasoning_trace:
        print("\n=== REASONING TRACE (first 500 chars) ===")
        print(result.reasoning_trace[:500])

    print("\n=== SOURCES ===")
    for s in result.source_chunks:
        print(f"  [{s['type']}] {s['file']} p.{s['page']} score={s['score']} — {s['summary']}")

    print(f"\n[Langfuse trace ID] {result.trace_id}")
    print("View at: https://cloud.langfuse.com")
