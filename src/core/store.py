"""
Vector store interface for chunk storage and retrieval.

Uses ChromaDB for persistent vector storage and BM25 for keyword-based
retrieval. Hybrid search fuses both via Reciprocal Rank Fusion (RRF).

Embedding backend is configurable via settings.json:
  - "sentence_transformers" (default): uses HuggingFace sentence-transformers
  - "ollama": uses Ollama embed API (e.g. kun432/cl-nagoya-ruri-large)
"""

import json
import logging
import re

import chromadb
from rank_bm25 import BM25Okapi

from src.core.embedder import BaseEmbedder, create_embedder
from src.core.models import ChunkType, ProcessedChunk

log = logging.getLogger(__name__)


class ChunkStore:
    """
    Manages chunk storage and retrieval using ChromaDB vector database and BM25.

    The embedding backend is selected via settings.json (embedder.backend).
    Supports sentence-transformers (local) and Ollama (remote API) backends.

    Attributes:
        _embedder: Embedder backend (BaseEmbedder protocol)
        _client: ChromaDB client
        _col: ChromaDB collection
        _bm25: BM25Okapi index (rebuilt on each upsert)
        _bm25_docs: Parallel list of raw texts for BM25 lookup
        _bm25_ids: Parallel list of chunk IDs for BM25 lookup
    """

    # Fallback model when no embedder is configured in settings
    _DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"

    # RRF constant — 60 is the standard default from the original paper
    _RRF_K = 60

    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Initialize chunk store with persistent ChromaDB.

        Reads embedder configuration from settings.json. If an existing
        ChromaDB collection has embeddings with a different dimension than
        the configured model, raises ValueError to prompt re-ingestion.

        Args:
            persist_dir: Directory for ChromaDB persistence

        Raises:
            ValueError: If existing collection has different embedding dimension
                        than the current embedder model.
        """
        from src.core.config import config as _config

        model_id = _config.get_model("embedder") or self._DEFAULT_EMBED_MODEL
        emb_cfg = _config.get("embedder") or {}
        backend = emb_cfg.get("backend", "sentence_transformers")
        query_prefix = emb_cfg.get("query_prefix")   # None → backend default
        passage_prefix = emb_cfg.get("passage_prefix")  # None → backend default
        batch_size = emb_cfg.get("batch_size", 32)
        ollama_url = _config.get("ollama_base_url") or "http://localhost:11434"

        self._embedder: BaseEmbedder = create_embedder(
            model_id=model_id,
            backend=backend,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            batch_size=batch_size,
            ollama_base_url=ollama_url,
        )
        log.info("ChunkStore embedder: backend=%s model=%s dim=%d", backend, model_id, self._embedder.embedding_dim)

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection("agentic_rag", metadata={"hnsw:space": "cosine"})

        self._check_dimension_mismatch()

        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_metas: list[dict] = []
        self._rebuild_bm25_from_store()

    def _check_dimension_mismatch(self) -> None:
        """
        Detect embedding dimension mismatch between existing store and current model.

        If the collection contains vectors with a different dimensionality than
        the current embedder, raises ValueError to inform the user to re-ingest.

        Raises:
            ValueError: If a dimension mismatch is detected.
        """
        try:
            peek = self._col.peek(limit=1)
            existing_embs = peek.get("embeddings")
            if not existing_embs or not existing_embs[0]:
                return  # Empty store — no mismatch possible
            existing_dim = len(existing_embs[0])
            current_dim = self._embedder.embedding_dim
            if existing_dim != current_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: existing store has dim={existing_dim}, "
                    f"but current embedder produces dim={current_dim}. "
                    f"Please delete the chroma_db directory and re-ingest all documents."
                )
        except ValueError:
            raise
        except Exception as e:
            log.warning("Could not verify embedding dimension (non-fatal): %s", e)

    def upsert(self, chunks: list[ProcessedChunk]) -> None:
        """
        Insert or update chunks in the vector database.

        Embeds chunk text using the configured passage prefix and upserts
        with metadata for filtering.

        Args:
            chunks: List of ProcessedChunk objects to upsert
        """
        texts = [f"{c.structured_text}\n\n{c.intuition_summary}" for c in chunks]
        embs = self._embedder.encode_passages(texts)
        metadatas = []
        for c in chunks:
            m = {
                "chunk_type": c.chunk_type.value,
                "page_num": c.page_num,
                "source_file": c.source_file,
                "intuition_summary": c.intuition_summary,
                "key_concepts": json.dumps(c.key_concepts, ensure_ascii=False),
                "confidence": c.confidence,
                "agent_notes": c.agent_notes,
            }
            if c.validation is not None:
                m["validation_score"] = c.validation.verdict_score
                m["validation_issues"] = "; ".join(c.validation.issues)
            metadatas.append(m)
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embs,
            documents=[c.structured_text for c in chunks],
            metadatas=metadatas,
        )
        log.info("Upserted %d chunks.", len(chunks))
        self._rebuild_bm25_from_store()

    def hybrid_query(
        self,
        question: str,
        n_results: int = 8,
        chunk_type: ChunkType | None = None,
        min_score: float = 0.0,
        bm25_weight: float = 0.5,
    ) -> list[dict]:
        """
        Hybrid vector + BM25 search fused with Reciprocal Rank Fusion (RRF).

        Performs both dense (vector) and sparse (BM25) retrieval then merges
        results using RRF. Effective for technical queries where exact keyword
        matching complements semantic similarity.

        Args:
            question: Query text to search for
            n_results: Number of results to return after fusion
            chunk_type: Optional filter for specific chunk type
            min_score: Minimum RRF score threshold (0.0 = no filter)
            bm25_weight: Weight of BM25 relative to vector (0.0=pure vector, 1.0=pure BM25)

        Returns:
            List of dicts with 'text', 'meta', and 'score' keys, sorted by RRF score
        """
        # Retrieve a larger candidate set for re-ranking
        candidate_n = min(n_results * 3, 30)

        # --- Dense (vector) retrieval ---
        vec_hits = self.query(question, n_results=candidate_n, chunk_type=chunk_type, min_score=0.0)

        # --- Sparse (BM25) retrieval ---
        bm25_hits = self._bm25_query(question, n_results=candidate_n, chunk_type=chunk_type)

        # --- RRF fusion ---
        # Build rank maps: chunk text → rank (1-indexed)
        vec_rank: dict[str, int] = {h["text"]: i + 1 for i, h in enumerate(vec_hits)}
        bm25_rank: dict[str, int] = {h["text"]: i + 1 for i, h in enumerate(bm25_hits)}

        # Collect all candidate texts
        all_texts = {h["text"]: h for h in vec_hits + bm25_hits}

        rrf_scores: dict[str, float] = {}
        k = self._RRF_K
        for text in all_texts:
            vec_r = vec_rank.get(text, candidate_n + 1)
            bm25_r = bm25_rank.get(text, candidate_n + 1)
            vec_rrf = (1 - bm25_weight) / (k + vec_r)
            bm25_rrf = bm25_weight / (k + bm25_r)
            rrf_scores[text] = vec_rrf + bm25_rrf

        # Sort by RRF score and return top-n
        sorted_texts = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)
        results = []
        for text in sorted_texts[:n_results]:
            hit = dict(all_texts[text])
            hit["score"] = rrf_scores[text]
            if min_score == 0.0 or hit["score"] >= min_score:
                results.append(hit)
        return results

    # ── BM25 internals ─────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25: split on whitespace and punctuation."""
        return re.findall(r"\w+", text.lower())

    def _rebuild_bm25_from_store(self) -> None:
        """Rebuild BM25 index from all documents currently in ChromaDB."""
        try:
            result = self._col.get(include=["documents", "metadatas"])
            docs = result.get("documents") or []
            ids = result.get("ids") or []
            metas = result.get("metadatas") or []
            if not docs:
                self._bm25 = None
                self._bm25_docs = []
                self._bm25_ids = []
                self._bm25_metas = []
                return
            tokenized = [self._tokenize(d) for d in docs]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_docs = list(docs)
            self._bm25_ids = list(ids)
            self._bm25_metas = list(metas)
            log.debug("BM25 index rebuilt with %d documents.", len(docs))
        except Exception as e:
            log.warning("BM25 index rebuild failed: %s", e)
            self._bm25 = None
            self._bm25_docs = []
            self._bm25_ids = []
            self._bm25_metas = []

    def _bm25_query(
        self,
        question: str,
        n_results: int = 6,
        chunk_type: ChunkType | None = None,
    ) -> list[dict]:
        """Run BM25 keyword search and return ranked results."""
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokens = self._tokenize(question)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked:
            if len(results) >= n_results:
                break
            meta = self._bm25_metas[idx] if hasattr(self, "_bm25_metas") else {}
            if chunk_type and meta.get("chunk_type") != chunk_type.value:
                continue
            results.append({
                "text": self._bm25_docs[idx],
                "meta": meta,
                "score": float(scores[idx]),
            })
        return results

    def query(
        self,
        question: str,
        n_results: int = 6,
        chunk_type: ChunkType | None = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Query the vector store for relevant chunks.

        Args:
            question: Query text to search for
            n_results: Number of results to return
            chunk_type: Optional filter for specific chunk type
            min_score: Minimum cosine similarity score threshold (0.0 = no filter)

        Returns:
            List of dicts with 'text', 'meta', and 'score' keys, sorted by score descending
        """
        # Encode query using the configured backend and query prefix
        vec = self._embedder.encode_query(question)
        where = {"chunk_type": chunk_type.value} if chunk_type else None
        res = self._col.query(
            query_embeddings=vec, n_results=n_results, where=where, include=["documents", "metadatas", "distances"]
        )
        hits = [
            {"text": doc, "meta": meta, "score": 1 - dist}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
        if min_score > 0.0:
            hits = [h for h in hits if h["score"] >= min_score]
        return hits