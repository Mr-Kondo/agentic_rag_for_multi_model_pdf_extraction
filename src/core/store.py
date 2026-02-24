"""
Vector store interface for chunk storage and retrieval.

Uses ChromaDB for persistent vector storage with e5-small embeddings.
"""

import json
import logging

import chromadb
from sentence_transformers import SentenceTransformer

from src.core.models import ChunkType, ProcessedChunk

log = logging.getLogger(__name__)


class ChunkStore:
    """
    Manages chunk storage and retrieval using ChromaDB vector database.

    Embeds chunks using multilingual e5-small model and stores in persistent
    ChromaDB collection with cosine similarity search.

    Attributes:
        EMBED_MODEL: Sentence transformer model for embeddings
        _embedder: SentenceTransformer instance
        _client: ChromaDB client
        _col: ChromaDB collection
    """

    EMBED_MODEL = "intfloat/multilingual-e5-small"

    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Initialize chunk store with persistent ChromaDB.

        Args:
            persist_dir: Directory for ChromaDB persistence
        """
        self._embedder = SentenceTransformer(self.EMBED_MODEL)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection("agentic_rag", metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[ProcessedChunk]) -> None:
        """
        Insert or update chunks in the vector database.

        Embeds chunk text and upserts with metadata for filtering.

        Args:
            chunks: List of ProcessedChunk objects to upsert
        """
        texts = [f"{c.structured_text}\n\n{c.intuition_summary}" for c in chunks]
        embs = self._embedder.encode(texts, normalize_embeddings=True).tolist()
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

    def query(self, question: str, n_results: int = 6, chunk_type: ChunkType | None = None) -> list[dict]:
        """
        Query the vector store for relevant chunks.

        Args:
            question: Query text to search for
            n_results: Number of results to return
            chunk_type: Optional filter for specific chunk type

        Returns:
            List of dicts with 'text', 'meta', and 'score' keys
        """
        vec = self._embedder.encode([question], normalize_embeddings=True).tolist()
        where = {"chunk_type": chunk_type.value} if chunk_type else None
        res = self._col.query(
            query_embeddings=vec, n_results=n_results, where=where, include=["documents", "metadatas", "distances"]
        )
        return [
            {"text": doc, "meta": meta, "score": 1 - dist}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
