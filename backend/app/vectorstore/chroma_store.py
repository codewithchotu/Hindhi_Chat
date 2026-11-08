"""
chroma_store.py — Persistent ChromaDB vector storage and indexing for Hindi RAG.

Features:
  1. Persistent local storage in configured directory (settings.CHROMA_PERSIST_DIR).
  2. Uses Cosine Similarity distance metric ("cosine") matching normalized embeddings.
  3. Stable chunk IDs prevent duplicate indexing across runs (idempotent upsert).
  4. Stores rich metadata: chunk_id, page, section, source_doc, content_type, char_count.
  5. Supports top-k vector search with metadata filtering.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.ingestion.chunker import DocumentChunk
from app.embeddings.embedder import MultilingualEmbedder, get_embedder

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages ChromaDB client, collections, indexing, and dense retrieval."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedder: Optional[MultilingualEmbedder] = None
    ):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.embedder = embedder or get_embedder()

        os.makedirs(self.persist_dir, exist_ok=True)
        logger.info(f"Connecting to Persistent ChromaDB at '{self.persist_dir}'")
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        # Cosine distance: distance = 1 - cosine_similarity
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"ChromaDB Collection '{self.collection_name}' ready with {self.collection.count()} items."
        )

    def count(self) -> int:
        """Return the number of vectors stored in the collection."""
        return self.collection.count()

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 32
    ) -> int:
        """
        Embed and upsert chunks into ChromaDB in batches.
        Uses chunk_id as the primary key to ensure idempotency.
        
        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0

        total = len(chunks)
        indexed_count = 0

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "chunk_id": c.chunk_id,
                    "page": int(c.page),
                    "section": c.section,
                    "source_doc": c.source_doc,
                    "content_type": c.content_type,
                    "char_count": int(c.char_count)
                }
                for c in batch
            ]

            # Generate normalized dense embeddings for batch
            embeddings = self.embedder.embed_texts(documents, batch_size=len(batch))

            # Upsert into ChromaDB (updates if exists, inserts if new)
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            indexed_count += len(batch)
            logger.info(f"Indexed {indexed_count}/{total} chunks in ChromaDB...")

        return indexed_count

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search collection by dense vector similarity.
        
        Args:
            query: Query string in English or Hindi.
            top_k: Number of nearest neighbors to retrieve.
            filter_metadata: Optional ChromaDB metadata filter.
            
        Returns:
            List of dicts:
              {
                "chunk_id": str,
                "text": str,
                "score": float,  # Cosine similarity (1.0 = identical)
                "metadata": dict
              }
        """
        query_embedding = self.embedder.embed_query(query)
        
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = self.collection.query(**kwargs)

        hits: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            # For cosine space in ChromaDB: similarity = 1 - cosine_distance
            similarity = max(0.0, min(1.0, 1.0 - dist))
            hits.append({
                "chunk_id": cid,
                "text": doc,
                "score": round(similarity, 4),
                "metadata": meta
            })

        return hits

    def reset_collection(self) -> None:
        """Clear all entries from collection (useful for fresh indexing)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Reset ChromaDB collection '{self.collection_name}'.")


_store_instance: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Singleton getter for ChromaVectorStore."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaVectorStore()
    return _store_instance
