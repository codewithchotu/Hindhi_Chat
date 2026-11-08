"""
retriever.py — Unified retrieval interface supporting Dense, BM25, and Hybrid RRF.

Provides a single entry point for Phase 4 retrieval benchmarks and LLM generation context.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import get_settings
from app.vectorstore.chroma_store import ChromaVectorStore, get_vector_store
from app.retrieval.hybrid_search import BM25Retriever, reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class UnifiedRetriever:
    """Combines Dense, BM25, and Hybrid RRF retrieval pipelines."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        chunks_path: Optional[str] = None,
        enable_reranker: bool = False
    ):
        settings = get_settings()
        self.vector_store = vector_store or get_vector_store()
        
        # Load chunks from disk to build BM25 index
        cpath = Path(chunks_path or settings.DATA_PROCESSED_DIR / "chunks.json")
        if not cpath.exists():
            raise FileNotFoundError(f"Chunks file not found at {cpath}")

        with open(cpath, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.bm25_retriever = BM25Retriever(self.chunks)
        self.reranker = CrossEncoderReranker(enabled=enable_reranker)
        logger.info(f"UnifiedRetriever initialized with {len(self.chunks)} chunks.")

    def retrieve_dense(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve using dense semantic embeddings only."""
        return self.vector_store.search(query, top_k=top_k, filter_metadata=filter_metadata)

    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using BM25 sparse keyword matching only."""
        return self.bm25_retriever.search(query, top_k=top_k)

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        candidate_k: int = 15,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using weighted Reciprocal Rank Fusion of Dense + BM25 results.
        """
        dense_results = self.retrieve_dense(query, top_k=candidate_k, filter_metadata=filter_metadata)
        sparse_results = self.retrieve_bm25(query, top_k=candidate_k)

        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            top_k=top_k
        )

        if self.reranker.enabled:
            return self.reranker.rerank(query, fused, top_k=top_k)

        return fused

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generic retrieval dispatcher supporting 'dense', 'bm25', and 'hybrid' modes.
        """
        mode = mode.lower().strip()
        if mode == "dense":
            return self.retrieve_dense(query, top_k=top_k, **kwargs)
        elif mode == "bm25":
            return self.retrieve_bm25(query, top_k=top_k)
        elif mode == "hybrid":
            return self.retrieve_hybrid(query, top_k=top_k, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Choose 'dense', 'bm25', or 'hybrid'.")


_retriever_instance: Optional[UnifiedRetriever] = None


def get_unified_retriever() -> UnifiedRetriever:
    """Singleton getter for UnifiedRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = UnifiedRetriever()
    return _retriever_instance
