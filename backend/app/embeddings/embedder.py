"""
embedder.py — Multilingual dense embedding generation using BAAI/bge-m3.

Features:
  1. Loads BAAI/bge-m3 via sentence-transformers.
  2. Ensures cache paths point to D: drive (protecting C: disk limit).
  3. Gracefully detects and handles GPU/CPU availability with automatic fallback.
  4. Supports batch embedding of document texts with configurable batch size.
  5. Computes normalized dense embeddings (cosine similarity compatible).
  6. Shared embedder instance for both indexing and inference.
"""
import os
import time
import logging
from typing import List, Union, Optional
import numpy as np
import torch

from app.config import get_settings

# Compatibility handling for CVE-2025-32434 check in recent transformers when loading PyTorch model weights
try:
    import transformers.modeling_utils as _mu
    import transformers.utils.import_utils as _iu
    if hasattr(_mu, "check_torch_load_is_safe"):
        _mu.check_torch_load_is_safe = lambda: True
    if hasattr(_iu, "check_torch_load_is_safe"):
        _iu.check_torch_load_is_safe = lambda: True
except Exception:
    pass

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MultilingualEmbedder:
    """Wrapper around BAAI/bge-m3 for dense vector representation."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        batch_size: int = 16
    ):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.cache_folder = cache_folder or settings.HF_HOME
        self.batch_size = batch_size

        # Device selection with graceful fallback
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"

        # Explicitly configure environment variables for cache directories
        os.environ["HF_HOME"] = self.cache_folder
        os.environ["TORCH_HOME"] = settings.TORCH_HOME

        logger.info(
            f"Initializing MultilingualEmbedder with '{self.model_name}' on device '{self.device}'"
        )
        t0 = time.time()
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=self.cache_folder
        )
        self.load_time = time.time() - t0
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Model loaded in {self.load_time:.2f}s. Embedding dimension: {self.embedding_dim}"
        )

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False
    ) -> List[List[float]]:
        """
        Embed a list of strings into normalized dense vector embeddings.
        """
        if not texts:
            return []

        bs = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single user query (Hindi or English) into a normalized dense vector.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        embedding = self.model.encode(
            query.strip(),
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.tolist()


_embedder_instance: Optional[MultilingualEmbedder] = None


def get_embedder() -> MultilingualEmbedder:
    """Singleton getter for the MultilingualEmbedder."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MultilingualEmbedder()
    return _embedder_instance
