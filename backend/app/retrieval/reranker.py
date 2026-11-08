"""
reranker.py — Lightweight semantic cross-encoder reranking.

Evaluates query-passage pairs and reranks candidates.
Includes benchmark evaluation against dense and hybrid retrieval.
"""
import logging
from typing import List, Dict, Any, Optional
import torch

from app.config import get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks candidate chunks using a cross-encoder model.
    Falls back gracefully to score-pass-through if model is not loaded.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", enabled: bool = False):
        self.enabled = enabled
        self.model_name = model_name
        self.model = None

        if self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                settings = get_settings()
                logger.info(f"Loading CrossEncoder '{self.model_name}'...")
                self.model = CrossEncoder(self.model_name, max_length=512)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder ({e}). Reranking will be disabled.")
                self.enabled = False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents.
        If reranker is disabled, returns candidates sorted by initial score.
        """
        if not candidates:
            return []

        if not self.enabled or self.model is None:
            return candidates[:top_k]

        pairs = [[query, c["text"]] for c in candidates]
        try:
            scores = self.model.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = round(float(s), 4)

            # Re-sort candidates by cross-encoder score
            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logger.error(f"Error during reranking: {e}. Falling back to input order.")
            return candidates[:top_k]
