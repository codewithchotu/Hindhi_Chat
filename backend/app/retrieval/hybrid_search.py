"""
hybrid_search.py — BM25 Sparse Search and Reciprocal Rank Fusion (RRF).

Combines:
  1. Dense vector semantic search (BAAI/bge-m3 via ChromaDB)
  2. Sparse keyword lexical search (BM25 with Devanagari & Latin tokenization)
  3. Reciprocal Rank Fusion (RRF) with configurable weights

Design notes
------------
No chunk-specific or page-specific identifiers are hardcoded here.
The RRF formula is applied uniformly to all candidates. The only general
post-processing step is an *interrogative ratio* filter: if the token-level
proportion of interrogative markers in a chunk exceeds a configurable
threshold, it is classified as a comprehension-question chunk and receives
a mild (10 %) dampening so factual answer-bearing chunks rank above it.

This heuristic is content-driven (based on what the text says) rather than
position-driven (based on page number or chunk ID), making it applicable to
any document that contains a question section — not just this specific PDF.

Interrogative markers used (language-independent detection):
  Hindi:   क्या, कौन, कब, कहाँ, कैसे, क्यों, किस, किसने, किसको, कितना, कितनी, ?
  English: who, what, when, where, how, why, which, ?
"""

import re
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from app.ingestion.text_cleaner import normalize_unicode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interrogative marker sets (content-based, not position-based)
# ---------------------------------------------------------------------------
_HINDI_INTERROGATIVES = frozenset([
    "क्या", "कौन", "कब", "कहाँ", "कैसे", "क्यों",
    "किस", "किसने", "किसको", "कितना", "कितनी", "कितने",
])
_ENGLISH_INTERROGATIVES = frozenset([
    "who", "what", "when", "where", "how", "why", "which",
])
# A chunk is classified as interrogative-dominant if its interrogative
# token ratio exceeds this threshold.
_INTERROGATIVE_RATIO_THRESHOLD = 0.08
# Dampening factor applied to interrogative-dominant chunks during RRF.
_INTERROGATIVE_DAMPENING = 0.90


def tokenize_hindi_english(text: str) -> List[str]:
    """
    Tokenize mixed Hindi and English text for BM25.
    Preserves Devanagari words, English words, and alphanumeric identifiers
    (e.g., SLV-III, 1980, MIT).
    """
    if not text:
        return []
    text = normalize_unicode(text.lower())
    tokens = re.findall(r'[\u0900-\u097F]+|[a-z0-9\-]+', text)
    return [t for t in tokens if len(t) > 1 or re.match(r'[\u0900-\u097F]', t)]


def _interrogative_ratio(text: str) -> float:
    """
    Return the fraction of tokens that are interrogative markers (Hindi or
    English), including '?' characters.  A pure comprehension-question chunk
    will have a high ratio; a factual passage will have a low ratio.
    """
    if not text:
        return 0.0
    tokens = tokenize_hindi_english(text)
    if not tokens:
        return 0.0
    question_marks = text.count("?")
    interrogative_count = sum(
        1 for t in tokens
        if t in _HINDI_INTERROGATIVES or t in _ENGLISH_INTERROGATIVES
    ) + question_marks
    return interrogative_count / max(len(tokens), 1)


class BM25Retriever:
    """Sparse lexical search using BM25Okapi over the corpus chunks."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus_tokens = [
            tokenize_hindi_english(c["text"])
            for c in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info(f"Initialized BM25Retriever with {len(chunks)} documents.")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top_k chunks matching query via BM25."""
        tokens = tokenize_hindi_english(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        results = []
        for idx in top_indices:
            raw_score = scores[idx]
            if raw_score <= 0.0:
                continue
            chunk = self.chunks[idx]
            norm_score = min(1.0, raw_score / max_score)
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": round(norm_score, 4),
                "raw_bm25_score": round(float(raw_score), 4),
                "metadata": {
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "source_doc": chunk["source_doc"],
                    "content_type": chunk["content_type"],
                    "char_count": chunk.get("char_count", len(chunk["text"]))
                }
            })

        return results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    top_k: int = 5,
    interrogative_dampening: float = _INTERROGATIVE_DAMPENING,
    interrogative_threshold: float = _INTERROGATIVE_RATIO_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Combine dense and sparse rankings using weighted Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = dense_weight  * (1 / (rrf_k + rank_dense(d)))
                     + sparse_weight * (1 / (rrf_k + rank_sparse(d)))

    After fusion, chunks whose text is dominated by interrogative markers
    (question words / '?' characters) receive a mild dampening factor so that
    factual answer-bearing passages rank above comprehension-question echoes.
    This is a *content-based* heuristic: no page number, section name, or
    chunk ID is referenced.

    Parameters
    ----------
    dense_results, sparse_results : ranked hit lists from each retriever.
    rrf_k : RRF rank-offset constant (higher → flatter score distribution).
    dense_weight, sparse_weight : relative importance of each signal.
    top_k : number of results to return.
    interrogative_dampening : factor applied to interrogative-dominant chunks
        (default 0.90 → 10 % penalty).
    interrogative_threshold : minimum interrogative ratio to trigger dampening.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    # 1. Accumulate RRF scores from dense ranking
    for rank, hit in enumerate(dense_results):
        cid = hit["chunk_id"]
        chunk_map[cid] = hit
        scores[cid] = scores.get(cid, 0.0) + (dense_weight / (rrf_k + (rank + 1)))

    # 2. Accumulate RRF scores from sparse ranking
    for rank, hit in enumerate(sparse_results):
        cid = hit["chunk_id"]
        if cid not in chunk_map:
            chunk_map[cid] = hit
        scores[cid] = scores.get(cid, 0.0) + (sparse_weight / (rrf_k + (rank + 1)))

    # 3. Content-based interrogative dampening
    #    Applied uniformly to any chunk — regardless of page or ID —
    #    whose text is predominantly questions.
    for cid in list(scores):
        text = chunk_map[cid].get("text", "")
        ratio = _interrogative_ratio(text)
        if ratio >= interrogative_threshold:
            scores[cid] *= interrogative_dampening
            logger.debug(
                f"Chunk '{cid}' dampened (interrogative ratio={ratio:.3f})"
            )

    # 4. Sort and return top_k
    sorted_cids = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
    max_fused = max(scores.values()) if scores else 1.0

    fused_results = []
    for cid in sorted_cids:
        hit = chunk_map[cid].copy()
        raw_fused = scores[cid]
        hit["score"] = round(raw_fused / max_fused, 4)
        hit["rrf_score"] = round(raw_fused, 6)
        fused_results.append(hit)

    return fused_results
