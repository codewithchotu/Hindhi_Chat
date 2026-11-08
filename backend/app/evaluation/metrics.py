"""
metrics.py — Statistical and evaluation metrics for RAG retrieval and answer generation.

Provides standard implementations of:
  - HitRate@K (Binary Recall@K)
  - Recall@K (Multi-relevant chunk recall)
  - Precision@K
  - Mean Reciprocal Rank (MRR)
  - Answer Correctness (Factual term coverage)
  - Citation Accuracy (Precision of cited chunk IDs)
  - Binary Classification Confusion Matrix (Accuracy, Precision, Recall, F1)
"""
import unicodedata
from typing import List, Set, Dict, Any, Tuple, Optional


def normalize_eval_text(text: str) -> str:
    """Normalize text for consistent string matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return text.lower().strip()


def compute_hit_rate(retrieved: List[Any], ground_truth: Set[Any], k: int) -> float:
    """
    HitRate@K (Binary Recall@K): Returns 1.0 if ANY relevant item appears in top-k, else 0.0.
    """
    if not ground_truth or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    return 1.0 if any(item in ground_truth for item in top_k) else 0.0


def compute_recall_at_k(retrieved: List[Any], ground_truth: Set[Any], k: int) -> float:
    """
    Recall@K (Multi-relevant): Proportion of ground-truth relevant items found in top-k.
    Formula: |G ∩ R_k| / |G|
    """
    if not ground_truth or k <= 0:
        return 0.0
    top_k_set = set(retrieved[:k])
    hits = len(top_k_set.intersection(ground_truth))
    return hits / len(ground_truth)


def compute_precision_at_k(retrieved: List[Any], ground_truth: Set[Any], k: int) -> float:
    """
    Precision@K: Proportion of top-k retrieved items that are relevant.
    Formula: |G ∩ R_k| / k
    """
    if k <= 0:
        return 0.0
    top_k_set = set(retrieved[:k])
    hits = len(top_k_set.intersection(ground_truth))
    return hits / k


def compute_mrr(retrieved: List[Any], ground_truth: Set[Any]) -> float:
    """
    Mean Reciprocal Rank (MRR) component:
    Formula: 1.0 / rank of first relevant item (1-indexed). Returns 0.0 if not found.
    """
    if not ground_truth:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in ground_truth:
            return 1.0 / rank
    return 0.0


def compute_answer_correctness(answer: str, key_terms: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Calculates factual term coverage of the generated answer against ground truth key terms.
    Returns: (score [0.0 - 1.0], matched_terms, missing_terms)
    """
    if not key_terms:
        return 1.0, [], []
    
    norm_answer = normalize_eval_text(answer)
    matched = []
    missing = []
    
    for term in key_terms:
        norm_term = normalize_eval_text(term)
        if norm_term in norm_answer:
            matched.append(term)
        else:
            missing.append(term)
            
    score = len(matched) / len(key_terms)
    return score, matched, missing


def compute_citation_accuracy(
    cited_cids: List[str],
    ground_truth_cids: Set[str],
    ground_truth_pages: Set[int],
    chunk_page_map: Dict[str, int]
) -> Tuple[float, List[str], List[str]]:
    """
    Computes precision of citations: proportion of cited chunk IDs that are relevant
    (either directly in ground_truth_cids or located on ground_truth_pages).
    
    Returns: (accuracy [0.0 - 1.0], accurate_citations, inaccurate_citations)
    """
    if not cited_cids:
        # If no citations were made:
        # If there are no ground truth chunks (e.g. unanswerable), 0 citations is 100% accurate.
        return (1.0 if not ground_truth_cids else 0.0), [], []
    
    accurate = []
    inaccurate = []
    
    for cid in cited_cids:
        page = chunk_page_map.get(cid)
        if cid in ground_truth_cids or (page is not None and page in ground_truth_pages):
            accurate.append(cid)
        else:
            inaccurate.append(cid)
            
    accuracy = len(accurate) / len(cited_cids)
    return accuracy, accurate, inaccurate


def compute_confusion_matrix(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    """
    Computes binary classification metrics:
      TP: true positive (expected ungrounded / invalid, predicted ungrounded)
      TN: true negative (expected grounded / valid, predicted grounded)
      FP: false positive (expected grounded, falsely predicted ungrounded)
      FN: false negative (expected ungrounded, falsely predicted grounded)
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
