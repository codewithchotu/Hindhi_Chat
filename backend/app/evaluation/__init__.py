"""
Evaluation package: Retrieval quality, faithfulness, answer correctness, and latency benchmarks.
"""
from app.evaluation.dataset import (
    EVAL_QUERIES,
    UNANSWERABLE_QUERIES,
    ADVERSARIAL_GROUNDING_CASES,
)
from app.evaluation.metrics import (
    compute_hit_rate,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_answer_correctness,
    compute_citation_accuracy,
    compute_confusion_matrix,
)
from app.evaluation.rag_evaluator import RAGEvaluator

__all__ = [
    "EVAL_QUERIES",
    "UNANSWERABLE_QUERIES",
    "ADVERSARIAL_GROUNDING_CASES",
    "compute_hit_rate",
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_mrr",
    "compute_answer_correctness",
    "compute_citation_accuracy",
    "compute_confusion_matrix",
    "RAGEvaluator",
]
