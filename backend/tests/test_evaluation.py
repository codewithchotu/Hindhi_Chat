"""
test_evaluation.py — Unit tests for evaluation metrics, dataset schema, and grounding evaluation.

Runs fast without loading BGE-M3 (compatible with Group A test suite).
"""
import pytest
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
    normalize_eval_text,
)
from app.evaluation.rag_evaluator import RAGEvaluator


class DummyRetriever:
    """Mock retriever for testing evaluator orchestrator without heavy embedding models."""
    def retrieve(self, query: str, mode: str = "hybrid", top_k: int = 5):
        return [
            {"chunk_id": "doc_p8_c3", "metadata": {"page": 8, "section": "6 · इसरो"}},
            {"chunk_id": "doc_p8_c2", "metadata": {"page": 8, "section": "6 · इसरो"}},
        ]


class DummyGenerator:
    """Mock generator for testing evaluator orchestrator without API calls."""
    pass


# ── 1. Statistical Metric Unit Tests ──────────────────────────────────────

def test_hit_rate():
    gt = {8, 17, 20}
    assert compute_hit_rate([8, 2, 3], gt, 1) == 1.0
    assert compute_hit_rate([1, 8, 3], gt, 1) == 0.0
    assert compute_hit_rate([1, 8, 3], gt, 2) == 1.0
    assert compute_hit_rate([1, 2, 3], gt, 3) == 0.0
    assert compute_hit_rate([], gt, 5) == 0.0
    assert compute_hit_rate([1, 2], set(), 5) == 0.0
    assert compute_hit_rate([8, 17], gt, 0) == 0.0


def test_recall_at_k():
    gt = {"c1", "c2", "c3", "c4"}
    retrieved = ["c1", "c2", "x1", "x2", "x3"]
    # 2 out of 4 found in top-5
    assert compute_recall_at_k(retrieved, gt, 5) == 0.5
    # 2 out of 4 found in top-2
    assert compute_recall_at_k(retrieved, gt, 2) == 0.5
    # 1 out of 4 found in top-1
    assert compute_recall_at_k(retrieved, gt, 1) == 0.25
    # No matches
    assert compute_recall_at_k(["y1", "y2"], gt, 2) == 0.0
    # Empty ground truth
    assert compute_recall_at_k(retrieved, set(), 5) == 0.0


def test_precision_at_k():
    gt = {"c1", "c2", "c3"}
    retrieved = ["c1", "x1", "c2", "x2", "x3"]
    # Top-1: 1/1 = 1.0
    assert compute_precision_at_k(retrieved, gt, 1) == 1.0
    # Top-2: 1/2 = 0.5
    assert compute_precision_at_k(retrieved, gt, 2) == 0.5
    # Top-3: 2/3 = 0.6666...
    assert round(compute_precision_at_k(retrieved, gt, 3), 4) == 0.6667
    # Top-5: 2/5 = 0.4
    assert compute_precision_at_k(retrieved, gt, 5) == 0.4
    # Zero k
    assert compute_precision_at_k(retrieved, gt, 0) == 0.0


def test_mrr():
    gt = {"c3", "c4"}
    # First match at rank 1
    assert compute_mrr(["c3", "c1", "c2"], gt) == 1.0
    # First match at rank 2
    assert compute_mrr(["c1", "c4", "c2"], gt) == 0.5
    # First match at rank 3
    assert round(compute_mrr(["c1", "c2", "c3"], gt), 4) == 0.3333
    # No match
    assert compute_mrr(["x1", "x2", "x3"], gt) == 0.0
    # Empty ground truth
    assert compute_mrr(["c3"], set()) == 0.0


def test_answer_correctness():
    key_terms = ["रोहिणी", "1980", "SLV-III"]
    answer_full = "18 जुलाई 1980 को SLV-III द्वारा रोहिणी उपग्रह को कक्षा में स्थापित किया गया।"
    score, matched, missing = compute_answer_correctness(answer_full, key_terms)
    assert score == 1.0
    assert len(matched) == 3
    assert len(missing) == 0

    answer_partial = "1980 में रोहिणी उपग्रह का प्रक्षेपण हुआ।"
    score, matched, missing = compute_answer_correctness(answer_partial, key_terms)
    assert round(score, 4) == 0.6667
    assert "SLV-III" in missing

    answer_none = "कलाम का जन्म रामेश्वरम में हुआ था।"
    score, matched, missing = compute_answer_correctness(answer_none, key_terms)
    assert score == 0.0
    assert len(matched) == 0
    assert len(missing) == 3


def test_citation_accuracy():
    gt_chunks = {"doc_p8_c3", "doc_p8_c2"}
    gt_pages = {8, 17}
    chunk_map = {
        "doc_p8_c3": 8,
        "doc_p8_c2": 8,
        "doc_p17_c1": 17,
        "doc_p4_c1": 4
    }

    # Perfect citation
    acc, matched, missing = compute_citation_accuracy(["doc_p8_c3"], gt_chunks, gt_pages, chunk_map)
    assert acc == 1.0
    assert matched == ["doc_p8_c3"]

    # 1 valid, 1 invalid
    acc, matched, missing = compute_citation_accuracy(["doc_p8_c3", "doc_p4_c1"], gt_chunks, gt_pages, chunk_map)
    assert acc == 0.5
    assert "doc_p4_c1" in missing

    # Unanswerable case: 0 citations expected and 0 provided -> 100% accuracy
    acc, matched, missing = compute_citation_accuracy([], set(), set(), chunk_map)
    assert acc == 1.0


def test_confusion_matrix():
    y_true = [True, True, False, False, True]
    y_pred = [True, False, False, False, True]
    # TP (expected False, predicted False): 2
    # TN (expected True, predicted True): 2
    # FP (expected True, falsely predicted False): 1
    # FN (expected False, falsely predicted True): 0
    matrix = compute_confusion_matrix(y_true, y_pred)
    assert matrix["tp"] == 2
    assert matrix["tn"] == 2
    assert matrix["fp"] == 1
    assert matrix["fn"] == 0
    assert matrix["accuracy"] == 0.8
    assert matrix["recall"] == 1.0


# ── 2. Dataset Schema & Integrity Tests ────────────────────────────────────

def test_evaluation_dataset_structure():
    assert len(EVAL_QUERIES) == 12
    mandatory = [q for q in EVAL_QUERIES if q["category"] == "mandatory"]
    additional = [q for q in EVAL_QUERIES if q["category"] == "additional"]
    assert len(mandatory) == 6
    assert len(additional) == 6

    for q in EVAL_QUERIES:
        assert "id" in q and isinstance(q["id"], int)
        assert "query" in q and len(q["query"]) > 5
        assert q["lang"] in ["Hindi", "English"]
        assert len(q["ground_truth_pages"]) > 0
        assert len(q["ground_truth_chunk_ids"]) > 0
        assert len(q["key_terms"]) > 0
        assert "reference_answer" in q and len(q["reference_answer"]) > 10


def test_unanswerable_dataset_structure():
    assert len(UNANSWERABLE_QUERIES) == 4
    for q in UNANSWERABLE_QUERIES:
        assert q["category"] == "unanswerable"
        assert q["expected_refusal"] is True
        assert len(q["ground_truth_pages"]) == 0
        assert len(q["ground_truth_chunk_ids"]) == 0


def test_adversarial_grounding_cases_structure():
    assert len(ADVERSARIAL_GROUNDING_CASES) == 10
    grounded_count = sum(1 for c in ADVERSARIAL_GROUNDING_CASES if c["expected_is_grounded"])
    ungrounded_count = sum(1 for c in ADVERSARIAL_GROUNDING_CASES if not c["expected_is_grounded"])
    # 4 valid cases (2 grounded + 2 valid refusals), 6 ungrounded adversarial cases
    assert grounded_count == 4
    assert ungrounded_count == 6


# ── 3. Grounding Verifier Evaluation Runner Test ──────────────────────────

def test_grounding_verifier_evaluation_runner():
    evaluator = RAGEvaluator(retriever=DummyRetriever(), generator=DummyGenerator())
    res = evaluator.evaluate_grounding_verifier()
    matrix = res["metrics"]
    
    assert "accuracy" in matrix
    assert "precision" in matrix
    assert "recall" in matrix
    assert "f1" in matrix
    assert matrix["total"] == 10
    # The GroundingVerifier should achieve 100% detection on all 10 adversarial cases
    assert matrix["accuracy"] == 1.0
    assert matrix["f1"] == 1.0
