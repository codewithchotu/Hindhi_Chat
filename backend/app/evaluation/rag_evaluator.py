"""
rag_evaluator.py — End-to-end evaluation engine for retrieval, generation, and grounding.

Executes:
  1. Multi-mode Retrieval Benchmarking (Dense vs BM25 vs Hybrid RRF)
  2. Answer Quality Assessment (Correctness, Faithfulness, Relevance, Citation Accuracy)
  3. Out-of-Domain / Unanswerable Refusal Benchmarking
  4. Grounding Verifier Performance on Adversarial Scenarios
  5. Fine-Grained Latency Profiling (Cold-Start vs Warm Query)
"""
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import get_settings
from app.retrieval.retriever import UnifiedRetriever, get_unified_retriever
from app.generation.answer_generator import AnswerGenerator, GenerationResult
from app.generation.grounding_verifier import GroundingVerifier, GroundingReport
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

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Comprehensive evaluation orchestrator for the Cross-Lingual Hindi RAG Engine."""

    def __init__(
        self,
        retriever: Optional[UnifiedRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        chunks_path: Optional[str] = None
    ):
        settings = get_settings()
        self.retriever = retriever or get_unified_retriever()
        self.generator = generator or AnswerGenerator(retriever=self.retriever)
        
        # Load processed chunks to map chunk_id -> page and metadata
        p = Path(chunks_path or (settings.DATA_PROCESSED_DIR / "chunks.json"))
        if p.exists():
            with open(p, encoding="utf-8") as f:
                raw_chunks = json.load(f)
            self.chunk_page_map = {
                c["chunk_id"]: c.get("page") or c.get("metadata", {}).get("page")
                for c in raw_chunks
            }
            self.chunks_by_id = {c["chunk_id"]: c for c in raw_chunks}
        else:
            self.chunk_page_map = {}
            self.chunks_by_id = {}

    # -----------------------------------------------------------------------
    # 1. Retrieval Benchmarking
    # -----------------------------------------------------------------------
    def evaluate_retrieval(
        self,
        queries: Optional[List[Dict[str, Any]]] = None,
        modes: Optional[List[str]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluates retrieval performance across dense, bm25, and hybrid modes.
        Computes page-level and chunk-level Recall@K, Precision@K, and MRR.
        """
        eval_set = queries or EVAL_QUERIES
        active_modes = modes or ["dense", "bm25", "hybrid"]
        
        mode_results = {}

        for mode in active_modes:
            per_query_stats = []
            
            p_r1_list, p_r3_list, p_r5_list, p_mrr_list = [], [], [], []
            p_prec1_list, p_prec3_list, p_prec5_list = [], [], []
            
            c_r1_list, c_r3_list, c_r5_list, c_mrr_list = [], [], [], []
            c_prec1_list, c_prec3_list, c_prec5_list = [], [], []

            latencies = []

            for q in eval_set:
                gt_pages = set(q["ground_truth_pages"])
                gt_chunks = set(q["ground_truth_chunk_ids"])

                t0 = time.time()
                hits = self.retriever.retrieve(q["query"], mode=mode, top_k=top_k)
                lat_ms = (time.time() - t0) * 1000
                latencies.append(lat_ms)

                ret_pages = [h.get("metadata", {}).get("page") for h in hits]
                ret_chunks = [h.get("chunk_id") for h in hits]

                # Page-level metrics
                p_r1 = compute_hit_rate(ret_pages, gt_pages, 1)
                p_r3 = compute_hit_rate(ret_pages, gt_pages, 3)
                p_r5 = compute_hit_rate(ret_pages, gt_pages, 5)
                p_prec1 = compute_precision_at_k(ret_pages, gt_pages, 1)
                p_prec3 = compute_precision_at_k(ret_pages, gt_pages, 3)
                p_prec5 = compute_precision_at_k(ret_pages, gt_pages, 5)
                p_mrr = compute_mrr(ret_pages, gt_pages)

                p_r1_list.append(p_r1)
                p_r3_list.append(p_r3)
                p_r5_list.append(p_r5)
                p_prec1_list.append(p_prec1)
                p_prec3_list.append(p_prec3)
                p_prec5_list.append(p_prec5)
                p_mrr_list.append(p_mrr)

                # Chunk-level metrics
                c_r1 = compute_hit_rate(ret_chunks, gt_chunks, 1)
                c_r3 = compute_hit_rate(ret_chunks, gt_chunks, 3)
                c_r5 = compute_hit_rate(ret_chunks, gt_chunks, 5)
                c_prec1 = compute_precision_at_k(ret_chunks, gt_chunks, 1)
                c_prec3 = compute_precision_at_k(ret_chunks, gt_chunks, 3)
                c_prec5 = compute_precision_at_k(ret_chunks, gt_chunks, 5)
                c_mrr = compute_mrr(ret_chunks, gt_chunks)

                c_r1_list.append(c_r1)
                c_r3_list.append(c_r3)
                c_r5_list.append(c_r5)
                c_prec1_list.append(c_prec1)
                c_prec3_list.append(c_prec3)
                c_prec5_list.append(c_prec5)
                c_mrr_list.append(c_mrr)

                per_query_stats.append({
                    "id": q["id"],
                    "query": q["query"],
                    "lang": q["lang"],
                    "category": q["category"],
                    "retrieved_pages": ret_pages,
                    "retrieved_chunks": ret_chunks,
                    "latency_ms": round(lat_ms, 2),
                    "page_metrics": {
                        "recall@1": p_r1,
                        "recall@3": p_r3,
                        "recall@5": p_r5,
                        "precision@1": round(p_prec1, 3),
                        "precision@3": round(p_prec3, 3),
                        "precision@5": round(p_prec5, 3),
                        "mrr": round(p_mrr, 4),
                    },
                    "chunk_metrics": {
                        "recall@1": c_r1,
                        "recall@3": c_r3,
                        "recall@5": c_r5,
                        "precision@1": round(c_prec1, 3),
                        "precision@3": round(c_prec3, 3),
                        "precision@5": round(c_prec5, 3),
                        "mrr": round(c_mrr, 4),
                    }
                })

            n = len(eval_set)
            mode_results[mode] = {
                "total_queries": n,
                "mean_latency_ms": round(sum(latencies) / n, 2) if n else 0.0,
                "page_level": {
                    "recall@1": round(sum(p_r1_list) / n, 4),
                    "recall@3": round(sum(p_r3_list) / n, 4),
                    "recall@5": round(sum(p_r5_list) / n, 4),
                    "precision@1": round(sum(p_prec1_list) / n, 4),
                    "precision@3": round(sum(p_prec3_list) / n, 4),
                    "precision@5": round(sum(p_prec5_list) / n, 4),
                    "mrr": round(sum(p_mrr_list) / n, 4),
                },
                "chunk_level": {
                    "recall@1": round(sum(c_r1_list) / n, 4),
                    "recall@3": round(sum(c_r3_list) / n, 4),
                    "recall@5": round(sum(c_r5_list) / n, 4),
                    "precision@1": round(sum(c_prec1_list) / n, 4),
                    "precision@3": round(sum(c_prec3_list) / n, 4),
                    "precision@5": round(sum(c_prec5_list) / n, 4),
                    "mrr": round(sum(c_mrr_list) / n, 4),
                },
                "queries": per_query_stats
            }

        return mode_results

    # -----------------------------------------------------------------------
    # 2. Answer Quality & Citation Benchmarking
    # -----------------------------------------------------------------------
    def evaluate_answers(
        self,
        queries: Optional[List[Dict[str, Any]]] = None,
        mode: str = "hybrid",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluates generated answers for:
          - Answer correctness (key term coverage)
          - Faithfulness (context grounding via GroundingVerifier)
          - Answer relevance (intent fulfillment)
          - Citation accuracy (precision of cited chunk IDs)
        """
        eval_set = queries or EVAL_QUERIES
        results = []
        
        correctness_scores = []
        faithfulness_scores = []
        citation_accuracies = []
        latencies = []

        for q in eval_set:
            t0 = time.time()
            gen_res: GenerationResult = self.generator.generate_answer(
                q["query"],
                retrieval_mode=mode,
                top_k=top_k
            )
            total_lat = (time.time() - t0) * 1000
            latencies.append(total_lat)

            # 1. Correctness
            corr_score, matched_terms, missing_terms = compute_answer_correctness(
                gen_res.answer,
                q.get("key_terms", [])
            )
            correctness_scores.append(corr_score)

            # 2. Faithfulness
            is_faithful = gen_res.is_grounded
            faith_score = gen_res.grounding_report.grounding_score if gen_res.grounding_report else 1.0
            faithfulness_scores.append(faith_score)

            # 3. Citation Accuracy
            inline_cids = gen_res.grounding_report.cited_chunk_ids if gen_res.grounding_report else []
            retrieved_cids = [c.chunk_id for c in gen_res.citations]
            gt_chunks = set(q.get("ground_truth_chunk_ids", []))
            gt_pages = set(q.get("ground_truth_pages", []))

            # Precision of chunks cited directly in the answer text
            if inline_cids:
                cite_acc, acc_cids, inacc_cids = compute_citation_accuracy(
                    inline_cids,
                    gt_chunks,
                    gt_pages,
                    self.chunk_page_map
                )
            else:
                cite_acc, acc_cids, inacc_cids = compute_citation_accuracy(
                    retrieved_cids,
                    gt_chunks,
                    gt_pages,
                    self.chunk_page_map
                )
            citation_accuracies.append(cite_acc)

            # Precision of all retrieved chunks attached to the response block
            ret_prec, _, _ = compute_citation_accuracy(
                retrieved_cids,
                gt_chunks,
                gt_pages,
                self.chunk_page_map
            )

            # 4. Relevance: non-empty, aligned with question language
            is_relevant = len(gen_res.answer.strip()) > 10 and not gen_res.is_refusal

            results.append({
                "id": q["id"],
                "query": q["query"],
                "lang": q["lang"],
                "category": q["category"],
                "answer": gen_res.answer,
                "is_grounded": gen_res.is_grounded,
                "is_refusal": gen_res.is_refusal,
                "correctness_score": round(corr_score, 4),
                "matched_key_terms": matched_terms,
                "missing_key_terms": missing_terms,
                "faithfulness_score": round(faith_score, 4),
                "citation_accuracy": round(cite_acc, 4),
                "context_citation_precision": round(ret_prec, 4),
                "is_relevant": is_relevant,
                "citations": [c.model_dump() for c in gen_res.citations],
                "grounding_report": gen_res.grounding_report.model_dump() if gen_res.grounding_report else None,
                "latency_ms": round(total_lat, 2)
            })

        n = len(eval_set)
        return {
            "total_queries": n,
            "mean_correctness": round(sum(correctness_scores) / n, 4) if n else 0.0,
            "mean_faithfulness": round(sum(faithfulness_scores) / n, 4) if n else 0.0,
            "mean_citation_accuracy": round(sum(citation_accuracies) / n, 4) if n else 0.0,
            "mean_latency_ms": round(sum(latencies) / n, 2) if n else 0.0,
            "queries": results
        }

    # -----------------------------------------------------------------------
    # 3. Unanswerable & Out-of-Domain Query Benchmarking
    # -----------------------------------------------------------------------
    def evaluate_unanswerable_queries(
        self,
        queries: Optional[List[Dict[str, Any]]] = None,
        mode: str = "hybrid",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Tests questions with facts outside the PDF to verify that the RAG pipeline
        refuses to answer rather than hallucinating false claims or inventing citations.
        """
        eval_set = queries or UNANSWERABLE_QUERIES
        results = []
        
        refusal_count = 0
        zero_invented_citations_count = 0

        for q in eval_set:
            gen_res: GenerationResult = self.generator.generate_answer(
                q["query"],
                retrieval_mode=mode,
                top_k=top_k
            )

            is_refusal = gen_res.is_refusal or GroundingVerifier.is_refusal_response(gen_res.answer)
            has_no_invented_cits = len(gen_res.grounding_report.invented_citations) == 0 if gen_res.grounding_report else True

            if is_refusal:
                refusal_count += 1
            if has_no_invented_cits:
                zero_invented_citations_count += 1

            results.append({
                "id": q["id"],
                "query": q["query"],
                "lang": q["lang"],
                "answer": gen_res.answer,
                "is_refusal": is_refusal,
                "is_grounded": gen_res.is_grounded,
                "citations_count": len(gen_res.citations),
                "unverified_warning_triggered": gen_res.unverified_draft is not None
            })

        n = len(eval_set)
        return {
            "total_unanswerable_queries": n,
            "refusal_rate": round(refusal_count / n, 4) if n else 0.0,
            "zero_invented_citation_rate": round(zero_invented_citations_count / n, 4) if n else 0.0,
            "queries": results
        }

    # -----------------------------------------------------------------------
    # 4. Grounding Verifier Performance on Adversarial Scenarios
    # -----------------------------------------------------------------------
    def evaluate_grounding_verifier(
        self,
        cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Tests the automated GroundingVerifier against 10 adversarial & controlled test cases.
        Calculates Confusion Matrix, Accuracy, Precision, Recall, and F1.
        """
        test_cases = cases or ADVERSARIAL_GROUNDING_CASES
        
        y_true = []
        y_pred = []
        case_details = []

        for c in test_cases:
            report: GroundingReport = GroundingVerifier.verify(
                c["answer"],
                c["context_chunks"]
            )

            expected = c["expected_is_grounded"]
            predicted = report.is_grounded

            y_true.append(expected)
            y_pred.append(predicted)

            case_details.append({
                "id": c["id"],
                "desc": c["desc"],
                "expected_grounded": expected,
                "predicted_grounded": predicted,
                "is_refusal": report.is_refusal,
                "grounding_score": report.grounding_score,
                "invented_citations": report.invented_citations,
                "unsupported_facts": report.unsupported_facts,
                "citation_mismatches": report.citation_mismatches,
                "notes": report.notes
            })

        matrix = compute_confusion_matrix(y_true, y_pred)
        return {
            "metrics": matrix,
            "cases": case_details
        }

    # -----------------------------------------------------------------------
    # 5. Fine-Grained Latency Profiling
    # -----------------------------------------------------------------------
    def profile_latencies(
        self,
        queries: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Profiles retrieval latency (dense, bm25, hybrid) and generation latency,
        measuring both cold-start and warm-query response times.
        """
        eval_set = queries or EVAL_QUERIES
        sample_q = eval_set[0]["query"]

        profile = {}

        for mode in ["dense", "bm25", "hybrid"]:
            # Cold run
            t0 = time.time()
            self.retriever.retrieve(sample_q, mode=mode, top_k=top_k)
            cold_ms = (time.time() - t0) * 1000

            # Warm runs (average of 3)
            warm_times = []
            for _ in range(3):
                t1 = time.time()
                self.retriever.retrieve(sample_q, mode=mode, top_k=top_k)
                warm_times.append((time.time() - t1) * 1000)
            avg_warm_ms = sum(warm_times) / len(warm_times)

            profile[mode] = {
                "cold_ms": round(cold_ms, 2),
                "warm_ms": round(avg_warm_ms, 2)
            }

        # End-to-end generation latency
        t0 = time.time()
        self.generator.generate_answer(sample_q, retrieval_mode="hybrid", top_k=top_k)
        cold_e2e_ms = (time.time() - t0) * 1000

        e2e_warm = []
        for _ in range(3):
            t1 = time.time()
            self.generator.generate_answer(sample_q, retrieval_mode="hybrid", top_k=top_k)
            e2e_warm.append((time.time() - t1) * 1000)
        avg_warm_e2e_ms = sum(e2e_warm) / len(e2e_warm)

        profile["end_to_end"] = {
            "cold_ms": round(cold_e2e_ms, 2),
            "warm_ms": round(avg_warm_e2e_ms, 2)
        }

        return profile
