"""
run_phase4_benchmark.py — Phase 4 comprehensive retrieval benchmark.

Evaluates Dense, BM25, and Hybrid RRF across 12 queries (6 mandatory + 6
additional).  Records per-query ranked results, Recall@K, MRR, cold-start
latency, warm latency, and chunk-count verification.

Run:
    cd backend
    $env:PYTHONPATH='.'
    python run_phase4_benchmark.py
"""
import sys
import time
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from app.retrieval.retriever import UnifiedRetriever

# ---------------------------------------------------------------------------
# Evaluation dataset — 12 queries with ground-truth page sets
# Ground-truth = pages that contain the factual answer to the question.
# Page 21 (comprehension questions) is excluded from ground-truth for every
# query because it reflects the query, not the answer.
# ---------------------------------------------------------------------------
EVAL_QUERIES = [
    # ── 6 Mandatory queries ─────────────────────────────────────────────
    {
        "id": 1,
        "query": "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?",
        "lang": "Hindi",
        "ground_truth_pages": [8, 17, 20, 22],
        "category": "mandatory",
        "primary_page": 8,
    },
    {
        "id": 2,
        "query": "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?",
        "lang": "Hindi",
        "ground_truth_pages": [16, 17, 20, 22],
        "category": "mandatory",
        "primary_page": 16,
    },
    {
        "id": 3,
        "query": "पोखरण-II में कलाम की क्या भूमिका थी?",
        "lang": "Hindi",
        "ground_truth_pages": [10, 17],
        "category": "mandatory",
        "primary_page": 10,
    },
    {
        "id": 4,
        "query": "Which institution did Kalam attend to study aeronautical engineering?",
        "lang": "English",
        "ground_truth_pages": [6, 17, 20, 22],
        "category": "mandatory",
        "primary_page": 6,
    },
    {
        "id": 5,
        "query": "Who co-wrote the autobiography, and in what year was it published?",
        "lang": "English",
        "ground_truth_pages": [1, 3, 17, 22],
        "category": "mandatory",
        "primary_page": 3,
    },
    {
        "id": 6,
        "query": "How and where did Kalam die in 2015?",
        "lang": "English",
        "ground_truth_pages": [15, 17, 22],
        "category": "mandatory",
        "primary_page": 15,
    },
    # ── 6 Additional queries ─────────────────────────────────────────────
    {
        "id": 7,
        "query": "पृथ्वी और अग्नि मिसाइल किस कार्यक्रम के तहत विकसित की गईं?",
        "lang": "Hindi",
        "ground_truth_pages": [9, 17, 20, 22],
        "category": "additional",
        "primary_page": 9,
    },
    {
        "id": 8,
        "query": "What were the names of Kalam's parents and where did he grow up?",
        "lang": "English",
        "ground_truth_pages": [4, 12, 17, 22],
        "category": "additional",
        "primary_page": 4,
    },
    {
        "id": 9,
        "query": "विक्रम साराभाई और सतीश धवन ने कलाम के जीवन में क्या भूमिका निभाई?",
        "lang": "Hindi",
        "ground_truth_pages": [7, 8, 12, 17],
        "category": "additional",
        "primary_page": 7,
    },
    {
        "id": 10,
        "query": "During which years did Kalam serve as the President of India?",
        "lang": "English",
        "ground_truth_pages": [3, 14, 17, 22],
        "category": "additional",
        "primary_page": 14,
    },
    {
        "id": 11,
        "query": "कलाम को 1981 और 1990 में कौन-से नागरिक पुरस्कार मिले?",
        "lang": "Hindi",
        "ground_truth_pages": [16, 17],
        "category": "additional",
        "primary_page": 16,
    },
    {
        "id": 12,
        "query": "What was Kalam's core message to youth about dreams?",
        "lang": "English",
        "ground_truth_pages": [3, 11, 18, 22],
        "category": "additional",
        "primary_page": 3,
    },
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_mrr(pages, ground_truth_pages):
    for rank, p in enumerate(pages):
        if p in ground_truth_pages:
            return 1.0 / (rank + 1)
    return 0.0


def evaluate_method(name, retrieve_fn, queries, top_k=5):
    """Run retrieval for every query and compute aggregate metrics."""
    per_query = []
    cold_latencies = []
    warm_latencies = []
    r1_hits, r3_hits, r5_hits, mrrs = [], [], [], []

    for q in queries:
        g_pages = set(q["ground_truth_pages"])

        # Cold run (first call in this process)
        t0 = time.time()
        hits = retrieve_fn(q["query"], top_k=top_k)
        cold_ms = (time.time() - t0) * 1000

        # Warm run (model already loaded)
        t1 = time.time()
        hits = retrieve_fn(q["query"], top_k=top_k)
        warm_ms = (time.time() - t1) * 1000

        cold_latencies.append(cold_ms)
        warm_latencies.append(warm_ms)

        pages = [h["metadata"]["page"] for h in hits]
        mrr   = compute_mrr(pages, g_pages)
        r1    = int(len(pages) > 0 and pages[0] in g_pages)
        r3    = int(any(p in g_pages for p in pages[:3]))
        r5    = int(any(p in g_pages for p in pages[:5]))

        r1_hits.append(r1)
        r3_hits.append(r3)
        r5_hits.append(r5)
        mrrs.append(mrr)

        ranked = [
            {
                "rank": i + 1,
                "chunk_id": h["chunk_id"],
                "page": h["metadata"]["page"],
                "section": h["metadata"]["section"],
                "score": h.get("score"),
                "rrf_score": h.get("rrf_score"),
            }
            for i, h in enumerate(hits)
        ]

        per_query.append({
            "id":           q["id"],
            "query":        q["query"],
            "lang":         q["lang"],
            "category":     q["category"],
            "ground_truth_pages": q["ground_truth_pages"],
            "primary_page": q["primary_page"],
            "cold_latency_ms": round(cold_ms, 2),
            "warm_latency_ms": round(warm_ms, 2),
            "retrieved_pages": pages,
            "recall@1": r1,
            "recall@3": r3,
            "recall@5": r5,
            "mrr": round(mrr, 4),
            "ranked_results": ranked,
        })

    n = len(queries)
    return {
        "method": name,
        "recall@1": round(sum(r1_hits) / n, 4),
        "recall@3": round(sum(r3_hits) / n, 4),
        "recall@5": round(sum(r5_hits) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "avg_cold_latency_ms": round(sum(cold_latencies) / n, 2),
        "avg_warm_latency_ms": round(sum(warm_latencies) / n, 2),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=" * 70)
print("Phase 4 — Cross-Lingual Hybrid Retrieval Benchmark")
print("=" * 70)

print("\n[1/4] Initialising UnifiedRetriever (loads bge-m3 + BM25)...")
t_init = time.time()
retriever = UnifiedRetriever(enable_reranker=False)
init_ms = (time.time() - t_init) * 1000
print(f"      Done in {init_ms:.0f} ms")

# Verify chunk count
chunk_count = retriever.vector_store.count()
bm25_count  = len(retriever.chunks)
print(f"\n[2/4] Verifying corpus integrity")
print(f"      ChromaDB vectors : {chunk_count}")
print(f"      BM25 corpus size : {bm25_count}")
assert chunk_count == 78, f"Expected 78 vectors, got {chunk_count}"
assert bm25_count == 78,  f"Expected 78 chunks, got {bm25_count}"
print("      ✓ All 78 chunks verified in both stores")

print("\n[3/4] Running three retrieval methods across 12 queries...")

print("  → Dense (BGE-M3)...")
dense_results = evaluate_method(
    "Dense (BGE-M3)", retriever.retrieve_dense, EVAL_QUERIES
)

print("  → BM25 (Sparse)...")
bm25_results = evaluate_method(
    "BM25 (Sparse)", retriever.retrieve_bm25, EVAL_QUERIES
)

print("  → Hybrid RRF (Dense 0.7 + BM25 0.3)...")
hybrid_results = evaluate_method(
    "Hybrid RRF", retriever.retrieve_hybrid, EVAL_QUERIES
)

# Save full results
out_path = Path("data/processed/retrieval_benchmark_phase4.json")
out_data = {
    "dense":      dense_results,
    "bm25":       bm25_results,
    "hybrid_rrf": hybrid_results,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_data, f, ensure_ascii=False, indent=2)
print(f"\n[4/4] Results saved → {out_path}")

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("AGGREGATE METRICS (12 queries)")
print("=" * 80)
FMT = "{:<28} {:>10} {:>10} {:>10} {:>8} {:>14} {:>14}"
print(FMT.format("Method", "Recall@1", "Recall@3", "Recall@5", "MRR",
                 "Cold(avg ms)", "Warm(avg ms)"))
print("-" * 80)
for m in [dense_results, bm25_results, hybrid_results]:
    print(FMT.format(
        m["method"],
        m["recall@1"],
        m["recall@3"],
        m["recall@5"],
        m["mrr"],
        m["avg_cold_latency_ms"],
        m["avg_warm_latency_ms"],
    ))
print("=" * 80)

# ---------------------------------------------------------------------------
# Per-query detail for mandatory queries
# ---------------------------------------------------------------------------
print("\nPER-QUERY DETAIL — 6 MANDATORY QUERIES")
print("-" * 80)
for method_data, label in [
    (dense_results, "Dense"),
    (bm25_results, "BM25 "),
    (hybrid_results, "Hybrid"),
]:
    print(f"\n  [{label}]")
    for r in method_data["per_query"]:
        if r["category"] != "mandatory":
            continue
        pages_str = str(r["retrieved_pages"])
        flag = "✓" if r["recall@1"] else ("~" if r["recall@3"] else "✗")
        print(
            f"    Q{r['id']} [{r['lang'][:2]}] {flag}  "
            f"top={r['retrieved_pages'][0] if r['retrieved_pages'] else '?'}  "
            f"pages={pages_str}  "
            f"MRR={r['mrr']:.2f}  "
            f"warm={r['warm_latency_ms']:.0f}ms"
        )

# ---------------------------------------------------------------------------
# Q2 special check (Page 16 vs Page 21)
# ---------------------------------------------------------------------------
print("\nQ2 RANKING CHECK  (bharat ratna — factual page 16 vs question page 21)")
print("-" * 60)
for method_data, label in [
    (dense_results, "Dense "),
    (hybrid_results, "Hybrid"),
]:
    q2 = next(r for r in method_data["per_query"] if r["id"] == 2)
    pages = q2["retrieved_pages"]
    rank16 = next((i + 1 for i, p in enumerate(pages) if p == 16), "not in top-5")
    rank21 = next((i + 1 for i, p in enumerate(pages) if p == 21), "not in top-5")
    top_cid = q2["ranked_results"][0]["chunk_id"] if q2["ranked_results"] else "?"
    print(f"  {label}: pages={pages}  "
          f"rank(p16)={rank16}  rank(p21)={rank21}  "
          f"top={top_cid}")

print()
