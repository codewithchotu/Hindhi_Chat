"""Phase 9: Final Submission Audit - 6 Mandatory Question Live Retrieval Verification."""
import sys, json, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'backend')

from app.retrieval.retriever import get_unified_retriever

retriever = get_unified_retriever()

queries = [
    ("Q1", "hi", "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?", "doc_p8_c3", 8),
    ("Q2", "hi", "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", "doc_p16_c2", 16),
    ("Q3", "hi", "पोखरण-II में कलाम की क्या भूमिका थी?", "doc_p10_c1", 10),
    ("Q4", "en", "Which institution did Kalam attend to study aeronautical engineering?", "doc_p6_c2", 6),
    ("Q5", "en", "Who co-wrote the autobiography, and in what year was it published?", "doc_p3_c3", 3),
    ("Q6", "en", "How and where did Kalam die in 2015?", "doc_p15_c3", 15),
]

print("=" * 80)
print("PHASE 9: 6 MANDATORY QUESTIONS - LIVE HYBRID RETRIEVAL VERIFICATION")
print("Fields: page and chunk_id live in hit['metadata']['page'] and hit['chunk_id']")
print("=" * 80)

all_pass = True
results_summary = []
for qid, lang, query, expected_chunk, expected_page in queries:
    t0 = time.perf_counter()
    hits = retriever.retrieve(query, top_k=5, mode="hybrid")
    elapsed = (time.perf_counter() - t0) * 1000

    top1 = hits[0] if hits else {}
    top1_chunk = top1.get("chunk_id", "N/A")
    top1_page  = top1.get("metadata", {}).get("page", -1)
    top1_section = top1.get("metadata", {}).get("section", "N/A")
    top1_score = top1.get("rrf_score", top1.get("score", 0))

    pages_in_5  = [h.get("metadata", {}).get("page") for h in hits]
    chunks_in_5 = [h.get("chunk_id") for h in hits]

    page_hit  = expected_page in pages_in_5
    chunk_hit_at1 = (top1_chunk == expected_chunk)
    chunk_hit_at5 = expected_chunk in chunks_in_5

    status = "PASS" if page_hit else "FAIL"
    if not page_hit:
        all_pass = False

    print(f"\n{qid} [{lang.upper()}] -> {status}  ({elapsed:.1f} ms)")
    print(f"  Query:    {query}")
    print(f"  Expected: chunk={expected_chunk}, page={expected_page}")
    print(f"  Top-1:    chunk={top1_chunk}, page={top1_page}, rrf_score={top1_score:.4f}")
    print(f"  Section:  {top1_section}")
    print(f"  Top-5 pages:  {pages_in_5}")
    print(f"  Top-5 chunks: {chunks_in_5}")
    print(f"  Page hit @5:    {page_hit}")
    print(f"  Chunk hit @1:   {chunk_hit_at1}")
    print(f"  Chunk hit @5:   {chunk_hit_at5}")

    results_summary.append({
        "qid": qid, "lang": lang, "status": status,
        "expected_chunk": expected_chunk, "expected_page": expected_page,
        "top1_chunk": top1_chunk, "top1_page": top1_page,
        "top1_rrf_score": round(top1_score, 4),
        "top5_pages": pages_in_5, "top5_chunks": chunks_in_5,
        "page_hit_at5": page_hit,
        "chunk_hit_at1": chunk_hit_at1,
        "chunk_hit_at5": chunk_hit_at5,
        "latency_ms": round(elapsed, 1)
    })

print("\n" + "=" * 80)
print(f"OVERALL: {'ALL 6 PASSED' if all_pass else 'SOME FAILED'}")
print("=" * 80)

with open("results/phase9_mandatory_query_audit.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print("\nDetailed results saved to results/phase9_mandatory_query_audit.json")
