import os
import sys
import time
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from app.config import get_settings
from app.embeddings.embedder import MultilingualEmbedder
from app.vectorstore.chroma_store import ChromaVectorStore
from app.ingestion.chunker import DocumentChunk

settings = get_settings()

print("=" * 60)
print("PHASE 3: INDEXING CHUNKS INTO CHROMADB WITH BAAI/BGE-M3")
print("=" * 60)

# 1. Load chunks from processed chunks.json
chunks_file = Path("data/processed/chunks.json")
if not chunks_file.exists():
    raise FileNotFoundError(f"Missing {chunks_file}")

with open(chunks_file, "r", encoding="utf-8") as f:
    chunks_raw = json.load(f)

chunks = [
    DocumentChunk(
        chunk_id=c["chunk_id"],
        text=c["text"],
        page=c["page"],
        section=c["section"],
        source_doc=c["source_doc"],
        content_type=c["content_type"],
        char_count=c["char_count"]
    )
    for c in chunks_raw
]

print(f"Loaded {len(chunks)} chunks from {chunks_file}.")

# 2. Initialize Embedder
print(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
t0_model = time.time()
embedder = MultilingualEmbedder(
    model_name=settings.EMBEDDING_MODEL,
    cache_folder=settings.HF_HOME,
    batch_size=16
)
t_model_load = time.time() - t0_model
print(f"Model loaded in {t_model_load:.2f} seconds.")
print(f"Embedding dimension: {embedder.embedding_dim}")
print(f"Device: {embedder.device}")

# 3. Initialize ChromaDB
print(f"Connecting to ChromaDB at: {settings.CHROMA_PERSIST_DIR}")
store = ChromaVectorStore(
    persist_dir=settings.CHROMA_PERSIST_DIR,
    collection_name=settings.COLLECTION_NAME,
    embedder=embedder
)

# 4. Index Chunks
print(f"Starting batch embedding and indexing ({len(chunks)} chunks)...")
t0_index = time.time()
indexed_count = store.add_chunks(chunks, batch_size=16)
t_indexing = time.time() - t0_index

print(f"Successfully indexed {indexed_count} chunks in {t_indexing:.2f} seconds.")
print(f"ChromaDB collection count: {store.count()}")

# 5. Verify Idempotence (re-adding same chunks shouldn't double count)
print("Testing duplicate prevention (upsert idempotence)...")
store.add_chunks(chunks[:5], batch_size=5)
print(f"ChromaDB collection count after re-upsert: {store.count()} (should still be {len(chunks)})")

# 6. Test Retrieval on the 6 Mandatory Questions
mandatory_queries = [
    {"id": 1, "lang": "Hindi", "query": "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?"},
    {"id": 2, "lang": "Hindi", "query": "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?"},
    {"id": 3, "lang": "Hindi", "query": "पोखरण-II में कलाम की क्या भूमिका थी?"},
    {"id": 4, "lang": "English", "query": "Which institution did Kalam attend to study aeronautical engineering?"},
    {"id": 5, "lang": "English", "query": "Who co-wrote the autobiography, and in what year was it published?"},
    {"id": 6, "lang": "English", "query": "How and where did Kalam die in 2015?"}
]

print("\n" + "=" * 60)
print("TESTING RETRIEVAL ON 6 MANDATORY QUESTIONS")
print("=" * 60)

retrieval_results = []

for q in mandatory_queries:
    t0_q = time.time()
    hits = store.search(q["query"], top_k=3)
    latency_ms = (time.time() - t0_q) * 1000
    
    top_hit = hits[0] if hits else None
    
    res = {
        "id": q["id"],
        "lang": q["lang"],
        "query": q["query"],
        "latency_ms": round(latency_ms, 2),
        "top_hit": {
            "chunk_id": top_hit["chunk_id"] if top_hit else None,
            "page": top_hit["metadata"]["page"] if top_hit else None,
            "section": top_hit["metadata"]["section"] if top_hit else None,
            "content_type": top_hit["metadata"]["content_type"] if top_hit else None,
            "score": top_hit["score"] if top_hit else None,
            "snippet": top_hit["text"][:150] if top_hit else None
        } if top_hit else None,
        "all_hits": [
            {
                "chunk_id": h["chunk_id"],
                "page": h["metadata"]["page"],
                "section": h["metadata"]["section"],
                "score": h["score"]
            }
            for h in hits
        ]
    }
    retrieval_results.append(res)

    print(f"\n[Q{q['id']} - {q['lang']}] {q['query']}")
    print(f"Latency: {latency_ms:.2f} ms")
    if top_hit:
        print(f"  Top Hit: {top_hit['chunk_id']} | Page {top_hit['metadata']['page']} | Score: {top_hit['score']}")
        print(f"  Section: {top_hit['metadata']['section']}")
        print(f"  Snippet: {top_hit['text'][:140]}...")
    else:
        print("  NO HITS FOUND!")

# Save results for reporting
out_file = Path("data/processed/phase3_retrieval_benchmark.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({
        "model_name": settings.EMBEDDING_MODEL,
        "embedding_dim": embedder.embedding_dim,
        "device": embedder.device,
        "model_load_time_sec": round(t_model_load, 2),
        "indexing_time_sec": round(t_indexing, 2),
        "total_chunks_indexed": store.count(),
        "results": retrieval_results
    }, f, ensure_ascii=False, indent=2)

print(f"\nSaved Phase 3 benchmark results to {out_file}")
