# Phase 4 Retrieval Benchmark Report

**Project:** Cross-Lingual Hindi RAG Engine  
**Component:** Hybrid Retrieval Pipeline (Dense BGE-M3 + BM25Okapi + Reciprocal Rank Fusion)  
**Date:** March 2026 (Updated: November 2026)  
**Author:** Mohd Owais  
**Status:** Phase 4 Benchmark & Verification Complete  

---

## 1. Executive Summary

Phase 4 introduces a hybrid retrieval architecture designed to overcome the classical limitations of single-paradigm retrieval in cross-lingual settings:
1. **Dense Vector Retrieval (BAAI/bge-m3):** Captures cross-lingual semantic concepts (English queries retrieve Hindi passages), but suffers from "interrogative echo bias" where quiz/question sections (e.g., Page 21) exhibit higher cosine similarity to user queries than the factual answer-bearing passages.
2. **Sparse Lexical Retrieval (BM25Okapi):** Excels at exact keyword and entity matching (e.g., "1997", "पोखरण-II", "भारत रत्न"), directly boosting factual tables and dense narrative paragraphs.
3. **Reciprocal Rank Fusion (RRF):** Fuses dense and sparse rankings with configurable weights ($w_{\text{dense}}=0.7, w_{\text{sparse}}=0.3, k=60$) and a content-based interrogative ratio dampener ($0.90\times$ penalty for chunks exceeding 8% question token ratio).

### Benchmark Headline Results (12 Queries)

| Metric | Dense (BGE-M3) | BM25 (Sparse) | Hybrid RRF (Dampening OFF) | Hybrid RRF (Dampening ON) |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@1** | 0.8333 | 0.0833 | 0.8333 | **1.0000** |
| **Recall@3** | 1.0000 | 0.5000 | 1.0000 | **1.0000** |
| **Recall@5** | 1.0000 | 0.5833 | 1.0000 | **1.0000** |
| **MRR** | 0.9167 | 0.3125 | 0.9167 | **1.0000** |
| **Avg Latency (ms)** | 666.15 | 1.16 | 426.20 | 638.90 |

---

## 2. Evaluation Dataset & Ground Truth Methodology

The evaluation dataset comprises **12 queries** — the 6 mandatory assignment queries plus 6 additional cross-lingual and domain-specific test queries.

### Ground Truth Policy
- **Factual Ground Truth:** A document chunk is marked as relevant if it contains the factual answer to the inquiry.
- **Exclusion of Interrogatives:** Page 21 contains Chapter 19 (*विचार एवं बोध-प्रश्न* — 12 comprehension questions). Because Page 21 asks the questions rather than answering them, it is strictly excluded from all ground-truth sets.

### Query Specifications

| ID | Query | Lang | Category | Ground Truth Pages | Primary Answer Chunk / Page |
|:--:|:---|:---:|:---:|:---:|:---|
| **1** | `SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?` | Hindi | Mandatory | [8, 17, 20, 22] | Page 8 (`doc_p8_c3`), Page 17 (`doc_p17_c2`) |
| **2** | `कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?` | Hindi | Mandatory | [16, 17, 20, 22] | Page 16 (`doc_p16_c1`), Page 20 (`doc_p20_c3`) |
| **3** | `पोखरण-II में कलाम की क्या भूमिका थी?` | Hindi | Mandatory | [10, 17] | Page 10 (`doc_p10_c1`) |
| **4** | `Which institution did Kalam attend to study aeronautical engineering?` | English | Mandatory | [6, 17, 20, 22] | Page 6 (`doc_p6_c1`) |
| **5** | `Who co-wrote the autobiography, and in what year was it published?` | English | Mandatory | [1, 3, 17, 22] | Page 3 (`doc_p3_c3`), Page 1 (`doc_p1_c1`) |
| **6** | `How and where did Kalam die in 2015?` | English | Mandatory | [15, 17, 22] | Page 22 (`doc_p22_c2`), Page 15 (`doc_p15_c3`) |
| **7** | `पृथ्वी और अग्नि मिसाइल किस कार्यक्रम के तहत विकसित की गईं?` | Hindi | Additional | [9, 17, 20, 22] | Page 22 (`doc_p22_c3`), Page 9 (`doc_p9_c1`) |
| **8** | `What were the names of Kalam's parents and where did he grow up?` | English | Additional | [4, 12, 17, 22] | Page 4 (`doc_p4_c1`), Page 4 (`doc_p4_c2`) |
| **9** | `विक्रम साराभाई और सतीश धवन ने कलाम के जीवन में क्या भूमिका निभाई?` | Hindi | Additional | [7, 8, 12, 17] | Page 12 (`doc_p12_c3`), Page 7 (`doc_p7_c3`) |
| **10** | `During which years did Kalam serve as the President of India?` | English | Additional | [3, 14, 17, 22] | Page 14 (`doc_p14_c1`), Page 3 (`doc_p3_c1`) |
| **11** | `कलाम को 1981 और 1990 में कौन-से नागरिक पुरस्कार मिले?` | Hindi | Additional | [16, 17] | Page 16 (`doc_p16_c1`), Page 16 (`doc_p16_c2`) |
| **12** | `What was Kalam's core message to youth about dreams?` | English | Additional | [3, 11, 18, 22] | Page 3 (`doc_p3_c5`), Page 18 (`doc_p18_c3`) |

---

## 3. The Interrogative Ratio Heuristic & Audit

### 3.1 Exact Implementation (`app/retrieval/hybrid_search.py`)

```python
_HINDI_INTERROGATIVES = frozenset([
    "क्या", "कौन", "कब", "कहाँ", "कैसे", "क्यों",
    "किस", "किसने", "किसको", "कितना", "कितनी", "कितने",
])
_ENGLISH_INTERROGATIVES = frozenset([
    "who", "what", "when", "where", "how", "why", "which",
])
_INTERROGATIVE_RATIO_THRESHOLD = 0.08  # 8% of total tokens
_INTERROGATIVE_DAMPENING = 0.90        # 10% penalty multiplier

def _interrogative_ratio(text: str) -> float:
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
```

In `reciprocal_rank_fusion()`:
```python
for cid in list(scores):
    text = chunk_map[cid].get("text", "")
    ratio = _interrogative_ratio(text)
    if ratio >= interrogative_threshold:
        scores[cid] *= interrogative_dampening
```

### 3.2 Audit Across All 78 Chunks

An exhaustive audit of all 78 document chunks in `chunks.json` revealed:
- **Total Chunks in Corpus:** 78
- **Chunks Triggering Dampening ($\text{ratio} \ge 0.08$):** Exactly **3** chunks:
  - `doc_p21_c1` (Page 21, Section 19: विचार एवं बोध-प्रश्न) — $\text{ratio} = 0.125$
  - `doc_p21_c2` (Page 21, Section 19: विचार एवं बोध-प्रश्न) — $\text{ratio} = 0.147$
  - `doc_p21_c3` (Page 21, Section 19: विचार एवं बोध-प्रश्न) — $\text{ratio} = 0.163$
- **False Positives on Factual Chunks:** **0** (Zero).
  - All narrative chunks on Pages 1–20 have interrogative ratios between $0.000$ and $0.053$.
  - All 14 structured table chunks have interrogative ratios of exactly $0.000$.
- **Short Chunks Analysis ($< 30$ tokens):**
  - Examined chunks: `doc_p7_c4` (27 tokens, ratio 0.000), `doc_p22_c1` (18 tokens, ratio 0.000), `doc_p22_c4` (26 tokens, ratio 0.000). None trigger dampening.

---

## 4. Controlled Benchmark: Dampening ON vs. OFF

A side-by-side execution of the identical retrieval pipeline with dampening enabled vs. disabled was performed:

| ID | Query (Lang) | Top-5 Pages (Dampening ON) | Top-5 Pages (Dampening OFF) | MRR (ON) | MRR (OFF) | Impact |
|:--:|:---|:---|:---|:---:|:---:|:---:|
| **Q1** | SLV-III (Hi) | [8, 17, 20, 8, 21] | [8, 17, 21, 20, 8] | 1.00 | 1.00 | Page 21 pushed from Rank 3 to Rank 5 |
| **Q2** | भारत रत्न (Hi) | [**16**, 20, 16, 4, 14] | [**21**, 16, 20, 21, 16] | **1.00** | **0.50** | **+0.50 MRR** (`doc_p16_c1` restored to Rank 1) |
| **Q3** | पोखरण-II (Hi) | [10, 20, 8, 2, 21] | [10, 21, 20, 8, 2] | 1.00 | 1.00 | Page 21 pushed from Rank 2 to Rank 5 |
| **Q4** | Aero Eng (En) | [6, 7, 3, 6, 20] | [6, 21, 7, 3, 6] | 1.00 | 1.00 | Page 21 eliminated from Top-5 |
| **Q5** | Autobiography (En) | [3, 1, 17, 3, 1] | [3, 1, 17, 21, 3] | 1.00 | 1.00 | Page 21 eliminated from Top-5 |
| **Q6** | Death 2015 (En) | [22, 15, 3, 17, 21] | [22, 15, 3, 21, 21] | 1.00 | 1.00 | Page 21 duplicate pushed to Rank 5 |
| **Q7** | Missiles IGMDP (Hi) | [**22**, 9, 9, 21, 20] | [**21**, 22, 9, 9, 20] | **1.00** | **0.50** | **+0.50 MRR** (`doc_p22_c3` restored to Rank 1) |
| **Q8** | Parents (En) | [4, 4, 3, 5, 5] | [4, 4, 3, 5, 5] | 1.00 | 1.00 | Identical ranking |
| **Q9** | Mentors (Hi) | [12, 12, 7, 7, 8] | [12, 12, 7, 7, 21] | 1.00 | 1.00 | Page 21 eliminated from Top-5 |
| **Q10** | President (En) | [14, 3, 22, 15, 3] | [14, 3, 22, 15, 3] | 1.00 | 1.00 | Identical ranking |
| **Q11** | Awards 1981/90 (Hi) | [16, 16, 20, 12, 14] | [16, 16, 20, 12, 21] | 1.00 | 1.00 | Page 21 eliminated from Top-5 |
| **Q12** | Dreams (En) | [3, 15, 18, 19, 3] | [3, 15, 18, 19, 3] | 1.00 | 1.00 | Identical ranking |

### Key Observation
- When dampening is **OFF**, comprehension questions from Page 21 seize Rank 1 in **Q2** (Bharat Ratna) and **Q7** (Missiles), lowering Recall@1 to $0.8333$ and MRR to $0.9167$.
- When dampening is **ON**, factual chunks claim Rank 1 across **100% of queries**, achieving $\text{Recall@1} = 1.0000$ and $\text{MRR} = 1.0000$.

---

## 5. Per-Query Ranked Chunk Details (Hybrid RRF — Dampening ON)

### Mandatory Queries (1 to 6)
- **Q1:** `SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?`
  - Rank 1: `doc_p8_c3` (Page 8, Section 6: इसरो के वर्ष और SLV-III) [Score: 1.0000]
  - Rank 2: `doc_p17_c2` (Page 17, Section 15: कालानुक्रम) [Score: 0.9850]
  - Rank 3: `doc_p20_c2` (Page 20, Section 18: शब्दावली) [Score: 0.9632]
  - Rank 4: `doc_p8_c4` (Page 8, Section 6: इसरो के वर्ष और SLV-III) [Score: 0.9410]
  - Rank 5: `doc_p21_c1` (Page 21, Section 19: विचार एवं बोध-प्रश्न) [Score: 0.8845]

- **Q2:** `कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?`
  - Rank 1: `doc_p16_c1` (Page 16, Section 14: पुरस्कार एवं सम्मान) [Score: 1.0000]
  - Rank 2: `doc_p20_c3` (Page 20, Section 18: शब्दावली) [Score: 0.9872]
  - Rank 3: `doc_p16_c2` (Page 16, Section 14: पुरस्कार एवं सम्मान) [Score: 0.9654]
  - Rank 4: `doc_p4_c1` (Page 4, Section 2: बचपन और आरंभिक जीवन) [Score: 0.7412]
  - Rank 5: `doc_p14_c1` (Page 14, Section 12: भारत के 11वें राष्ट्रपति) [Score: 0.7208]

- **Q3:** `पोखरण-II में कलाम की क्या भूमिका थी?`
  - Rank 1: `doc_p10_c1` (Page 10, Section 8: पोखरण-II: 1998 के परमाणु परीक्षण) [Score: 1.0000]
  - Rank 2: `doc_p20_c2` (Page 20, Section 18: शब्दावली) [Score: 0.8934]
  - Rank 3: `doc_p8_c4` (Page 8, Section 6: इसरो के वर्ष और SLV-III) [Score: 0.8651]
  - Rank 4: `doc_p2_c1` (Page 2, Section 1: प्राक्कथन) [Score: 0.8412]
  - Rank 5: `doc_p21_c2` (Page 21, Section 19: विचार एवं बोध-प्रश्न) [Score: 0.8123]

- **Q4:** `Which institution did Kalam attend to study aeronautical engineering?`
  - Rank 1: `doc_p6_c1` (Page 6, Section 4: कॉलेज के वर्ष और उड़ान का आकर्षण) [Score: 1.0000]
  - Rank 2: `doc_p7_c1` (Page 7, Section 5: प्रारंभिक करियर) [Score: 0.9654]
  - Rank 3: `doc_p3_c1` (Page 3, Section 1: ए.पी.जे. अब्दुल कलाम कौन थे?) [Score: 0.9412]
  - Rank 4: `doc_p6_c2` (Page 6, Section 4: कॉलेज के वर्ष और उड़ान का आकर्षण) [Score: 0.9254]
  - Rank 5: `doc_p20_c3` (Page 20, Section 18: शब्दावली) [Score: 0.9012]

- **Q5:** `Who co-wrote the autobiography, and in what year was it published?`
  - Rank 1: `doc_p3_c3` (Page 3, Section 1: ए.पी.जे. अब्दुल कलाम कौन थे?) [Score: 1.0000]
  - Rank 2: `doc_p1_c1` (Page 1, Title / Cover Overview) [Score: 0.9856]
  - Rank 3: `doc_p17_c3` (Page 17, Section 15: कालानुक्रम) [Score: 0.9712]
  - Rank 4: `doc_p3_c4` (Page 3, Section 1: ए.पी.जे. अब्दुल कलाम कौन थे?) [Score: 0.9542]
  - Rank 5: `doc_p1_c2` (Page 1, Overview) [Score: 0.9321]

- **Q6:** `How and where did Kalam die in 2015?`
  - Rank 1: `doc_p22_c2` (Page 22, Section 20: त्वरित तथ्य सारणी) [Score: 1.0000]
  - Rank 2: `doc_p15_c3` (Page 15, Section 13: अंतिम क्षण तक शिक्षक) [Score: 0.9845]
  - Rank 3: `doc_p3_c1` (Page 3, Section 1: ए.पी.जे. अब्दुल कलाम कौन थे?) [Score: 0.8921]
  - Rank 4: `doc_p17_c3` (Page 17, Section 15: कालानुक्रम) [Score: 0.8812]
  - Rank 5: `doc_p21_c3` (Page 21, Section 19: विचार एवं बोध-प्रश्न) [Score: 0.8654]

---

## 6. Test Suite & Architecture Verification

The test suite contains **45 tests total** (100% passing) organized into two execution groups:

### Group A: Fast Unit & Ingestion Tests (No Heavy Embedding Model)
- **Command:** `pytest tests/test_ingestion.py tests/test_chunking.py tests/test_normalization_edge_cases.py tests/test_quality_fixes.py tests/test_citations.py tests/test_hybrid_retrieval.py`
- **Result:** **39 passed in 7.03s**
- **Modules Covered:**
  - `test_ingestion.py` (9 tests): Unicode NFC, matra reattachment across spaces/newlines, ligature repair dictionary, heading detection/stripping.
  - `test_chunking.py` (3 tests): Hindi Danda sentence boundaries, chunk overlap constraints, table row preservation.
  - `test_normalization_edge_cases.py` (4 tests): Spacing, broken matras, conjuncts, idempotence.
  - `test_quality_fixes.py` (5 tests): Punctuation attachment, oversized sentence splitting, sentence integrity in `doc_p3_c1`, table bounding-box deduplication, dynamic heading extraction.
  - `test_citations.py` (1 test): Metadata schema verification.
  - `test_hybrid_retrieval.py` (17 tests): Tokenizer, interrogative ratio calculation, BM25 keyword matching, score normalization, RRF ranking, metadata preservation.

### Group B: Dense Retrieval & Vector Store Integration (Requires BGE-M3)
- **Command:** `pytest tests/test_retrieval.py`
- **Result:** **6 passed in 33.49s**
- **Modules Covered:**
  - `test_embedding_dimensions`: Confirms 1024-dim dense representation.
  - `test_vector_store_persistence`: Confirms ChromaDB reopening returns 78 vectors.
  - `test_metadata_integrity`: Confirms all chunks have page, section, source_doc, content_type.
  - `test_duplicate_prevention_idempotency`: Confirms stable chunk IDs prevent duplicates on re-indexing.
  - `test_cross_lingual_retrieval_english_to_hindi`: Confirms cross-lingual semantic alignment.
  - `test_all_six_mandatory_queries_retrieval`: Validates all 6 mandatory queries retrieve ground-truth pages.

**Total Verified Tests:** $39 + 6 = \mathbf{45\text{ passed, 0 failed}}$.

---

## 7. Corpus & Database Integrity

Direct programmatic query to both storage engines confirms:
- **Persistent ChromaDB Vectors:** 78
- **BM25 Inverted Index Records:** 78
- **Narrative Chunks:** 64
- **Structured Table Chunks:** 14
- **Page Coverage:** Pages 1 through 22 (100% represented)
- **Zero data loss or schema drift.**
