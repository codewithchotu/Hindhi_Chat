# AGENTS.md — Rules and Instructions for AI Engineers Working on Cross-Lingual Hindi RAG Engine

## 1. Project Identity & Purpose
- **Project:** Cross-Lingual Hindi RAG Engine
- **Candidate:** Mohd Owais
- **Organization:** MegaMindz AI Solutions Pvt Ltd
- **Assignment:** Internship Screening Assignment — Level 1: RAG over a Hindi Document
- **Core Mission:** A production-quality, cross-lingual Retrieval-Augmented Generation (RAG) system accepting user queries in both English and Hindi, retrieving relevant passages from a Devanagari Hindi PDF (*Agni Ki Udaan Adhyayan Sahayika* on Dr. A.P.J. Abdul Kalam), generating grounded answers using Gemini, and automatically rendering verifiable source citations from vector DB metadata.

---

## 2. Fundamental Engineering Guidelines

### 2.1 No Hallucination, No Hardcoding
- **Never hardcode answers or citations** for the test questions or sample queries.
- Citations must be generated dynamically from vector database metadata (`page`, `section`, `chunk_id`, `score`).
- Never claim unmeasured or fabricated benchmark numbers. If evaluating Faithfulness or Time-to-First-Token (TTFT), run actual measurement scripts and report measured values honestly.

### 2.2 Hindi & Devanagari Preprocessing Integrity
- **Unicode Normalization:** Always apply `unicodedata.normalize('NFC', text)` to standardize composite glyphs (e.g., nuqta characters like ज़, फ़).
- **Devanagari Spacing Heuristics:** PDF font extraction frequently separates consonants from subsequent matras (e.g., `कला म` for `कलाम`, `वि चा र` for `विचार`) or produces duplicate syllable artifacts (e.g., `है,है`). Text cleaning must use conservative, linguistically sound regular expressions without corrupting valid separate Hindi words.
- **Sentence Segmentation:** Hindi sentence boundaries are demarcated by the Hindi Danda (`।` - U+0964) and Double Danda (`॥` - U+0965), along with standard punctuation (`?`, `!`, newline). Never segment Hindi text using English periods (`.`) alone.
- **Tables Extraction:** Structured data (Missiles table, Mentors table, Awards table, Timeline table, Glossary, Quick Facts) must preserve column-value relationships rather than becoming garbled flat strings.

### 2.3 Cross-Lingual Semantic Retrieval
- **Embedding Model:** Use multilingual embeddings capable of projecting English queries and Hindi passages into a shared vector space (`BAAI/bge-m3` or high-performance multilingual sentence transformers).
- **Vector Store:** ChromaDB with persistent storage. Chunks must be tagged with complete metadata:
  ```json
  {
    "page": 8,
    "section": "6. इसरो के वर्ष और SLV-III",
    "chunk_id": "doc_p8_c1",
    "source_doc": "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf",
    "content_type": "narrative | table"
  }
  ```
- **Similarity & Metric:** Use Cosine Similarity or Inner Product on normalized vectors.
- **Hybrid Search & Reranking:** Combine dense vector search with sparse keyword matching (BM25) and cross-encoder reranking where appropriate for challenging queries.

### 2.4 Grounded Generation & Citation Rules
- **Evidence-Based Generation:** Prompts must instruct the LLM to synthesize answers solely from retrieved context.
- **Strict Fallback:** If retrieved context is insufficient or irrelevant, the LLM must explicitly state that the document does not contain sufficient information. It must not generate plausible-sounding hallucinations from pre-training.
- **Response Language:** Default to answering in the language of the query (English query $\to$ English answer; Hindi query $\to$ Hindi answer).
- **Mandatory Citations:** Every response must display the source citation containing:
  - Page number
  - Section title / heading
  - Chunk ID
  - Similarity / Relevance score

---

## 3. UI/UX Principles (Clean, ChatGPT-Inspired)
- **Minimalist & Modern:** Neutral tones, clean typography, comfortable whitespace.
- **No Flashy Dashboard Clutter:** Avoid glowing AI effects, gratuitous 3D badges, excessive gradients, or irrelevant charts.
- **Dual Themes:** Clean Dark Mode (charcoal/slate) and Light Mode (soft grey/white), persisted via `localStorage`.
- **User Convenience:** Quick-start chips for the 6 mandatory questions, copy-to-clipboard, collapsible source view with exact snippet inspection, and conversation reset.

---

## 4. Hardware & System Constraints Awareness
- **Storage Management:** Drive `C:` on the host has constrained disk space (~1 GB free), whereas Drive `D:` has 193+ GB free.
- Large artifacts, HuggingFace model caches (`HF_HOME`), PyTorch models (`TORCH_HOME`), and environments must be pointed to `D:\` if needed to prevent disk exhaustion.
- Python 3.11 (`C:\Users\owais\AppData\Local\Programs\Python\Python311\python.exe`) is preferred over 3.14 for PyTorch, Sentence-Transformers, and ChromaDB binary wheel compatibility.

---

## 5. Development Phases Workflow
Work phase by phase with clear verification before proceeding:
1. **Phase 0:** Inspection, Discrepancy Analysis & Architectural Planning.
2. **Phase 1:** Project Setup, Directory Scaffolding & Virtual Environment.
3. **Phase 2:** Document Ingestion, Table Extraction & Devanagari Normalization.
4. **Phase 3:** Multilingual Embeddings & Persistent ChromaDB Store.
5. **Phase 4:** Cross-Lingual & Native Retrieval Pipeline (Dense + Hybrid).
6. **Phase 5:** Grounded Generation with Gemini & Dynamic Citation Builder.
7. **Phase 6:** End-to-End Evaluation & Metric Benchmarking.
8. **Phase 7:** Minimalist ChatGPT-Style React Frontend.
9. **Phase 8:** Containerization (Docker), Documentation, and Submission Readiness.
