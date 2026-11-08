# Architecture Documentation — Cross-Lingual Hindi RAG Engine

## 1. Overview & Architecture Diagram

```mermaid
flowchart TD
    subgraph DataIngestion ["1. Document Ingestion Pipeline"]
        PDF["Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf\n(22 Pages, 20 Sections)"] --> Extractor["PyMuPDF / pdfplumber\nTable & Text Extractor"]
        Extractor --> Normalizer["Devanagari Normalizer\n(Unicode NFC + Ligature Spacing Repair)"]
        Normalizer --> Chunker["Danda-Aware Chunker\n(Hindi '।' boundaries, Token-window overlap)"]
    end

    subgraph EmbeddingStorage ["2. Representation & Persistence"]
        Chunker --> Embedder["BAAI/bge-m3 Multilingual Embedder\n(1024-dim Dense Vectors, Cache on D:)"]
        Embedder --> ChromaDB[("ChromaDB Vector Store\n(Persistent disk collection with page & section metadata)")]
        Chunker --> BM25Index["BM25 Lexical Inverted Index\n(Sparse Keyword Matching)"]
    end

    subgraph RetrievalLayer ["3. Cross-Lingual Semantic Retrieval"]
        Query["User Query\n(Hindi or English)"] --> DenseSearch["Dense Vector Search\n(Cosine Similarity via bge-m3)"]
        Query --> SparseSearch["BM25 Keyword Search\n(Hindi tokenized matching)"]
        ChromaDB --> DenseSearch
        BM25Index --> SparseSearch
        DenseSearch --> HybridReranker["Hybrid Rank Fusion\n(score = α·dense + (1-α)·bm25)"]
        SparseSearch --> HybridReranker
        HybridReranker --> TopK["Top-K Retrieved Chunks + Metadata"]
    end

    subgraph GenerationLayer ["4. Grounded Synthesis & Citations"]
        TopK --> GroundedPrompt["Grounded Context Injection Prompt\n(Strict anti-hallucination guardrails)"]
        GroundedPrompt --> GeminiLLM["Google Gemini API\n(gemini-2.5-flash / gemini-1.5-flash)"]
        GeminiLLM --> GroundedAnswer["Factual Synthesized Answer\n(Language matches user query)"]
        TopK --> CitationBuilder["Dynamic Citation Builder\n(Page, Section, Chunk ID, Score)"]
        GroundedAnswer --> FinalResponse["API Response Payload\n(Answer + Citations + Latency)"]
        CitationBuilder --> FinalResponse
    end

    subgraph FrontendPresentation ["5. Presentation & UI"]
        FinalResponse --> ReactUI["React + Vite Minimalist Chat Interface\n(Dark/Light Themes, Question Chips, Source Inspector)"]
    end
```

## 2. Core Architectural Principles

1. **Cross-Lingual Projection:** English questions and Hindi passages are projected into a common semantic space via `BAAI/bge-m3`.
2. **Deterministic Source Traceability:** Every response is anchored by metadata extracted during ingestion (`page`, `section`, `chunk_id`, `score`).
3. **Linguistic Sensitivity:** Hindi sentence segmentation respects the Hindi Danda (`।` - `U+0964`) and handles ligature spacing anomalies specific to Devanagari PDF typography.
4. **Resilient Hardware Footprint:** High-capacity secondary storage (`D:\`) handles model weights, HuggingFace caches (`HF_HOME`), and PyTorch checkpoints, preventing host system drive exhaustion.
