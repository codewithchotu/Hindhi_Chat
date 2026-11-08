"""
query.py — FastAPI router for Cross-Lingual Hindi RAG query endpoints.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.generation.answer_generator import get_answer_generator, GenerationResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG Query"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question in English or Hindi.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of passages to retrieve.")
    use_hybrid: bool = Field(default=True, description="Enable hybrid search (Dense + BM25 with RRF).")
    retrieval_mode: Optional[str] = Field(default="hybrid", description="Explicit mode: 'hybrid', 'dense', or 'bm25'.")


class CitationResponse(BaseModel):
    chunk_id: str
    page: int
    section: str
    score: float
    score_type: str = "dense_similarity"
    snippet: str = ""
    source_doc: str = "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"


class QueryResponse(BaseModel):
    query: str
    answer: str
    language: str
    model_name: str
    citations: List[CitationResponse] = Field(default_factory=list)
    citations_markdown: str = ""
    full_response: str = ""
    is_grounded: bool = True
    is_refusal: bool = False
    unverified_draft: Optional[str] = None
    latency_ms: float = 0.0


@router.post("/query", response_model=QueryResponse)
def execute_query(payload: QueryRequest):
    """
    Execute grounded RAG query over the Hindi Devanagari corpus.
    Supports English and Hindi questions with verifiable citations and grounding verification.
    """
    try:
        generator = get_answer_generator()
        mode = payload.retrieval_mode or ("hybrid" if payload.use_hybrid else "dense")
        
        result: GenerationResult = generator.generate_answer(
            query=payload.query,
            retrieval_mode=mode,
            top_k=payload.top_k
        )
        
        citations_data = [
            CitationResponse(
                chunk_id=c.chunk_id,
                page=c.page,
                section=c.section,
                score=c.score,
                score_type=c.score_type,
                snippet=c.snippet,
                source_doc=c.source_doc,
            )
            for c in result.citations
        ]
        
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            language=result.language,
            model_name=result.model_name,
            citations=citations_data,
            citations_markdown=result.citations_markdown,
            full_response=result.full_response,
            is_grounded=result.is_grounded,
            is_refusal=result.is_refusal,
            unverified_draft=result.unverified_draft,
            latency_ms=result.latency_ms,
        )
    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/corpus-info")
def get_corpus_info():
    """Returns metadata about the indexed Hindi corpus."""
    return {
        "document_name": "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf",
        "title": "अग्नि की उड़ान: अध्ययन एवं संदर्भ सहायिका (डॉ. ए.पी.जे. अब्दुल कलाम)",
        "total_pages": 22,
        "total_chunks": 78,
        "embedding_model": "BAAI/bge-m3",
        "vector_store": "ChromaDB",
        "hybrid_search": "Dense (BGE-M3) + BM25Okapi + Reciprocal Rank Fusion",
    }
