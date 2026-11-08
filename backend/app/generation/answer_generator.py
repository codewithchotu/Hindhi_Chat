"""
answer_generator.py — Grounded answer generation using Google Gemini and dynamic citations.

Features:
  1. Integrates with the modern, maintained google-genai SDK (v3.0.0+), with fallback to google-generativeai.
  2. Strict Grounding: Synthesizes answers solely from retrieved vector store context.
  3. Safe Fallback Guard: Failed grounding checks (invented citations, unsupported facts,
     or citation mismatches) are NEVER presented as verified answers.
  4. Cross-Lingual: Automatically responds in Hindi for Hindi queries, English for English queries.
  5. Verifiable Citations: Attaches dynamic citations with Page, Section, Chunk ID, and Score.
  6. Refusal Fallback: Explicitly states when context is insufficient to answer the question.
  7. Testability: Supports mock model client injection for testing without network or quota usage.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Try importing modern google-genai SDK first, then legacy google-generativeai
try:
    from google import genai
    from google.genai import types as genai_types
    _HAS_GENAI_SDK = True
except ImportError:
    genai = None
    genai_types = None
    _HAS_GENAI_SDK = False

try:
    import google.generativeai as legacy_genai
    _HAS_LEGACY_SDK = True
except ImportError:
    legacy_genai = None
    _HAS_LEGACY_SDK = False

from app.config import get_settings
from app.citations.citation_builder import Citation, CitationBuilder
from app.generation.prompts import (
    detect_query_language,
    build_generation_prompt,
    FALLBACK_HINDI,
    FALLBACK_ENGLISH
)
from app.generation.grounding_verifier import GroundingVerifier, GroundingReport
from app.retrieval.retriever import UnifiedRetriever, get_unified_retriever

logger = logging.getLogger(__name__)


# Safe unverified response warnings (when grounding check fails)
UNVERIFIED_WARNING_HINDI = (
    "⚠️ सत्यापन चेतावनी: उत्पन्न उत्तर स्रोत संदर्भ से पूर्णतः सत्यापित नहीं हो सका "
    "(असमर्थित तथ्य, अमान्य खंड पहचान, या संदर्भ बेमेल पाया गया)। "
    "विश्वसनीयता नीति के तहत इसे सत्यापित उत्तर के रूप में प्रस्तुत नहीं किया जा सकता। "
    "कृपया नीचे दिए गए मूल दस्तावेज़ उद्धरणों का सीधा अवलोकन करें।"
)

UNVERIFIED_WARNING_ENGLISH = (
    "⚠️ Verification Warning: The generated response failed source verification checks "
    "(detected unsupported claims, invalid citation IDs, or citation mismatches). "
    "Under the strict grounding policy, unverified responses cannot be presented as factual answers. "
    "Please inspect the verified document citations below directly."
)


class GenerationResult(BaseModel):
    """Complete output of the grounded RAG generation pipeline."""
    model_config = {"protected_namespaces": ()}

    query: str
    answer: str
    language: str
    model_name: str
    citations: List[Citation] = Field(default_factory=list)
    citations_markdown: str = ""
    full_response: str = ""
    is_grounded: bool = True
    is_refusal: bool = False
    unverified_draft: Optional[str] = None
    grounding_report: Optional[GroundingReport] = None
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0


class AnswerGenerator:
    """Orchestrates RAG generation with Gemini and dynamic citation assembly."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        retriever: Optional[UnifiedRetriever] = None,
        mock_client: Optional[Any] = None
    ):
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self.retriever = retriever
        self.mock_client = mock_client
        self.client = None
        self.legacy_model = None

        if self.mock_client:
            logger.info("AnswerGenerator initialized with mock client for deterministic testing.")
        elif self.api_key and not self.api_key.startswith("your_gemini") and len(self.api_key) > 15:
            self._init_live_client()
        else:
            logger.info("AnswerGenerator in standby mode (no valid live GEMINI_API_KEY provided).")

    def _init_live_client(self):
        """Initialize the modern or legacy Google Gemini client."""
        if _HAS_GENAI_SDK and genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key, http_options=genai_types.HttpOptions(api_version="v1beta"))
                logger.info(f"AnswerGenerator configured with modern google-genai SDK (model: {self.model_name}).")
                return
            except Exception as e:
                logger.warning(f"Could not initialize modern google-genai client ({e}). Trying legacy SDK...")

        if _HAS_LEGACY_SDK and legacy_genai is not None:
            try:
                legacy_genai.configure(api_key=self.api_key)
                self.legacy_model = legacy_genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={"temperature": 0.2, "top_p": 0.95, "max_output_tokens": 1024}
                )
                logger.info(f"AnswerGenerator configured with legacy google-generativeai SDK (model: {self.model_name}).")
            except Exception as e:
                logger.warning(f"Could not initialize legacy Gemini model: {e}")

    def _get_retriever(self) -> UnifiedRetriever:
        """Lazy load singleton retriever if not supplied."""
        if self.retriever is None:
            self.retriever = get_unified_retriever()
        return self.retriever

    def _call_gemini_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call Gemini API using modern SDK, fallback legacy SDK, or raise exception."""
        if self.client and genai_types:
            config = genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=1024,
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config
            )
            if response and response.text:
                return response.text.strip()
            raise ValueError("Empty response received from Gemini API.")

        if self.legacy_model:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.legacy_model.generate_content(full_prompt)
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
            raise ValueError("Empty response received from legacy Gemini API.")

        raise RuntimeError("No active Gemini client available.")

    def generate(
        self,
        query: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> GenerationResult:
        """
        End-to-end grounded RAG generation for a user query.

        Args:
            query: User question in English or Hindi.
            retrieved_chunks: Pre-retrieved passages. If None, retrieves via Hybrid RRF.
            top_k: Number of passages to consider.
            dense_weight: Weight for dense vector search in RRF.
            sparse_weight: Weight for BM25 search in RRF.

        Returns:
            GenerationResult containing verified answer, citations, and grounding report.
        """
        t0 = time.time()
        query = query.strip()
        lang = detect_query_language(query)

        # 1. Retrieve context chunks if not provided
        if retrieved_chunks is None:
            retriever = self._get_retriever()
            retrieved_chunks = retriever.retrieve_hybrid(
                query=query,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )

        # Check if retrieval yielded nothing
        if not retrieved_chunks:
            refusal_text = FALLBACK_HINDI if lang == "hi" else FALLBACK_ENGLISH
            dt_ms = (time.time() - t0) * 1000
            return GenerationResult(
                query=query,
                answer=refusal_text,
                language=lang,
                model_name=self.model_name,
                citations=[],
                citations_markdown="",
                full_response=refusal_text,
                is_grounded=True,
                is_refusal=True,
                unverified_draft=None,
                grounding_report=GroundingReport(
                    is_grounded=True,
                    is_refusal=True,
                    notes="No context chunks retrieved."
                ),
                retrieved_chunks=[],
                latency_ms=round(dt_ms, 2)
            )

        # 2. Build prompts
        prompts = build_generation_prompt(query, retrieved_chunks)
        system_prompt = prompts["system_prompt"]
        user_prompt = prompts["user_prompt"]

        # 3. Generate answer via Gemini (or mock client)
        raw_answer = ""
        api_error_occurred = False
        api_error_msg = ""

        if self.mock_client:
            try:
                raw_answer = self.mock_client.generate(query=query, context=retrieved_chunks, lang=lang)
            except Exception as e:
                api_error_occurred = True
                api_error_msg = str(e)
        elif self.client or self.legacy_model:
            try:
                raw_answer = self._call_gemini_api(system_prompt, user_prompt)
            except Exception as e:
                logger.error(f"Gemini API error during generation: {e}")
                api_error_occurred = True
                api_error_msg = str(e)
        else:
            # Standby mode when API key is unconfigured
            refusal = FALLBACK_HINDI if lang == "hi" else FALLBACK_ENGLISH
            raw_answer = (
                f"{refusal}\n\n"
                f"[नोट: सक्रिय GEMINI_API_KEY कॉन्फ़िगर नहीं है। संदर्भ सफलतापूर्वक पुनः प्राप्त हुआ।]"
                if lang == "hi" else
                f"{refusal}\n\n"
                f"[Note: Active GEMINI_API_KEY is not configured. Context was retrieved successfully.]"
            )

        # Handle API execution error
        if api_error_occurred:
            err_answer = (
                f"त्रुटि: उत्तर उत्पन्न करते समय API से संपर्क विफल रहा ({api_error_msg})।"
                if lang == "hi" else
                f"Error: API communication failed during answer generation ({api_error_msg})."
            )
            citations = CitationBuilder.build_citations(retrieved_chunks, max_citations=top_k)
            citations_md = CitationBuilder.format_citations_block(citations, lang=lang)
            dt_ms = (time.time() - t0) * 1000
            return GenerationResult(
                query=query,
                answer=err_answer,
                language=lang,
                model_name=self.model_name,
                citations=citations,
                citations_markdown=citations_md,
                full_response=f"{err_answer}{citations_md}",
                is_grounded=False,
                is_refusal=False,
                unverified_draft=None,
                grounding_report=GroundingReport(
                    is_grounded=False,
                    is_refusal=False,
                    grounding_score=0.0,
                    notes=f"API call failed: {api_error_msg}"
                ),
                retrieved_chunks=retrieved_chunks,
                latency_ms=round(dt_ms, 2)
            )

        # 4. Build dynamic citations from retrieved chunks
        citations = CitationBuilder.build_citations(retrieved_chunks, max_citations=top_k)
        citations_md = CitationBuilder.format_citations_block(citations, lang=lang)

        # 5. Verify source grounding
        grounding_report = GroundingVerifier.verify(raw_answer, retrieved_chunks)

        # 6. Apply Safe Fallback Guard: If grounding fails, do NOT present raw answer as verified!
        if not grounding_report.is_grounded and not grounding_report.is_refusal:
            unverified_draft = raw_answer
            safe_warning = UNVERIFIED_WARNING_HINDI if lang == "hi" else UNVERIFIED_WARNING_ENGLISH
            verified_answer = safe_warning
            full_response = f"{verified_answer}{citations_md}"
        else:
            unverified_draft = None
            verified_answer = raw_answer
            full_response = f"{raw_answer}{citations_md}"

        dt_ms = (time.time() - t0) * 1000

        return GenerationResult(
            query=query,
            answer=verified_answer,
            language=lang,
            model_name=self.model_name,
            citations=citations,
            citations_markdown=citations_md,
            full_response=full_response,
            is_grounded=grounding_report.is_grounded,
            is_refusal=grounding_report.is_refusal,
            unverified_draft=unverified_draft,
            grounding_report=grounding_report,
            retrieved_chunks=retrieved_chunks,
            latency_ms=round(dt_ms, 2)
        )

    def generate_answer(
        self,
        query: str,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer with dynamic retrieval using specified mode ('dense', 'bm25', 'hybrid').
        """
        retriever = self._get_retriever()
        chunks = retriever.retrieve(query, mode=retrieval_mode, top_k=top_k, **kwargs)
        return self.generate(query=query, retrieved_chunks=chunks, top_k=top_k)


_generator_instance: Optional[AnswerGenerator] = None


def get_answer_generator() -> AnswerGenerator:
    """Singleton getter for AnswerGenerator."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = AnswerGenerator()
    return _generator_instance
