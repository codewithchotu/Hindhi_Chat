"""app.generation package — Grounded LLM generation with Gemini."""
from app.generation.prompts import (
    detect_query_language,
    build_generation_prompt,
    FALLBACK_HINDI,
    FALLBACK_ENGLISH
)
from app.generation.grounding_verifier import GroundingVerifier, GroundingReport
from app.generation.answer_generator import (
    GenerationResult,
    AnswerGenerator,
    get_answer_generator
)

__all__ = [
    "detect_query_language",
    "build_generation_prompt",
    "FALLBACK_HINDI",
    "FALLBACK_ENGLISH",
    "GroundingVerifier",
    "GroundingReport",
    "GenerationResult",
    "AnswerGenerator",
    "get_answer_generator",
]
