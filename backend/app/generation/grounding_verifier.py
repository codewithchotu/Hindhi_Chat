"""
grounding_verifier.py — Automated verification of factual grounding and citation attribution.

Checks:
  1. Refusal Detection: Checks if response represents an explicit refusal due to lack of evidence.
  2. Invented Citation IDs: Verifies that any chunk IDs cited in the answer (e.g. [doc_p8_c3])
     exist in the actual retrieved context chunks (zero invented citations).
  3. Chunk-Specific Attribution: Verifies that when a specific chunk is cited in a sentence,
     the factual tokens (years, acronyms, key terms) in that sentence are actually present in
     that cited chunk (preventing correct IDs from being attached to unsupported claims).
  4. Context-Wide Factual Overlap: Confirms critical named entities, years, and figures mentioned
     in the answer are present in the retrieved passages.
"""
import re
from typing import List, Dict, Any, Set
from pydantic import BaseModel, Field

from app.generation.prompts import FALLBACK_HINDI, FALLBACK_ENGLISH


class GroundingReport(BaseModel):
    """Result of source-grounding check."""
    is_grounded: bool = Field(description="Whether answer passes all grounding checks")
    is_refusal: bool = Field(description="Whether answer is an explicit refusal due to lack of evidence")
    cited_chunk_ids: List[str] = Field(default_factory=list, description="Chunk IDs mentioned in answer text")
    valid_citations: List[str] = Field(default_factory=list, description="Cited IDs that exist in retrieved context")
    invented_citations: List[str] = Field(default_factory=list, description="Cited IDs that do NOT exist in context")
    citation_mismatches: List[str] = Field(default_factory=list, description="Citations attached to claims not supported by the cited chunk")
    unsupported_facts: List[str] = Field(default_factory=list, description="Entities/years found in answer but missing from all retrieved context")
    grounding_score: float = Field(default=1.0, description="Algorithmic verification score in [0.0, 1.0]")
    notes: str = Field(default="", description="Summary of verification findings")


class GroundingVerifier:
    """Verifies that generated text is strictly grounded in retrieved passages."""

    # Cross-lingual alias mapping for English acronyms vs Devanagari equivalents
    CROSS_LINGUAL_ALIASES = {
        "MIT": ["एमआईटी", "मद्रास इंस्टीट्यूट"],
        "DRDO": ["डीआरडीओ", "रक्षा अनुसंधान एवं विकास संगठन", "रक्षा अनुसंधान"],
        "DAE": ["डीएई", "परमाणु ऊर्जा विभाग"],
        "IIM": ["आईआईएम", "भारतीय प्रबंधन संस्थान"],
        "ISRO": ["इसरो", "भारतीय अंतरिक्ष अनुसंधान संगठन"],
        "SLV": ["एसएलवी"],
        "SLV-III": ["एसएलवी-III", "एसएलवी-3", "एसएलवी"],
        "IGMDP": ["आईजीएमडीपी", "एकीकृत निर्देशित मिसाइल विकास कार्यक्रम", "निर्देशित मिसाइल"],
        "HAL": ["एचएएल", "हिंदुस्तान एयरोनॉटिक्स"],
        "INCOSPAR": ["इंकोस्पार", "इनकोस्पार", "भारतीय राष्ट्रीय अंतरिक्ष"],
    }

    REFUSAL_TRIGGERS = [
        "उत्तर उपलब्ध नहीं",
        "दस्तावेज़ में नहीं",
        "जानकारी उपलब्ध नहीं",
        "पर्याप्त जानकारी नहीं",
        "not contain sufficient information",
        "could not be found",
        "not mentioned in the provided",
        "no information",
        FALLBACK_HINDI.lower(),
        FALLBACK_ENGLISH.lower()
    ]

    @classmethod
    def is_refusal_response(cls, text: str) -> bool:
        """Check if response is an explicit fallback/refusal."""
        lower = text.lower()
        return any(trigger in lower for trigger in cls.REFUSAL_TRIGGERS)

    @staticmethod
    def extract_chunk_tags(text: str) -> List[str]:
        """Extract chunk citation tags like [doc_p16_c1] or [doc_fake_chunk] from text."""
        return re.findall(r'\[(doc_[a-zA-Z0-9_]+)\]', text)

    @staticmethod
    def extract_key_tokens(text: str) -> Set[str]:
        """
        Extract years (e.g. 1980, 1997, 2015) and uppercase acronyms (e.g. SLV-III, MIT, DRDO)
        for factual overlap checking.
        """
        years = set(re.findall(r'\b(19\d{2}|20\d{2})\b', text))
        acronyms = set(re.findall(r'\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b', text))
        return years.union(acronyms)

    @classmethod
    def token_present_in_text(cls, token: str, text: str, token_set: Set[str]) -> bool:
        """Check if token is present in token_set or its cross-lingual alias appears in text."""
        if token in token_set:
            return True
        aliases = cls.CROSS_LINGUAL_ALIASES.get(token, [])
        return any(alias in text for alias in aliases)

    @classmethod
    def split_into_sentences(cls, text: str) -> List[str]:
        """Split text into sentences using Hindi and English delimiters."""
        raw_sentences = re.split(r'[।॥\n\.\?!]+', text)
        return [s.strip() for s in raw_sentences if s.strip()]

    @classmethod
    def verify(
        cls,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> GroundingReport:
        """
        Perform automated grounding check between generated answer and retrieved context.
        """
        if not answer or not answer.strip():
            return GroundingReport(
                is_grounded=False,
                is_refusal=False,
                grounding_score=0.0,
                notes="Empty answer received."
            )

        # 1. Check refusal
        if cls.is_refusal_response(answer):
            return GroundingReport(
                is_grounded=True,
                is_refusal=True,
                grounding_score=1.0,
                notes="Response correctly identifies insufficient evidence in context."
            )

        # Map available chunks by ID and collect all text
        chunk_map = {
            c.get("chunk_id") or c.get("metadata", {}).get("chunk_id"): c.get("text", "")
            for c in retrieved_chunks
        }
        chunk_map = {k: v for k, v in chunk_map.items() if k is not None}
        available_cids = set(chunk_map.keys())

        # 2. Check cited chunk IDs
        cited_cids = cls.extract_chunk_tags(answer)
        valid_citations = [cid for cid in cited_cids if cid in available_cids]
        invented_citations = [cid for cid in cited_cids if cid not in available_cids]

        # 3. Check context-wide factual presence
        full_context_text = " ".join(c.get("text", "") for c in retrieved_chunks)
        answer_key_tokens = cls.extract_key_tokens(answer)
        context_key_tokens = cls.extract_key_tokens(full_context_text)

        unsupported_facts = [
            token for token in sorted(answer_key_tokens)
            if not cls.token_present_in_text(token, full_context_text, context_key_tokens)
        ]

        # 4. Check chunk-specific claim attribution (sentence-level)
        #    If a sentence cites [doc_pX_cY], its specific key facts must be in that chunk.
        citation_mismatches = []
        sentences = cls.split_into_sentences(answer)
        for sent in sentences:
            sent_citations = cls.extract_chunk_tags(sent)
            sent_tokens = cls.extract_key_tokens(sent)
            for cid in sent_citations:
                if cid in chunk_map:
                    chunk_text = chunk_map[cid]
                    chunk_tokens = cls.extract_key_tokens(chunk_text)
                    missing_tokens = [
                        tok for tok in sent_tokens
                        if not cls.token_present_in_text(tok, chunk_text, chunk_tokens)
                    ]
                    if missing_tokens:
                        citation_mismatches.append(
                            f"Chunk '{cid}' does not contain tokens {missing_tokens} from sentence: \"{sent[:80]}...\""
                        )

        # Scoring & notes
        score = 1.0
        notes = []

        if invented_citations:
            score -= 0.4
            notes.append(f"Invented citation IDs detected: {invented_citations}")

        if unsupported_facts:
            score -= 0.3
            notes.append(f"Factual tokens absent from all context: {unsupported_facts}")

        if citation_mismatches:
            score -= 0.3
            notes.append(f"Citation attribution mismatches: {len(citation_mismatches)} instances")

        score = max(0.0, min(1.0, score))
        is_grounded = (
            len(invented_citations) == 0 and
            len(unsupported_facts) == 0 and
            len(citation_mismatches) == 0
        )

        if is_grounded:
            notes.append("All cited chunks and key factual claims verified against retrieved context.")

        return GroundingReport(
            is_grounded=is_grounded,
            is_refusal=False,
            cited_chunk_ids=cited_cids,
            valid_citations=valid_citations,
            invented_citations=invented_citations,
            citation_mismatches=citation_mismatches,
            unsupported_facts=unsupported_facts,
            grounding_score=score,
            notes="; ".join(notes)
        )
