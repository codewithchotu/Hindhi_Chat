"""
citation_builder.py — Dynamic citation building and verification for retrieved passages.

Ensures that:
  1. Every citation maps strictly to an actual retrieved chunk from ChromaDB / BM25.
  2. PDF page number, section name, chunk ID, and score are always included.
  3. Retrieval score is clearly identified by type:
       - 'dense_cosine' (Cosine Similarity in [0, 1])
       - 'hybrid_rrf'   (Reciprocal Rank Fusion normalized score in [0, 1] + raw RRF score)
       - 'sparse_bm25'  (Normalized BM25 score in [0, 1])
  4. Explicit score semantics: Normalized RRF / similarity scores are clearly labeled as
     relative retrieval ranking metrics, NOT as probabilities or factual confidence levels.
  5. Provides formatting helpers for text/markdown citations and structured API responses.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Structured verifiable citation from vector database metadata."""
    chunk_id: str = Field(description="Unique stable identifier of the chunk, e.g. doc_p16_c1")
    page: int = Field(description="PDF page number (1-indexed, 1-22)")
    section: str = Field(description="Document section or table heading")
    source_doc: str = Field(default="Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf", description="Source document file name")
    content_type: str = Field(default="narrative", description="'narrative' or 'table'")
    score: float = Field(description="Relative retrieval ranking score in [0.0, 1.0]. Note: Algorithmic ranking score, not a statistical probability.")
    score_type: str = Field(description="Type of score: 'dense_cosine', 'hybrid_rrf', or 'sparse_bm25'")
    raw_rrf_score: Optional[float] = Field(default=None, description="Raw Reciprocal Rank Fusion score if hybrid")
    snippet: str = Field(default="", description="Relevant excerpt from the chunk text")

    def to_citation_tag(self, lang: str = "hi") -> str:
        """Format as a compact inline citation tag."""
        if lang == "en":
            return f"[Page {self.page} | {self.section} | {self.chunk_id}]"
        return f"[पृष्ठ {self.page} | {self.section} | {self.chunk_id}]"

    def to_markdown(self, lang: str = "hi") -> str:
        """
        Format as a detailed markdown citation item.
        Accurately labels scores as algorithmic retrieval rank scores,
        never as statistical probabilities or factual confidence values.
        """
        if self.score_type == "hybrid_rrf":
            type_label = "सापेक्ष RRF रैंक स्कोर (Relative RRF Rank Score)" if lang == "hi" else "Relative RRF Rank Score"
            score_desc = f"{self.score:.4f} ({type_label}"
            if self.raw_rrf_score is not None:
                score_desc += f", कच्चा RRF: {self.raw_rrf_score:.6f}" if lang == "hi" else f", Raw RRF: {self.raw_rrf_score:.6f}"
            score_desc += ")"
        elif self.score_type == "dense_cosine":
            type_label = "कोसाइन समानता स्कोर (Cosine Similarity)" if lang == "hi" else "Dense Cosine Similarity"
            score_desc = f"{self.score:.4f} ({type_label})"
        else:
            type_label = "BM25 शाब्दिक स्कोर (Lexical Score)" if lang == "hi" else "BM25 Lexical Score"
            score_desc = f"{self.score:.4f} ({type_label})"

        if lang == "en":
            return (
                f"- **Source:** Page {self.page} ({self.section})\n"
                f"  - **Chunk ID:** `{self.chunk_id}`\n"
                f"  - **Retrieval Relevance:** {score_desc}\n"
                f"  - **Excerpt:** *\"{self.snippet}\"*"
            )
        return (
            f"- **स्रोत:** पृष्ठ {self.page} ({self.section})\n"
            f"  - **खंड पहचान (Chunk ID):** `{self.chunk_id}`\n"
            f"  - **पुनर्प्राप्ति प्रासंगिकता:** {score_desc}\n"
            f"  - **अंश:** *\"{self.snippet}\"*"
        )


class CitationBuilder:
    """Builds and validates citations from retrieval hit dictionaries."""

    @staticmethod
    def identify_score_type(hit: Dict[str, Any]) -> str:
        """Determine score type based on hit fields."""
        if "rrf_score" in hit and hit["rrf_score"] is not None:
            return "hybrid_rrf"
        if "raw_bm25_score" in hit:
            return "sparse_bm25"
        return "dense_cosine"

    @staticmethod
    def extract_snippet(text: str, max_chars: int = 160) -> str:
        """Extract a clean, readable excerpt from chunk text."""
        if not text:
            return ""
        clean = " ".join(text.strip().split())
        if len(clean) <= max_chars:
            return clean
        cutoff = clean[:max_chars].rfind(" ")
        if cutoff > 40:
            return clean[:cutoff] + "..."
        return clean[:max_chars] + "..."

    @classmethod
    def from_hit(cls, hit: Dict[str, Any]) -> Citation:
        """Construct a Citation object from a single retriever hit dictionary."""
        metadata = hit.get("metadata", {})
        score_type = cls.identify_score_type(hit)
        text = hit.get("text", "")

        return Citation(
            chunk_id=hit.get("chunk_id") or metadata.get("chunk_id", "unknown"),
            page=int(metadata.get("page", 1)),
            section=str(metadata.get("section", "अज्ञात अनुभाग")),
            source_doc=str(metadata.get("source_doc", "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")),
            content_type=str(metadata.get("content_type", "narrative")),
            score=round(float(hit.get("score", 0.0)), 4),
            score_type=score_type,
            raw_rrf_score=hit.get("rrf_score"),
            snippet=cls.extract_snippet(text)
        )

    @classmethod
    def build_citations(cls, hits: List[Dict[str, Any]], max_citations: int = 5) -> List[Citation]:
        """
        Build verified citations from a list of retrieved chunks.
        Deduplicates by chunk_id while preserving rank order.
        """
        citations = []
        seen_ids = set()

        for hit in hits:
            cid = hit.get("chunk_id") or hit.get("metadata", {}).get("chunk_id")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            citations.append(cls.from_hit(hit))
            if len(citations) >= max_citations:
                break

        return citations

    @classmethod
    def format_citations_block(cls, citations: List[Citation], lang: str = "hi") -> str:
        """
        Format list of citations into a final markdown section for the user.
        Includes an explicit disclaimer that scores reflect algorithmic retrieval ranking.
        """
        if not citations:
            return ""

        if lang == "hi":
            header = "### 📚 संदर्भ एवं उद्धरण (Sources & Citations)"
            disclaimer = "> *[नोट: स्कोर खोज एल्गोरिदम की सापेक्ष प्रासंगिकता (रैंकिंग) दर्शाते हैं, सांख्यिकीय संभाव्यता या तथ्यात्मक सटीकता नहीं।]*\n"
        else:
            header = "### 📚 Sources & Citations"
            disclaimer = "> *[Note: Scores indicate relative algorithmic retrieval ranking, not a statistical probability or factual confidence level.]*\n"

        items = "\n".join(c.to_markdown(lang=lang) for c in citations)
        return f"\n\n{header}\n{disclaimer}\n{items}"
