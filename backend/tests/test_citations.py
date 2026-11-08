"""
test_citations.py — Comprehensive unit tests for dynamic citation builder.

Validates:
  1. Score type identification (dense_cosine vs hybrid_rrf vs sparse_bm25).
  2. Snippet extraction with proper word boundary truncation.
  3. Structured Citation field extraction from vector hit metadata.
  4. Deduplication of citation IDs while preserving rank order.
  5. Multi-language markdown and citation tag formatting (Hindi and English).
  6. Non-probability score semantics (scores labeled as relative algorithmic ranking, never probability).
"""
import pytest
from app.citations.citation_builder import Citation, CitationBuilder


@pytest.fixture
def sample_dense_hit():
    return {
        "chunk_id": "doc_p8_c3",
        "text": "18 जुलाई 1980 को SLV-III द्वारा रोहिणी उपग्रह को सफलतापूर्वक कक्षा में स्थापित किया गया।",
        "score": 0.6073,
        "metadata": {
            "chunk_id": "doc_p8_c3",
            "page": 8,
            "section": "6. इसरो के वर्ष और SLV-III",
            "source_doc": "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf",
            "content_type": "narrative",
            "char_count": 88
        }
    }


@pytest.fixture
def sample_hybrid_hit():
    return {
        "chunk_id": "doc_p16_c1",
        "text": "1997 में डॉ. कलाम को भारत का सर्वोच्च नागरिक सम्मान 'भारत रत्न' प्रदान किया गया।",
        "score": 1.0000,
        "rrf_score": 0.016314,
        "metadata": {
            "chunk_id": "doc_p16_c1",
            "page": 16,
            "section": "14. पुरस्कार एवं सम्मान सारणी",
            "source_doc": "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf",
            "content_type": "table",
            "char_count": 83
        }
    }


def test_identify_score_type(sample_dense_hit, sample_hybrid_hit):
    """Verify score type identification between dense and hybrid RRF."""
    assert CitationBuilder.identify_score_type(sample_dense_hit) == "dense_cosine"
    assert CitationBuilder.identify_score_type(sample_hybrid_hit) == "hybrid_rrf"
    bm25_hit = {"chunk_id": "doc_p1_c1", "score": 0.8, "raw_bm25_score": 4.5}
    assert CitationBuilder.identify_score_type(bm25_hit) == "sparse_bm25"


def test_extract_snippet():
    """Verify snippet extraction cleans whitespace and respects word boundaries."""
    long_text = "यह एक बहुत लंबा विवरण है जो डॉ. कलाम के प्रारंभिक जीवन के बारे में बताता है और उनके संघर्षों को रेखांकित करता है।"
    snippet = CitationBuilder.extract_snippet(long_text, max_chars=40)
    assert len(snippet) <= 45
    assert not snippet.endswith(" ")
    assert "..." in snippet or len(snippet) < 40


def test_citation_from_hit(sample_dense_hit, sample_hybrid_hit):
    """Verify complete metadata population in Citation model."""
    c_dense = CitationBuilder.from_hit(sample_dense_hit)
    assert c_dense.chunk_id == "doc_p8_c3"
    assert c_dense.page == 8
    assert c_dense.section == "6. इसरो के वर्ष और SLV-III"
    assert c_dense.score == 0.6073
    assert c_dense.score_type == "dense_cosine"
    assert c_dense.raw_rrf_score is None
    assert "SLV-III" in c_dense.snippet

    c_hybrid = CitationBuilder.from_hit(sample_hybrid_hit)
    assert c_hybrid.chunk_id == "doc_p16_c1"
    assert c_hybrid.page == 16
    assert c_hybrid.score == 1.0000
    assert c_hybrid.score_type == "hybrid_rrf"
    assert c_hybrid.raw_rrf_score == 0.016314
    assert c_hybrid.content_type == "table"


def test_build_citations_deduplication(sample_dense_hit):
    """Verify duplicate hits with identical chunk_id are filtered out."""
    hits = [sample_dense_hit, sample_dense_hit, sample_dense_hit]
    citations = CitationBuilder.build_citations(hits, max_citations=5)
    assert len(citations) == 1
    assert citations[0].chunk_id == "doc_p8_c3"


def test_citation_tags_formatting(sample_dense_hit):
    """Verify compact citation tag formatting in Hindi and English."""
    citation = CitationBuilder.from_hit(sample_dense_hit)
    tag_hi = citation.to_citation_tag(lang="hi")
    tag_en = citation.to_citation_tag(lang="en")

    assert tag_hi == "[पृष्ठ 8 | 6. इसरो के वर्ष और SLV-III | doc_p8_c3]"
    assert tag_en == "[Page 8 | 6. इसरो के वर्ष और SLV-III | doc_p8_c3]"


def test_citation_markdown_formatting(sample_hybrid_hit):
    """Verify markdown output contains all required fields and scores."""
    citation = CitationBuilder.from_hit(sample_hybrid_hit)
    md_hi = citation.to_markdown(lang="hi")
    md_en = citation.to_markdown(lang="en")

    assert "पृष्ठ 16" in md_hi
    assert "doc_p16_c1" in md_hi
    assert "Page 16" in md_en
    assert "doc_p16_c1" in md_en
    assert "0.016314" in md_en


def test_citation_score_semantics_not_labeled_as_probability(sample_hybrid_hit, sample_dense_hit):
    """
    Requirement 7: Confirm citation scores are accurately labeled and that a
    normalized RRF score is NOT described as a probability or factual confidence.
    """
    c_hybrid = CitationBuilder.from_hit(sample_hybrid_hit)
    md_en = c_hybrid.to_markdown(lang="en")
    md_hi = c_hybrid.to_markdown(lang="hi")

    # Score must be labeled as rank score, not probability or factual confidence
    assert "probability" not in md_en.lower()
    assert "confidence" not in md_en.lower()
    assert "संभाव्यता" not in md_hi
    assert "Relative RRF Rank Score" in md_en or "रैंक स्कोर" in md_hi

    # Also verify citations block includes disclaimer
    block_en = CitationBuilder.format_citations_block([c_hybrid], lang="en")
    assert "algorithmic retrieval ranking" in block_en
    assert "not a statistical probability" in block_en


def test_format_citations_block(sample_dense_hit, sample_hybrid_hit):
    """Verify aggregate citations block has proper section headers and disclaimers."""
    citations = [
        CitationBuilder.from_hit(sample_dense_hit),
        CitationBuilder.from_hit(sample_hybrid_hit)
    ]
    block_hi = CitationBuilder.format_citations_block(citations, lang="hi")
    block_en = CitationBuilder.format_citations_block(citations, lang="en")

    assert "संदर्भ एवं उद्धरण" in block_hi
    assert "doc_p8_c3" in block_hi
    assert "doc_p16_c1" in block_hi

    assert "Sources & Citations" in block_en
    assert "doc_p8_c3" in block_en
    assert "doc_p16_c1" in block_en
