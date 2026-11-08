"""
test_hybrid_retrieval.py — Tests for Hybrid RRF, BM25, and Cross-Lingual ranking.

Validates:
  1. BM25 tokenizer and lexical matching
  2. BM25 retrieval against the actual 78-chunk corpus (no embedder needed)
  3. Reciprocal Rank Fusion (RRF) logic, score normalization
  4. Content-based interrogative dampening (not page-number-based)
  5. Metadata preservation through the BM25 and RRF pipeline

These tests are designed to run WITHOUT loading the bge-m3 embedding model,
avoiding MemoryError on memory-constrained systems.  Dense retrieval and full
end-to-end hybrid integration are validated in:
  - tests/test_retrieval.py  (loads the model; run separately)
  - backend/run_phase4_benchmark.py  (full 12-query benchmark)
"""
import json
import pytest
from pathlib import Path

from app.retrieval.hybrid_search import (
    tokenize_hindi_english,
    BM25Retriever,
    reciprocal_rank_fusion,
    _interrogative_ratio,
    _INTERROGATIVE_RATIO_THRESHOLD,
)
from app.config import get_settings


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def corpus_chunks():
    """Load the actual 78 chunks from disk.  No embedder is needed."""
    settings = get_settings()
    chunks_path = Path(settings.DATA_PROCESSED_DIR) / "chunks.json"
    assert chunks_path.exists(), f"chunks.json not found at {chunks_path}"
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def bm25(corpus_chunks):
    """BM25Retriever built from the actual corpus."""
    return BM25Retriever(corpus_chunks)


# ── Helpers ───────────────────────────────────────────────────────────────

def _hit(chunk_id, page, section, text="Sample factual text.", score=0.8,
         content_type="narrative"):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "metadata": {
            "page": page,
            "section": section,
            "source_doc": "test.pdf",
            "content_type": content_type,
            "char_count": len(text),
        },
    }


# ── 1. Tokenizer ──────────────────────────────────────────────────────────

def test_hindi_english_tokenizer():
    """Correctly segments Devanagari and Latin tokens."""
    tokens = tokenize_hindi_english("SLV-III ने 1980 में Rohini उपग्रह लॉन्च किया।")
    assert "slv-iii" in tokens
    assert "1980" in tokens
    assert "rohini" in tokens
    assert "उपग्रह" in tokens


def test_tokenizer_empty_input():
    assert tokenize_hindi_english("") == []
    assert tokenize_hindi_english("   ") == []


def test_tokenizer_pure_hindi():
    tokens = tokenize_hindi_english("कलाम को भारत रत्न प्राप्त हुआ")
    assert "कलाम" in tokens
    assert "भारत" in tokens
    assert "रत्न" in tokens


# ── 2. Interrogative Ratio ────────────────────────────────────────────────

def test_interrogative_ratio_factual_text():
    """Factual passages have a low interrogative ratio (below threshold)."""
    text = "कलाम को 1997 में भारत रत्न से सम्मानित किया गया।"
    assert _interrogative_ratio(text) < _INTERROGATIVE_RATIO_THRESHOLD


def test_interrogative_ratio_question_chunk():
    """A comprehension-question chunk has a high interrogative ratio."""
    # Simulates a chunk from a 'विचार एवं बोध-प्रश्न' section
    text = (
        "1. कलाम को भारत रत्न किस वर्ष प्राप्त हुआ? "
        "2. SLV-III ने किस उपग्रह को कक्षा में स्थापित किया? "
        "3. पोखरण-II में कलाम की क्या भूमिका थी? "
        "4. कलाम के माता-पिता कौन थे? "
        "5. किस संस्थान से उन्होंने वैमानिकी इंजीनियरिंग की?"
    )
    assert _interrogative_ratio(text) >= _INTERROGATIVE_RATIO_THRESHOLD


def test_interrogative_ratio_english_questions():
    """English comprehension questions are also detected."""
    text = "Who was Kalam? What did he study? Where did he work? When did he die?"
    assert _interrogative_ratio(text) >= _INTERROGATIVE_RATIO_THRESHOLD


# ── 3. BM25 Retrieval ─────────────────────────────────────────────────────

def test_bm25_retrieval_pokhran(bm25):
    """BM25 finds Pokhran-II content via Hindi keyword matching."""
    results = bm25.search("पोखरण-II परमाणु परीक्षण", top_k=5)
    assert len(results) > 0
    pages = [r["metadata"]["page"] for r in results]
    assert 10 in pages


def test_bm25_retrieval_awards(bm25):
    """BM25 finds awards table content."""
    results = bm25.search("भारत रत्न पुरस्कार 1997", top_k=5)
    assert len(results) > 0
    pages = [r["metadata"]["page"] for r in results]
    assert 16 in pages


def test_bm25_q2_boosts_awards_page(bm25):
    """
    For Q2 (भारत रत्न), BM25 boosts the awards page (16) which contains
    keyword-rich factual content.  This is the BM25 contribution that lets
    Hybrid RRF promote Page 16 above the comprehension-question chunk.
    """
    results = bm25.search("कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", top_k=5)
    pages = [r["metadata"]["page"] for r in results]
    assert 16 in pages, f"Expected Page 16 in BM25 results, got: {pages}"


def test_bm25_metadata_completeness(bm25):
    """BM25 results carry all required citation fields."""
    results = bm25.search("SLV-III रोहिणी उपग्रह", top_k=3)
    assert len(results) > 0
    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert 0.0 <= r["score"] <= 1.0
        for key in ("page", "section", "source_doc", "content_type", "char_count"):
            assert key in r["metadata"], f"Missing metadata key: {key}"


def test_bm25_score_normalization(bm25):
    """Top BM25 result is normalised to 1.0; all others in [0, 1]."""
    results = bm25.search("कलाम रामेश्वरम", top_k=5)
    if results:
        assert results[0]["score"] == 1.0
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


def test_bm25_cross_lingual_english_does_not_crash(bm25):
    """
    English queries against a Hindi BM25 corpus should not raise exceptions.
    Recall is expected to be low (documented limitation of pure lexical search).
    """
    results = bm25.search(
        "Which institution did Kalam attend for aeronautical engineering?",
        top_k=5
    )
    assert isinstance(results, list)


# ── 4. Reciprocal Rank Fusion ─────────────────────────────────────────────

def test_rrf_basic_fusion():
    """RRF merges dense and sparse rankings; chunks in both lists score higher."""
    dense = [
        _hit("c1", 8, "SLV-III", score=0.9),
        _hit("c2", 17, "Timeline", score=0.7),
        _hit("c3", 10, "Pokhran", score=0.6),
    ]
    sparse = [
        _hit("c3", 10, "Pokhran", score=1.0),
        _hit("c1", 8, "SLV-III", score=0.8),
        _hit("c4", 16, "Awards", score=0.5),
    ]
    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=5)
    ids = [h["chunk_id"] for h in fused]
    # c1 and c3 appear in both lists → should be in results
    assert "c1" in ids
    assert "c3" in ids


def test_rrf_top_score_is_one():
    """RRF normalises the top result's score to 1.0."""
    dense = [_hit("c1", 8, "SLV-III")]
    sparse = [_hit("c1", 8, "SLV-III")]
    fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
    assert fused[0]["score"] == 1.0


def test_rrf_interrogative_dampening_is_content_based():
    """
    The interrogative dampener fires based on chunk TEXT, not page number.
    A synthetic factual chunk on page 99 is NOT dampened.
    A synthetic question chunk on page 1 IS dampened.
    """
    # Factual chunk on a deliberately unusual page
    factual = _hit(
        "factual_p99", page=99, section="Factual",
        text="कलाम को 1997 में भारत रत्न से सम्मानित किया गया।"
    )
    # Question chunk on page 1 (normally a factual page)
    question_text = (
        "1. कलाम को भारत रत्न किस वर्ष मिला? "
        "2. SLV-III ने किस उपग्रह को कक्षा में स्थापित किया? "
        "3. पोखरण-II में उनकी क्या भूमिका थी? "
        "4. कौन-से पुरस्कार मिले? "
        "5. किस संस्थान से पढ़े?"
    )
    question = _hit("q_p1", page=1, section="Questions", text=question_text)

    dense = [question, factual]  # question-chunk ranks first in dense
    sparse = [question, factual]

    fused = reciprocal_rank_fusion(dense, sparse, top_k=5)

    rank_factual = next(i for i, h in enumerate(fused) if h["chunk_id"] == "factual_p99")
    rank_question = next(i for i, h in enumerate(fused) if h["chunk_id"] == "q_p1")
    assert rank_factual < rank_question, (
        f"Factual chunk (rank {rank_factual}) should beat question chunk "
        f"(rank {rank_question}) after content-based dampening"
    )


def test_rrf_preserves_all_metadata():
    """Every fused result carries chunk_id, score, rrf_score and all metadata."""
    dense = [_hit("c1", 8, "SLV-III")]
    sparse = [_hit("c2", 16, "Awards")]
    for hit in reciprocal_rank_fusion(dense, sparse, top_k=5):
        assert "chunk_id" in hit
        assert "score" in hit
        assert "rrf_score" in hit
        for key in ("page", "section", "source_doc", "content_type", "char_count"):
            assert key in hit["metadata"]


def test_rrf_empty_inputs():
    """RRF is robust to empty input from either retriever."""
    assert reciprocal_rank_fusion([], [], top_k=5) == []
    dense = [_hit("c1", 8, "SLV-III")]
    fused = reciprocal_rank_fusion(dense, [], top_k=5)
    assert len(fused) == 1
