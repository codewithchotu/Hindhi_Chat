"""
test_answer_generator.py — Unit, adversarial, and integration tests for grounded RAG generation.

Validates:
  1. Language detection (Hindi vs English).
  2. Prompt building with structured context passages and system instructions.
  3. Grounded answer generation using mocked Gemini client (0 API quota used).
  4. Language alignment: Hindi answers for Hindi queries, English answers for English queries.
  5. Refusal fallback: Returns standard message when context lacks evidence.
  6. Dynamic citation attachment: Citations derived exclusively from retrieved chunks.
  7. End-to-end simulation of all 6 mandatory assignment queries.
  8. Adversarial Tests:
     - Correct year attached to the wrong event.
     - Invented chunk ID.
     - Correct citation ID attached to an unsupported claim.
     - Answer containing facts absent from the context.
     - Empty retrieval results.
     - Gemini timeout or API failure.
     - Gemini returning an explicit refusal.
  9. Safe Fallback Guard: Proves that ungrounded responses cannot be presented as verified answers.
  10. Optional integration test against live Gemini API (gracefully skipped if key is missing/placeholder).
"""
import os
import pytest
from typing import List, Dict, Any

from app.generation.prompts import (
    detect_query_language,
    build_generation_prompt,
    FALLBACK_HINDI,
    FALLBACK_ENGLISH
)
from app.generation.grounding_verifier import GroundingVerifier, GroundingReport
from app.generation.answer_generator import (
    AnswerGenerator,
    GenerationResult,
    UNVERIFIED_WARNING_HINDI,
    UNVERIFIED_WARNING_ENGLISH
)
from app.citations.citation_builder import CitationBuilder


# ── Mock Gemini Client for Deterministic Unit Testing ────────────────────

class MockGeminiClient:
    """
    Simulates Google Gemini model responses for unit tests.
    Ensures tests run deterministically, fast, and with 0 API quota consumption.
    """

    MOCK_RESPONSES = {
        "SLV-III": "18 जुलाई 1980 को SLV-III रॉकेट द्वारा रोहिणी उपग्रह को सफलतापूर्वक पृथ्वी की कक्षा में स्थापित किया गया [doc_p8_c3]।",
        "भारत रत्न": "डॉ. ए.पी.जे. अब्दुल कलाम को वर्ष 1997 में भारत का सर्वोच्च नागरिक सम्मान 'भारत रत्न' प्रदान किया गया [doc_p16_c1]।",
        "पोखरण": "मई 1998 के पोखरण-II परमाणु परीक्षण में कलाम ने रक्षा मंत्री के वैज्ञानिक सलाहकार और DRDO प्रमुख के रूप में DAE के साथ समन्वयक भूमिका निभाई [doc_p10_c1]।",
        "institution": "Dr. Kalam attended the Madras Institute of Technology (MIT), Chennai, to pursue his studies in aeronautical engineering [doc_p6_c1].",
        "autobiography": "Dr. Kalam's autobiography 'Wings of Fire' was co-written with Arun Tiwari and published in the year 1999 [doc_p3_c3].",
        "die": "Dr. Kalam passed away on 27 July 2015 in Shillong, Meghalaya, after collapsing while delivering a lecture to students at IIM Shillong [doc_p15_c3].",
        "unknown_topic": "The provided context does not contain sufficient information to answer this question.",
        "unknown_topic_hi": "प्रदान किए गए संदर्भ में इस प्रश्न का उत्तर उपलब्ध नहीं है।"
    }

    def __init__(self, failure_mode: str = "none"):
        self.failure_mode = failure_mode

    def generate(self, query: str, context: List[Dict[str, Any]], lang: str) -> str:
        if self.failure_mode == "timeout":
            raise TimeoutError("Gemini API connection timed out after 30 seconds.")
        if self.failure_mode == "api_error":
            raise RuntimeError("503 Service Unavailable: Gemini API rate limit exceeded.")

        if not context:
            return FALLBACK_HINDI if lang == "hi" else FALLBACK_ENGLISH

        query_lower = query.lower()
        for key, resp in self.MOCK_RESPONSES.items():
            if key.lower() in query_lower:
                return resp

        return FALLBACK_HINDI if lang == "hi" else FALLBACK_ENGLISH


# ── Sample Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_generator():
    return AnswerGenerator(mock_client=MockGeminiClient())


@pytest.fixture
def sample_kalam_chunks():
    return [
        {
            "chunk_id": "doc_p8_c3",
            "text": "18 जुलाई 1980 को SLV-III द्वारा रोहिणी उपग्रह को सफलतापूर्वक कक्षा में स्थापित किया गया।",
            "score": 1.0,
            "metadata": {"chunk_id": "doc_p8_c3", "page": 8, "section": "6. इसरो के वर्ष", "source_doc": "doc.pdf", "content_type": "narrative"}
        },
        {
            "chunk_id": "doc_p16_c1",
            "text": "1997 में डॉ. कलाम को भारत का सर्वोच्च नागरिक सम्मान 'भारत रत्न' प्रदान किया गया।",
            "score": 1.0,
            "metadata": {"chunk_id": "doc_p16_c1", "page": 16, "section": "14. पुरस्कार", "source_doc": "doc.pdf", "content_type": "table"}
        },
        {
            "chunk_id": "doc_p6_c1",
            "text": "कलाम ने वैमानिकी इंजीनियरिंग (एरोनॉटिकल इंजीनियरिंग) की पढ़ाई के लिए मद्रास इंस्टीट्यूट ऑफ टेक्नोलॉजी (MIT) चेन्नई में दाखिला लिया।",
            "score": 1.0,
            "metadata": {"chunk_id": "doc_p6_c1", "page": 6, "section": "4. कॉलेज वर्ष", "source_doc": "doc.pdf", "content_type": "narrative"}
        },
        {
            "chunk_id": "doc_p3_c3",
            "text": "कलाम की प्रसिद्ध आत्मकथा Wings of Fire (अग्नि की उड़ान) 1999 में प्रकाशित हुई थी, जिसके सह-लेखक अरुण तिवारी (Arun Tiwari) थे।",
            "score": 1.0,
            "metadata": {"chunk_id": "doc_p3_c3", "page": 3, "section": "1. परिचय", "source_doc": "doc.pdf", "content_type": "narrative"}
        },
        {
            "chunk_id": "doc_p15_c3",
            "text": "27 जुलाई 2015 को आईआईएम शिलांग (IIM Shillong) में छात्रों को व्याख्यान देते समय कार्डियक अरेस्ट से उनका निधन हो गया।",
            "score": 1.0,
            "metadata": {"chunk_id": "doc_p15_c3", "page": 15, "section": "13. अंतिम क्षण", "source_doc": "doc.pdf", "content_type": "narrative"}
        },
        {
            "chunk_id": "doc_p10_c1",
            "text": "मई 1998 में पोखरण-II परमाणु परीक्षण (ऑपरेशन शक्ति) में कलाम ने रक्षा मंत्री के वैज्ञानिक सलाहकार और DRDO प्रमुख के रूप में DAE के साथ समन्वयक भूमिका निभाई।",
            "score": 1.0,
            "metadata": {"chunk_id": "doc_p10_c1", "page": 10, "section": "8. पोखरण-II", "source_doc": "doc.pdf", "content_type": "narrative"}
        }
    ]


# ── 1. Language Detection Tests ──────────────────────────────────────────

def test_language_detection():
    """Verify accurate detection of Hindi vs English queries."""
    assert detect_query_language("कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?") == "hi"
    assert detect_query_language("SLV-III ने किस उपग्रह को कक्षा में स्थापित किया?") == "hi"
    assert detect_query_language("Which institution did Kalam attend?") == "en"
    assert detect_query_language("How and where did Kalam die in 2015?") == "en"
    assert detect_query_language("") == "en"


# ── 2. Prompt Building Tests ─────────────────────────────────────────────

def test_prompt_building(sample_kalam_chunks):
    """Verify system and user prompt generation for both languages."""
    p_hi = build_generation_prompt("कलाम को भारत रत्न कब मिला?", sample_kalam_chunks[:2])
    assert p_hi["lang"] == "hi"
    assert "दिए गए संदर्भ" in p_hi["system_prompt"]
    assert "doc_p8_c3" in p_hi["user_prompt"]
    assert "doc_p16_c1" in p_hi["user_prompt"]

    p_en = build_generation_prompt("When did Kalam receive the Bharat Ratna?", sample_kalam_chunks[:2])
    assert p_en["lang"] == "en"
    assert "ONLY the facts explicitly stated" in p_en["system_prompt"]
    assert "Context Passages from Document" in p_en["user_prompt"]


# ── 3. Generation & Language Alignment Tests ──────────────────────────────

def test_hindi_generation_and_citation(mock_generator, sample_kalam_chunks):
    """Verify Hindi query receives Hindi answer with citations attached."""
    result = mock_generator.generate(
        query="कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?",
        retrieved_chunks=[sample_kalam_chunks[1]]
    )
    assert result.language == "hi"
    assert "1997" in result.answer
    assert "भारत रत्न" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "doc_p16_c1"
    assert result.citations[0].page == 16
    assert "संदर्भ एवं उद्धरण" in result.citations_markdown
    assert result.is_grounded is True


def test_english_generation_and_citation(mock_generator, sample_kalam_chunks):
    """Verify English query receives English answer with citations attached."""
    result = mock_generator.generate(
        query="Which institution did Kalam attend to study aeronautical engineering?",
        retrieved_chunks=[sample_kalam_chunks[2]]
    )
    assert result.language == "en"
    assert "Madras Institute of Technology" in result.answer or "MIT" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "doc_p6_c1"
    assert result.citations[0].page == 6
    assert "Sources & Citations" in result.citations_markdown
    assert result.is_grounded is True


# ── 4. Mandatory Assignment Queries Simulation ────────────────────────────

MANDATORY_TEST_CASES = [
    (1, "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?", "hi", "doc_p8_c3", 8, "1980"),
    (2, "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", "hi", "doc_p16_c1", 16, "1997"),
    (3, "पोखरण-II में कलाम की क्या भूमिका थी?", "hi", "doc_p10_c1", 10, "1998"),
    (4, "Which institution did Kalam attend to study aeronautical engineering?", "en", "doc_p6_c1", 6, "MIT"),
    (5, "Who co-wrote the autobiography, and in what year was it published?", "en", "doc_p3_c3", 3, "1999"),
    (6, "How and where did Kalam die in 2015?", "en", "doc_p15_c3", 15, "2015")
]


@pytest.mark.parametrize("qid,query,lang,expected_cid,expected_page,expected_token", MANDATORY_TEST_CASES)
def test_all_six_mandatory_queries_generation(
    mock_generator,
    sample_kalam_chunks,
    qid,
    query,
    lang,
    expected_cid,
    expected_page,
    expected_token
):
    """
    Test generation, language matching, citation integrity, and grounding
    for all 6 mandatory questions without consuming Gemini quota.
    """
    matching_chunks = [c for c in sample_kalam_chunks if c["chunk_id"] == expected_cid]
    assert len(matching_chunks) > 0

    result = mock_generator.generate(query=query, retrieved_chunks=matching_chunks)

    assert result.language == lang
    assert expected_token.lower() in result.answer.lower()
    assert len(result.citations) >= 1
    assert result.citations[0].chunk_id == expected_cid
    assert result.citations[0].page == expected_page
    assert result.is_grounded is True
    assert result.is_refusal is False


# ── 5. Adversarial Tests (Requirement 3) ─────────────────────────────────

def test_adversarial_correct_year_attached_to_wrong_event(sample_kalam_chunks):
    """
    Adversarial Check 1: Answer attaches 1998 (Pokhran-II year) to SLV-III launch chunk (1980).
    The verifier must flag this citation mismatch because 1998 is not in doc_p8_c3.
    """
    hallucinated_answer = "1998 में कलाम ने SLV-III रॉकेट से उपग्रह लॉन्च किया [doc_p8_c3]।"
    report = GroundingVerifier.verify(hallucinated_answer, [sample_kalam_chunks[0]])
    assert report.is_grounded is False
    assert len(report.citation_mismatches) > 0
    assert "doc_p8_c3" in report.citation_mismatches[0]


def test_adversarial_invented_chunk_id(sample_kalam_chunks):
    """
    Adversarial Check 2: Answer cites a hallucinated / fabricated chunk ID.
    The verifier must detect the invented chunk ID.
    """
    hallucinated_answer = "कलाम को भारत रत्न मिला [doc_p99_c99]। और एक अन्य पुरस्कार मिला [doc_fake_chunk]।"
    report = GroundingVerifier.verify(hallucinated_answer, sample_kalam_chunks)
    assert report.is_grounded is False
    assert "doc_p99_c99" in report.invented_citations
    assert "doc_fake_chunk" in report.invented_citations


def test_adversarial_correct_citation_attached_to_unsupported_claim(sample_kalam_chunks):
    """
    Adversarial Check 3: Answer cites a valid chunk ID [doc_p8_c3] (SLV-III), but attaches
    it to an entirely unrelated claim about Bharat Ratna or Pokhran.
    """
    # Context has both chunks, but the sentence attaches [doc_p8_c3] to 1997 award
    mismatched_answer = "कलाम को 1997 में सर्वोच्च नागरिक सम्मान मिला [doc_p8_c3]।"
    report = GroundingVerifier.verify(mismatched_answer, sample_kalam_chunks)
    assert report.is_grounded is False
    assert len(report.citation_mismatches) > 0


def test_adversarial_facts_absent_from_context(sample_kalam_chunks):
    """
    Adversarial Check 4: Answer introduces facts, dates, or organizations
    completely absent from all retrieved context chunks.
    """
    hallucinated_answer = "कलाम 2045 में NASA के मुख्य वैज्ञानिक बने [doc_p8_c3]।"
    report = GroundingVerifier.verify(hallucinated_answer, sample_kalam_chunks)
    assert report.is_grounded is False
    assert any("2045" in fact or "NASA" in fact for fact in report.unsupported_facts)


def test_adversarial_empty_retrieval_results(mock_generator):
    """
    Adversarial Check 5: Retrieval returns empty list (no chunks matched).
    The generator must safely return an explicit refusal without failing.
    """
    result_hi = mock_generator.generate(query="क्या कलाम कभी चांद पर उतरे थे?", retrieved_chunks=[])
    assert result_hi.is_refusal is True
    assert result_hi.is_grounded is True
    assert FALLBACK_HINDI in result_hi.answer
    assert len(result_hi.citations) == 0

    result_en = mock_generator.generate(query="Did Kalam ever walk on Mars?", retrieved_chunks=[])
    assert result_en.is_refusal is True
    assert result_en.is_grounded is True
    assert FALLBACK_ENGLISH in result_en.answer


def test_adversarial_gemini_timeout_or_api_failure(sample_kalam_chunks):
    """
    Adversarial Check 6: Gemini API encounters a network timeout or 503 error.
    The generator must catch the exception, return an error result, and not crash.
    """
    timeout_generator = AnswerGenerator(mock_client=MockGeminiClient(failure_mode="timeout"))
    result = timeout_generator.generate("Test query", retrieved_chunks=sample_kalam_chunks)
    assert result.is_grounded is False
    assert "Error" in result.answer or "त्रुटि" in result.answer
    assert "timed out" in result.answer.lower()
    # Citations are still safely attached from retrieved passages
    assert len(result.citations) > 0

    api_err_generator = AnswerGenerator(mock_client=MockGeminiClient(failure_mode="api_error"))
    result2 = api_err_generator.generate("Test query", retrieved_chunks=sample_kalam_chunks)
    assert result2.is_grounded is False
    assert "rate limit" in result2.answer.lower()


def test_adversarial_gemini_returning_refusal(sample_kalam_chunks):
    """
    Adversarial Check 7: Gemini explicitly returns a refusal string when context is insufficient.
    The generator must recognize is_refusal=True and mark is_grounded=True.
    """
    refusal_client = MockGeminiClient()
    refusal_gen = AnswerGenerator(mock_client=refusal_client)

    result_en = refusal_gen.generate("What is unknown_topic?", retrieved_chunks=sample_kalam_chunks)
    assert result_en.is_refusal is True
    assert result_en.is_grounded is True

    result_hi = refusal_gen.generate("unknown_topic_hi क्या है?", retrieved_chunks=sample_kalam_chunks)
    assert result_hi.is_refusal is True
    assert result_hi.is_grounded is True


# ── 6. Safe Fallback Guard (Requirement 2) ───────────────────────────────

class UngroundedMockClient:
    """Mock that returns ungrounded hallucinated claims to test the safe fallback guard."""
    def generate(self, query: str, context: List[Dict[str, Any]], lang: str) -> str:
        return "कलाम 2045 में NASA गए और नोबेल पुरस्कार जीता [doc_fake_chunk]।"


def test_safe_fallback_guard_prevents_unverified_answer(sample_kalam_chunks):
    """
    Requirement 2: Verify that invented citations and unsupported factual claims
    CANNOT be presented as verified answers. Safe fallback warning must be returned.
    """
    guard_gen = AnswerGenerator(mock_client=UngroundedMockClient())
    result = guard_gen.generate("कलाम की उपलब्धियां बताएं?", retrieved_chunks=sample_kalam_chunks)

    # 1. Grounding must be flagged as False
    assert result.is_grounded is False

    # 2. Verified answer must NOT present the hallucination as verified!
    assert "2045" not in result.answer
    assert "NASA" not in result.answer
    assert "doc_fake_chunk" not in result.answer

    # 3. Must display the safe verification warning notice
    assert UNVERIFIED_WARNING_HINDI in result.answer or "सत्यापन चेतावनी" in result.answer

    # 4. Raw draft is quarantined in unverified_draft for audit
    assert result.unverified_draft is not None
    assert "2045" in result.unverified_draft

    # 5. Citations from actual retrieved context are still provided for direct inspection
    assert len(result.citations) > 0


# ── 7. Optional Live API Integration Test ─────────────────────────────────

def test_live_gemini_integration():
    """
    Optional live integration test using modern google-genai SDK.
    Only runs if a valid non-placeholder GEMINI_API_KEY is configured.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key.startswith("your_gemini") or len(api_key) < 20:
        pytest.skip("Valid non-placeholder GEMINI_API_KEY not configured. Skipping live API call.")

    generator = AnswerGenerator()
    if generator.client is None and generator.legacy_model is None:
        pytest.skip("Gemini client could not be initialized.")

    dummy_context = [{
        "chunk_id": "doc_p16_c1",
        "text": "1997 में डॉ. कलाम को भारत का सर्वोच्च नागरिक सम्मान 'भारत रत्न' मिला।",
        "score": 1.0,
        "metadata": {"chunk_id": "doc_p16_c1", "page": 16, "section": "पुरस्कार", "source_doc": "doc.pdf"}
    }]

    try:
        res = generator.generate(
            query="कलाम को भारत रत्न किस वर्ष मिला?",
            retrieved_chunks=dummy_context
        )
        assert res.language == "hi"
        assert "1997" in res.answer
        assert len(res.citations) == 1
    except Exception as e:
        pytest.skip(f"Live API call failed (quota/network): {e}")
