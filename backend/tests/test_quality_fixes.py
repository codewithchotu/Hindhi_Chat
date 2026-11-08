"""Regression tests for Phase 2 code quality fixes:
1. Punctuation preservation without leading spaces
2. Oversized sentence handling without truncation
3. Table deduplication between narrative and structured tables
4. Dynamic section heading detection
5. Sentence integrity in doc_p3_c1
"""
import pytest
from app.ingestion.chunker import (
    split_hindi_sentences,
    chunk_narrative_text,
    _subdivide_oversized_sentence,
    DocumentChunk
)
from app.ingestion.pdf_loader import (
    _detect_heading_from_raw,
    _get_section_for_page,
    extract_prose_excluding_tables,
    extract_pages
)
import fitz


def test_punctuation_attached_directly():
    """Requirement 1: Verify Hindi punctuation remains attached to the sentence without leading spaces."""
    raw = "कलाम एक वैज्ञानिक थे। उन्होंने मिसाइलें बनाईं॥ क्या यह सच है? हाँ!"
    sents = split_hindi_sentences(raw)
    assert len(sents) == 4
    assert sents[0] == "कलाम एक वैज्ञानिक थे।"
    assert not sents[0].endswith(" ।")
    assert sents[1] == "उन्होंने मिसाइलें बनाईं॥"
    assert not sents[1].endswith(" ॥")
    assert sents[2] == "क्या यह सच है?"
    assert sents[3] == "हाँ!"


def test_oversized_sentence_subdivision():
    """Requirement 2: Verify sentences longer than chunk_size are not truncated."""
    long_sentence = (
        "कलाम का जीवन एक असंभव-सी यात्रा थी: दक्षिण भारत के एक छोटे-से द्वीपीय कस्बे से, "
        "जहाँ बचपन में वे परिवार की मदद के लिए अख़बार बेचते थे, देश के सबसे महत्वाकांक्षी "
        "विज्ञान कार्यक्रमों के नेतृत्व तक, और अंततः देश के सर्वोच्च संवैधानिक पद तक।"
    )
    max_chunk_size = 100
    parts = _subdivide_oversized_sentence(long_sentence, max_size=max_chunk_size)
    assert len(parts) > 1
    # Verify no content was lost: all major keywords must be present across parts
    recombined = " ".join(parts)
    assert "कलाम का जीवन" in recombined
    assert "द्वीपीय कस्बे" in recombined
    assert "अख़बार बेचते थे" in recombined
    assert "संवैधानिक पद तक" in recombined


def test_doc_p3_sentence_integrity():
    """Requirement 3: Verify doc_p3_c1 does not truncate with incomplete phrase."""
    pdf_path = r'c:\Users\owais\OneDrive\Pictures\Desktop\hindi-rag-engine\reference\Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf'
    pages = extract_pages(pdf_path)
    p3 = pages[2]
    chunks = chunk_narrative_text(p3.cleaned_text, page=3, section=p3.section, source_doc="test.pdf")
    
    # Check chunk 0
    c1_text = chunks[0].text
    # Should end at a complete sentence boundary
    assert c1_text.endswith("।") or c1_text.endswith("॥")
    # Must NOT end with the chopped phrase "जहाँ बचपन में वे परिवार की"
    assert not c1_text.endswith("जहाँ बचपन में वे परिवार की")
    # In fact, that clause should be part of a complete sentence in chunk 1
    assert "कलाम का जीवन एक असंभव-सी यात्रा थी" in chunks[1].text
    assert chunks[1].text.endswith("संवैधानिक पद तक।")


def test_table_deduplication():
    """Requirement 4: Verify narrative extraction excludes table contents."""
    pdf_path = r'c:\Users\owais\OneDrive\Pictures\Desktop\hindi-rag-engine\reference\Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf'
    pages = extract_pages(pdf_path)
    p9 = pages[8]  # Page 9 (Missiles table)
    cleaned = p9.cleaned_text
    
    # The narrative should have the surrounding text
    assert "निर्देशित मिसाइल कार्यक्रम" in p9.section
    assert "मिसाइल मैन" in cleaned
    # The narrative MUST NOT contain the missile table rows (e.g., table cells like 'टैंक-रोधी')
    assert "टैंक-रोधी निर्देशित मिसाइल" not in cleaned
    assert '"नाग/कोबरा"' not in cleaned


def test_dynamic_heading_detection_with_fallback():
    """Requirement 5: Verify dynamic section heading detection is preferred over static map."""
    raw_sample = "8 · पोखरण-II: 1998 क\nे परमाणु परीक्षण\nमई 1998 में, भारत ने राजस्थान के पोखरण में..."
    detected = _detect_heading_from_raw(raw_sample)
    assert detected == "8 · पोखरण-II: 1998 के परमाणु परीक्षण"
    
    # Check fallback mechanism
    fallback_title = _get_section_for_page(1, "कुछ अज्ञात पाठ बिना शीर्षक के")
    assert fallback_title == "शीर्षक पृष्ठ"
