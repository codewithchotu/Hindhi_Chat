"""Tests for Hindi text ingestion, cleaning, table extraction, and PDF parsing."""
import pytest
import unicodedata
from app.config import get_settings
from app.main import app
from fastapi.testclient import TestClient

from app.ingestion.text_cleaner import (
    clean_text,
    normalize_unicode,
    reattach_matras,
    repair_ligatures,
    normalize_whitespace,
    LIGATURE_REPAIRS
)
from app.ingestion.pdf_loader import _detect_heading_from_raw, _remove_heading_from_text

client = TestClient(app)


def test_settings_load():
    """Verify application settings initialize with expected defaults."""
    settings = get_settings()
    assert settings.APP_NAME == "Cross-Lingual Hindi RAG Engine"
    assert settings.EMBEDDING_MODEL == "BAAI/bge-m3"
    assert "D:" in settings.HF_HOME


def test_health_endpoint():
    """Verify healthcheck endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_unicode_normalization():
    """Verify that composite and decomposed Devanagari characters normalize to NFC."""
    decomposed = "\u091c\u093c"
    normalized = normalize_unicode(decomposed)
    assert normalized == unicodedata.normalize("NFC", decomposed)


def test_matra_reattachment_newline():
    """Verify consonant split by newline from matra gets joined."""
    raw = "भारत क\nे मिसाइल मैन"
    cleaned = reattach_matras(raw)
    assert "भारत के" in cleaned
    assert "मिसाइल" in cleaned


def test_matra_reattachment_space():
    """Verify space before solitary matra is rejoined to consonant."""
    raw = "कलाम क े विचार"
    cleaned = reattach_matras(raw)
    assert "कलाम के" in cleaned


def test_ligature_repair_table_words():
    """Verify garbled conjuncts from PDF tables are fixed."""
    raw = "अकू्टबर में उनका जन।्म हुआ और उन्होंने शिका्ष पाई।"
    cleaned = repair_ligatures(raw)
    assert "अक्टूबर" in cleaned
    assert "जन्म" in cleaned
    assert "शिक्षा" in cleaned


def test_legitimate_spaces_preserved():
    """Ensure legitimate Hindi word spaces are never stripped."""
    raw = "कलाम का जन्म रामेश्वरम में हुआ था।"
    cleaned = clean_text(raw)
    assert cleaned == "कलाम का जन्म रामेश्वरम में हुआ था।"


def test_heading_detection():
    """Verify regex correctly parses section headings."""
    sample_raw = "6 · इसरो क\nे वर्ष और SLV-III\nकुछ अन्य पाठ..."
    heading = _detect_heading_from_raw(sample_raw)
    assert heading is not None
    assert "6 · इसरो के वर्ष और SLV-III" in heading


def test_heading_removal_from_body():
    """Ensure heading is removed from body text to avoid duplication."""
    section = "1 · ए.पी.जे. अब्दुल कलाम कौन थे?"
    raw = "1 · ए.पी.जे. अब्दुल कलाम कौन थे?\nकलाम एक महान वैज्ञानिक थे।"
    cleaned = _remove_heading_from_text(raw, section)
    assert "1 · ए.पी.जे." not in cleaned
    assert "कलाम एक महान वैज्ञानिक थे।" in cleaned
