"""
test_api.py — Unit tests for FastAPI endpoints (/api/query, /api/corpus-info, /health).
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.generation.answer_generator import GenerationResult
from app.citations.citation_builder import Citation

client = TestClient(app)


def test_api_health_endpoint():
    """Verify health endpoint."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}


def test_api_corpus_info():
    """Verify corpus-info endpoint returns expected document metadata."""
    res = client.get("/api/corpus-info")
    assert res.status_code == 200
    data = res.json()
    assert data["total_pages"] == 22
    assert data["total_chunks"] == 78
    assert "Agni_Ki_Udaan" in data["document_name"]


@patch("app.api.query.get_answer_generator")
def test_api_query_successful(mock_get_gen):
    """Verify POST /api/query returns correctly formatted response."""
    mock_gen = MagicMock()
    mock_gen.generate_answer.return_value = GenerationResult(
        query="SLV-III question",
        answer="1980 में रोहिणी उपग्रह स्थापित किया।",
        language="hi",
        model_name="mock-gemini",
        citations=[
            Citation(
                chunk_id="doc_p8_c3",
                page=8,
                section="6. इसरो के वर्ष",
                score=0.95,
                score_type="rrf_score",
                snippet="रोहिणी उपग्रह",
                source_doc="Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"
            )
        ],
        citations_markdown="* [1] पृष्ठ 8",
        full_response="1980 में रोहिणी उपग्रह स्थापित किया। * [1] पृष्ठ 8",
        is_grounded=True,
        is_refusal=False,
        latency_ms=120.5
    )
    mock_get_gen.return_value = mock_gen

    payload = {
        "query": "SLV-III ने किस उपग्रह को स्थापित किया?",
        "top_k": 5,
        "use_hybrid": True
    }
    res = client.post("/api/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "SLV-III question"
    assert "1980" in data["answer"]
    assert data["is_grounded"] is True
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "doc_p8_c3"
    assert data["citations"][0]["page"] == 8
    assert data["citations"][0]["score_type"] == "rrf_score"


def test_api_query_validation_error():
    """Verify empty query string yields 422 Unprocessable Entity."""
    res = client.post("/api/query", json={"query": ""})
    assert res.status_code == 422
