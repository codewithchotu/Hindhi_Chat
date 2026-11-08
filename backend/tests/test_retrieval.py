import gc
import pytest
from app.config import get_settings
from app.embeddings.embedder import get_embedder
from app.vectorstore.chroma_store import ChromaVectorStore
from app.ingestion.chunker import DocumentChunk


@pytest.fixture(scope="module")
def vector_store():
    """Shared vector store fixture (reuses initialized embedder singleton)."""
    gc.collect()
    embedder = get_embedder()
    store = ChromaVectorStore(embedder=embedder)
    return store


def test_embedding_dimensions(vector_store):
    """Verify that BAAI/bge-m3 outputs 1024-dimensional normalized vectors."""
    text = "अग्नि की उड़ान एक महान आत्मकथा है।"
    vec = vector_store.embedder.embed_query(text)
    assert len(vec) == 1024
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


def test_vector_store_persistence(vector_store):
    """Verify vectors persist and can be queried after connecting."""
    count = vector_store.count()
    assert count == 78


def test_metadata_integrity(vector_store):
    """Verify all required metadata fields are stored with every vector."""
    results = vector_store.search("रोहिणी उपग्रह", top_k=1)
    assert len(results) > 0
    meta = results[0]["metadata"]
    for field in ["chunk_id", "page", "section", "source_doc", "content_type", "char_count"]:
        assert field in meta, f"Missing metadata field: {field}"
    assert isinstance(meta["page"], int)
    assert 1 <= meta["page"] <= 22


def test_duplicate_prevention_idempotency(vector_store):
    """Verify re-adding existing chunks does not duplicate them in ChromaDB."""
    initial_count = vector_store.count()
    dummy_chunk = DocumentChunk(
        chunk_id="doc_p8_c1",
        text="परीक्षण पाठ",
        page=8,
        section="6 · इसरो के वर्ष और SLV-III",
        source_doc="test.pdf",
        content_type="narrative",
        char_count=12
    )
    vector_store.add_chunks([dummy_chunk])
    assert vector_store.count() == initial_count


def test_cross_lingual_retrieval_english_to_hindi(vector_store):
    """Verify an English query retrieves relevant Hindi passages."""
    query = "Where did Kalam study aeronautical engineering?"
    hits = vector_store.search(query, top_k=3)
    assert len(hits) > 0
    retrieved_pages = [h["metadata"]["page"] for h in hits]
    assert 6 in retrieved_pages or 20 in retrieved_pages or 22 in retrieved_pages
    assert hits[0]["score"] > 0.45


def test_all_six_mandatory_queries_retrieval(vector_store):
    """Verify retrieval quality for all six mandatory questions."""
    test_cases = [
        {"q": "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?", "expected_pages": [8, 17, 20, 22]},
        {"q": "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", "expected_pages": [16, 17, 20, 21, 22]},
        {"q": "पोखरण-II में कलाम की क्या भूमिका थी?", "expected_pages": [10, 17, 20, 21]},
        {"q": "Which institution did Kalam attend to study aeronautical engineering?", "expected_pages": [6, 17, 20, 21, 22]},
        {"q": "Who co-wrote the autobiography, and in what year was it published?", "expected_pages": [1, 3, 17, 22]},
        {"q": "How and where did Kalam die in 2015?", "expected_pages": [15, 17, 21, 22]},
    ]
    for tc in test_cases:
        hits = vector_store.search(tc["q"], top_k=3)
        assert len(hits) > 0
        pages = [h["metadata"]["page"] for h in hits]
        intersection = set(pages).intersection(set(tc["expected_pages"]))
        assert len(intersection) > 0, f"Query '{tc['q']}' failed: got pages {pages}, expected from {tc['expected_pages']}"
