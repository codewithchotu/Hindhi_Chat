"""Tests for Hindi sentence segmentation, chunking, and metadata tagging."""
import pytest
from app.ingestion.chunker import (
    split_hindi_sentences,
    chunk_narrative_text,
    chunk_table_rows,
    DocumentChunk
)
from app.ingestion.table_extractor import ExtractedTable, TableRow


def test_danda_sentence_splitting():
    """Verify sentence boundaries respect Hindi Danda (।) and Double Danda (॥)."""
    text = "कलाम एक महान वैज्ञानिक थे। उन्होंने भारत को कई मिसाइलें दीं॥ उनका जीवन प्रेरणादायक था।"
    sentences = split_hindi_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].strip().endswith("।")
    assert sentences[1].strip().endswith("॥")
    assert sentences[2].strip().endswith("।")


def test_chunk_size_and_overlap():
    """Verify narrative text gets chunked with proper length and overlap."""
    sample_text = (
        "कलाम का जन्म 15 अक्टूबर 1931 को हुआ था। वे रामेश्वरम में पले-बढ़े। "
        "उनकी प्रारंभिक शिक्षा श्वार्ट्ज़ हाई स्कूल में हुई। उन्होंने भौतिकी का अध्ययन किया। "
        "बाद में वे मद्रास इंस्टीट्यूट ऑफ टेक्नोलॉजी गए जहाँ उन्होंने एयरोनॉटिकल इंजीनियरिंग पढ़ी।"
    )
    chunks = chunk_narrative_text(
        text=sample_text,
        page=4,
        section="2 · रामेश्वरम में बचपन",
        source_doc="test.pdf",
        chunk_size=120,
        chunk_overlap=40
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.page == 4
        assert chunk.section == "2 · रामेश्वरम में बचपन"
        assert chunk.content_type == "narrative"
        assert chunk.chunk_id.startswith("doc_p4_c")
        assert len(chunk.text) > 0


def test_table_row_chunking():
    """Verify structured table rows produce valid chunks with correct metadata."""
    rows = [
        TableRow(
            row_index=1,
            headers=["मिसाइल", "प्रकार"],
            values=["पृथ्वी", "सतह-से-सतह"],
            linearized_text="मिसाइल: पृथ्वी। भूमिका: सतह-से-सतह।"
        ),
        TableRow(
            row_index=2,
            headers=["मिसाइल", "प्रकार"],
            values=["अग्नि", "दूरी वाली"],
            linearized_text="मिसाइल: अग्नि। भूमिका: दूरी वाली।"
        )
    ]
    table = ExtractedTable(
        page_number=9,
        section="7 · निर्देशित मिसाइल कार्यक्रम",
        headers=["मिसाइल", "प्रकार"],
        rows=rows,
        source_doc="test.pdf"
    )
    chunks = chunk_table_rows(table, chunk_size=300)
    assert len(chunks) >= 1
    assert chunks[0].page == 9
    assert chunks[0].content_type == "table"
    assert "पृथ्वी" in chunks[0].text
    assert "अग्नि" in chunks[0].text
