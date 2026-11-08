"""Tests for normalization edge cases: spaces, matras, ligatures, idempotence."""
import pytest
from app.ingestion.text_cleaner import (
    clean_text,
    normalize_unicode,
    reattach_matras,
    repair_ligatures,
)


def test_correctly_spaced_hindi():
    """Verify that correctly spaced Hindi text is unchanged and spaces are preserved."""
    sample = "अग्नि की उड़ान एक मौलिक अध्ययन सहायिका है।"
    assert clean_text(sample) == sample


def test_broken_matras_spaces_and_newlines():
    """Verify matras split across spaces and newlines get reattached properly."""
    # Newline split
    newline_split = "भारत क\nे मिसाइल मैन"
    assert "भारत के" in reattach_matras(newline_split)

    # Space split
    space_split = "कलाम क े विचार"
    assert "कलाम के" in reattach_matras(space_split)

    # Isolated matra
    isolated = "वैज्ञानिक करियर क\nे पहले कदम"
    assert "वैज्ञानिक करियर के पहले कदम" in clean_text(isolated)


def test_conjuncts_and_ligatures():
    """Verify known corrupted table conjuncts are properly repaired."""
    raw = "कका्ष में उपगह्र का प्रकेप्ष ण और राषप्ट्र ति द्वारा पुरस्रका मिला।"
    repaired = repair_ligatures(raw)
    assert "कक्षा" in repaired
    assert "उपग्रह" in repaired
    assert "प्रक्षेपण" in repaired
    assert "राष्ट्रपति" in repaired
    assert "पुरस्कार" in repaired


def test_idempotence_already_correct_hindi():
    """Verify that running clean_text on already-correct text does not alter it."""
    correct_text = (
        "अवुल पाकिर जैनुलाब्दीन अब्दुल कलाम (1931–2015) एक भारतीय एयरोस्पेस वैज्ञानिक थे। "
        "उन्होंने भारत को कई मिसाइलें दीं।"
    )
    assert clean_text(correct_text) == correct_text
