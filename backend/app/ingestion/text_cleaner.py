"""
text_cleaner.py — Devanagari text normalization and PDF artifact repair.

Handles:
  1. Unicode NFC normalization for consistent Devanagari representation.
  2. Matra reattachment: fixes consonant + linebreak + matra splits
     caused by PDF font metric positioning (e.g., "क\\nे" → "के").
  3. Known ligature corruption repair via a curated lookup table,
     targeting garbled conjuncts produced by PDF table extraction.
  4. Whitespace normalization: collapses multiple spaces and trims lines.

Design principle: every repair is targeted and conservative.
We never blindly strip spaces between legitimate Hindi words.
"""
import re
import unicodedata
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 1. Curated Ligature / Garble Repair Dictionary
#    Built from empirical inspection of PyMuPDF output on this specific PDF.
#    Each entry: broken_form → correct_form
# ---------------------------------------------------------------------------
LIGATURE_REPAIRS: dict[str, str] = {
    # Conjuncts with क्ष (ksha)
    "कका्ष": "कक्षा",
    "ककी्षय": "कक्षीय",
    "शिका्ष": "शिक्षा",
    "शिकण्ष": "शिक्षण",
    "परीकण्ष ों": "परीक्षणों",
    "परीकण्ष": "परीक्षण",
    "लक ्ष्य": "लक्ष्य",
    "अपेका्ष": "अपेक्षा",
    "प्रकेप्ष ण": "प्रक्षेपण",
    "प्रकेप्षण": "प्रक्षेपण",
    # Conjuncts with ष्ट्र (shtra)
    "राषप्ट्र ति": "राष्ट्रपति",
    "राषप्ट्रति": "राष्ट्रपति",
    "राष ्ट्र": "राष्ट्र",
    # Conjuncts with प्र (pra)
    "पम्र ुख": "प्रमुख",
    "पय्र ास": "प्रयास",
    "पक्र ाश": "प्रकाश",
    # Other garbled conjuncts
    "अंतरिक ्ष": "अंतरिक्ष",
    "वा्यख्यान": "व्याख्यान",
    "बह्र ्म": "ब्रह्म",
    "उपगह्र": "उपग्रह",
    "चिरसा्थयी": "चिरस्थायी",
    "त्शरि ूल": "त्रिशूल",
    # Reordered / space-split words
    "द्रवा ा": "द्वारा",
    "अकू्टबर": "अक्टूबर",
    "जन।्म": "जन्म",
    "पढ़।ी": "पढ़ी।",
    "पद ्म": "पद्म",
    "आरभंिक": "आरंभिक",
    "आरभं": "आरंभ",
    "अरणु": "अरुण",
    "डॉक्टरटे": "डॉक्टरेट",
    "ऑपरशे न": "ऑपरेशन",
    "पुरस्रका": "पुरस्कार",
    "विश्वविद्यालयो ं": "विश्वविद्यालयों",
    "जिन्नहों े": "जिन्होंने",
    "केंद्यरी": "केंद्रीय",
    "रवे .": "रेव.",
    "इंजी नि यरिं ग": "इंजीनियरिंग",
    "इंजी नि यरिंग": "इंजीनियरिंग",
    "इंजीनियरि ंग": "इंजीनियरिंग",
    "विद्यार् थियों": "विद्यार्थियों",
    "स ्कूल": "स्कूल",
    "ऊ ँचा": "ऊँचा",
    "कि ंतु": "किंतु",
    "क ुछ": "कुछ",
}

# Sort by descending length so longer patterns are matched first
_SORTED_REPAIRS: List[Tuple[str, str]] = sorted(
    LIGATURE_REPAIRS.items(), key=lambda kv: len(kv[0]), reverse=True
)

# ---------------------------------------------------------------------------
# 2. Regex Patterns for Matra Reattachment
# ---------------------------------------------------------------------------
# Devanagari consonants: U+0915–U+0939, with optional nuqta U+093C
# Dependent vowel signs (matras): U+093E–U+094D, U+0901–U+0903
# Matches consonant at end-of-line followed by matra on next line
_MATRA_NEWLINE_RE = re.compile(
    r'([\u0915-\u0939]\u093C?)\s*[\r\n]+\s*([\u093E-\u094D\u0901-\u0903])'
)

# Same pattern but for intra-line spaces (consonant + space + lone matra)
_MATRA_SPACE_RE = re.compile(
    r'([\u0915-\u0939]\u093C?) ([\u093E-\u094D\u0901-\u0903])(?=\s|$|[^\u0900-\u097F])'
)


def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalization to standardize Devanagari representations."""
    return unicodedata.normalize('NFC', text)


def reattach_matras(text: str) -> str:
    """
    Rejoin Devanagari matras that were split from their base consonant
    by PDF line-breaking or font-metric spacing.

    Example: "भारत क\\nे" → "भारत के"
    """
    text = _MATRA_NEWLINE_RE.sub(r'\1\2', text)
    text = _MATRA_SPACE_RE.sub(r'\1\2', text)
    return text


def repair_ligatures(text: str) -> str:
    """
    Apply the curated ligature repair dictionary to fix known garbled
    conjunct consonants from PDF table extraction.
    """
    for broken, correct in _SORTED_REPAIRS:
        if broken in text:
            text = text.replace(broken, correct)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and trim lines, preserving paragraph breaks."""
    text = re.sub(r'[ \t]+', ' ', text)
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
        elif cleaned and cleaned[-1] != '':
            cleaned.append('')  # preserve paragraph break
    return '\n'.join(cleaned)


def clean_text(raw_text: str) -> str:
    """
    Full cleaning pipeline for Devanagari text extracted from PDF.

    Order matters:
      1. Unicode NFC normalization first (standardize codepoints).
      2. Matra reattachment (fix line-break splits).
      3. Ligature repair (fix known garbled conjuncts).
      4. Whitespace normalization (collapse spaces, trim).
    """
    text = normalize_unicode(raw_text)
    text = reattach_matras(text)
    text = repair_ligatures(text)
    text = normalize_whitespace(text)
    return text
