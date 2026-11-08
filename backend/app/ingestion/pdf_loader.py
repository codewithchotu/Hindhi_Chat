"""
pdf_loader.py — Extract text and metadata from the Hindi PDF using PyMuPDF.

Responsibilities:
  1. Open the PDF and iterate page-by-page.
  2. Dynamically detect section headings from the "N · Title" pattern.
     Fallback to SECTION_TITLES lookup if extraction fails.
  3. Detect which pages contain tables (for table_extractor to handle).
  4. Exclude detected table bounding boxes from narrative extraction to prevent
     content from being indexed twice as both prose and structured table chunks.
  5. Apply text_cleaner.clean_text() to narrative text.
  6. Return structured PageContent objects with full metadata.
"""
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

import fitz  # PyMuPDF

from app.ingestion.text_cleaner import clean_text, reattach_matras, normalize_unicode

logger = logging.getLogger(__name__)

# Fallback section titles (post-cleaning)
SECTION_TITLES: Dict[int, str] = {
    1: "शीर्षक पृष्ठ",
    2: "विषय-सूची",
    3: "1 · ए.पी.जे. अब्दुल कलाम कौन थे?",
    4: "2 · रामेश्वरम में बचपन",
    5: "3 · प्रारंभिक शिक्षा एवं प्रेरणाएँ",
    6: "4 · कॉलेज के वर्ष और उड़ान का आकर्षण",
    7: "5 · वैज्ञानिक करियर के पहले कदम",
    8: "6 · इसरो के वर्ष और SLV-III",
    9: "7 · निर्देशित मिसाइल कार्यक्रम",
    10: "8 · पोखरण-II: 1998 के परमाणु परीक्षण",
    11: "9 · अग्नि की उड़ान के मुख्य विषय",
    12: "10 · गुरु एवं शिक्षक",
    13: "11 · कलाम के नेतृत्व के सबक",
    14: "12 · जनता के राष्ट्रपति (2002–2007)",
    15: "13 · अंतिम क्षण तक शिक्षक (2007–2015)",
    16: "14 · पुरस्कार एवं सम्मान",
    17: "15 · जीवन-रेखा (टाइमलाइन)",
    18: "16 · कलाम के जीवन-मूल्य",
    19: "17 · विरासत एवं प्रभाव",
    20: "18 · संस्थाओं एवं शब्दों की शब्दावली",
    21: "19 · विचार एवं बोध-प्रश्न",
    22: "20 · त्वरित तथ्य-सारांश",
}

TABLE_PAGES = {9, 12, 16, 17, 20, 22}

_HEADING_RE = re.compile(r'^(\d{1,2})\s*[·•]\s*(.+)')


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int
    section: str
    raw_text: str
    cleaned_text: str
    has_tables: bool
    source_doc: str
    content_type: str = "narrative"


def _detect_heading_from_raw(raw_text: str) -> Optional[str]:
    """
    Dynamically detect section heading from raw page text.
    Applies clean_text to top text lines and matches 'N · Title'.
    """
    cleaned = clean_text(raw_text)
    for line in cleaned.splitlines()[:3]:
        line = line.strip()
        match = _HEADING_RE.match(line)
        if match:
            section_num = match.group(1)
            section_title = match.group(2).strip()
            return f"{section_num} · {section_title}"

    return None


def _get_section_for_page(page_num: int, raw_text: str) -> str:
    """
    Determine section title preferring dynamic detection first,
    falling back to SECTION_TITLES dictionary.
    """
    detected = _detect_heading_from_raw(raw_text)
    if detected:
        return detected

    if page_num in SECTION_TITLES:
        return SECTION_TITLES[page_num]

    return "Unknown Section"


def _remove_heading_from_text(text: str, section: str) -> str:
    """
    Remove the section heading line from body text to prevent duplication.
    """
    cleaned = clean_text(text)
    lines = cleaned.splitlines()
    if not lines:
        return ""

    first_line = lines[0].strip()
    match = _HEADING_RE.match(first_line)
    if match:
        return "\n".join(lines[1:]).strip()

    return cleaned


def extract_prose_excluding_tables(page: fitz.Page) -> str:
    """
    Extract text from page, excluding any block whose bounding box
    significantly overlaps with detected tables on the page.
    """
    tabs = page.find_tables()
    if not tabs or not tabs.tables:
        return page.get_text()

    table_bboxes = [fitz.Rect(t.bbox) for t in tabs.tables]
    blocks = page.get_text("blocks")
    prose_blocks = []

    for b in blocks:
        if b[6] != 0:  # non-text block
            continue
        block_rect = fitz.Rect(b[:4])
        in_table = False
        for t_rect in table_bboxes:
            intersect = block_rect & t_rect
            if not intersect.is_empty:
                overlap_ratio = intersect.get_area() / max(block_rect.get_area(), 1.0)
                if overlap_ratio > 0.4:
                    in_table = True
                    break
        if not in_table:
            text = b[4].strip()
            if text:
                prose_blocks.append(text)

    return "\n".join(prose_blocks)


def extract_pages(
    pdf_path: str,
    source_doc_name: Optional[str] = None,
) -> List[PageContent]:
    """
    Extract all pages from the Hindi PDF with metadata.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if source_doc_name is None:
        source_doc_name = path.name

    doc = fitz.open(str(path))
    pages: List[PageContent] = []

    logger.info(f"Extracting {len(doc)} pages from {source_doc_name}")

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]

        raw_page_text = page.get_text()
        section = _get_section_for_page(page_num, raw_page_text)
        has_tables = page_num in TABLE_PAGES

        if page_num == 1:
            content_type = "title"
        elif page_num == 2:
            content_type = "toc"
        elif page_num == 21:
            content_type = "questions"
        elif has_tables:
            content_type = "table"
        else:
            content_type = "narrative"

        # Exclude table bboxes from narrative extraction
        prose_raw = extract_prose_excluding_tables(page)
        body_text = _remove_heading_from_text(prose_raw, section)
        cleaned = clean_text(body_text)

        pages.append(PageContent(
            page_number=page_num,
            section=section,
            raw_text=raw_page_text,
            cleaned_text=cleaned,
            has_tables=has_tables,
            source_doc=source_doc_name,
            content_type=content_type,
        ))

    doc.close()
    logger.info(f"Extracted {len(pages)} pages")
    return pages


def get_heading_map(pdf_path: str) -> Dict[int, str]:
    """Return mapping of page_number → section_title preferring dynamic detection."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    heading_map: Dict[int, str] = {}

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        raw_text = doc[page_idx].get_text()
        heading_map[page_num] = _get_section_for_page(page_num, raw_text)

    doc.close()
    return heading_map
