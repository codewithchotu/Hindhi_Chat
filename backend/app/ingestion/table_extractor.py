"""
table_extractor.py — Extract structured tables from the Hindi PDF using PyMuPDF.

Handles extraction and linearization of tables from specific pages:
  - Page 9: Guided Missiles Table (6 rows x 3 cols: मिसाइल, प्रकार, नाम का अर्थ)
  - Page 12: Mentors and Teachers Table (6 rows x 2 cols: व्यक्ति, भूमिका)
  - Page 16: Awards and Honors Table (6 rows x 3 cols: वर्ष, सम्मान, टिप्पणी)
  - Page 17: Timeline Table (14 rows x 2 cols: वर्ष, पड़ाव)
  - Page 20: Glossary / Institutions (10 rows x 2 cols: शब्द, अर्थ)
  - Page 22: Quick Facts Summary (13 rows x 2 cols: क्षेत्र, तथ्य)

Converts each table into:
  1. Cleaned structured row representation.
  2. Searchable natural language sentences with metadata.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF

from app.ingestion.text_cleaner import clean_text

logger = logging.getLogger(__name__)


@dataclass
class TableRow:
    """Represents a single row from an extracted table."""
    row_index: int
    headers: List[str]
    values: List[str]
    linearized_text: str


@dataclass
class ExtractedTable:
    """Represents an extracted table with its metadata and rows."""
    page_number: int
    section: str
    headers: List[str]
    rows: List[TableRow]
    source_doc: str
    table_index: int = 0

    @property
    def searchable_passages(self) -> List[str]:
        """Return all linearized rows as searchable text sentences."""
        return [r.linearized_text for r in self.rows if r.linearized_text.strip()]


def _clean_cell(cell: Optional[str]) -> str:
    """Clean and normalize a single table cell."""
    if not cell:
        return ""
    # Clean up linebreaks in cells and run the Devanagari text cleaner
    cell = cell.replace("\n", " ").strip()
    return clean_text(cell)


def _linearize_row(headers: List[str], values: List[str], page_number: int) -> str:
    """
    Convert a row of headers and values into a semantically dense,
    searchable Hindi sentence.
    """
    pairs = []
    for h, v in zip(headers, values):
        if h and v:
            pairs.append(f"{h}: {v}")
    
    # Specific natural phrasing based on page context if desirable
    if page_number == 9:
        # Missile table: मिसाइल, प्रकार (भूमिका), नाम का अर्थ
        missile = values[0] if len(values) > 0 else ""
        role = values[1] if len(values) > 1 else ""
        meaning = values[2] if len(values) > 2 else ""
        return f"मिसाइल: {missile}। भूमिका: {role}। नाम का अर्थ: {meaning}।"
    elif page_number == 12:
        # Mentors: व्यक्ति, कलाम के जीवन में भूमिका
        person = values[0] if len(values) > 0 else ""
        role = values[1] if len(values) > 1 else ""
        return f"व्यक्ति/गुरु: {person}। कलाम के जीवन में भूमिका: {role}।"
    elif page_number == 16:
        # Awards: वर्ष, सम्मान, टिप्पणी
        year = values[0] if len(values) > 0 else ""
        award = values[1] if len(values) > 1 else ""
        note = values[2] if len(values) > 2 else ""
        return f"वर्ष: {year}। सम्मान: {award}। विवरण: {note}।"
    elif page_number == 17:
        # Timeline: वर्ष, पड़ाव
        year = values[0] if len(values) > 0 else ""
        milestone = values[1] if len(values) > 1 else ""
        return f"वर्ष: {year}। जीवन पड़ाव/घटना: {milestone}।"
    elif page_number == 20:
        # Glossary: शब्द, अर्थ
        term = values[0] if len(values) > 0 else ""
        def_ = values[1] if len(values) > 1 else ""
        return f"पारिभाषिक शब्द: {term}। अर्थ/परिभाषा: {def_}।"
    elif page_number == 22:
        # Quick facts: क्षेत्र, तथ्य
        field_ = values[0] if len(values) > 0 else ""
        fact = values[1] if len(values) > 1 else ""
        return f"तथ्य क्षेत्र: {field_}। विवरण: {fact}।"

    return "। ".join(pairs) + "।"


def extract_tables_from_page(
    page: fitz.Page,
    page_number: int,
    section: str,
    source_doc: str
) -> List[ExtractedTable]:
    """Extract and parse all tables from a given PyMuPDF page."""
    tab_finder = page.find_tables()
    extracted_tables: List[ExtractedTable] = []

    if not tab_finder or not tab_finder.tables:
        return extracted_tables

    for t_idx, tab in enumerate(tab_finder.tables):
        raw_data = tab.extract()
        if not raw_data or len(raw_data) < 2:
            continue

        # Extract & clean headers
        raw_headers = raw_data[0]
        headers = [_clean_cell(h) for h in raw_headers]

        rows: List[TableRow] = []
        for r_idx, raw_row in enumerate(raw_data[1:]):
            values = [_clean_cell(val) for val in raw_row]
            # Ensure values length matches headers
            if len(values) < len(headers):
                values.extend([""] * (len(headers) - len(values)))
            elif len(values) > len(headers):
                values = values[:len(headers)]

            linearized = _linearize_row(headers, values, page_number)
            rows.append(TableRow(
                row_index=r_idx + 1,
                headers=headers,
                values=values,
                linearized_text=linearized
            ))

        extracted_tables.append(ExtractedTable(
            page_number=page_number,
            section=section,
            headers=headers,
            rows=rows,
            source_doc=source_doc,
            table_index=t_idx
        ))

    return extracted_tables


def extract_all_tables(pdf_path: str, heading_map: Dict[int, str]) -> List[ExtractedTable]:
    """
    Extract all tables across the document for known table pages.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    all_tables: List[ExtractedTable] = []

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]
        section = heading_map.get(page_num, f"Section Page {page_num}")
        tables = extract_tables_from_page(page, page_num, section, path.name)
        all_tables.extend(tables)

    doc.close()
    return all_tables
