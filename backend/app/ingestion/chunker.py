"""
chunker.py — Hindi-aware sentence splitting and metadata-rich document chunking.

Features:
  1. Respects Hindi Danda (।) and Double Danda (॥) as sentence delimiters,
     along with standard delimiters (?, !, paragraph breaks).
  2. Hindi punctuation remains attached to the preceding sentence without
     introducing extra space before । or ॥.
  3. Single newlines within a paragraph are normalized to spaces so sentences
     are not broken across arbitrary line-wrap boundaries.
  4. Explicitly handles sentences longer than chunk_size by splitting at
     natural clause/word boundaries without truncation.
  5. Correct chunk size accounting including intra-sentence space separators.
  6. Attaches complete metadata to every chunk:
     - chunk_id
     - page
     - section
     - source_doc
     - content_type ('narrative' | 'table')
     - char_count
"""
import re
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from app.ingestion.pdf_loader import PageContent
from app.ingestion.table_extractor import ExtractedTable

logger = logging.getLogger(__name__)

# Matches Hindi Danda (। - U+0964), Double Danda (॥ - U+0965),
# question mark, exclamation mark, or multiple newlines.
_SENTENCE_SPLIT_RE = re.compile(r'([।॥\?\!]+|\n{2,})')


@dataclass
class DocumentChunk:
    """Represents a text chunk ready for vectorization and metadata indexing."""
    chunk_id: str
    text: str
    page: int
    section: str
    source_doc: str
    content_type: str  # "narrative" or "table"
    char_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def split_hindi_sentences(text: str) -> List[str]:
    """
    Split Hindi text into sentences respecting Danda (।) and Double Danda (॥).
    Punctuation remains attached directly to the sentence without leading spaces.
    Single linebreaks within paragraphs are collapsed to spaces.
    """
    if not text or not text.strip():
        return []

    # Normalize paragraph newlines vs intra-paragraph line wraps
    paras = re.split(r'\n{2,}', text)
    cleaned_paras = []
    for p in paras:
        p_clean = re.sub(r'\s*\n\s*', ' ', p.strip())
        p_clean = re.sub(r'[ \t]+', ' ', p_clean)
        if p_clean:
            cleaned_paras.append(p_clean)

    full_text = "\n\n".join(cleaned_paras)
    tokens = _SENTENCE_SPLIT_RE.split(full_text)
    sentences = []

    for i in range(0, len(tokens) - 1, 2):
        sent_body = tokens[i].strip()
        delimiter = tokens[i + 1].strip()
        if delimiter == "\n\n":
            if sent_body:
                sentences.append(sent_body)
        else:
            # Punctuation directly attached (NO space before danda)
            full_sent = f"{sent_body}{delimiter}".strip()
            if full_sent:
                sentences.append(full_sent)

    if len(tokens) % 2 == 1:
        trailing = tokens[-1].strip()
        if trailing:
            sentences.append(trailing)

    return [s for s in sentences if len(s.strip()) > 0]


def _subdivide_oversized_sentence(sentence: str, max_size: int) -> List[str]:
    """
    Subdivide an individual sentence that exceeds max_size into sub-chunks,
    preferring clause delimiters (;, :, ,, —) or word spaces.
    Never truncates text.
    """
    if len(sentence) <= max_size:
        return [sentence]

    parts: List[str] = []
    # Split by major clause punctuation if available
    clauses = re.split(r'([;:,—]+)', sentence)
    current = ""

    for i in range(0, len(clauses) - 1, 2):
        clause_body = clauses[i].strip()
        clause_delim = clauses[i + 1].strip()
        clause_full = f"{clause_body}{clause_delim} "

        if len(current) + len(clause_full) > max_size and current:
            parts.append(current.strip())
            current = clause_full
        else:
            current += clause_full

    if len(clauses) % 2 == 1 and clauses[-1].strip():
        trailing = clauses[-1].strip()
        if len(current) + len(trailing) > max_size and current:
            parts.append(current.strip())
            current = trailing
        else:
            current += (" " if current else "") + trailing

    if current.strip():
        parts.append(current.strip())

    # Fallback if clause splitting didn't break down adequately
    final_parts: List[str] = []
    for p in parts:
        if len(p) <= max_size:
            final_parts.append(p)
        else:
            # Word-based breakdown
            words = p.split()
            w_curr = ""
            for w in words:
                if len(w_curr) + len(w) + 1 > max_size and w_curr:
                    final_parts.append(w_curr.strip())
                    w_curr = w
                else:
                    w_curr = f"{w_curr} {w}".strip()
            if w_curr.strip():
                final_parts.append(w_curr.strip())

    return final_parts if final_parts else [sentence]


def chunk_narrative_text(
    text: str,
    page: int,
    section: str,
    source_doc: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    chunk_start_idx: int = 1
) -> List[DocumentChunk]:
    """
    Group sentences into coherent chunks with overlap.
    Correctly accounts for space separator overhead and handles oversized sentences.
    """
    raw_sentences = split_hindi_sentences(text)
    if not raw_sentences:
        return []

    # Flatten sentences, subdividing any individual oversized sentences
    sentences: List[str] = []
    for s in raw_sentences:
        if len(s) > chunk_size:
            sentences.extend(_subdivide_oversized_sentence(s, chunk_size))
        else:
            sentences.append(s)

    chunks: List[DocumentChunk] = []
    current_sentences: List[str] = []
    current_len = 0
    c_idx = chunk_start_idx

    for sent in sentences:
        sent_len = len(sent)
        # Separator overhead is 1 char per existing sentence in group
        sep_len = 1 if current_sentences else 0
        added_len = sent_len + sep_len

        if current_len + added_len > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            chunks.append(DocumentChunk(
                chunk_id=f"doc_p{page}_c{c_idx}",
                text=chunk_text,
                page=page,
                section=section,
                source_doc=source_doc,
                content_type="narrative",
                char_count=len(chunk_text)
            ))
            c_idx += 1

            # Manage overlap based on trailing sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_sentences):
                needed = len(s) + (1 if overlap_sentences else 0)
                if overlap_len + needed <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += needed
                else:
                    break

            current_sentences = overlap_sentences
            current_len = overlap_len
            sep_len = 1 if current_sentences else 0
            added_len = sent_len + sep_len

        current_sentences.append(sent)
        current_len += added_len

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        chunks.append(DocumentChunk(
            chunk_id=f"doc_p{page}_c{c_idx}",
            text=chunk_text,
            page=page,
            section=section,
            source_doc=source_doc,
            content_type="narrative",
            char_count=len(chunk_text)
        ))

    return chunks


def chunk_table_rows(
    table: ExtractedTable,
    chunk_size: int = 400,
    chunk_start_idx: int = 1
) -> List[DocumentChunk]:
    """Convert extracted table rows into semantic chunks."""
    chunks: List[DocumentChunk] = []
    c_idx = chunk_start_idx

    current_group: List[str] = []
    current_len = 0

    for row in table.rows:
        row_text = row.linearized_text.strip()
        if not row_text:
            continue

        sep_len = 1 if current_group else 0
        added_len = len(row_text) + sep_len

        if current_len + added_len > chunk_size and current_group:
            chunk_text = " ".join(current_group).strip()
            chunks.append(DocumentChunk(
                chunk_id=f"doc_p{table.page_number}_c{c_idx}",
                text=chunk_text,
                page=table.page_number,
                section=table.section,
                source_doc=table.source_doc,
                content_type="table",
                char_count=len(chunk_text)
            ))
            c_idx += 1
            current_group = []
            current_len = 0
            added_len = len(row_text)

        current_group.append(row_text)
        current_len += added_len

    if current_group:
        chunk_text = " ".join(current_group).strip()
        chunks.append(DocumentChunk(
            chunk_id=f"doc_p{table.page_number}_c{c_idx}",
            text=chunk_text,
            page=table.page_number,
            section=table.section,
            source_doc=table.source_doc,
            content_type="table",
            char_count=len(chunk_text)
        ))

    return chunks


def build_all_chunks(
    pages: List[PageContent],
    tables: List[ExtractedTable],
    chunk_size: int = 400,
    chunk_overlap: int = 80
) -> List[DocumentChunk]:
    """Build complete set of chunks combining narrative and tables."""
    tables_by_page: Dict[int, List[ExtractedTable]] = {}
    for t in tables:
        tables_by_page.setdefault(t.page_number, []).append(t)

    all_chunks: List[DocumentChunk] = []

    for page in pages:
        chunk_counter = 1

        if page.cleaned_text.strip():
            narrative_chunks = chunk_narrative_text(
                text=page.cleaned_text,
                page=page.page_number,
                section=page.section,
                source_doc=page.source_doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_start_idx=chunk_counter
            )
            all_chunks.extend(narrative_chunks)
            chunk_counter += len(narrative_chunks)

        if page.page_number in tables_by_page:
            for tab in tables_by_page[page.page_number]:
                t_chunks = chunk_table_rows(
                    table=tab,
                    chunk_size=chunk_size,
                    chunk_start_idx=chunk_counter
                )
                all_chunks.extend(t_chunks)
                chunk_counter += len(t_chunks)

    return all_chunks
