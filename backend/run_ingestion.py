import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from app.ingestion.pdf_loader import extract_pages, get_heading_map
from app.ingestion.table_extractor import extract_all_tables
from app.ingestion.chunker import build_all_chunks

pdf_path = r'c:\Users\owais\OneDrive\Pictures\Desktop\hindi-rag-engine\reference\Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf'
heading_map = get_heading_map(pdf_path)
pages = extract_pages(pdf_path)
tables = extract_all_tables(pdf_path, heading_map)
chunks = build_all_chunks(pages, tables)

print(f"Total pages extracted: {len(pages)}")
print(f"Total tables extracted: {len(tables)}")
print(f"Total chunks created: {len(chunks)}")

out_dir = Path(r'c:\Users\owais\OneDrive\Pictures\Desktop\hindi-rag-engine\backend\data\processed')
out_dir.mkdir(parents=True, exist_ok=True)

samples = []
for p in pages[2:6]:
    samples.append({
        'page_number': p.page_number,
        'section': p.section,
        'raw_text_preview': p.raw_text[:250],
        'cleaned_text_preview': p.cleaned_text[:250]
    })

with open(out_dir / 'extraction_samples.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

chunks_data = [c.to_dict() for c in chunks]
with open(out_dir / 'chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks_data, f, ensure_ascii=False, indent=2)

print("Saved extraction_samples.json and chunks.json successfully!")
