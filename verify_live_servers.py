import urllib.request, json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Hindi query (transliterated — no Gemini key so retrieval will still run)
payload = json.dumps({
    'query': 'SLV-III rocket rohini satellite orbit 1980',
    'top_k': 3, 'use_hybrid': True, 'retrieval_mode': 'hybrid'
}).encode()
req = urllib.request.Request(
    'http://localhost:8000/api/query', data=payload,
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req, timeout=30) as r:
    resp = json.loads(r.read())
    print('=== QUERY RESULT ===')
    print('Language:', resp['language'])
    print('Is refusal:', resp['is_refusal'])
    print('Is grounded:', resp['is_grounded'])
    print('Citations returned:', len(resp['citations']))
    for c in resp['citations'][:3]:
        snippet = c.get('snippet', '')[:80]
        print(f"  Chunk={c['chunk_id']}  Page={c['page']}  Score={c['score']:.4f}  Type={c['score_type']}")
        print(f"  Snippet: {snippet}")
    print('Answer:', resp['answer'][:200])

# Check frontend is reachable
with urllib.request.urlopen('http://localhost:5173/', timeout=5) as r:
    html = r.read().decode('utf-8', errors='replace')
    print()
    print('=== FRONTEND REACHABILITY ===')
    print('HTTP Status:', r.status)
    print('Content-Type:', r.headers.get('Content-Type'))
    print('Has React app:', '<div id="root">' in html)
    print('Has Vite script:', 'src/main.jsx' in html or 'type="module"' in html)
