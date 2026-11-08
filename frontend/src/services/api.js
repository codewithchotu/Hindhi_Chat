/**
 * api.js — Frontend service for communicating with the Cross-Lingual Hindi RAG Engine backend.
 * Uses configurable VITE_API_BASE_URL (or VITE_API_URL, fallback to http://localhost:8000).
 */

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_API_URL ||
  'http://localhost:8000';

export { API_BASE_URL };

/**
 * Check backend health status.
 */
export async function checkBackendHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const res = await fetch(`${API_BASE_URL}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) throw new Error(`Healthcheck HTTP ${res.status}`);
    const data = await res.json();
    return { status: data.status || 'healthy', online: true };
  } catch (err) {
    return {
      status: 'offline',
      online: false,
      error: err.name === 'AbortError' ? 'Connection timed out' : err.message,
    };
  }
}

/**
 * Fetch indexed corpus metadata (pages, chunks, models).
 */
export async function getCorpusInfo() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/corpus-info`);
    if (!res.ok) throw new Error(`Corpus info HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Could not fetch corpus info:', err.message);
    return {
      document_name: 'Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf',
      title: 'अग्नि की उड़ान: अध्ययन एवं संदर्भ सहायिका',
      total_pages: 22,
      total_chunks: 78,
      embedding_model: 'BAAI/bge-m3',
      vector_store: 'ChromaDB',
      hybrid_search: 'Dense (BGE-M3) + BM25Okapi + Reciprocal Rank Fusion',
    };
  }
}

/**
 * Submit a grounded RAG query to the backend.
 *
 * @param {Object} options
 * @param {string} options.query - Question text in English or Hindi
 * @param {number} [options.topK=5] - Number of passages to retrieve
 * @param {boolean} [options.useHybrid=true] - Enable hybrid search
 * @param {string} [options.retrievalMode='hybrid'] - 'hybrid', 'dense', or 'bm25'
 * @returns {Promise<Object>} GenerationResult from backend
 */
export async function submitRAGQuery({
  query,
  topK = 5,
  useHybrid = true,
  retrievalMode = 'hybrid',
}) {
  if (!query || !query.trim()) {
    throw new Error('Query string cannot be empty.');
  }

  const payload = {
    query: query.trim(),
    top_k: topK,
    use_hybrid: useHybrid,
    retrieval_mode: retrievalMode,
  };

  const res = await fetch(`${API_BASE_URL}/api/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `Query failed with HTTP status ${res.status}`
    );
  }

  return await res.json();
}
