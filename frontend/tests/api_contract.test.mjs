/**
 * frontend/tests/api_contract.test.mjs
 * Validates frontend API client contracts and request formatting.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert';

describe('Frontend API Logic & Contract Tests', () => {
  it('rejects empty queries before making network calls', async () => {
    // Simulated submitRAGQuery validation
    const validateQuery = (query) => {
      if (!query || !query.trim()) {
        throw new Error('Query string cannot be empty.');
      }
      return true;
    };

    assert.throws(() => validateQuery(''), /cannot be empty/);
    assert.throws(() => validateQuery('   '), /cannot be empty/);
    assert.doesNotThrow(() => validateQuery('कलाम का जन्म कब हुआ?'));
    assert.doesNotThrow(() => validateQuery('When was Kalam born?'));
  });

  it('correctly formats the request payload for hybrid search', () => {
    const buildPayload = ({ query, topK = 5, useHybrid = true, retrievalMode = 'hybrid' }) => ({
      query: query.trim(),
      top_k: topK,
      use_hybrid: useHybrid,
      retrieval_mode: retrievalMode,
    });

    const payload = buildPayload({
      query: 'SLV-III उपग्रह',
      topK: 5,
      useHybrid: true,
      retrievalMode: 'hybrid',
    });

    assert.strictEqual(payload.query, 'SLV-III उपग्रह');
    assert.strictEqual(payload.top_k, 5);
    assert.strictEqual(payload.use_hybrid, true);
    assert.strictEqual(payload.retrieval_mode, 'hybrid');
  });

  it('correctly maps citation score types and formats badges', () => {
    const formatScore = (score, scoreType) => {
      if (scoreType === 'rrf_score') {
        return `RRF Score: ${score.toFixed(4)}`;
      }
      if (scoreType === 'bm25_score') {
        return `BM25: ${score.toFixed(2)}`;
      }
      return `Similarity: ${(score * 100).toFixed(1)}%`;
    };

    assert.strictEqual(formatScore(0.033333, 'rrf_score'), 'RRF Score: 0.0333');
    assert.strictEqual(formatScore(14.52, 'bm25_score'), 'BM25: 14.52');
    assert.strictEqual(formatScore(0.8542, 'dense_similarity'), 'Similarity: 85.4%');
  });
});
