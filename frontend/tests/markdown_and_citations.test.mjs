/**
 * frontend/tests/markdown_and_citations.test.mjs
 * Validates markdown inline parsing and citation marker extraction logic.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert';

describe('Markdown & Citation Tag Parsing Tests', () => {
  it('correctly extracts Devanagari and chunk ID citation tags', () => {
    const text = '18 जुलाई 1980 को रोहिणी उपग्रह स्थापित किया गया [doc_p8_c3]। संदर्भ [Page 8] भी देखें।';
    const pattern = /(\[doc_p\d+_c\d+\]|\[(?:Page|पृष्ठ)\s*\d+\])/g;

    const matches = [...text.matchAll(pattern)].map((m) => m[0]);
    assert.deepStrictEqual(matches, ['[doc_p8_c3]', '[Page 8]']);
  });

  it('detects Devanagari script in mixed text', () => {
    const isDevanagari = (str) => /[\u0900-\u097F]/.test(str);

    assert.strictEqual(isDevanagari('कलाम'), true);
    assert.strictEqual(isDevanagari('Dr. A.P.J. Abdul Kalam'), false);
    assert.strictEqual(isDevanagari('SLV-III उपग्रह'), true);
    assert.strictEqual(isDevanagari('Pokhran-II nuclear test'), false);
  });

  it('correctly parses bold and inline code', () => {
    const text = 'डॉ. कलाम को **भारत रत्न** वर्ष `1997` में प्राप्त हुआ।';
    const boldMatch = text.match(/\*\*([^*]+)\*\*/);
    const codeMatch = text.match(/`([^`]+)`/);

    assert.ok(boldMatch);
    assert.strictEqual(boldMatch[1], 'भारत रत्न');
    assert.ok(codeMatch);
    assert.strictEqual(codeMatch[1], '1997');
  });
});
