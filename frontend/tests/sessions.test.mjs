/**
 * frontend/tests/sessions.test.mjs
 * Validates conversation session creation, naming, and deletion logic.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert';

describe('Chat Session Management Logic Tests', () => {
  it('auto-titles session based on the first query (truncating to 36 chars)', () => {
    const formatSessionTitle = (query) => {
      const trimmed = query.trim();
      return trimmed.slice(0, 36) + (trimmed.length > 36 ? '...' : '');
    };

    const shortQuery = 'कलाम का जन्म कब हुआ?';
    assert.strictEqual(formatSessionTitle(shortQuery), 'कलाम का जन्म कब हुआ?');

    const longQuery = 'Which institution did Kalam attend to study aeronautical engineering in Chennai?';
    assert.strictEqual(
      formatSessionTitle(longQuery),
      'Which institution did Kalam attend t...'
    );
  });

  it('handles session deletion and always maintains at least one active session', () => {
    let sessions = [
      { id: 's1', title: 'Session 1' },
      { id: 's2', title: 'Session 2' },
    ];
    let activeId = 's1';

    // Delete active session s1
    sessions = sessions.filter((s) => s.id !== 's1');
    if (activeId === 's1') {
      activeId = sessions[0].id;
    }

    assert.strictEqual(sessions.length, 1);
    assert.strictEqual(activeId, 's2');

    // Delete remaining session s2
    sessions = sessions.filter((s) => s.id !== 's2');
    if (sessions.length === 0) {
      const fresh = { id: 's-new', title: 'New Conversation' };
      sessions.push(fresh);
      activeId = fresh.id;
    }

    assert.strictEqual(sessions.length, 1);
    assert.strictEqual(activeId, 's-new');
  });
});
