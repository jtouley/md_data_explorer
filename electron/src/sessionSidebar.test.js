// SPDX-License-Identifier: MIT
/** @vitest-environment jsdom */

import { describe, expect, it, vi } from 'vitest';
import {
  UNTITLED_SESSION,
  ensureServerSession,
  formatRelativeTime,
  groupSessionsByDate,
  normalizeSessionListResponse,
  renderSessionList,
  sessionTitle,
  sortSessionsNewestFirst,
  titleFromQuestion,
} from './sessionSidebar.js';

describe('normalizeSessionListResponse', () => {
  it('returns empty for null/undefined', () => {
    expect(normalizeSessionListResponse(null)).toEqual({ sessions: [], total: 0 });
    expect(normalizeSessionListResponse(undefined)).toEqual({ sessions: [], total: 0 });
  });

  it('uses API total when present', () => {
    const raw = { sessions: [{ session_id: 's1' }], total: 5 };
    const result = normalizeSessionListResponse(raw);
    expect(result.total).toBe(5);
    expect(result.sessions).toHaveLength(1);
  });

  it('falls back total to length', () => {
    const raw = { sessions: [{ session_id: 'a' }, { session_id: 'b' }] };
    expect(normalizeSessionListResponse(raw).total).toBe(2);
  });

  it('treats non-array sessions as empty', () => {
    expect(normalizeSessionListResponse({ sessions: 'nope' })).toEqual({ sessions: [], total: 0 });
  });
});

describe('sortSessionsNewestFirst', () => {
  it('orders by updated_at descending', () => {
    const sessions = [
      { session_id: '1', updated_at: '2024-01-01' },
      { session_id: '3', updated_at: '2024-03-01' },
      { session_id: '2', updated_at: '2024-02-01' },
    ];
    const sorted = sortSessionsNewestFirst(sessions);
    expect(sorted.map((s) => s.session_id)).toEqual(['3', '2', '1']);
  });

  it('falls back to created_at', () => {
    const sessions = [
      { session_id: 'a', created_at: '2024-01-01' },
      { session_id: 'b', created_at: '2024-06-01' },
    ];
    expect(sortSessionsNewestFirst(sessions)[0].session_id).toBe('b');
  });

  it('does not mutate input', () => {
    const sessions = [{ updated_at: 'z' }, { updated_at: 'a' }];
    sortSessionsNewestFirst(sessions);
    expect(sessions[0].updated_at).toBe('z');
  });

  it('returns empty array for empty input', () => {
    expect(sortSessionsNewestFirst([])).toEqual([]);
  });

  it('prefers updated_at over created_at when both exist', () => {
    const sessions = [
      { session_id: 'old_created_new_updated', created_at: '2020-01-01', updated_at: '2025-01-01' },
      { session_id: 'new_created_old_updated', created_at: '2025-06-01', updated_at: '2023-01-01' },
    ];
    const sorted = sortSessionsNewestFirst(sessions);
    expect(sorted[0].session_id).toBe('old_created_new_updated');
  });
});

describe('formatRelativeTime', () => {
  const NOW = Date.parse('2026-06-15T12:00:00Z');
  const ago = (ms) => new Date(NOW - ms).toISOString();

  it.each([
    ['', ''],
    [null, ''],
    ['not-a-date', ''],
  ])('formatRelativeTime_invalid_%s_returnsEmpty', (input, expected) => {
    // Arrange / Act
    const result = formatRelativeTime(input, NOW);
    // Assert
    expect(result).toBe(expected);
  });

  it.each([
    [10 * 1000, 'just now'],
    [5 * 60 * 1000, '5 min ago'],
    [59 * 60 * 1000, '59 min ago'],
    [60 * 60 * 1000, '1 h ago'],
    [2 * 60 * 60 * 1000, '2 h ago'],
    [24 * 60 * 60 * 1000, '1 day ago'],
    [3 * 24 * 60 * 60 * 1000, '3 days ago'],
  ])('formatRelativeTime_%ims_returnsReadableText', (ms, expected) => {
    // Arrange / Act
    const result = formatRelativeTime(ago(ms), NOW);
    // Assert
    expect(result).toBe(expected);
  });

  it('formatRelativeTime_over30Days_returnsLocaleDate', () => {
    // Arrange / Act
    const result = formatRelativeTime(ago(60 * 24 * 60 * 60 * 1000), NOW);
    // Assert
    expect(result).not.toBe('');
    expect(result).not.toContain('ago');
  });
});

describe('titleFromQuestion', () => {
  it.each([
    ['Compare LDL by treatment', 'Compare LDL by treatment'],
    ['  spaced   out \n second line', 'spaced out'],
    ['', ''],
    [null, ''],
  ])('titleFromQuestion_%s_returnsFirstLine', (input, expected) => {
    // Arrange / Act
    const result = titleFromQuestion(input);
    // Assert
    expect(result).toBe(expected);
  });

  it('titleFromQuestion_longQuestion_truncatesTo80WithEllipsis', () => {
    // Arrange
    const long = 'a'.repeat(200);
    // Act
    const result = titleFromQuestion(long);
    // Assert
    expect(result).toHaveLength(80);
    expect(result.endsWith('…')).toBe(true);
  });
});

describe('sessionTitle', () => {
  it.each([
    [{ session_id: 's1', title: 'From the API' }, new Map([['s1', 'Local']]), 'From the API'],
    [{ session_id: 's1' }, new Map([['s1', 'Local']]), 'Local'],
    [{ session_id: 's1', title: '   ' }, undefined, UNTITLED_SESSION],
    [{ session_id: 's2' }, new Map([['s1', 'Local']]), UNTITLED_SESSION],
  ])('sessionTitle_case%#_prefersApiThenLocalThenUntitled', (session, titles, expected) => {
    // Arrange / Act
    const result = sessionTitle(session, titles);
    // Assert
    expect(result).toBe(expected);
  });
});

describe('groupSessionsByDate', () => {
  const NOW = new Date(2026, 5, 15, 12, 0, 0).getTime();
  const at = (days, hours = 0) => new Date(NOW - days * 864e5 - hours * 36e5).toISOString();

  it('groupSessionsByDate_mixedAges_groupsInFixedOrder', () => {
    // Arrange
    const sessions = [
      { session_id: 'old', updated_at: at(40) },
      { session_id: 'today', updated_at: at(0, 1) },
      { session_id: 'week', updated_at: at(4) },
      { session_id: 'yesterday', updated_at: at(1) },
    ];
    // Act
    const groups = groupSessionsByDate(sessions, NOW);
    // Assert
    expect(groups.map((g) => g.label)).toEqual(['Today', 'Yesterday', 'Previous 7 days', 'Older']);
    expect(groups.map((g) => g.sessions[0].session_id)).toEqual([
      'today',
      'yesterday',
      'week',
      'old',
    ]);
  });

  it.each([[[]], [[{ session_id: 'x', updated_at: 'nope' }]]])(
    'groupSessionsByDate_emptyOrInvalidDates_noEmptyGroups',
    (sessions) => {
      // Arrange / Act
      const groups = groupSessionsByDate(sessions, NOW);
      // Assert
      expect(groups.every((g) => g.sessions.length > 0)).toBe(true);
      if (sessions.length) expect(groups[0].label).toBe('Older');
    }
  );
});

describe('renderSessionList', () => {
  const handlers = (over = {}) => ({ onSelect: () => {}, onDelete: () => {}, ...over });

  it('renderSessionList_noSessions_showsEmptyMessage', () => {
    // Arrange
    const ul = document.createElement('ul');
    // Act
    renderSessionList(ul, [], null, handlers());
    // Assert
    expect(ul.querySelector('.session-list-empty').textContent).toBe(
      'Your conversations will appear here.'
    );
  });

  it('renderSessionList_activeSession_marksItemAndAriaCurrent', () => {
    // Arrange
    const ul = document.createElement('ul');
    const sessions = [
      { session_id: 's1', dataset_id: 'd1', updated_at: '2024-01-01' },
      { session_id: 's2', dataset_id: 'd2', updated_at: '2024-02-01' },
    ];
    // Act
    renderSessionList(ul, sessions, 's2', handlers());
    // Assert
    expect(ul.querySelectorAll('.session-item')).toHaveLength(2);
    const active = ul.querySelector('.session-item--active');
    expect(active.dataset.sessionId).toBe('s2');
    expect(active.querySelector('.session-item-main').getAttribute('aria-current')).toBe('true');
  });

  it('renderSessionList_clickRow_callsOnSelect', () => {
    // Arrange
    const ul = document.createElement('ul');
    let selected = null;
    renderSessionList(
      ul,
      [{ session_id: 'x', updated_at: '2024-01-01' }],
      null,
      handlers({ onSelect: (id) => (selected = id) })
    );
    // Act
    ul.querySelector('.session-item-main').click();
    // Assert
    expect(selected).toBe('x');
  });

  it('renderSessionList_rowIsAButton_keyboardReachable', () => {
    // Arrange
    const ul = document.createElement('ul');
    // Act
    renderSessionList(ul, [{ session_id: 'k', updated_at: '2024-01-01' }], null, handlers());
    // Assert
    const main = ul.querySelector('.session-item-main');
    expect(main.tagName).toBe('BUTTON');
    expect(main.type).toBe('button');
  });

  it('renderSessionList_deleteThenConfirm_callsOnDeleteOnce', () => {
    // Arrange
    const ul = document.createElement('ul');
    document.body.appendChild(ul);
    const deleted = [];
    let selected = null;
    renderSessionList(
      ul,
      [{ session_id: 'y', updated_at: '2024-01-01' }],
      null,
      handlers({ onDelete: (id) => deleted.push(id), onSelect: (id) => (selected = id) })
    );
    // Act
    ul.querySelector('.btn-session-delete').click();
    const confirmBtn = ul.querySelector('[data-testid="session-delete-confirm"]');
    confirmBtn.click();
    confirmBtn.click();
    // Assert
    expect(deleted).toEqual(['y']);
    expect(selected).toBeNull();
    ul.remove();
  });

  it('renderSessionList_deleteThenKeep_restoresRowWithoutDeleting', () => {
    // Arrange
    const ul = document.createElement('ul');
    document.body.appendChild(ul);
    const deleted = [];
    renderSessionList(
      ul,
      [{ session_id: 'y', updated_at: '2024-01-01' }],
      null,
      handlers({ onDelete: (id) => deleted.push(id) })
    );
    ul.querySelector('.btn-session-delete').click();
    // Act
    const keep = ul.querySelector('[data-testid="session-delete-cancel"]');
    expect(document.activeElement).toBe(keep);
    keep.click();
    // Assert
    expect(deleted).toEqual([]);
    expect(ul.querySelector('.session-item-main')).toBeTruthy();
    expect(ul.querySelector('.session-item--confirming')).toBeNull();
    expect(document.activeElement).toBe(ul.querySelector('.btn-session-delete'));
    ul.remove();
  });

  it('renderSessionList_escapeWhileConfirming_cancels', () => {
    // Arrange
    const ul = document.createElement('ul');
    const deleted = [];
    renderSessionList(
      ul,
      [{ session_id: 'y', updated_at: '2024-01-01' }],
      null,
      handlers({ onDelete: (id) => deleted.push(id) })
    );
    ul.querySelector('.btn-session-delete').click();
    // Act
    ul.querySelector('.session-item').dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })
    );
    // Assert
    expect(deleted).toEqual([]);
    expect(ul.querySelector('.session-item--confirming')).toBeNull();
  });

  it('renderSessionList_titlesAndNames_showFirstQuestionAndCohortName', () => {
    // Arrange
    const ul = document.createElement('ul');
    const now = Date.parse('2026-06-15T12:00:00Z');
    // Act
    renderSessionList(
      ul,
      [
        {
          session_id: 's1',
          dataset_id: 'd1',
          message_count: 5,
          updated_at: new Date(now).toISOString(),
        },
      ],
      null,
      handlers(),
      {
        titles: new Map([['s1', 'Compare LDL by treatment']]),
        datasetNames: new Map([['d1', 'cohort']]),
        now,
      }
    );
    // Assert
    expect(ul.querySelector('.session-item-label').textContent).toBe('Compare LDL by treatment');
    expect(ul.querySelector('.session-item-meta').textContent).toBe('cohort · just now');
    expect(ul.querySelector('.btn-session-delete').getAttribute('aria-label')).toBe(
      'Delete conversation: Compare LDL by treatment'
    );
  });

  it('renderSessionList_unknownTitleAndName_fallsBackToUntitledAndId', () => {
    // Arrange
    const ul = document.createElement('ul');
    // Act
    renderSessionList(
      ul,
      [{ session_id: 's1', dataset_id: 'd9', updated_at: '2024-01-01' }],
      null,
      handlers()
    );
    // Assert
    expect(ul.querySelector('.session-item-label').textContent).toBe(UNTITLED_SESSION);
    expect(ul.querySelector('.session-item-meta').textContent).toContain('d9');
  });

  it('renderSessionList_groups_rendersDateHeadings', () => {
    // Arrange
    const ul = document.createElement('ul');
    const now = new Date(2026, 5, 15, 12).getTime();
    // Act
    renderSessionList(
      ul,
      [
        { session_id: 'a', updated_at: new Date(now - 36e5).toISOString() },
        { session_id: 'b', updated_at: new Date(now - 40 * 864e5).toISOString() },
      ],
      null,
      handlers(),
      { now }
    );
    // Assert
    const headings = [...ul.querySelectorAll('.session-group-label')].map((h) => h.textContent);
    expect(headings).toEqual(['Today', 'Older']);
  });
});

describe('ensureServerSession: one session per conversation (round 2)', () => {
  const deferred = () => {
    let resolve;
    const promise = new Promise((r) => {
      resolve = r;
    });
    return { promise, resolve };
  };

  it('ensureServerSession_askAgainWhileTheFirstIsInFlight_createsOneSession', async () => {
    // Arrange: Stop stopped waiting for the first creation; the user asks again before it returns
    const pending = deferred();
    const api = { createSession: vi.fn(() => pending.promise) };
    const state = {
      currentDatasetId: 'ds',
      sessionId: 'local',
      sessionOnServer: false,
      sessionCreating: null,
    };

    // Act
    const first = ensureServerSession(state, api);
    const second = ensureServerSession(state, api);
    pending.resolve({ session_id: 'server-1' });
    await Promise.all([first, second]);

    // Assert
    expect(api.createSession).toHaveBeenCalledTimes(1);
    expect(state.sessionId).toBe('server-1');
    expect(state.sessionOnServer).toBe(true);
    expect(state.sessionCreating).toBeNull();
  });

  it('ensureServerSession_conversationResetWhileInFlight_staleSessionIsIgnored', async () => {
    // Arrange
    const pending = deferred();
    const api = { createSession: vi.fn(() => pending.promise) };
    const state = {
      currentDatasetId: 'ds',
      sessionId: 'local',
      sessionOnServer: false,
      sessionCreating: null,
    };
    const first = ensureServerSession(state, api);

    // Act: New chat or another cohort drops the in-flight creation
    state.sessionCreating = null;
    state.sessionId = 'local-2';
    pending.resolve({ session_id: 'server-1' });
    await first;

    // Assert
    expect(state.sessionId).toBe('local-2');
    expect(state.sessionOnServer).toBe(false);
  });

  it.each([
    ['rejects', () => Promise.reject(new Error('offline'))],
    ['returns no id', () => Promise.resolve({})],
  ])('ensureServerSession_creation%s_allowsATryNextTime', async (_name, create) => {
    // Arrange
    const api = { createSession: vi.fn(create) };
    const state = {
      currentDatasetId: 'ds',
      sessionId: 'local',
      sessionOnServer: false,
      sessionCreating: null,
    };

    // Act
    await ensureServerSession(state, api, { warn: () => {} });
    await ensureServerSession(state, api, { warn: () => {} });

    // Assert
    expect(api.createSession).toHaveBeenCalledTimes(2);
    expect(state.sessionOnServer).toBe(false);
    expect(state.sessionCreating).toBeNull();
  });

  it('ensureServerSession_alreadyOnServer_createsNothing', async () => {
    // Arrange
    const api = { createSession: vi.fn() };

    // Act
    await ensureServerSession({ sessionOnServer: true, sessionCreating: null }, api);

    // Assert
    expect(api.createSession).not.toHaveBeenCalled();
  });
});

describe('heading levels (round 2)', () => {
  it('renderSessionList_dateGroups_areH2UnderTheAppH1', () => {
    // Arrange
    const ul = document.createElement('ul');
    const sessions = [{ session_id: 's1', dataset_id: 'd1', updated_at: '2024-01-01' }];

    // Act
    renderSessionList(ul, sessions, null, { onSelect: () => {}, onDelete: () => {} });

    // Assert: the sidebar's h1 is followed by h2, never a skipped level
    expect(ul.querySelector('.session-group-label').tagName).toBe('H2');
  });
});
