/** @vitest-environment jsdom */

import { describe, expect, it } from 'vitest';
import {
  formatRelativeTime,
  normalizeSessionListResponse,
  renderSessionList,
  sortSessionsNewestFirst,
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
  it('returns empty for falsy input', () => {
    expect(formatRelativeTime('')).toBe('');
    expect(formatRelativeTime(null)).toBe('');
  });

  it('returns "just now" for very recent timestamps', () => {
    const now = new Date().toISOString();
    expect(formatRelativeTime(now)).toBe('just now');
  });

  it('returns minutes for recent timestamps', () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    expect(formatRelativeTime(fiveMinAgo)).toBe('5m ago');
  });

  it('returns hours for older timestamps', () => {
    const twoHrsAgo = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(twoHrsAgo)).toBe('2h ago');
  });

  it('returns empty for invalid date string', () => {
    expect(formatRelativeTime('not-a-date')).toBe('');
  });

  it('returns days for multi-day timestamps', () => {
    const threeDaysAgo = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(threeDaysAgo)).toBe('3d ago');
  });

  it('returns locale date for 30+ day timestamps', () => {
    const sixtyDaysAgo = new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString();
    const result = formatRelativeTime(sixtyDaysAgo);
    expect(result).not.toBe('');
    expect(result).not.toContain('ago');
  });

  it('returns 1h at 60 minute boundary', () => {
    const exactlyOneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(exactlyOneHourAgo)).toBe('1h ago');
  });
});

describe('renderSessionList', () => {
  it('shows empty message when no sessions', () => {
    const ul = document.createElement('ul');
    renderSessionList(ul, [], null, { onSelect: () => {}, onDelete: () => {} });
    expect(ul.querySelector('.session-list-empty')).toBeTruthy();
    expect(ul.querySelector('.session-list-empty').textContent).toBe('No sessions yet.');
  });

  it('renders session items and marks active', () => {
    const ul = document.createElement('ul');
    const sessions = [
      { session_id: 's1', dataset_id: 'd1', message_count: 3, updated_at: '2024-01-01' },
      { session_id: 's2', dataset_id: 'd2', message_count: 0, updated_at: '2024-02-01' },
    ];
    renderSessionList(ul, sessions, 's2', { onSelect: () => {}, onDelete: () => {} });
    const items = ul.querySelectorAll('.session-item');
    expect(items).toHaveLength(2);
    const activeItem = ul.querySelector('.session-item--active');
    expect(activeItem).toBeTruthy();
    expect(activeItem.dataset.sessionId).toBe('s2');
  });

  it('calls onSelect when item is clicked', () => {
    const ul = document.createElement('ul');
    let selected = null;
    renderSessionList(ul, [{ session_id: 'x', updated_at: '2024-01-01' }], null, {
      onSelect: (id) => {
        selected = id;
      },
      onDelete: () => {},
    });
    ul.querySelector('.session-item').click();
    expect(selected).toBe('x');
  });

  it('calls onDelete when delete button is clicked', () => {
    const ul = document.createElement('ul');
    let deleted = null;
    renderSessionList(ul, [{ session_id: 'y', updated_at: '2024-01-01' }], null, {
      onSelect: () => {},
      onDelete: (id) => {
        deleted = id;
      },
    });
    ul.querySelector('.btn-session-delete').click();
    expect(deleted).toBe('y');
  });

  it('delete click does not trigger onSelect', () => {
    const ul = document.createElement('ul');
    let selected = null;
    let deleted = null;
    renderSessionList(ul, [{ session_id: 'z', updated_at: '2024-01-01' }], null, {
      onSelect: (id) => {
        selected = id;
      },
      onDelete: (id) => {
        deleted = id;
      },
    });
    ul.querySelector('.btn-session-delete').click();
    expect(deleted).toBe('z');
    expect(selected).toBeNull();
  });

  it('shows "Session" label when no dataset_id', () => {
    const ul = document.createElement('ul');
    renderSessionList(ul, [{ session_id: 's1', updated_at: '2024-01-01' }], null, {
      onSelect: () => {},
      onDelete: () => {},
    });
    const label = ul.querySelector('.session-item-label');
    expect(label.textContent).toBe('Session');
  });

  it('displays message count and timestamp in meta', () => {
    const ul = document.createElement('ul');
    const recent = new Date().toISOString();
    renderSessionList(
      ul,
      [{ session_id: 's1', dataset_id: 'd1', message_count: 5, updated_at: recent }],
      null,
      { onSelect: () => {}, onDelete: () => {} }
    );
    const meta = ul.querySelector('.session-item-meta');
    expect(meta.textContent).toContain('5 msgs');
    expect(meta.textContent).toContain('just now');
  });
});
