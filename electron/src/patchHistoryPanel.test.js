/** @vitest-environment jsdom */

import { describe, expect, it } from 'vitest';
import {
  canRevertPatch,
  normalizeHistoryResponse,
  renderPatchHistoryTable,
  sortPatchesNewestFirst,
} from './patchHistoryPanel.js';

describe('normalizeHistoryResponse', () => {
  it('returns empty patches for invalid payload', () => {
    expect(normalizeHistoryResponse(null)).toEqual({ patches: [], total: 0 });
    expect(normalizeHistoryResponse({})).toEqual({ patches: [], total: 0 });
  });

  it('uses API total when present', () => {
    const raw = {
      patches: [{ patch_id: 'a' }],
      total: 3,
    };
    expect(normalizeHistoryResponse(raw)).toEqual({
      patches: [{ patch_id: 'a' }],
      total: 3,
    });
  });

  it('falls back total to patches length', () => {
    const raw = {
      patches: [{ patch_id: 'x' }, { patch_id: 'y' }],
    };
    expect(normalizeHistoryResponse(raw).total).toBe(2);
  });

  it('treats non-array patches as empty', () => {
    expect(normalizeHistoryResponse({ patches: 'bad' })).toEqual({ patches: [], total: 0 });
    expect(normalizeHistoryResponse({ patches: 42 })).toEqual({ patches: [], total: 0 });
  });
});

describe('sortPatchesNewestFirst', () => {
  it('orders by created_at descending', () => {
    const patches = [
      { patch_id: '1', created_at: '2024-01-01T00:00:00Z' },
      { patch_id: '3', created_at: '2024-03-01T00:00:00Z' },
      { patch_id: '2', created_at: '2024-02-01T00:00:00Z' },
    ];
    const sorted = sortPatchesNewestFirst(patches);
    expect(sorted.map((p) => p.patch_id)).toEqual(['3', '2', '1']);
  });

  it('does not mutate input', () => {
    const patches = [{ created_at: 'b' }, { created_at: 'a' }];
    sortPatchesNewestFirst(patches);
    expect(patches[0].created_at).toBe('b');
  });

  it('returns empty array for empty input', () => {
    expect(sortPatchesNewestFirst([])).toEqual([]);
  });

  it('handles null created_at gracefully', () => {
    const patches = [
      { patch_id: 'a', created_at: null },
      { patch_id: 'b', created_at: '2024-06-01' },
    ];
    const sorted = sortPatchesNewestFirst(patches);
    expect(sorted[0].patch_id).toBe('b');
  });
});

describe('canRevertPatch', () => {
  it('allows accepted without reverted_at', () => {
    expect(canRevertPatch({ status: 'ACCEPTED', patch_id: 'p1', reverted_at: null })).toBe(true);
  });

  it('rejects non-accepted', () => {
    expect(canRevertPatch({ status: 'PENDING', patch_id: 'p1' })).toBe(false);
  });

  it('rejects when reverted_at set', () => {
    expect(
      canRevertPatch({
        status: 'ACCEPTED',
        patch_id: 'p1',
        reverted_at: '2024-01-02T00:00:00Z',
      })
    ).toBe(false);
  });

  it('returns false for null/undefined input', () => {
    expect(canRevertPatch(null)).toBe(false);
    expect(canRevertPatch(undefined)).toBe(false);
  });

  it('is case-insensitive for accepted status', () => {
    expect(canRevertPatch({ status: 'accepted', patch_id: 'p2' })).toBe(true);
    expect(canRevertPatch({ status: 'Accepted', patch_id: 'p3' })).toBe(true);
  });

  it('rejects REJECTED status', () => {
    expect(canRevertPatch({ status: 'REJECTED', patch_id: 'p4' })).toBe(false);
  });
});

describe('renderPatchHistoryTable', () => {
  it('shows empty message when no patches', () => {
    const container = document.createElement('div');
    renderPatchHistoryTable(container, [], { onRevert: () => {} });
    expect(container.querySelector('.patch-history-empty')).toBeTruthy();
  });

  it('renders revert for revertable patch and calls handler', () => {
    const container = document.createElement('div');
    let called = null;
    renderPatchHistoryTable(
      container,
      [
        {
          patch_id: 'abc',
          column: 'age',
          operation: 'SET',
          value: '1',
          status: 'ACCEPTED',
          created_at: '2024-01-01',
        },
      ],
      {
        onRevert: (id) => {
          called = id;
        },
      }
    );
    const btn = container.querySelector('button.btn-patch-revert');
    expect(btn).toBeTruthy();
    btn.click();
    expect(called).toBe('abc');
  });

  it('does not render revert button for non-revertable patch', () => {
    const container = document.createElement('div');
    renderPatchHistoryTable(
      container,
      [
        {
          patch_id: 'r1',
          column: 'col',
          operation: 'SET',
          value: 'v',
          status: 'REJECTED',
          created_at: '2024-01-01',
        },
      ],
      { onRevert: () => {} }
    );
    expect(container.querySelector('button.btn-patch-revert')).toBeNull();
  });

  it('renders correct number of rows and column headers', () => {
    const container = document.createElement('div');
    renderPatchHistoryTable(
      container,
      [
        {
          patch_id: 'p1',
          column: 'a',
          operation: 'SET',
          value: '1',
          status: 'ACCEPTED',
          created_at: '2024-02-01',
        },
        {
          patch_id: 'p2',
          column: 'b',
          operation: 'DEL',
          value: '2',
          status: 'PENDING',
          created_at: '2024-01-01',
        },
      ],
      { onRevert: () => {} }
    );
    const rows = container.querySelectorAll('tbody tr');
    expect(rows).toHaveLength(2);
    const headers = container.querySelectorAll('thead th');
    expect(headers).toHaveLength(6);
    expect(headers[0].textContent).toBe('Column');
    expect(headers[3].textContent).toBe('Status');
  });

  it('escapes HTML in cell values', () => {
    const container = document.createElement('div');
    renderPatchHistoryTable(
      container,
      [
        {
          patch_id: 'xss',
          column: '<script>alert(1)</script>',
          operation: 'SET',
          value: 'v',
          status: 'PENDING',
          created_at: '2024-01-01',
        },
      ],
      { onRevert: () => {} }
    );
    const firstCell = container.querySelector('tbody td');
    expect(firstCell.textContent).toBe('<script>alert(1)</script>');
    expect(firstCell.innerHTML).not.toContain('<script>');
  });
});
