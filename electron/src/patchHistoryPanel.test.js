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
});
