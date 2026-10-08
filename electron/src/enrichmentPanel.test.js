import { describe, expect, it } from 'vitest';
import {
  normalizePendingResponse,
  suggestionToRows,
} from './enrichmentPanel.js';

describe('normalizePendingResponse', () => {
  it('returns empty suggestions for invalid payload', () => {
    expect(normalizePendingResponse(null)).toEqual({ suggestions: [], total: 0 });
    expect(normalizePendingResponse({})).toEqual({ suggestions: [], total: 0 });
  });

  it('uses API total when present', () => {
    const raw = {
      suggestions: [{ patch_id: 'a' }],
      total: 5,
    };
    expect(normalizePendingResponse(raw)).toEqual({
      suggestions: [{ patch_id: 'a' }],
      total: 5,
    });
  });

  it('falls back total to suggestions length', () => {
    const raw = {
      suggestions: [{ patch_id: 'x' }, { patch_id: 'y' }],
    };
    expect(normalizePendingResponse(raw).total).toBe(2);
  });
});

describe('suggestionToRows', () => {
  it('maps API fields to display rows', () => {
    const rows = suggestionToRows({
      operation: 'SET',
      column: 'age',
      suggested_value: 'numeric',
      current_value: null,
      confidence: 0.9,
      model_id: 'm1',
    });
    expect(rows).toContainEqual(['Operation', 'SET']);
    expect(rows).toContainEqual(['Column', 'age']);
    expect(rows).toContainEqual(['Suggested', 'numeric']);
    expect(rows).toContainEqual(['Current', '—']);
    expect(rows).toContainEqual(['Confidence', '0.9']);
    expect(rows).toContainEqual(['Model', 'm1']);
  });

  it('shows current value when set', () => {
    const rows = suggestionToRows({
      operation: 'X',
      column: 'c',
      suggested_value: 's',
      current_value: 'old',
      confidence: 0,
      model_id: 'z',
    });
    expect(rows).toContainEqual(['Current', 'old']);
  });
});
