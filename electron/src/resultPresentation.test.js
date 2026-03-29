import { describe, expect, it } from 'vitest';
import {
  buildCompletionMessage,
  buildQueryCompletedPresentation,
  extractTableFromResultPreview,
  normalizeIntentType,
  resultTitleForIntent,
} from './resultPresentation.js';

describe('normalizeIntentType', () => {
  it('lowercases and maps hyphenated values', () => {
    expect(normalizeIntentType('DESCRIBE')).toBe('describe');
    expect(normalizeIntentType('Compare_Groups')).toBe('compare_groups');
  });

  it('returns unknown for empty input', () => {
    expect(normalizeIntentType(null)).toBe('unknown');
    expect(normalizeIntentType('')).toBe('unknown');
  });
});

describe('resultTitleForIntent', () => {
  it('maps describe to descriptive statistics label', () => {
    expect(resultTitleForIntent('describe')).toBe('Descriptive statistics');
  });

  it('returns null for unknown intent', () => {
    expect(resultTitleForIntent('unknown')).toBeNull();
  });
});

describe('extractTableFromResultPreview', () => {
  it('extracts flat table shape from API preview', () => {
    const preview = {
      table: {
        columns: ['a'],
        rows: [{ a: 1 }],
      },
      row_count: 10,
    };
    const got = extractTableFromResultPreview(preview);
    expect(got).not.toBeNull();
    expect(got.rowCount).toBe(10);
    expect(got.table.columns).toEqual(['a']);
  });

  it('extracts nested block with table key', () => {
    const preview = {
      metrics: {
        table: { columns: ['x'], rows: [{ x: 2 }] },
        row_count: 3,
      },
    };
    const got = extractTableFromResultPreview(preview);
    expect(got).not.toBeNull();
    expect(got.rowCount).toBe(3);
  });
});

describe('buildQueryCompletedPresentation', () => {
  it('produces titled result for describe intent and flat table', () => {
    const pres = buildQueryCompletedPresentation('describe', {
      table: { columns: ['m'], rows: [{ m: 1 }] },
      row_count: 1,
    });
    expect(pres.content).toContain('Descriptive statistics');
    expect(pres.result).not.toBeNull();
    expect(pres.result.title).toBe('Descriptive statistics');
    expect(pres.result.intentType).toBe('describe');
  });

  it('handles missing table', () => {
    const pres = buildQueryCompletedPresentation('describe', {});
    expect(pres.result).toBeNull();
    expect(pres.content).toContain('Descriptive statistics');
  });
});

describe('buildCompletionMessage', () => {
  it('uses row count when table present', () => {
    expect(buildCompletionMessage('describe', 5, true)).toMatch(/5 rows/);
  });
});
