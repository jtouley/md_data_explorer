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

  it('handles missing table by still returning intent presentation', () => {
    const pres = buildQueryCompletedPresentation('describe', {});
    expect(pres.result).not.toBeNull();
    expect(pres.result.intentType).toBe('describe');
    expect(pres.content).toContain('Descriptive statistics');
  });

  it('builds descriptive summary from typed payload', () => {
    const pres = buildQueryCompletedPresentation('describe', {
      summary: { sample_size: 120, numeric_columns: ['age', 'bmi'] },
      table: { columns: ['metric', 'value'], rows: [{ metric: 'mean_age', value: 57.2 }] },
    });
    expect(pres.result?.summary).toContain('N=120');
    expect(pres.result?.summary).toContain('age');
  });

  it('builds comparison summary from typed payload', () => {
    const pres = buildQueryCompletedPresentation('compare_groups', {
      comparison: {
        group_by: 'sex',
        metric: 'mortality_rate',
        p_value: 0.031,
      },
      table: { columns: ['group', 'mean'], rows: [{ group: 'F', mean: 0.12 }] },
    });
    expect(pres.result?.summary).toContain('sex');
    expect(pres.result?.summary).toContain('p=');
  });

  it('builds predictor summary from typed payload', () => {
    const pres = buildQueryCompletedPresentation('find_predictors', {
      model: { target: 'mortality', top_predictors: ['age', 'creatinine'] },
      table: { columns: ['feature', 'odds_ratio'], rows: [{ feature: 'age', odds_ratio: 1.4 }] },
    });
    expect(pres.result?.summary).toContain('mortality');
    expect(pres.result?.summary).toContain('age');
  });

  it('builds survival summary from typed payload', () => {
    const pres = buildQueryCompletedPresentation('examine_survival', {
      survival: {
        time_column: 'days_to_event',
        event_column: 'death',
        median_survival_days: 42,
      },
      table: { columns: ['stratum', 'median_days'], rows: [{ stratum: 'all', median_days: 42 }] },
    });
    expect(pres.result?.summary).toContain('42');
    expect(pres.result?.summary).toContain('days_to_event');
  });

  it('builds relationship summary from typed payload', () => {
    const pres = buildQueryCompletedPresentation('explore_relationships', {
      correlations: {
        method: 'pearson',
        strongest_pair: ['age', 'bmi'],
        strongest_r: 0.81,
      },
      table: { columns: ['var_a', 'var_b', 'r'], rows: [{ var_a: 'age', var_b: 'bmi', r: 0.81 }] },
    });
    expect(pres.result?.summary).toContain('pearson');
    expect(pres.result?.summary).toContain('0.81');
  });
});

describe('buildCompletionMessage', () => {
  it('uses row count when table present', () => {
    expect(buildCompletionMessage('describe', 5, true)).toMatch(/5 rows/);
  });
});
