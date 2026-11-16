// SPDX-License-Identifier: MIT
import { describe, expect, it } from 'vitest';
import {
  HIDDEN_CELL_FOOTNOTE,
  SMALL_CELL_THRESHOLD,
  buildCompletionMessage,
  buildConfirmationHoldPresentation,
  buildExampleQuestions,
  buildQueryCompletedPresentation,
  claimDecimals,
  classifyCell,
  columnLabel,
  columnWithLabel,
  describeQueryFailure,
  formatNumber,
  isPValueColumn,
  extractTableFromResultPreview,
  isNumericColumn,
  normalizeIntentType,
  numbersIn,
  parsePlanLine,
  plainFailureMessage,
  resultTitleForIntent,
  splitClaim,
  stageLabel,
  ENGINE_UNREACHABLE_MESSAGE,
  TABLE_LEVEL_LIMIT,
  holdNextStep,
  isFullyHiddenTable,
  needsHiddenFootnote,
  plainLevelLabel,
  suggestColumns,
  tableLevelCount,
  httpFailureText,
  isGroupLevelTable,
  buildIntentSummary,
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

describe('buildQueryCompletedPresentation with API claim', () => {
  const claimPreview = {
    analysis_type: 'comparison',
    claim:
      'Mean ldl was 16.14 lower in drug than in placebo (95% CI -18.83 to -13.44; t-test p<0.001; n=300).',
    table: {
      columns: ['treatment', 'mean ldl', 'n'],
      rows: [
        { treatment: 'drug', 'mean ldl': 113.96, n: 160 },
        { treatment: 'placebo', 'mean ldl': 130.09, n: 140 },
      ],
    },
    row_count: 2,
  };

  it('leads with the claim and keeps the statistics table', () => {
    const pres = buildQueryCompletedPresentation('compare_groups', claimPreview);
    expect(pres.content).toBe('Group comparison');
    expect(pres.result.summary).toBe(claimPreview.claim);
    expect(pres.result.table).toEqual(claimPreview.table);
  });

  it('falls back to a generic heading for an unknown intent', () => {
    const pres = buildQueryCompletedPresentation('mystery', {
      claim: 'Total count: 3.',
      table: null,
    });
    expect(pres.content).toBe('Analysis complete.');
    expect(pres.result.summary).toBe('Total count: 3.');
  });

  it('ignores an empty claim and uses the row-count message', () => {
    const pres = buildQueryCompletedPresentation('count', {
      claim: '',
      table: { columns: ['n'], rows: [{ n: 1 }] },
    });
    expect(pres.content).toBe('Counts: 1 row (preview)');
  });
});

describe('buildCompletionMessage', () => {
  it('uses row count when table present', () => {
    expect(buildCompletionMessage('describe', 5, true)).toMatch(/5 rows/);
  });
});

describe('buildQueryCompletedPresentation claim-first', () => {
  it('prefers claim arg over preview and attaches claim on result', () => {
    const claim = 'Mean ldl was 5.0 higher in Female than Male.';
    const pres = buildQueryCompletedPresentation(
      'compare_groups',
      { table: { columns: ['g'], rows: [{ g: 1 }] }, row_count: 1 },
      claim
    );
    expect(pres.content).toBe('Group comparison');
    expect(pres.result?.summary).toBe(claim);
    expect(pres.result?.claim).toBe(claim);
  });
});

describe('buildConfirmationHoldPresentation', () => {
  it('surfaces plan line and pending query for Confirm', () => {
    const hold = buildConfirmationHoldPresentation(
      {
        plan_line:
          'COMPARE_GROUPS · outcome ldl · grouping sex · filters none · test group comparison',
        failure_reason: 'Confirmation required before execution.',
        confidence: 0.4,
      },
      'compare ldl by sex'
    );
    expect(hold.result.awaitingConfirmation).toBe(true);
    expect(hold.result.pendingQuery).toBe('compare ldl by sex');
    expect(hold.result.planLine).toContain('COMPARE_GROUPS');
    expect(hold.content).toContain('Confirmation required');
  });

  it('keeps the plan line out of the message text so it renders once', () => {
    const hold = buildConfirmationHoldPresentation(
      {
        plan_line: 'DESCRIBE · outcome age',
        failure_reason: 'Confirmation required before execution.',
      },
      'average age'
    );
    expect(hold.content).toBe('Confirmation required before execution.');
    expect(hold.content).not.toContain('DESCRIBE');
  });

  it('records the dataset and session the plan was held for', () => {
    const hold = buildConfirmationHoldPresentation({ plan_line: 'p' }, 'q', {
      datasetId: 'ds_a',
      sessionId: 'sess_1',
    });
    expect(hold.result.datasetId).toBe('ds_a');
    expect(hold.result.sessionId).toBe('sess_1');
  });

  it('keeps the plan hash so Confirm can bind to the exact plan', () => {
    const hold = buildConfirmationHoldPresentation({ plan_line: 'p', plan_hash: 'abc123' }, 'q');
    expect(hold.result.planHash).toBe('abc123');
  });

  it('leaves the binding empty when none is given', () => {
    const hold = buildConfirmationHoldPresentation({ plan_line: 'p' }, 'q');
    expect(hold.result.datasetId).toBeNull();
    expect(hold.result.sessionId).toBeNull();
  });
});

// Claims below are copied from tests/core/test_result_presentation.py (the API's exact output).
const API_CLAIMS = {
  tTest:
    'Mean ldl was 16.14 higher in placebo than drug (95% CI 13.44 to 18.83; t-test p<0.001; n=300). ' +
    'Unadjusted comparison by treatment.',
  notSignificant:
    'Mean ldl was 2.50 higher in drug than placebo, not statistically significant (t-test p=0.200; n=300). ' +
    'Unadjusted comparison by treatment.',
  anova: 'Mean ldl differed across 3 site groups (ANOVA p=0.030). Unadjusted.',
  predictor:
    'age had the strongest evidence of association with outcome (OR 1.05; 95% CI 1.02 to 1.08; p=0.001). ' +
    'Logistic regression of outcome on age; unadjusted.',
  noPredictor:
    'No predictor reached p<0.05 for outcome. Logistic regression of outcome on hdl; unadjusted.',
  correlation: 'The strongest correlation was moderate: age and ldl (r=-0.42; n=250).',
  describe: 'Mean ldl was 121.49 (SD 14.40; median 121.50; n=300).',
  smallExclusion:
    'Mean ldl was 16.14 higher in placebo than drug (95% CI 13.44 to 18.83; Welch t-test p<0.001; ' +
    'n hidden). Unadjusted comparison by treatment.',
  hiddenExclusions:
    'Mean ldl was 16.14 higher in placebo than drug (95% CI 13.44 to 18.83; Welch t-test p<0.001; ' +
    'n=377; some excluded for missing data). Unadjusted comparison by treatment.',
  hiddenTopCount: 'Most common smoker: no (count hidden).',
  hiddenSurvivalMedian:
    'Median time to event=1 (days) is not shown because n is hidden (n hidden).',
  hiddenDescriptive: 'Statistics for ldl are not shown because n is hidden (n hidden).',
  missingNSurvivalMedian: 'Median time to event=1 (days) is not shown because n is missing.',
  missingNDescriptive: 'Statistics for ldl are not shown because n is missing.',
  count: 'Total count: 400.',
};

describe('splitClaim', () => {
  it.each([
    [
      'tTest',
      'Mean ldl was 16.14 higher in placebo than drug.',
      ['95% CI 13.44 to 18.83', 't-test p<0.001', 'n=300'],
      'Unadjusted comparison by treatment.',
    ],
    [
      'notSignificant',
      'Mean ldl was 2.50 higher in drug than placebo, not statistically significant.',
      ['t-test p=0.200', 'n=300'],
      'Unadjusted comparison by treatment.',
    ],
    ['anova', 'Mean ldl differed across 3 site groups.', ['ANOVA p=0.030'], 'Unadjusted.'],
    [
      'predictor',
      'age had the strongest evidence of association with outcome.',
      ['OR 1.05', '95% CI 1.02 to 1.08', 'p=0.001'],
      'Logistic regression of outcome on age; unadjusted.',
    ],
    [
      'noPredictor',
      'No predictor reached p<0.05 for outcome.',
      [],
      'Logistic regression of outcome on hdl; unadjusted.',
    ],
    [
      'correlation',
      'The strongest correlation was moderate: age and ldl.',
      ['r=-0.42', 'n=250'],
      '',
    ],
    ['describe', 'Mean ldl was 121.49.', ['SD 14.40', 'median 121.50', 'n=300'], ''],
    [
      'smallExclusion',
      'Mean ldl was 16.14 higher in placebo than drug.',
      ['95% CI 13.44 to 18.83', 'Welch t-test p<0.001', 'n hidden'],
      'Unadjusted comparison by treatment.',
    ],
    [
      'hiddenExclusions',
      'Mean ldl was 16.14 higher in placebo than drug.',
      ['95% CI 13.44 to 18.83', 'Welch t-test p<0.001', 'n=377', 'some excluded for missing data'],
      'Unadjusted comparison by treatment.',
    ],
    ['hiddenTopCount', 'Most common smoker: no (count hidden).', [], ''],
    [
      'hiddenSurvivalMedian',
      'Median time to event=1 (days) is not shown because n is hidden.',
      ['n hidden'],
      '',
    ],
    [
      'hiddenDescriptive',
      'Statistics for ldl are not shown because n is hidden.',
      ['n hidden'],
      '',
    ],
    [
      'missingNSurvivalMedian',
      'Median time to event=1 (days) is not shown because n is missing.',
      [],
      '',
    ],
    ['missingNDescriptive', 'Statistics for ldl are not shown because n is missing.', [], ''],
    ['count', 'Total count: 400.', [], ''],
  ])('splitClaim_%s_headlineStatsNotes', (key, headline, stats, notes) => {
    // Arrange
    const claim = API_CLAIMS[key];
    // Act
    const result = splitClaim(claim);
    // Assert
    expect(result).toEqual({ headline, stats, notes });
  });

  it.each(Object.keys(API_CLAIMS))('splitClaim_%s_keepsEveryNumber', (key) => {
    // Arrange
    const claim = API_CLAIMS[key];
    // Act
    const { headline, stats, notes } = splitClaim(claim);
    // Assert
    expect(numbersIn([headline, ...stats, notes].join(' '))).toEqual(numbersIn(claim));
  });

  it.each([
    ['', { headline: '', stats: [], notes: '' }],
    [null, { headline: '', stats: [], notes: '' }],
    [undefined, { headline: '', stats: [], notes: '' }],
    ['   ', { headline: '', stats: [], notes: '' }],
  ])('splitClaim_empty_%s_returnsEmptyParts', (claim, expected) => {
    // Arrange / Act
    const result = splitClaim(claim);
    // Assert
    expect(result).toEqual(expected);
  });

  it('splitClaim_parenthesisWithoutStats_staysInHeadline', () => {
    // Arrange
    const claim =
      'Median time to death (days) by arm: A 12.0, B not shown (fewer than 11 patients) (log-rank p=0.040; n=200).';
    // Act
    const { headline, stats } = splitClaim(claim);
    // Assert
    expect(headline).toBe(
      'Median time to death (days) by arm: A 12.0, B not shown (fewer than 11 patients).'
    );
    expect(stats).toEqual(['log-rank p=0.040', 'n=200']);
  });

  it.each([
    ['twoStatsGroups', 'LDL was higher in placebo (p=0.03), but not after adjustment (p=0.20).'],
    ['statsGroupMidSentence', 'LDL was higher (p=0.03) in placebo than drug.'],
    [
      'twoStatsGroupsWithNotes',
      'Age differed (n=200) and so did sex (n=198). Unadjusted comparison.',
    ],
  ])('splitClaim_%s_keepsHeadlineWhole', (_key, claim) => {
    // Arrange
    const sentence = claim.split('. ')[0].replace(/\.?$/, '.');
    // Act
    const { headline, stats } = splitClaim(claim);
    // Assert
    expect(stats).toEqual([]);
    expect(headline.startsWith(sentence.slice(0, -1))).toBe(true);
  });

  it('splitClaim_unbalancedParenthesis_neverDropsText', () => {
    // Arrange
    const claim = 'Mean x was 3 (n=10; p=0.2';
    // Act
    const { headline, stats, notes } = splitClaim(claim);
    // Assert
    expect([headline, ...stats, notes].join(' ')).toContain('n=10; p=0.2');
  });
});

describe('classifyCell', () => {
  it.each([
    [null, 'empty', ''],
    [undefined, 'empty', ''],
    ['', 'empty', ''],
    [0, 'number', '0'],
    [12.5, 'number', '12.50'],
    [Number.NaN, 'text', 'NaN'],
    ['suppressed', 'hidden', 'hidden'],
    ['Suppressed', 'text', 'Suppressed'],
    ['<11', 'small', '<11'],
    ['< 11', 'small', '<11'],
    ['1,234', 'number', '1,234'],
    ['31%', 'number', '31%'],
    ['-0.42', 'number', '-0.42'],
    ['placebo', 'text', 'placebo'],
    [{ a: 1 }, 'text', '{"a":1}'],
  ])('classifyCell_%s_is_%s', (value, kind, text) => {
    // Arrange / Act
    const result = classifyCell(value);
    // Assert
    expect(result).toEqual({ kind, text });
  });

  it.each([
    ['<65', '<65'],
    ['<18', '<18'],
    ['<5', '<5'],
    ['< 10', '< 10'],
    ['<110', '<110'],
    ['<12', '<12'],
    ['<011', '<011'],
    ['< 011', '< 011'],
  ])('classifyCell_inequalityLabel_%s_staysText', (value, text) => {
    // Arrange / Act
    const result = classifyCell(value);
    // Assert: only the API's own threshold token means "hidden"
    expect(result).toEqual({ kind: 'text', text });
  });

  it('classifyCell_thresholdToken_isBuiltFromThreshold', () => {
    // Arrange
    const token = `<${SMALL_CELL_THRESHOLD}`;
    // Act / Assert: the token tracks the threshold; one below it is a label
    expect(classifyCell(token)).toEqual({ kind: 'small', text: token });
    expect(classifyCell(`<${SMALL_CELL_THRESHOLD - 1}`).kind).toBe('text');
  });

  it.each([
    ['0', '<0.001'],
    ['1', '1'],
    [' 0 ', '<0.001'],
  ])('classifyCell_pValueIntegerString_%s_isFormatted', (value, text) => {
    // Arrange / Act
    const result = classifyCell(value, { pValue: true });
    // Assert: a stringified p of 0 is never shown as 0
    expect(result).toEqual({ kind: 'number', text });
  });

  it('hiddenCellFootnote_usesThreshold11_neverFive', () => {
    // Arrange / Act / Assert
    expect(SMALL_CELL_THRESHOLD).toBe(11);
    expect(HIDDEN_CELL_FOOTNOTE).toContain('from 1 to 10');
    expect(HIDDEN_CELL_FOOTNOTE).not.toMatch(/<\s*5|fewer than 5/);
  });
});

describe('formatNumber', () => {
  it.each([
    [64.23400000000001, {}, '64.23'],
    [64.23400000000001, { decimals: 1 }, '64.2'],
    [190, {}, '190'],
    [0, {}, '0'],
    [-0.4251, {}, '-0.43'],
    [0.000123, {}, '0.00012'],
    [12.5, { decimals: 3 }, '12.500'],
    [64.00001, { decimals: 1 }, '64.0'],
    [0.00004, { pValue: true }, '<0.001'],
    [0.0412, { pValue: true }, '0.041'],
    [0.12, { pValue: true, decimals: 1 }, '0.120'],
    [0.001, { pValue: true }, '0.001'],
    [0, { pValue: true }, '<0.001'],
    [1, { pValue: true }, '1'],
  ])('formatNumber_%s_%o_is_%s', (value, opts, expected) => {
    // Arrange / Act
    const result = formatNumber(value, opts);
    // Assert
    expect(result).toBe(expected);
  });

  it.each([
    ['Mean was 16.14 higher (p<0.001; n=377).', 2],
    ['Mean ldl was 16.97 higher (95% CI 13.96 to 19.99; Welch t-test p<0.001; n=377).', 2],
    ['No difference (chi-square p=0.120; n=388).', 2],
    ['Mean age was 64.2 (n=190).', 1],
    ['Count was 12 (n=12).', 2],
    ['', 2],
    [null, 2],
    ['Value 1.23456.', 3],
  ])('claimDecimals_%s_is_%s', (claim, expected) => {
    // Arrange / Act / Assert
    expect(claimDecimals(claim)).toBe(expected);
  });

  it.each([
    ['p', true],
    ['p_value', true],
    ['P value', true],
    ['pvalue', true],
    ['adj_p_value', true],
    ['pct_female', false],
    ['peak', false],
    ['', false],
  ])('isPValueColumn_%s_is_%s', (name, expected) => {
    // Arrange / Act / Assert
    expect(isPValueColumn(name)).toBe(expected);
  });

  it.each([
    [64.23400000000001, { decimals: 1 }, 'number', '64.2'],
    ['64.23400000000001', { decimals: 1 }, 'number', '64.2'],
    ['<11', { decimals: 1 }, 'small', '<11'],
    ['suppressed', { decimals: 1 }, 'hidden', 'hidden'],
    ['1,234', { decimals: 1 }, 'number', '1,234'],
  ])('classifyCell_withFormat_%s_is_%s', (value, format, kind, text) => {
    // Arrange / Act
    const result = classifyCell(value, format);
    // Assert
    expect(result).toEqual({ kind, text });
  });
});

describe('columnLabel', () => {
  it.each([
    ['ldl_mg_dl', 'LDL (mg/dL)', 'LDL (mg/dL) — ldl_mg_dl'],
    ['first_visit', 'First visit', 'First visit — first_visit'],
    ['age', 'Age', 'Age'],
    ['hba1c_pct', 'HbA1c (%)', 'HbA1c (%) — hba1c_pct'],
    ['sbp_mm_hg', 'SBP (mmHg)', 'SBP (mmHg) — sbp_mm_hg'],
    ['bmi_kg_m2', 'BMI (kg/m²)', 'BMI (kg/m²) — bmi_kg_m2'],
    ['mg_dl', 'Mg dl', 'Mg dl — mg_dl'],
    ['', '', ''],
  ])('columnLabel_%s', (column, label, withName) => {
    // Arrange / Act / Assert
    expect(columnLabel(column)).toBe(label);
    expect(columnWithLabel(column)).toBe(withName);
  });
});

describe('isNumericColumn', () => {
  it.each([
    [[[1], [2]], 0, true],
    [[['suppressed'], [3]], 0, true],
    [[['<11'], ['suppressed']], 0, true],
    [[['<65'], ['<18']], 0, false],
    [[[''], [null]], 0, false],
    [[['drug'], [3]], 0, false],
    [[], 0, false],
  ])('isNumericColumn_case%#', (rows, index, expected) => {
    // Arrange / Act
    const result = isNumericColumn(rows, index);
    // Assert
    expect(result).toBe(expected);
  });
});

describe('parsePlanLine', () => {
  it('parsePlanLine_fullApiLine_readableLabels', () => {
    // Arrange
    const line =
      'COMPARE_GROUPS · outcome ldl · grouping treatment · filters age > 65 · test Welch t-test';
    // Act
    const result = parsePlanLine(line);
    // Assert
    expect(result).toEqual({
      analysis: 'Compare groups',
      items: [
        { key: 'outcome', label: 'Outcome', value: 'LDL' },
        { key: 'grouping', label: 'Compared across', value: 'Treatment' },
        { key: 'included', label: 'Included', value: 'Age over 65' },
        { key: 'counting', label: 'Counting', value: 'Rows' },
        { key: 'test', label: 'Test', value: 'Welch t-test' },
      ],
    });
  });

  it.each([
    ['==', 'is'],
    ['!=', 'is not'],
    ['>', 'over'],
    ['>=', 'at least'],
    ['<', 'under'],
    ['<=', 'at most'],
  ])('parsePlanLine_operator_%s_is_%s', (operator, word) => {
    // Arrange / Act
    const { items } = parsePlanLine(`DESCRIBE · filters ldl_mg_dl ${operator} 130`);
    // Assert
    expect(items[0].value).toBe(`LDL (mg/dL) (ldl_mg_dl) ${word} 130`);
  });

  it.each([
    ['sex IN [F, M]', 'Sex is one of F, M'],
    ['sex NOT_IN [F]', 'Sex is not one of F'],
    [
      'first_visit >= 2024-01-01 (missing kept)',
      'First visit (first_visit) at least 2024-01-01; patients with no value are kept',
    ],
    ['age > 65; sex == F', 'Age over 65 and Sex is F'],
    ['odd text', 'odd text'],
  ])('parsePlanLine_filters_%s', (filters, expected) => {
    // Arrange / Act
    const { items } = parsePlanLine(`DESCRIBE · filters ${filters}`);
    // Assert
    expect(items[0]).toEqual({ key: 'included', label: 'Included', value: expected });
  });

  it('parsePlanLine_predictors_eachLabelled', () => {
    // Arrange / Act
    const { items } = parsePlanLine('FIND_PREDICTORS · predictors age, ldl_mg_dl');
    // Assert
    expect(items[0]).toEqual({
      key: 'predictors',
      label: 'Predictors',
      value: 'Age, LDL (mg/dL) — ldl_mg_dl',
    });
  });

  it.each([
    ['grouping none', { key: 'grouping', label: 'Compared across', value: 'Not set' }],
    ['outcome none', { key: 'outcome', label: 'Outcome', value: 'Not set' }],
    ['time days', { key: 'time', label: 'Time', value: 'Days' }],
    ['event died', { key: 'event', label: 'Event', value: 'Died' }],
    ['statistic mean', { key: 'statistic', label: 'Statistic', value: 'Mean' }],
    ['weird part', { key: 'other', label: 'Other', value: 'weird part' }],
  ])('parsePlanLine_part_%s', (part, expected) => {
    // Arrange / Act
    const { items } = parsePlanLine(`DESCRIBE · ${part}`);
    // Assert
    expect(items).toEqual([expected]);
  });

  it.each([
    ['', ''],
    [null, ''],
    ['NEW_INTENT', 'New intent'],
  ])('parsePlanLine_intent_%s', (line, analysis) => {
    // Arrange / Act
    const result = parsePlanLine(line);
    // Assert
    expect(result.analysis).toBe(analysis);
  });
});

describe('describeQueryFailure', () => {
  it.each([
    [{ failure_reason: 'No date column for 2019.' }, 'refused', 'No date column for 2019.'],
    [
      { error: 'This would list individual patients.' },
      'refused',
      'This would list individual patients.',
    ],
    [
      { error: 'Something went wrong.', error_type: 'ValueError' },
      'error',
      'Something went wrong.',
    ],
    [{}, 'error', 'The analysis stopped before it could answer.'],
    [null, 'error', 'The analysis stopped before it could answer.'],
    [{ failure_reason: '   ', error: 'Fallback' }, 'refused', 'Fallback'],
    [
      { failure_reason: "Execution error: 'StringColumn' object has no attribute 'mean'" },
      'error',
      'The analysis couldn’t finish this question. Ask again in other words, naming the variables you mean; if it keeps happening, restart the app.',
    ],
  ])('describeQueryFailure_case%#', (data, kind, message) => {
    // Arrange / Act
    const result = describeQueryFailure(data);
    // Assert
    expect(result).toMatchObject({ kind, message });
  });

  it('describeQueryFailure_engineWording_plainMessageWithExactDetail', () => {
    // Arrange
    const raw = 'DESCRIBE intent requires metric (primary_variable)';
    // Act
    const result = describeQueryFailure({ failure_reason: raw });
    // Assert
    expect(result.kind).toBe('refused');
    expect(result.detail).toBe(raw);
    expect(result.message).not.toMatch(/DESCRIBE|primary_variable|intent/);
  });

  // Every validator message in src/clinical_analytics/core/semantic.py that reaches the UI.
  it.each([
    "Column 'LDL' not found in dataset",
    "Filter column 'ldl_mg_dl' not found",
    "Column 'x' not found in schema. Pre-execution validation should have caught this.",
    "Type mismatch for column 'age': expected numeric value, got string 'old'. LLM validation should have caught this.",
    'DESCRIBE intent requires metric (primary_variable)',
    'COMPARE_GROUPS intent requires both metric and group_by',
    'This query requires grouping, but no grouping_variable provided.',
    "Cannot group by entity key 'patient_id'. Grouping by entity key yields near-unique groups.",
    'Time and event variables required for survival analysis',
    "Cannot use scope='all' with filters. Use scope='filtered' or remove filters.",
    "scope='filtered' requires filters, but no filters provided.",
    "Invalid operator 'LIKE'",
    'All parsing tiers failed',
    'UNKNOWN_INTENT could not be planned',
  ])('plainFailureMessage_%s_hasNoEngineWords', (raw) => {
    // Arrange / Act
    const message = plainFailureMessage(raw);
    // Assert
    expect(message).not.toBe(raw);
    expect(message).not.toMatch(
      /\bLLM\b|[A-Z]{3,}_[A-Z_]+|primary_variable|group_by|grouping_variable|entity key|scope=|\bintent\b|validation/
    );
    expect(message).toMatch(/\.$|”\.$/);
  });

  it('plainFailureMessage_columnNotFound_namesTheColumnAndNextStep', () => {
    // Arrange / Act
    const message = plainFailureMessage("Column 'LDL' not found in dataset");
    // Assert
    expect(message).toContain('“LDL”');
    expect(message).toContain('cohort details');
  });

  it('plainFailureMessage_productRefusal_keptAsWritten', () => {
    // Arrange
    const refusal = 'This would list individual patients. Try asking for counts.';
    // Act / Assert
    expect(plainFailureMessage(refusal)).toBe(refusal);
  });
});

describe('stageLabel', () => {
  it.each([
    [{ stage: 'parsing', message: 'Analyzing query...' }, 'Reading your question'],
    [{ stage: 'plan' }, 'Choosing the variables and the test'],
    [{ stage: 'executing', message: 'Running analysis...' }, 'Running the analysis'],
    [{ stage: 'other', message: 'Custom' }, 'Custom'],
    [{}, ''],
    [null, ''],
  ])('stageLabel_case%#', (data, expected) => {
    // Arrange / Act
    const result = stageLabel(data);
    // Assert
    expect(result).toBe(expected);
  });
});

describe('buildExampleQuestions', () => {
  it('buildExampleQuestions_schema_usesNumericAndGroupColumns', () => {
    // Arrange
    const columns = ['patient_id', 'age', 'sex', 'mortality'];
    const schema = { patient_id: 'id', age: 'numeric', sex: 'categorical', mortality: 'binary' };
    // Act
    const result = buildExampleQuestions(columns, [], schema);
    // Assert
    expect(result).toEqual([
      'What is the average age?',
      'Compare age between sex groups',
      'How many patients are in each mortality group?',
    ]);
  });

  it('buildExampleQuestions_noSchema_infersFromPreviewRows', () => {
    // Arrange
    const rows = [1, 2, 3, 4, 5, 6].map((i) => ({
      id: i,
      ldl: 100 + i,
      arm: i % 2 ? 'drug' : 'placebo',
    }));
    // Act
    const result = buildExampleQuestions(['id', 'ldl', 'arm'], rows);
    // Assert
    expect(result[0]).toBe('What is the average ldl?');
    expect(result.join(' ')).not.toContain(' id');
  });

  it.each([
    [[], [], {}],
    [null, null, undefined],
    [['patient_id'], [{ patient_id: 'S1' }], {}],
  ])('buildExampleQuestions_nothingUsable_returnsEmpty_case%#', (columns, rows, schema) => {
    // Arrange / Act
    const result = buildExampleQuestions(columns, rows, schema);
    // Assert
    expect(result).toEqual([]);
  });
});

describe('re-audit follow-ups', () => {
  it.each([
    [0, '<0.001'],
    [0.0, '<0.001'],
    [1, '1'],
  ])('formatNumber_pValue_%s_isNeverShownAsZero', (value, expected) => {
    // Arrange: scipy returns 0.0 for very large effects

    // Act
    const text = formatNumber(value, { pValue: true });

    // Assert
    expect(text).toBe(expected);
  });

  it('plainFailureMessage_realRefusalMentioningIntentToTreat_isKeptAsWritten', () => {
    // Arrange: a product refusal, not engine jargon
    const reason =
      'Patients in the intent-to-treat population could not be listed individually; ask for a count instead.';

    // Act
    const text = plainFailureMessage(reason);

    // Assert
    expect(text).toBe(reason);
  });

  it.each([
    ["KeyError: 'group_sizes'"],
    ['polars.exceptions.ColumnNotFoundError: ldl'],
    ['Unexpected NoneType for plan.metric'],
    ["Metric column 'x' not in active version schema"],
    ['DESCRIBE intent needs a metric'],
  ])('plainFailureMessage_engineText_%s_isNotShownRaw', (raw) => {
    // Arrange: raw engine/exception text from the reviewer's probes

    // Act
    const text = plainFailureMessage(raw);

    // Assert
    expect(text).not.toBe(raw);
  });
});

// Synthetic cohort columns (no real data): the variables a doctor's file might hold.
const COHORT_COLUMNS = [
  'patient_id',
  'age',
  'sex',
  'treatment',
  'site',
  'ldl',
  'hdl',
  'bmi',
  'diabetes',
  'outcome',
  'followup_days',
  'event',
];

describe('answer plan (H1, H5)', () => {
  it('parsePlanLine_filteredCount_limitAndPatientUnitAlwaysShown', () => {
    // Arrange
    const line =
      'COUNT · outcome none · grouping none · filters age > 83 · counting distinct patient_id · test count';

    // Act
    const { analysis, items } = parsePlanLine(line, { omitUnset: true });

    // Assert
    expect(analysis).toBe('Count');
    expect(items).toEqual([
      { key: 'included', label: 'Included', value: 'Age over 83' },
      { key: 'counting', label: 'Counting', value: 'Patients (each Patient ID once)' },
    ]);
  });

  it.each([
    ['filters none', 'All rows'],
    ['filters none · counting distinct patient_id', 'All patients'],
  ])('parsePlanLine_noLimits_%s_saysWhoWasIncluded', (parts, expected) => {
    // Arrange / Act
    const { items } = parsePlanLine(`COUNT · ${parts}`, { omitUnset: true });

    // Assert
    expect(items.find((i) => i.key === 'included').value).toBe(expected);
  });

  it('parsePlanLine_rowsNotPatients_whenNoDistinctCount', () => {
    // Arrange / Act
    const { items } = parsePlanLine('COUNT · grouping sex · filters age > 82 · test count');

    // Assert
    expect(items).toContainEqual({ key: 'counting', label: 'Counting', value: 'Rows' });
  });

  it.each([
    ['limits in order 1) patient_id = S0001', 'Patient ID (patient_id) is S0001'],
    ['limits in order 1) age > 65; 2) ldl ≥ 130', 'Age over 65, then LDL at least 130'],
    ['limits in order 1) ldl is recorded', 'LDL is recorded'],
    [
      'limits in order 1) bmi above its median among the patients so far',
      'BMI above its median among the patients so far',
    ],
  ])('parsePlanLine_limits_%s_inWords', (part, expected) => {
    // Arrange / Act
    const { items } = parsePlanLine(`DESCRIBE · filters none · ${part}`);

    // Assert
    const included = items.find((i) => i.key === 'included');
    expect(included.value).toBe(expected);
    expect(items.map((i) => i.value).join(' ')).not.toMatch(/limits in order|\d\)/);
  });

  it('parsePlanLine_filtersThenLimits_bothInOrder', () => {
    // Arrange / Act
    const { items } = parsePlanLine('DESCRIBE · filters sex == F · limits in order 1) age > 65');

    // Assert
    expect(items.filter((i) => i.key === 'included')).toEqual([
      { key: 'included', label: 'Included', value: 'Sex is F, then Age over 65' },
    ]);
  });

  it.each([
    ['test count'],
    ['test not selected'],
    ['scope filtered'],
    ['filters required'],
    ['grouping required'],
  ])('parsePlanLine_internalPart_%s_isNotShown', (part) => {
    // Arrange / Act
    const { items } = parsePlanLine(`COUNT · ${part}`);

    // Assert
    expect(items).toEqual([]);
  });

  it('parsePlanLine_testLabel_sentenceCase', () => {
    // Arrange / Act
    const { items } = parsePlanLine('DESCRIBE · test descriptive statistics');

    // Assert
    expect(items).toEqual([{ key: 'test', label: 'Test', value: 'Descriptive statistics' }]);
  });

  it('parsePlanLine_neverShowsNone', () => {
    // Arrange
    const line =
      'SURVIVAL · outcome none · grouping none · time none · event none · filters none · test Kaplan–Meier';

    // Act
    const shown = parsePlanLine(line)
      .items.map((i) => i.value)
      .join(' | ');
    const answer = parsePlanLine(line, { omitUnset: true })
      .items.map((i) => i.key)
      .join(',');

    // Assert
    expect(shown).not.toMatch(/\bNone\b|\bnone\b/);
    expect(shown).toContain('Not set');
    expect(answer).toBe('included,counting,test');
  });

  it('buildQueryCompletedPresentation_keepsPlanLine', () => {
    // Arrange
    const planLine = 'COUNT · filters age > 83 · test count';

    // Act
    const pres = buildQueryCompletedPresentation('COUNT', {}, 'Total count hidden.', planLine);

    // Assert
    expect(pres.result.planLine).toBe(planLine);
  });
});

describe('hidden counts and table guards (H2, M5, B1)', () => {
  it.each([
    ['Total count hidden.', false, true],
    ['15 total across 2 sex groups (n hidden).', false, true],
    ['Total count: <11.', false, true],
    ['Total count: 400.', false, false],
    ['Mean age was 58.2 (n=400).', true, true],
    ['Age band <65 was most common (n=200).', false, false],
    [null, false, false],
  ])('needsHiddenFootnote_%s_table%s_is_%s', (claim, tableHasHidden, expected) => {
    // Arrange / Act / Assert
    expect(needsHiddenFootnote(claim, tableHasHidden)).toBe(expected);
  });

  it.each([
    [
      {
        columns: ['outcome', 'F', 'M'],
        rows: [
          { outcome: 'outcome=0', F: 'suppressed', M: 'suppressed' },
          { outcome: 'outcome=1', F: '<11', M: 'suppressed' },
        ],
      },
      true,
    ],
    [
      {
        columns: ['sex', 'n'],
        rows: [
          ['F', 'suppressed'],
          ['M', 204],
        ],
      },
      false,
    ],
    [{ columns: ['sex', 'n'], rows: [] }, false],
    [{ columns: ['sex', 'n'], rows: [['F', null]] }, false],
  ])('isFullyHiddenTable_case%#', (table, expected) => {
    // Arrange / Act / Assert
    expect(isFullyHiddenTable(table)).toBe(expected);
  });

  it.each([
    [
      {
        columns: ['treatment', ...Array.from({ length: 289 }, (_, i) => `ldl=${80 + i}`)],
        rows: [[], []],
      },
      289,
    ],
    [{ columns: ['sex', 'n'], rows: Array.from({ length: 30 }, () => ['x', 12]) }, 30],
    [
      {
        columns: ['sex', 'mean age', 'n'],
        rows: [
          ['F', 58, 204],
          ['M', 59, 196],
        ],
      },
      2,
    ],
    [null, 0],
  ])('tableLevelCount_case%#_is_%s', (table, expected) => {
    // Arrange / Act / Assert
    expect(tableLevelCount(table)).toBe(expected);
  });

  it('TABLE_LEVEL_LIMIT_is_about_twenty', () => {
    expect(TABLE_LEVEL_LIMIT).toBe(20);
  });

  it.each([
    ['outcome=0', 'Outcome: 0'],
    ['ldl=80.9', 'LDL: 80.9'],
    ['drug', 'drug'],
    ['a = b', 'a = b'],
    ['', ''],
  ])('plainLevelLabel_%s_is_%s', (raw, expected) => {
    // Arrange / Act / Assert
    expect(plainLevelLabel(raw)).toBe(expected);
  });
});

describe('caught messages (H3)', () => {
  it.each([
    ['Failed to fetch'],
    ['TypeError: NetworkError when attempting to fetch resource.'],
    [''],
    [undefined],
  ])('plainFailureMessage_%s_isTheEngineUnreachableFallback', (raw) => {
    // Arrange / Act / Assert
    expect(plainFailureMessage(raw)).toBe(ENGINE_UNREACHABLE_MESSAGE);
  });

  it('ENGINE_UNREACHABLE_MESSAGE_wording', () => {
    expect(ENGINE_UNREACHABLE_MESSAGE).toBe(
      'Couldn’t reach the analysis engine. Ask again; if it keeps happening, restart the app.'
    );
  });

  it('plainFailureMessage_retryWord_followsTheContext', () => {
    expect(plainFailureMessage('Failed to fetch', { context: 'upload' })).toBe(
      'Couldn’t reach the analysis engine. Try again; if it keeps happening, restart the app.'
    );
  });

  it.each([
    ["Execution error: 'StringColumn' object has no attribute 'mean'"],
    ['Execution failed: unsupported operand type(s) for +'],
    ["'NoneType' object is not subscriptable"],
    ['Binder Error: Referenced column "x" not found'],
  ])('plainFailureMessage_exception_%s_isNeverShownRaw', (raw) => {
    // Arrange / Act
    const text = plainFailureMessage(raw);

    // Assert
    expect(text).not.toMatch(/StringColumn|attribute|operand|NoneType|Binder|Execution/);
  });

  it('describeQueryFailure_executionError_isAnErrorWithDetailForCopying', () => {
    // Arrange
    const raw = "Execution error: 'StringColumn' object has no attribute 'mean'";

    // Act
    const result = describeQueryFailure({ failure_reason: raw });

    // Assert
    expect(result.kind).toBe('error');
    expect(result.detail).toBe(raw);
    expect(result.message).not.toContain('StringColumn');
  });
});

describe('hold next step (H5)', () => {
  it.each([
    ['cholesterol', ['ldl', 'hdl']],
    ['diabete', ['diabetes']],
    ['gender', ['sex']],
    ['average', []],
    ['by', []],
    ['ids', []],
  ])('suggestColumns_%s_is_%o', (word, expected) => {
    // Arrange / Act / Assert
    expect(suggestColumns([word], COHORT_COLUMNS)).toEqual(expected);
  });

  it('suggestColumns_neverSuggestsIdentifiers_andCapsAtThree', () => {
    // Arrange / Act
    const out = suggestColumns(['patient', 'cholesterol', 'diabetes', 'gender'], COHORT_COLUMNS);

    // Assert
    expect(out).not.toContain('patient_id');
    expect(out.length).toBeLessThanOrEqual(3);
  });

  it('holdNextStep_unplacedWords_namesClosestColumns_andRejectIsPrimary', () => {
    // Arrange
    const reason =
      "I couldn't place the words 'average' and 'cholesterol', so they are not in the plan.";

    // Act
    const step = holdNextStep(reason, COHORT_COLUMNS);

    // Assert
    expect(step.dropped).toBe(true);
    expect(step.text).toBe(
      'This cohort has LDL and HDL. If you meant one of them, choose “Not what I meant” and ask again using its name.'
    );
  });

  it('holdNextStep_unplacedWordWithNoMatch_pointsToVariableNames', () => {
    // Arrange
    const reason = "I couldn't place the word 'recently', so it is not in the plan.";

    // Act
    const step = holdNextStep(reason, COHORT_COLUMNS);

    // Assert
    expect(step.dropped).toBe(true);
    expect(step.text).toBe(
      'If that word matters, choose “Not what I meant” and ask again using the cohort’s variable names (see cohort details).'
    );
  });

  it('holdNextStep_confidenceHold_saysWhatToCheck', () => {
    // Arrange / Act
    const step = holdNextStep('Confirmation required before execution.', COHORT_COLUMNS);

    // Assert
    expect(step.dropped).toBe(false);
    expect(step.text).toBe(
      'If the plan below matches your question, run it. If not, choose “Not what I meant” and ask again in other words.'
    );
  });
});

describe('failure messages by context (round 2)', () => {
  const JARGON = [
    "Error saving upload: polars.exceptions.ComputeError: could not parse 'abc' as dtype i64",
    "Execution error: 'StringColumn' object has no attribute 'mean'",
    "Column 'LDL' not found in dataset",
    'DESCRIBE intent requires metric (primary_variable)',
  ];

  it.each(['upload', 'delete', 'panel'].flatMap((context) => JARGON.map((raw) => [context, raw])))(
    'plainFailureMessage_%s_%s_neverTalksAboutAQuestion',
    (context, raw) => {
      // Arrange / Act
      const text = plainFailureMessage(raw, { context });

      // Assert
      expect(text).not.toMatch(/question/i);
      expect(text).not.toBe(raw);
      expect(text).toMatch(/[Tt]ry again/);
    }
  );

  it('plainFailureMessage_uploadThatCannotBeParsed_saysCheckTheFile', () => {
    // Arrange
    const raw =
      "Error saving upload: polars.exceptions.ComputeError: could not parse 'abc' as dtype i64";

    // Act
    const text = plainFailureMessage(raw, { context: 'upload' });

    // Assert
    expect(text).toBe(
      'The file couldn’t be read. Check that it is a CSV, Excel or SPSS file with a header row, then try again.'
    );
  });

  it.each([
    [
      httpFailureText(
        500,
        'Dataset only partly deleted, try again: OSError: [Errno 13] Permission denied'
      ),
      'Dataset only partly deleted, try again.',
    ],
    [
      httpFailureText(500, 'Dataset files deleted but its chats were not; try the delete again'),
      'Dataset files deleted but its chats were not; try the delete again.',
    ],
  ])('plainFailureMessage_serverSentenceForPeople_%s_isKept', (raw, expected) => {
    // Arrange / Act
    const text = plainFailureMessage(raw, { context: 'delete' });

    // Assert
    expect(text).toBe(expected);
  });

  it.each([
    [httpFailureText(500, 'Internal Server Error')],
    [httpFailureText(502, '')],
    ['Internal Server Error'],
    ['Failed to submit query: Internal Server Error'],
  ])('plainFailureMessage_5xx_%s_isAProblemWithTheQuestion_notUnreachable', (raw) => {
    // Arrange / Act
    const text = plainFailureMessage(raw);

    // Assert
    expect(text).toBe(
      'The analysis engine ran into a problem with this question. Ask again in other words; if it keeps happening, restart the app.'
    );
  });

  it('plainFailureMessage_5xxOutsideAQuestion_saysTryAgain', () => {
    // Arrange / Act
    const text = plainFailureMessage(httpFailureText(500, ''), { context: 'panel' });

    // Assert
    expect(text).toBe(
      'The analysis engine ran into a problem. Try again; if it keeps happening, restart the app.'
    );
  });

  it.each([
    [httpFailureText(404, 'Dataset not found: abc'), 'panel'],
    [httpFailureText(404, "Dataset 'abc' not found"), 'delete'],
    [httpFailureText(404, 'Dataset data not found: abc'), 'question'],
  ])('plainFailureMessage_404ForACohort_%s_saysItNoLongerExists', (raw, context) => {
    // Arrange / Act
    const text = plainFailureMessage(raw, { context });

    // Assert
    expect(text).toBe(
      'This cohort no longer exists. Choose another cohort, or refresh the list of cohorts.'
    );
  });

  it.each([
    [
      [
        {
          type: 'string_too_short',
          loc: ['body', 'query_text'],
          msg: 'String should have at least 1 character',
        },
      ],
      'The question is empty. Type a question, then ask again.',
    ],
    [
      [{ type: 'missing', loc: ['body', 'dataset_id'], msg: 'Field required' }],
      'No cohort was chosen. Choose a cohort, then ask again.',
    ],
    [
      [
        {
          type: 'extra_forbidden',
          loc: ['body', 'surprise'],
          msg: 'Extra inputs are not permitted',
        },
      ],
      'The analysis engine didn’t accept the request from the app. Restart the app, then ask again.',
    ],
  ])('plainFailureMessage_422_%#_givesTheReasonInPlainWords', (detail, expected) => {
    // Arrange
    const raw = httpFailureText(422, detail);

    // Act
    const text = plainFailureMessage(raw);

    // Assert
    expect(text).toBe(expected);
    expect(text).not.toMatch(/reach/);
  });

  it('plainFailureMessage_4xxWithAReasonForPeople_keepsTheReason', () => {
    // Arrange
    const raw = httpFailureText(
      400,
      "File extension '.txt' not allowed. Accepted: .csv, .sav, .xls, .xlsx"
    );

    // Act
    const text = plainFailureMessage(raw, { context: 'upload' });

    // Assert
    expect(text).toBe("File extension '.txt' not allowed. Accepted: .csv, .sav, .xls, .xlsx");
  });

  it.each([
    ['Failed to load dataset preview: Not Found'],
    ['Failed to list sessions: Bad Request'],
  ])('plainFailureMessage_failedToPrefix_%s_isNotUnreachable', (raw) => {
    // Arrange / Act / Assert
    expect(plainFailureMessage(raw, { context: 'panel' })).not.toMatch(/reach/);
  });

  it('httpFailureText_encodesStatusAndReason', () => {
    // Arrange / Act / Assert
    expect(httpFailureText(404, 'Dataset not found: abc')).toBe(
      '[HTTP 404] Dataset not found: abc'
    );
    expect(httpFailureText(500, null)).toBe('[HTTP 500]');
  });

  it.each([
    'This cohort has no variable',
    'The question asks for a breakdown',
    'That variable identifies',
    'using a variable name',
  ])('engineMessages_useVariableNotColumn_%s', (start) => {
    // Arrange
    const raws = [
      "Column 'LDL' not found in dataset",
      'This query requires grouping, but no grouping_variable provided.',
      "Cannot group by entity key 'patient_id'.",
      'All parsing tiers failed',
    ];

    // Act
    const texts = raws.map((raw) => plainFailureMessage(raw));

    // Assert
    const text = texts.find((t) => t.includes(start));
    expect(text).toBeDefined();
    expect(text).not.toMatch(/\bcolumns?\b/);
  });

  it('stageLabel_planning_saysVariables', () => {
    expect(stageLabel({ stage: 'planning' })).toBe('Choosing the variables and the test');
  });

  it.each([
    [httpFailureText(500, '')],
    ['Failed to fetch'],
    ["Execution error: 'x'"],
    ['DESCRIBE intent requires metric (primary_variable)'],
  ])('plainFailureMessage_%s_usesCurlyApostrophes', (raw) => {
    // Arrange / Act / Assert
    expect(plainFailureMessage(raw)).not.toContain("'");
  });
});

describe('single suggestion and lists (round 2)', () => {
  it('holdNextStep_oneSuggestion_saysItAndShowsTheAcceptedName', () => {
    // Arrange
    const reason = "I couldn't place the word 'cholesterol', so it is not in the plan.";

    // Act
    const step = holdNextStep(reason, ['ldl_mg_dl', 'age']);

    // Assert
    expect(step.text).toBe(
      'This cohort has LDL (mg/dL) (ldl_mg_dl). If you meant it, choose “Not what I meant” and ask again using that name.'
    );
  });

  it('holdNextStep_threeSuggestions_listsThemWithIntl', () => {
    // Arrange
    const reason =
      "I couldn't place the words 'cholesterol' and 'diabete', so they are not in the plan.";

    // Act
    const step = holdNextStep(reason, ['ldl', 'hdl', 'diabetes']);

    // Assert
    expect(step.text).toBe(
      'This cohort has LDL, HDL and Diabetes. If you meant one of them, choose “Not what I meant” and ask again using its name.'
    );
  });
});

describe('level guard scope (round 2)', () => {
  const pairs = {
    columns: ['variable 1', 'variable 2', 'r'],
    rows: Array.from({ length: 21 }, (_, i) => [`v${i}`, `w${i}`, 0.1]),
  };
  const crossTab = {
    columns: ['treatment', ...Array.from({ length: 30 }, (_, i) => `ldl=${80 + i}`)],
    rows: [[], []],
  };
  const overview = {
    columns: ['variable', 'mean', 'n'],
    rows: Array.from({ length: 25 }, (_, i) => [`v${i}`, 1, 400]),
  };

  it.each([
    ['relationship', pairs, false],
    ['predictor', { columns: ['variable', 'odds ratio'], rows: pairs.rows }, false],
    ['descriptive', overview, false],
    ['comparison', crossTab, true],
    ['count', { columns: ['site', 'count'], rows: pairs.rows }, true],
    ['grouped_statistic', { columns: ['site', 'mean age', 'n'], rows: pairs.rows }, true],
    [undefined, crossTab, true],
  ])('isGroupLevelTable_%s_case%#', (analysisType, table, expected) => {
    // Arrange / Act / Assert
    expect(isGroupLevelTable(table, analysisType)).toBe(expected);
  });

  it('buildQueryCompletedPresentation_keepsTheAnalysisType', () => {
    // Arrange / Act
    const pres = buildQueryCompletedPresentation(
      'CORRELATIONS',
      { analysis_type: 'relationship', claim: 'r=0.1.', table: pairs },
      null,
      null
    );

    // Assert
    expect(pres.result.analysisType).toBe('relationship');
  });
});

describe('vocabulary in summaries (round 2)', () => {
  it('buildIntentSummary_describe_namesVariables', () => {
    // Arrange / Act
    const text = buildIntentSummary(
      'describe',
      { summary: { sample_size: 400, numeric_columns: ['age', 'ldl'] } },
      0
    );

    // Assert
    expect(text).toBe('N=400 | Variables: age, ldl');
  });
});
