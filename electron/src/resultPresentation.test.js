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
    'n=280; <11 excluded for missing data). Unadjusted comparison by treatment.',
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
      ['95% CI 13.44 to 18.83', 'Welch t-test p<0.001', 'n=280', '<11 excluded for missing data'],
      'Unadjusted comparison by treatment.',
    ],
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
    ['Suppressed', 'hidden', 'hidden'],
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
    ['Mean was 16.14 higher (p<0.001; n=377).', 3],
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
        { label: 'Outcome', value: 'LDL' },
        { label: 'Compared across', value: 'Treatment' },
        { label: 'Patients included', value: 'Age over 65' },
        { label: 'Test', value: 'Welch t-test' },
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
    expect(items[0]).toEqual({ label: 'Patients included', value: expected });
  });

  it('parsePlanLine_predictors_eachLabelled', () => {
    // Arrange / Act
    const { items } = parsePlanLine('FIND_PREDICTORS · predictors age, ldl_mg_dl');
    // Assert
    expect(items[0]).toEqual({ label: 'Predictors', value: 'Age, LDL (mg/dL) — ldl_mg_dl' });
  });

  it.each([
    ['filters none', { label: 'Patients included', value: 'All patients' }],
    ['grouping none', { label: 'Compared across', value: 'None' }],
    ['time days', { label: 'Time', value: 'Days' }],
    ['event died', { label: 'Event', value: 'Died' }],
    ['weird part', { label: '', value: 'weird part' }],
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
  ])('describeQueryFailure_case%#', (data, kind, message) => {
    // Arrange / Act
    const result = describeQueryFailure(data);
    // Assert
    expect(result).toEqual({ kind, message });
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
    [{ stage: 'plan' }, 'Choosing the columns and the test'],
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
