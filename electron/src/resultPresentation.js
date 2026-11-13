// SPDX-License-Identifier: MIT
/**
 * Pure helpers for mapping API intent + result_preview into chat UI copy.
 * Keeps parity labels aligned with AnalysisIntent enum values (lowercase).
 */

/** @typedef {{ columns: string[], rows: unknown[] }} TablePayload */

/**
 * Normalize intent strings from SSE (enum value or legacy uppercase).
 * @param {string | null | undefined} raw
 * @returns {string}
 */
export function normalizeIntentType(raw) {
  if (raw == null || raw === '') {
    return 'unknown';
  }
  return String(raw).trim().toLowerCase().replace(/-/g, '_');
}

/**
 * Human-readable title for result card (maps to Streamlit analysis pages / omnibus P20–P24).
 * @param {string} normalizedIntent
 * @returns {string | null} null → generic card without section title
 */
export function resultTitleForIntent(normalizedIntent) {
  switch (normalizedIntent) {
    case 'describe':
      return 'Descriptive statistics';
    case 'compare_groups':
      return 'Group comparison';
    case 'find_predictors':
      return 'Risk factors';
    case 'examine_survival':
      return 'Survival analysis';
    case 'explore_relationships':
      return 'Correlations';
    case 'count':
      return 'Counts';
    default:
      return null;
  }
}

/**
 * Extract first tabular payload from API result_preview (handles flat and nested shapes).
 * @param {Record<string, unknown> | null | undefined} preview
 * @returns {{ table: TablePayload, rowCount: number } | null}
 */
export function extractTableFromResultPreview(preview) {
  if (!preview || typeof preview !== 'object') {
    return null;
  }

  const top = /** @type {Record<string, unknown>} */ (preview);
  const topTable = top.table;
  if (
    topTable &&
    typeof topTable === 'object' &&
    Array.isArray(/** @type {{ columns?: unknown }} */ (topTable).columns) &&
    Array.isArray(/** @type {{ rows?: unknown }} */ (topTable).rows)
  ) {
    const t = /** @type {TablePayload} */ (topTable);
    const rowCount = typeof top.row_count === 'number' ? top.row_count : t.rows.length;
    return { table: t, rowCount };
  }

  for (const value of Object.values(top)) {
    if (!value || typeof value !== 'object') {
      continue;
    }
    const v = /** @type {Record<string, unknown>} */ (value);
    if (v.table && typeof v.table === 'object') {
      const inner = /** @type {{ columns?: unknown, rows?: unknown }} */ (v.table);
      if (Array.isArray(inner.columns) && Array.isArray(inner.rows)) {
        const t = /** @type {TablePayload} */ (v.table);
        const rowCount = typeof v.row_count === 'number' ? v.row_count : t.rows.length;
        return { table: t, rowCount };
      }
    }
  }

  return null;
}

/**
 * One-line assistant message when a query completes.
 * @param {string} normalizedIntent
 * @param {number} rowCount
 * @param {boolean} hasTable
 * @returns {string}
 */
export function buildCompletionMessage(normalizedIntent, rowCount, hasTable) {
  const title = resultTitleForIntent(normalizedIntent);
  if (hasTable && rowCount >= 0) {
    const label = title || 'Results';
    return `${label}: ${rowCount} row${rowCount === 1 ? '' : 's'} (preview)`;
  }
  if (title) {
    return `${title}: analysis complete.`;
  }
  return 'Analysis complete.';
}

/**
 * @param {unknown} value
 * @returns {Record<string, unknown>}
 */
function asRecord(value) {
  return value && typeof value === 'object' ? /** @type {Record<string, unknown>} */ (value) : {};
}

/**
 * @param {unknown} value
 * @returns {string}
 */
function formatNumeric(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '';
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  return value.toFixed(2).replace(/\.00$/, '');
}

/**
 * Intent-aware summary for typed result payload blocks.
 * @param {string} normalizedIntent
 * @param {Record<string, unknown> | null | undefined} resultPreview
 * @param {number} rowCount
 * @returns {string | null}
 */
export function buildIntentSummary(normalizedIntent, resultPreview, rowCount) {
  const preview = asRecord(resultPreview);

  if (normalizedIntent === 'describe') {
    const summary = asRecord(preview.summary);
    const sampleSize = summary.sample_size;
    const numericColumns = Array.isArray(summary.numeric_columns)
      ? summary.numeric_columns.filter((v) => typeof v === 'string').slice(0, 2)
      : [];
    const parts = [];
    if (typeof sampleSize === 'number') {
      parts.push(`N=${sampleSize}`);
    } else if (rowCount > 0) {
      parts.push(`N=${rowCount}`);
    }
    if (numericColumns.length > 0) {
      parts.push(`Variables: ${numericColumns.join(', ')}`);
    }
    return parts.length > 0 ? parts.join(' | ') : null;
  }

  if (normalizedIntent === 'compare_groups') {
    const comparison = asRecord(preview.comparison);
    const groupBy = typeof comparison.group_by === 'string' ? comparison.group_by : null;
    const metric = typeof comparison.metric === 'string' ? comparison.metric : null;
    const pValue = formatNumeric(comparison.p_value);
    const parts = [];
    if (groupBy) parts.push(`Grouped by ${groupBy}`);
    if (metric) parts.push(`Metric: ${metric}`);
    if (pValue) parts.push(`p=${pValue}`);
    return parts.length > 0 ? parts.join(' | ') : null;
  }

  if (normalizedIntent === 'find_predictors') {
    const model = asRecord(preview.model);
    const target = typeof model.target === 'string' ? model.target : null;
    const predictors = Array.isArray(model.top_predictors)
      ? model.top_predictors.filter((v) => typeof v === 'string').slice(0, 2)
      : [];
    const parts = [];
    if (target) parts.push(`Target: ${target}`);
    if (predictors.length > 0) parts.push(`Top: ${predictors.join(', ')}`);
    return parts.length > 0 ? parts.join(' | ') : null;
  }

  if (normalizedIntent === 'examine_survival') {
    const survival = asRecord(preview.survival);
    const timeCol = typeof survival.time_column === 'string' ? survival.time_column : null;
    const eventCol = typeof survival.event_column === 'string' ? survival.event_column : null;
    const median = formatNumeric(survival.median_survival_days);
    const parts = [];
    if (timeCol) parts.push(`Time: ${timeCol}`);
    if (eventCol) parts.push(`Event: ${eventCol}`);
    if (median) parts.push(`Median: ${median} days`);
    return parts.length > 0 ? parts.join(' | ') : null;
  }

  if (normalizedIntent === 'explore_relationships') {
    const correlations = asRecord(preview.correlations);
    const method = typeof correlations.method === 'string' ? correlations.method : null;
    const strongestPair = Array.isArray(correlations.strongest_pair)
      ? correlations.strongest_pair.filter((v) => typeof v === 'string').slice(0, 2)
      : [];
    const strongestR = formatNumeric(correlations.strongest_r);
    const parts = [];
    if (method) parts.push(`Method: ${method}`);
    if (strongestPair.length === 2)
      parts.push(`Strongest: ${strongestPair[0]} ~ ${strongestPair[1]}`);
    if (strongestR) parts.push(`r=${strongestR}`);
    return parts.length > 0 ? parts.join(' | ') : null;
  }

  return null;
}

/**
 * @param {string | null | undefined} intentRaw
 * @param {Record<string, unknown> | null | undefined} resultPreview
 * @param {string | null | undefined} [claimArg] optional claim from SSE top-level field
 * @param {string | null | undefined} [planLineArg] the plan that was computed (query_completed)
 * @returns {{ content: string, result: { intentType: string, title: string | null, summary: string | null, claim?: string, planLine?: string, table?: TablePayload } | null }}
 */
export function buildQueryCompletedPresentation(intentRaw, resultPreview, claimArg, planLineArg) {
  const intentType = normalizeIntentType(intentRaw);
  const extracted = extractTableFromResultPreview(resultPreview);
  const hasTable = extracted != null;
  const rowCount = extracted ? extracted.rowCount : 0;
  const title = resultTitleForIntent(intentType);
  // Prefer explicit SSE claim, then claim embedded in the preview payload.
  const fromArg = typeof claimArg === 'string' && claimArg.trim() ? claimArg.trim() : null;
  const fromPreview =
    typeof resultPreview?.claim === 'string' && resultPreview.claim ? resultPreview.claim : null;
  const claim = fromArg || fromPreview;
  const planLine =
    typeof planLineArg === 'string' && planLineArg.trim() ? planLineArg.trim() : null;
  const content = claim
    ? title || 'Analysis complete.'
    : buildCompletionMessage(intentType, rowCount, hasTable);
  const summary = claim || buildIntentSummary(intentType, resultPreview, rowCount);

  if (!extracted && !title && !summary && !claim && !planLine) {
    return { content, result: null };
  }

  const analysisType =
    typeof resultPreview?.analysis_type === 'string' ? resultPreview.analysis_type : null;
  const result = {
    intentType,
    ...(analysisType ? { analysisType } : {}),
    title,
    summary,
    ...(claim ? { claim } : {}),
    ...(planLine ? { planLine } : {}),
  };

  if (extracted) {
    return {
      content,
      result: {
        ...result,
        table: extracted.table,
      },
    };
  }

  return {
    content,
    result,
  };
}

/**
 * Build a confirmation-hold presentation for low-confidence plans.
 * @param {{ plan_line?: string, plan_hash?: string, failure_reason?: string, confidence?: number, confirm_options?: { label: string, plan_hash: string }[] } | null | undefined} data
 * @param {string} pendingQuery
 * @param {{ datasetId?: string | null, sessionId?: string | null }} [binding] where the plan was held
 * @returns {{ content: string, result: { awaitingConfirmation: true, planLine: string, planHash: string, failureReason: string, pendingQuery: string, confidence: number | null, confirmOptions: { label: string, planHash: string }[] } }}
 */
export function buildConfirmationHoldPresentation(data, pendingQuery, binding = {}) {
  const planLine = typeof data?.plan_line === 'string' ? data.plan_line : '';
  // Hash of the exact executable plan on this card; Confirm sends it back and the backend
  // releases only that plan.
  const planHash = typeof data?.plan_hash === 'string' ? data.plan_hash : '';
  const failureReason =
    typeof data?.failure_reason === 'string' && data.failure_reason
      ? data.failure_reason
      : 'Confirmation required before execution.';
  const confidence = typeof data?.confidence === 'number' ? data.confidence : null;
  // A unit hold is settled by picking one option; each carries its own hash.
  const confirmOptions = Array.isArray(data?.confirm_options)
    ? data.confirm_options
        .filter((o) => typeof o?.label === 'string' && typeof o?.plan_hash === 'string')
        .map((o) => ({ label: o.label, planHash: o.plan_hash }))
    : [];
  // The plan line renders once, inside the card; the message text carries only the reason.
  return {
    content: failureReason,
    result: {
      awaitingConfirmation: true,
      planLine,
      planHash,
      failureReason,
      pendingQuery: pendingQuery || '',
      confidence,
      confirmOptions,
      // The dataset and session the plan was held for; Confirm must not run it anywhere else.
      datasetId: binding.datasetId || null,
      sessionId: binding.sessionId || null,
    },
  };
}

// ============================================================================
// Answer anatomy (docs/app/DESIGN.md): headline, stats line, caveats.
// These helpers only re-arrange the API's text; they never change a word or a number.
// ============================================================================

/** Counts from 1 to SMALL_CELL_THRESHOLD - 1 are hidden by the API (never write "<5"). */
export const SMALL_CELL_THRESHOLD = 11;

/** The API's marker for every hidden cell (primary and secondary alike, by design). */
export const HIDDEN_CELL_MARKER = 'suppressed';
/** The API's own small-cell token (`suppress_count` writes exactly this). */
export const SMALL_CELL_TOKEN = `<${SMALL_CELL_THRESHOLD}`;

/**
 * Only the API's exact tokens mean "hidden": a category label such as "<65" or "<5" is data.
 * @param {string} s trimmed cell text
 * @returns {boolean}
 */
export function isSmallCellToken(s) {
  // The characters the API writes, not any number equal to the threshold ("<011" is a label).
  const m = /^<\s*(\d+)$/.exec(s);
  return m !== null && m[1] === String(SMALL_CELL_THRESHOLD);
}

/**
 * The API writes the lowercase marker; "Suppressed" as a real category label stays text.
 * @param {string} s trimmed cell text
 * @returns {boolean}
 */
export function isHiddenCellMarker(s) {
  return s === HIDDEN_CELL_MARKER;
}

export const HIDDEN_CELL_FOOTNOTE =
  `Hidden to protect privacy: counts from 1 to ${SMALL_CELL_THRESHOLD - 1} are not shown, ` +
  'and some other cells are hidden so those counts can’t be worked out from totals.';

/** "hidden" or the API's "<11" anywhere in the claim (the stats line is part of the claim). */
const CLAIM_HIDES_A_COUNT = new RegExp(`\\bhidden\\b|(^|[^\\w<])${SMALL_CELL_TOKEN}(?!\\d)`, 'i');

/**
 * The hidden-count footnote belongs under an answer whenever anything in it is hidden: a table
 * cell, or a count the claim or its stats line reports as hidden ("Total count hidden.").
 * @param {string | null | undefined} claim
 * @param {boolean} tableHasHidden
 * @returns {boolean}
 */
export function needsHiddenFootnote(claim, tableHasHidden) {
  return Boolean(tableHasHidden) || CLAIM_HIDES_A_COUNT.test(String(claim || ''));
}

/**
 * Rows as arrays of cells, in column order (the API may send Polars to_dicts() row objects).
 * @param {unknown[]} columns
 * @param {unknown[]} rows
 * @returns {unknown[][]}
 */
export function normalizeTableRows(columns, rows) {
  if (!Array.isArray(rows) || rows.length === 0) return [];
  return rows.map((row) => {
    if (Array.isArray(row)) return row;
    if (row && typeof row === 'object' && Array.isArray(columns)) {
      return columns.map((col) => /** @type {Record<string, unknown>} */ (row)[String(col)]);
    }
    return [];
  });
}

/**
 * True when every value cell (all but the label column) is hidden: such a table shows nothing
 * but its labels, so the answer draws no table, only the footnote.
 * @param {{ columns?: unknown[], rows?: unknown[] } | null | undefined} table
 * @returns {boolean}
 */
export function isFullyHiddenTable(table) {
  const columns = Array.isArray(table?.columns) ? table.columns : [];
  const rows = normalizeTableRows(columns, table?.rows);
  let seen = 0;
  for (const row of rows) {
    for (let i = 1; i < columns.length; i += 1) {
      const { kind } = classifyCell(row[i]);
      if (kind !== 'hidden' && kind !== 'small') return false;
      seen += 1;
    }
  }
  return seen > 0;
}

/**
 * Most groups a table may show. With more, groups get small enough to describe single patients
 * (a cross-tab headed by every distinct LDL value names one patient per column).
 */
export const TABLE_LEVEL_LIMIT = 20;

/**
 * Number of groups a table shows: its value columns or its rows, whichever is more.
 * @param {{ columns?: unknown[], rows?: unknown[] } | null | undefined} table
 * @returns {number}
 */
export function tableLevelCount(table) {
  const columns = Array.isArray(table?.columns) ? table.columns.length : 0;
  const rows = Array.isArray(table?.rows) ? table.rows.length : 0;
  return Math.max(columns - 1, rows, 0);
}

/** Analyses whose table rows are variables, never levels of a grouping variable. */
const VARIABLE_ROW_ANALYSES = new Set(['relationship', 'predictor']);

/**
 * Whether a table's rows or columns are levels of a grouping variable (cross-tabs, group
 * comparisons, grouped counts), so the level limit applies. Correlation pairs, predictors and
 * the per-variable overview list variables, not groups. An unknown analysis is treated as
 * grouped. This is defence in depth: the server applies the real rule before the table is sent.
 * @param {{ columns?: unknown[], rows?: unknown[] } | null | undefined} table
 * @param {string | null | undefined} analysisType the payload's `analysis_type`
 * @returns {boolean}
 */
export function isGroupLevelTable(table, analysisType) {
  if (!table) return false;
  if (analysisType && VARIABLE_ROW_ANALYSES.has(analysisType)) return false;
  const first = Array.isArray(table.columns) ? String(table.columns[0] ?? '') : '';
  if (analysisType === 'descriptive' && first === 'variable') return false;
  return true;
}

/**
 * What to say instead of a table with too many groups.
 * @param {number} levels
 * @returns {string}
 */
export function tableWithheldMessage(levels) {
  return (
    `The table isn’t shown because it has ${levels} groups, and groups that small can describe ` +
    'single patients. Ask again with fewer groups, for example with a measurement put into ranges.'
  );
}

const LEVEL_LABEL = /^([A-Za-z][A-Za-z0-9_]*)=(.+)$/;

/**
 * A group label the engine wrote as "column=value" ("outcome=0"), in words: "Outcome: 0".
 * @param {unknown} text
 * @returns {string}
 */
export function plainLevelLabel(text) {
  const s = String(text ?? '');
  const m = LEVEL_LABEL.exec(s);
  return m ? `${columnLabel(m[1])}: ${m[2]}` : s;
}

const STATS_TOKEN =
  /(\bp\s*[<=>]|\bn\s*=|\bn hidden\b|\bCI\b|\bOR\b|\br\s*=|\bSD\b|\bmedian\b|\bexcluded\b|log-rank|t-test|chi-square|ANOVA|Fisher)/;

/**
 * Index just past the first sentence (a '.' at paren depth 0 followed by a space and a capital,
 * or the end of the text).
 * @param {string} text
 * @returns {number}
 */
function firstSentenceEnd(text) {
  let depth = 0;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === '(') depth += 1;
    else if (ch === ')' && depth > 0) depth -= 1;
    else if (ch === '.' && depth === 0) {
      if (i === text.length - 1) return i + 1;
      if (text[i + 1] === ' ' && /[A-Z]/.test(text[i + 2] || '')) return i + 1;
    }
  }
  return text.length;
}

/**
 * The statistics parenthetical of `sentence`, only when it is the sentence's only top-level
 * group carrying statistics and it closes the sentence. "higher (p=0.03), but not after
 * adjustment (p=0.20)." has two, so nothing is extracted and the headline stays whole.
 * @param {string} sentence
 * @returns {{ start: number, end: number, inner: string } | null}
 */
function findStatsGroup(sentence) {
  let depth = 0;
  let start = -1;
  /** @type {{ start: number, end: number, inner: string }[]} */
  const found = [];
  for (let i = 0; i < sentence.length; i += 1) {
    const ch = sentence[i];
    if (ch === '(') {
      if (depth === 0) start = i;
      depth += 1;
    } else if (ch === ')' && depth > 0) {
      depth -= 1;
      if (depth === 0) {
        const inner = sentence.slice(start + 1, i);
        if (STATS_TOKEN.test(inner)) found.push({ start, end: i, inner });
      }
    }
  }
  if (found.length !== 1) return null;
  const [group] = found;
  return /^\s*\.?\s*$/.test(sentence.slice(group.end + 1)) ? group : null;
}

/**
 * Numbers in a text, sorted (used to prove a split kept every number).
 * @param {string} text
 * @returns {string[]}
 */
export function numbersIn(text) {
  return (String(text).match(/\d+(?:\.\d+)?/g) || []).sort();
}

/**
 * Split an API claim into a headline sentence, its statistics and the remaining caveats.
 * Falls back to the whole claim as the headline if any number would be lost.
 * @param {string | null | undefined} claim
 * @returns {{ headline: string, stats: string[], notes: string }}
 */
export function splitClaim(claim) {
  const text = typeof claim === 'string' ? claim.trim() : '';
  if (!text) return { headline: '', stats: [], notes: '' };

  const cut = firstSentenceEnd(text);
  const sentence = text.slice(0, cut);
  const notes = text.slice(cut).trim();
  const group = findStatsGroup(sentence);

  let headline = sentence;
  let stats = [];
  if (group) {
    const before = sentence.slice(0, group.start).trimEnd();
    const after = sentence.slice(group.end + 1);
    headline = `${before}${after}`.trim();
    stats = group.inner
      .split(';')
      .map((part) => part.trim())
      .filter(Boolean);
  }

  const kept = numbersIn([headline, ...stats, notes].join(' '));
  const original = numbersIn(text);
  if (kept.length !== original.length || kept.some((n, i) => n !== original[i])) {
    return { headline: text, stats: [], notes: '' };
  }
  return { headline, stats, notes };
}

const DEFAULT_DECIMALS = 2;
const MAX_DECIMALS = 3;
const P_FLOOR = 0.001;
const P_COLUMN = /^p([\s_.-]?val(ue)?)?$|(^|[\s_])p[\s_.-]?val(ue)?$/i;

/**
 * Decimal places to show in a table: the most the claim itself shows (1–3), else 2.
 * @param {string | null | undefined} claim
 * @returns {number}
 */
export function claimDecimals(claim) {
  // A p-value's places ("p<0.001", "p=0.120") are its own convention, not the claim's precision.
  const text = String(claim || '').replace(/\bp\s*[<=>≤≥]\s*\d*\.?\d+/gi, '');
  const places = (text.match(/\d+\.(\d+)/g) || []).map((n) => n.split('.')[1].length);
  if (places.length === 0) return DEFAULT_DECIMALS;
  return Math.min(MAX_DECIMALS, Math.max(1, ...places));
}

/**
 * A p-value column, by name ("p", "p_value", "P value", "pvalue").
 * @param {unknown} name
 * @returns {boolean}
 */
export function isPValueColumn(name) {
  return P_COLUMN.test(String(name || '').trim());
}

/**
 * A number with sensible precision: integers unchanged, float noise removed, other numbers
 * at a fixed `decimals` places so a column lines up, and never rounded to zero. P-values keep at least three places and show
 * "<0.001" below that, so a small p is never shown as 0.
 * @param {number} value
 * @param {{ decimals?: number, pValue?: boolean }} [opts]
 * @returns {string}
 */
export function formatNumber(value, opts = {}) {
  if (!Number.isFinite(value)) return String(value);
  // A p-value is never shown as 0 (scipy returns 0.0 for very large effects); check before the integer path.
  if (opts.pValue && value === 0) return `<${P_FLOOR}`;
  if (Number.isInteger(value)) return String(value);
  const clean = Number(value.toPrecision(12));
  if (opts.pValue) {
    if (clean > 0 && clean < P_FLOOR) return `<${P_FLOOR}`;
    const places = Math.max(MAX_DECIMALS, opts.decimals ?? 0);
    return clean.toFixed(places);
  }
  const places = opts.decimals ?? DEFAULT_DECIMALS;
  const rounded = Number(clean.toFixed(places));
  if (rounded === 0 && clean !== 0) return String(Number(clean.toPrecision(2)));
  return clean.toFixed(places);
}

/**
 * How a table cell should be shown.
 * @param {unknown} value
 * @param {{ decimals?: number, pValue?: boolean }} [format] precision for numbers
 * @returns {{ kind: 'empty' | 'hidden' | 'small' | 'number' | 'text', text: string }}
 */
export function classifyCell(value, format = {}) {
  if (value === null || value === undefined || value === '') return { kind: 'empty', text: '' };
  if (typeof value === 'number') {
    return Number.isFinite(value)
      ? { kind: 'number', text: formatNumber(value, format) }
      : { kind: 'text', text: String(value) };
  }
  if (typeof value === 'object') return { kind: 'text', text: JSON.stringify(value) };
  const s = String(value).trim();
  if (isHiddenCellMarker(s)) return { kind: 'hidden', text: 'hidden' };
  if (isSmallCellToken(s)) return { kind: 'small', text: SMALL_CELL_TOKEN };
  if (/^[-+]?\d+\.\d+$/.test(s)) return { kind: 'number', text: formatNumber(Number(s), format) };
  if (format.pValue && /^[-+]?\d+$/.test(s)) {
    return { kind: 'number', text: formatNumber(Number(s), format) };
  }
  if (/^[-+]?\d[\d,]*(\.\d+)?%?$/.test(s)) return { kind: 'number', text: s };
  return { kind: 'text', text: s };
}

/**
 * A column is numeric (right-aligned) when every non-empty cell is a number or a hidden count.
 * @param {unknown[][]} rows
 * @param {number} index
 * @returns {boolean}
 */
export function isNumericColumn(rows, index) {
  let seen = false;
  for (const row of rows) {
    const { kind } = classifyCell(row[index]);
    if (kind === 'empty') continue;
    if (kind === 'text') return false;
    seen = true;
  }
  return seen;
}

const INTENT_LABELS = {
  COMPARE_GROUPS: 'Compare groups',
  DESCRIBE: 'Describe',
  FIND_PREDICTORS: 'Find risk factors',
  CORRELATIONS: 'Correlations',
  COUNT: 'Count',
  SURVIVAL: 'Survival analysis',
};

const PLAN_KEYS = {
  outcome: 'Outcome',
  grouping: 'Compared across',
  predictors: 'Predictors',
  variables: 'Variables',
  time: 'Time',
  event: 'Event',
  statistic: 'Statistic',
  result: 'Result',
  test: 'Test',
};

/** Plan parts whose value is one or more column names. */
const COLUMN_KEYS = new Set(['outcome', 'grouping', 'predictors', 'variables', 'time', 'event']);

/** Shown when a plan part has no value (never the engine's "none"). */
export const NOT_SET = 'Not set';

/**
 * Plan parts that only steer validation or the hold, never what is computed: not shown.
 * "test count" / "test not selected" restate the analysis or say nothing.
 */
const HIDDEN_PLAN_PARTS = new Set([
  'filters required',
  'grouping required',
  'test count',
  'test not selected',
]);

const OPERATOR_WORDS = {
  '==': 'is',
  '!=': 'is not',
  '>': 'over',
  '>=': 'at least',
  '<': 'under',
  '<=': 'at most',
  IN: 'is one of',
  NOT_IN: 'is not one of',
};

const ACRONYMS = new Set([
  'ldl',
  'hdl',
  'bmi',
  'a1c',
  'egfr',
  'crp',
  'sbp',
  'dbp',
  'ecg',
  'ekg',
  'icu',
  'id',
  'hiv',
  'alt',
  'ast',
  'tsh',
  'mri',
  'ct',
  'copd',
  'ckd',
  'mi',
  'inr',
  'wbc',
  'rbc',
  'bp',
  'hr',
]);
const SPECIAL_WORDS = { hba1c: 'HbA1c', ldlc: 'LDL-C', hdlc: 'HDL-C' };
const UNITS = [
  [['mg', 'dl'], 'mg/dL'],
  [['mmol', 'l'], 'mmol/L'],
  [['g', 'dl'], 'g/dL'],
  [['kg', 'm2'], 'kg/m²'],
  [['mm', 'hg'], 'mmHg'],
  [['mmhg'], 'mmHg'],
  [['pct'], '%'],
  [['percent'], '%'],
  [['kg'], 'kg'],
  [['cm'], 'cm'],
  [['yrs'], 'years'],
  [['bpm'], 'bpm'],
];

/**
 * A readable label for a column name: "ldl_mg_dl" → "LDL (mg/dL)", "first_visit" → "First visit".
 * @param {string} column
 * @returns {string}
 */
export function columnLabel(column) {
  let words = String(column || '')
    .trim()
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((w) => w.toLowerCase());
  if (words.length === 0) return '';
  let unit = '';
  for (const [parts, text] of UNITS) {
    if (
      words.length > parts.length &&
      parts.every((p, i) => words[words.length - parts.length + i] === p)
    ) {
      unit = text;
      words = words.slice(0, words.length - parts.length);
      break;
    }
  }
  const shown = words.map((w, i) => {
    if (SPECIAL_WORDS[w]) return SPECIAL_WORDS[w];
    if (ACRONYMS.has(w)) return w.toUpperCase();
    return i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w;
  });
  return unit ? `${shown.join(' ')} (${unit})` : shown.join(' ');
}

/**
 * "LDL (mg/dL) — ldl_mg_dl": the readable label first, the exact column name kept.
 * @param {string} column
 * @returns {string}
 */
export function columnWithLabel(column) {
  const name = String(column || '').trim();
  const label = columnLabel(name);
  return label && label.toLowerCase() !== name.toLowerCase() ? `${label} — ${name}` : label || name;
}

const FILTER_PATTERN = /^(.+?)\s+(==|!=|>=|<=|>|<|NOT_IN|IN)\s+(.*)$/;
const MISSING_KEPT = /\s*\(missing kept\)\s*$/;

/**
 * One filter from the plan line in words: "ldl_mg_dl >= 130 (missing kept)" →
 * "LDL (mg/dL) (ldl_mg_dl) at least 130; patients with no value are kept".
 * @param {string} text
 * @returns {string}
 */
function filterInWords(text) {
  const keepsMissing = MISSING_KEPT.test(text);
  const core = text.replace(MISSING_KEPT, '').trim();
  const match = FILTER_PATTERN.exec(core);
  if (!match) return text;
  const [, column, operator, rawValue] = match;
  const value = rawValue.replace(/^\[(.*)\]$/, '$1');
  const sentence = `${nameInWords(column)} ${OPERATOR_WORDS[operator]} ${value}`;
  return keepsMissing ? `${sentence}; patients with no value are kept` : sentence;
}

const STEP_OPERATORS = { '=': 'is', '>': 'over', '<': 'under', '≥': 'at least', '≤': 'at most' };
const STEP_PATTERN = /^(\S+)\s+(=|>|<|≥|≤)\s+(.+)$/;
const STEP_PHRASE = /^(\S+)\s+(is recorded|above its median\b.*|at or above its\b.*)$/;

/** "LDL (mg/dL) (ldl_mg_dl)": the label, and the exact name when they differ. */
function nameInWords(column) {
  const label = columnLabel(column);
  return label && label.toLowerCase() !== column.toLowerCase()
    ? `${label} (${column})`
    : label || column;
}

/**
 * One cohort limit from "limits in order" in words: "age > 65" → "Age over 65".
 * @param {string} text
 * @returns {string}
 */
function stepInWords(text) {
  const op = STEP_PATTERN.exec(text);
  if (op) return `${nameInWords(op[1])} ${STEP_OPERATORS[op[2]]} ${op[3]}`;
  const phrase = STEP_PHRASE.exec(text);
  if (phrase) return `${nameInWords(phrase[1])} ${phrase[2]}`;
  return text;
}

/**
 * "1) age > 65; 2) ldl ≥ 130" → ["age > 65", "ldl ≥ 130"].
 * @param {string} raw
 * @returns {string[]}
 */
function splitSteps(raw) {
  return raw
    .split(/;\s*(?=\d+\)\s)/)
    .map((step) => step.replace(/^\d+\)\s*/, '').trim())
    .filter(Boolean);
}

const sentenceCase = (text) => text.charAt(0).toUpperCase() + text.slice(1);

/**
 * @typedef {{ key: string, label: string, value: string }} PlanItem
 */

/**
 * Readable form of the API's one-line plan ("COMPARE_GROUPS · outcome ldl · …").
 * Display only: Confirm always sends the original line unchanged.
 *
 * Who was included (filters, then cohort limits in order) is one item, and "Counting" says
 * whether each row or each patient was counted, so an answer never reads as patients when it
 * counted rows. Internal flags are dropped and a missing value reads "Not set".
 * @param {string | null | undefined} planLine
 * @param {{ omitUnset?: boolean }} [opts] omitUnset: leave out parts with no value (answers)
 * @returns {{ analysis: string, items: PlanItem[] }}
 */
export function parsePlanLine(planLine, opts = {}) {
  const parts = String(planLine || '')
    .split(' · ')
    .map((p) => p.trim())
    .filter(Boolean);
  if (parts.length === 0) return { analysis: '', items: [] };
  const [first, ...rest] = parts;
  const analysis =
    INTENT_LABELS[first] ||
    first.charAt(0).toUpperCase() + first.slice(1).toLowerCase().replace(/_/g, ' ');

  const countsPatientsBy = rest
    .map((part) => /^counting distinct (\S+)$/.exec(part))
    .find(Boolean)?.[1];
  const filters = [];
  const limits = [];
  let includedAt = -1;
  /** @type {PlanItem[]} */
  const items = [];

  for (const part of rest) {
    if (HIDDEN_PLAN_PARTS.has(part)) continue;
    const space = part.indexOf(' ');
    const key = space > 0 ? part.slice(0, space) : part;
    const raw = space > 0 ? part.slice(space + 1) : '';
    if (key === 'scope' || key === 'counting') continue;
    if (key === 'filters' || (key === 'limits' && raw.startsWith('in order '))) {
      if (includedAt === -1) includedAt = items.length;
      if (key === 'filters' && raw !== 'none') {
        filters.push(...raw.split('; ').map(filterInWords));
      } else if (key === 'limits') {
        limits.push(...splitSteps(raw.slice('in order '.length)).map(stepInWords));
      }
      continue;
    }
    const label = PLAN_KEYS[key];
    if (!label) {
      items.push({ key: 'other', label: 'Other', value: part });
      continue;
    }
    if (raw === 'none' || raw === '') {
      if (!opts.omitUnset) items.push({ key, label, value: NOT_SET });
      continue;
    }
    if (COLUMN_KEYS.has(key)) {
      items.push({ key, label, value: raw.split(', ').map(columnWithLabel).join(', ') });
    } else {
      items.push({ key, label, value: sentenceCase(raw) });
    }
  }

  if (includedAt !== -1) {
    const who = [filters.join(' and '), limits.join(', then ')].filter(Boolean).join(', then ');
    const everyone = countsPatientsBy ? 'All patients' : 'All rows';
    items.splice(
      includedAt,
      0,
      { key: 'included', label: 'Included', value: who || everyone },
      {
        key: 'counting',
        label: 'Counting',
        value: countsPatientsBy ? `Patients (each ${columnLabel(countsPatientsBy)} once)` : 'Rows',
      }
    );
  } else if (countsPatientsBy) {
    items.push({
      key: 'counting',
      label: 'Counting',
      value: `Patients (each ${columnLabel(countsPatientsBy)} once)`,
    });
  }
  return { analysis, items };
}

const quoteColumn = (c) => `“${c}”`;

/**
 * Known engine and validator messages, each as a plain sentence with a next step.
 * @type {[RegExp, (m: RegExpExecArray) => string][]}
 */
const ENGINE_MESSAGES = [
  [
    /(?:Filter )?[Cc]olumn '([^']+)' not found/,
    (m) =>
      `This cohort has no variable called ${quoteColumn(m[1])}. Check the spelling, or open the cohort details to see its variables.`,
  ],
  [
    /Type mismatch for column '([^']+)'.*got string '([^']*)'/,
    (m) =>
      `${quoteColumn(m[2])} isn’t a number, but ${columnLabel(m[1])} holds numbers. Ask again with a number, for example “over 65”.`,
  ],
  [
    /\bDESCRIBE\b.*requires/,
    () =>
      'The question didn’t name what to summarise. Ask again naming the measurement, for example “What is the average age?”.',
  ],
  [
    /\bCOMPARE_GROUPS\b.*requires/,
    () =>
      'To compare groups, the question needs a measurement and the groups to compare. Ask again in the form “Compare age by sex”.',
  ],
  [
    /requires grouping/,
    () =>
      'The question asks for a breakdown but doesn’t say by what. Add “by” and a variable, for example “by sex”.',
  ],
  [
    /Cannot group by entity key/,
    () =>
      'That variable identifies individual patients, so it can’t form groups. Choose a variable such as sex or treatment.',
  ],
  [
    /Time and event variables required/,
    () =>
      'Survival analysis needs a follow-up time and an event, such as death. Ask again naming both.',
  ],
  [
    /scope='all' with filters/,
    () => 'The question asks about all patients and a subgroup at once. Ask about one of them.',
  ],
  [
    /scope='filtered' requires filters/,
    () =>
      'The question asks about a subgroup but doesn’t say which patients. Say who to include, for example “patients over 65”.',
  ],
  [
    /Invalid operator/,
    () =>
      'The question uses a comparison the analysis can’t apply. Use words like is, is not, over, under, at least or at most.',
  ],
  [
    /All parsing tiers failed/,
    () =>
      'The question couldn’t be matched to an analysis. Ask again using a variable name from the cohort.',
  ],
];

/** Words that only make sense to the people who built the engine. */
const ENGINE_JARGON = new RegExp(
  [
    '\\bLLM\\b',
    '\\b[A-Z]{3,}_[A-Z_]+\\b',
    '\\b(primary_variable|group_by|grouping_variable|entity_key)\\b',
    '\\bintent requires\\b',
    '\\b[A-Z]+_?[A-Z]*\\s+intent\\b',
    'QueryPlan',
    '\\bscope=',
    'Traceback',
    'Exception',
    '\\b\\w+Error:',
    'NoneType',
    '\\bactive version schema\\b',
    '\\bvalidation should\\b',
    // Python and engine exception text
    'has no attribute',
    "'\\w+' object is not",
    'object is not (callable|subscriptable|iterable)',
    'unsupported operand',
    '\\b\\w+Column\\b',
    '\\bdtype\\b',
    '\\b(polars|duckdb|pyarrow|numpy|scipy)\\b',
    '\\b(Binder|Catalog|Conversion|Parser|Invalid Input|Internal) Error\\b',
    '\\bExecution (error|failed)\\b',
  ].join('|')
);

const GENERIC_ENGINE_MESSAGE =
  'The analysis couldn’t use this question as written. Ask again, naming the variables you mean.';

/** An analysis that started and then crashed: an error, never a refusal. */
const EXECUTION_FAILURE = /^\s*Execution (?:error|failed)\b/i;
const EXECUTION_FAILURE_MESSAGE =
  'The analysis couldn’t finish this question. Ask again in other words, naming the variables ' +
  'you mean; if it keeps happening, restart the app.';

/** The request never got an answer from the engine: network failures and aborted fetches only. */
const ENGINE_UNREACHABLE = new RegExp(
  [
    'Failed to fetch',
    'NetworkError',
    'Load failed',
    'ECONNREFUSED',
    'ECONNRESET',
    'ETIMEDOUT',
    'socket hang up',
    'net::ERR_',
    '^Stream failed$',
  ].join('|'),
  'i'
);

/** HTTP 5xx status text with no status attached: the engine answered, then failed. */
const SERVER_FAILURE_TEXT =
  /\b(Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout)\b/i;

/** "[HTTP 404] Dataset not found: abc": how preload hands a failed response to the renderer. */
const HTTP_FAILURE = /^\[HTTP (\d{3})\]\s*([\s\S]*)$/;

/**
 * Server sentences already written for people. They are kept as written (with any exception
 * text after them cut), before the jargon test, so their recovery step reaches the user.
 */
const SERVER_SENTENCES = [
  /^(Dataset only partly deleted, try again)\b/,
  /^(Dataset files deleted but its chats were not; try the delete again)\b/,
  /^(The saved data contract for this cohort is damaged)\b/,
];

/** 404 details naming what no longer exists, and what to say instead. */
const GONE = [
  [
    /\bDataset\b.*\bnot found\b|\bdataset_version not found\b/i,
    'This cohort no longer exists. Choose another cohort, or refresh the list of cohorts.',
  ],
  [/\bSession\b.*\bnot found\b/i, 'This conversation no longer exists. Start a new question.'],
  [/\bQuery\b.*\bnot found\b/i, 'This answer is no longer available. Ask the question again.'],
  [
    /\bPatch not found\b/i,
    'That change no longer exists. Reopen the cohort details to see its changes.',
  ],
];

/** @typedef {'question' | 'upload' | 'delete' | 'panel'} FailureContext */

/** What the user names when a request field is missing or empty, by request field. */
const FIELD_WORDS = {
  query_text: ['The question is empty.', 'Type a question'],
  dataset_id: ['No cohort was chosen.', 'Choose a cohort'],
  session_id: ['No conversation was open.', 'Start a new question'],
  file: ['No file was chosen.', 'Choose a file'],
};

/**
 * Encode a failed HTTP response so the status survives the context bridge (only an Error's
 * message crosses it). A FastAPI validation list is kept as JSON.
 * @param {number} status
 * @param {unknown} detail the response's `detail`
 * @returns {string}
 */
export function httpFailureText(status, detail) {
  let reason = '';
  if (typeof detail === 'string') reason = detail.trim();
  else if (Array.isArray(detail)) reason = JSON.stringify(detail);
  return reason ? `[HTTP ${status}] ${reason}` : `[HTTP ${status}]`;
}

/**
 * A FastAPI/Pydantic 422 list in plain words, or '' when no entry is recognised.
 * @param {string} reason JSON from httpFailureText
 * @param {string} retry
 * @returns {string}
 */
function validationInWords(reason, retry) {
  let items;
  try {
    items = JSON.parse(reason);
  } catch {
    return '';
  }
  if (!Array.isArray(items)) return '';
  for (const item of items) {
    const loc = Array.isArray(item?.loc) ? item.loc : [];
    const field = String(loc[loc.length - 1] ?? '');
    const type = String(item?.type ?? '');
    if (type === 'extra_forbidden') {
      return `The analysis engine didn’t accept the request from the app. Restart the app, then ${retry.toLowerCase()}.`;
    }
    const words = FIELD_WORDS[field];
    if (words && /missing|too_short|string_type|none_required/.test(type)) {
      return `${words[0]} ${words[1]}, then ${retry.toLowerCase()}.`;
    }
  }
  return `The analysis engine didn’t accept the request from the app. Restart the app, then ${retry.toLowerCase()}.`;
}

/**
 * @param {string} retry what to do first ("Ask again", "Try again")
 * @returns {string}
 */
function engineUnreachable(retry) {
  return `Couldn’t reach the analysis engine. ${retry}; if it keeps happening, restart the app.`;
}

/**
 * @param {FailureContext} context
 * @returns {string}
 */
function serverFailure(context) {
  return context === 'question'
    ? 'The analysis engine ran into a problem with this question. Ask again in other words; if ' +
        'it keeps happening, restart the app.'
    : 'The analysis engine ran into a problem. Try again; if it keeps happening, restart the app.';
}

/** Exception text outside a question: what to check, for the action the user took. */
const CONTEXT_FALLBACK = {
  upload:
    'The file couldn’t be read. Check that it is a CSV, Excel or SPSS file with a header row, then try again.',
  delete:
    'The analysis engine couldn’t finish the delete. Try again; if it keeps happening, restart the app.',
  panel: 'Try again; if it keeps happening, restart the app.',
};

/** Shown when a request to the engine failed without a message written for people. */
export const ENGINE_UNREACHABLE_MESSAGE = engineUnreachable('Ask again');

/**
 * Any message on its way to the screen, in plain language, for the action that failed.
 * Network failures get "couldn't reach"; an HTTP 5xx gets "ran into a problem"; a 404 says what
 * no longer exists; a 422 gives the request's problem in words. Server sentences written for
 * people are kept. Engine wording gets its own sentence only for questions; elsewhere exception
 * text gets a fallback about that action, never about "this question".
 * @param {unknown} text
 * @param {{ context?: FailureContext }} [opts]
 * @returns {string}
 */
export function plainFailureMessage(text, opts = {}) {
  const context = opts.context ?? 'question';
  const retry = context === 'question' ? 'Ask again' : 'Try again';
  let raw = typeof text === 'string' ? text.trim() : '';
  let status = 0;
  const http = HTTP_FAILURE.exec(raw);
  if (http) {
    status = Number(http[1]);
    raw = http[2].trim();
  }
  if (!status && (!raw || ENGINE_UNREACHABLE.test(raw))) return engineUnreachable(retry);

  for (const pattern of SERVER_SENTENCES) {
    const match = pattern.exec(raw);
    if (match) return `${match[1]}.`;
  }
  if (status === 404) {
    for (const [pattern, words] of GONE) if (pattern.test(raw)) return words;
  }
  if (status === 422) {
    const words = validationInWords(raw, retry);
    if (words) return words;
  }
  if (status >= 500 && (!raw || SERVER_FAILURE_TEXT.test(raw))) return serverFailure(context);
  if (!status && SERVER_FAILURE_TEXT.test(raw)) return serverFailure(context);
  if (!raw) {
    return status >= 400 && status < 500 && context !== 'question'
      ? CONTEXT_FALLBACK[context]
      : serverFailure(context);
  }

  if (context === 'question') {
    if (EXECUTION_FAILURE.test(raw)) return EXECUTION_FAILURE_MESSAGE;
    for (const [pattern, toPlain] of ENGINE_MESSAGES) {
      const match = pattern.exec(raw);
      if (match) return toPlain(match);
    }
    if (ENGINE_JARGON.test(raw)) return GENERIC_ENGINE_MESSAGE;
  } else if (ENGINE_JARGON.test(raw) || EXECUTION_FAILURE.test(raw) || /\bnot found\b/.test(raw)) {
    return status >= 500 && context !== 'upload'
      ? serverFailure(context)
      : CONTEXT_FALLBACK[context];
  }
  return raw;
}

/**
 * A failed query is either a refusal (the product declining, with a reason) or an error (the
 * analysis crashed, or nothing came back). `detail` holds the engine's exact words when the
 * message was reworded; it is offered for copying, never shown.
 * @param {Record<string, unknown> | null | undefined} data query_failed payload
 * @returns {{ kind: 'refused' | 'error', message: string, detail?: string }}
 */
export function describeQueryFailure(data) {
  const d = asRecord(data);
  const pick = (v) => (typeof v === 'string' && v.trim() ? v.trim() : '');
  const reason = pick(d.failure_reason) || pick(d.error);
  const kind = d.error_type || !reason || EXECUTION_FAILURE.test(reason) ? 'error' : 'refused';
  if (!reason) return { kind, message: 'The analysis stopped before it could answer.' };
  const message = plainFailureMessage(reason);
  return message === reason ? { kind, message } : { kind, message, detail: reason };
}

/**
 * Plain-language progress text for a query_progress event.
 * @param {Record<string, unknown> | null | undefined} data
 * @returns {string}
 */
export function stageLabel(data) {
  const d = asRecord(data);
  const stage = typeof d.stage === 'string' ? d.stage.toLowerCase() : '';
  if (/pars|read/.test(stage)) return 'Reading your question';
  if (/plan/.test(stage)) return 'Choosing the variables and the test';
  if (/exec|run/.test(stage)) return 'Running the analysis';
  return typeof d.message === 'string' ? d.message : '';
}

const ID_COLUMN = /(^|_)(id|mrn|uuid)$|^id_/i;

/**
 * Up to three example questions built from the cohort's own columns.
 * @param {string[]} columns
 * @param {Record<string, unknown>[]} rows preview rows
 * @param {Record<string, string>} [schema] column → type from the dataset detail
 * @returns {string[]}
 */
export function buildExampleQuestions(columns, rows, schema = {}) {
  const usable = (Array.isArray(columns) ? columns : []).filter(
    (c) => typeof c === 'string' && c && !ID_COLUMN.test(c)
  );
  const sample = Array.isArray(rows) ? rows : [];
  const numeric = [];
  const groups = [];
  for (const col of usable) {
    const type = String(schema?.[col] ?? '').toLowerCase();
    const values = sample
      .map((r) => r?.[col])
      .filter((v) => v !== null && v !== undefined && v !== '');
    const distinct = new Set(values.map(String)).size;
    const allNumbers = values.length > 0 && values.every((v) => typeof v === 'number');
    if (
      /categor|binary|bool/.test(type) ||
      (!type && !allNumbers && distinct > 0 && distinct <= 6)
    ) {
      groups.push(col);
    } else if (/numer|contin|float|int/.test(type) || (!type && allNumbers && distinct > 5)) {
      numeric.push(col);
    }
  }
  const out = [];
  if (numeric[0]) out.push(`What is the average ${numeric[0]}?`);
  if (numeric.length && groups[0]) {
    out.push(`Compare ${numeric[1] || numeric[0]} between ${groups[0]} groups`);
  }
  if (groups.length) out.push(`How many patients are in each ${groups[1] || groups[0]} group?`);
  return out.slice(0, 3);
}

// ============================================================================
// Hold next step: name the variables the cohort has that are closest to the words left out.
// ============================================================================

/**
 * Everyday clinical words and the variable-name fragments they usually mean. A word that is a
 * typo or a form of a variable name ("diabete") is matched by spelling instead.
 */
const CLINICAL_HINTS = {
  cholesterol: ['ldl', 'hdl', 'chol', 'trig'],
  lipid: ['ldl', 'hdl', 'chol', 'trig'],
  lipids: ['ldl', 'hdl', 'chol', 'trig'],
  sugar: ['glucose', 'a1c', 'hba1c'],
  glucose: ['glucose', 'a1c', 'hba1c'],
  pressure: ['sbp', 'dbp', 'bp', 'systolic', 'diastolic'],
  weight: ['bmi', 'weight'],
  obesity: ['bmi', 'weight'],
  kidney: ['egfr', 'creatinine', 'ckd'],
  renal: ['egfr', 'creatinine', 'ckd'],
  death: ['death', 'died', 'mortality', 'event'],
  died: ['death', 'died', 'mortality', 'event'],
  mortality: ['death', 'died', 'mortality', 'event'],
  survival: ['followup', 'event', 'death'],
  gender: ['sex', 'gender'],
  male: ['sex', 'gender'],
  female: ['sex', 'gender'],
  women: ['sex', 'gender'],
  men: ['sex', 'gender'],
  drug: ['treatment', 'arm', 'drug'],
  therapy: ['treatment', 'arm'],
  arm: ['treatment', 'arm'],
  hospital: ['site', 'centre', 'center', 'hospital'],
  centre: ['site', 'centre', 'center'],
  center: ['site', 'centre', 'center'],
  smoking: ['smok'],
  smoker: ['smok'],
  older: ['age'],
  elderly: ['age'],
};

/** Edit distance, capped: enough to tell a typo from a different word. */
function editDistance(a, b) {
  const row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i += 1) {
    let prev = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const next = Math.min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = row[j];
      row[j] = next;
    }
  }
  return row[b.length];
}

/** Same word, or a typo or another form of it (at least four letters). */
function spelledLike(word, part) {
  if (word === part) return true;
  if (word.length < 4 || part.length < 4) return false;
  if (part.startsWith(word) || word.startsWith(part)) return true;
  return editDistance(word, part) <= (Math.min(word.length, part.length) >= 7 ? 2 : 1);
}

/**
 * Up to `limit` of the cohort's variables closest to words the plan could not place.
 * Identifier columns are never suggested.
 * @param {string[]} words
 * @param {string[]} columns
 * @param {number} [limit]
 * @returns {string[]}
 */
export function suggestColumns(words, columns, limit = 3) {
  const usable = (Array.isArray(columns) ? columns : []).filter(
    (c) => typeof c === 'string' && c && !ID_COLUMN.test(c)
  );
  const parts = new Map(
    usable.map((c) => [
      c,
      c
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter(Boolean),
    ])
  );
  const out = [];
  for (const rawWord of words) {
    const word = String(rawWord || '').toLowerCase();
    const hints = CLINICAL_HINTS[word];
    for (const column of usable) {
      if (out.includes(column)) continue;
      const pieces = parts.get(column);
      const matches = hints
        ? pieces.some((p) => hints.some((h) => p.startsWith(h)))
        : pieces.some((p) => spelledLike(word, p));
      if (matches) out.push(column);
    }
  }
  return out.slice(0, limit);
}

const DROPPED_WORDS =
  /couldn't place the words?\s+(.+?),\s+so\s+(?:it is|they are)\s+not in the plan/i;

const LIST_FORMAT = new Intl.ListFormat('en-GB', { style: 'long', type: 'conjunction' });

/** "A", "A and B", "A, B and C". */
function listed(items) {
  return LIST_FORMAT.format(items);
}

/**
 * The next step under a held plan's reason. When words were left out, name the cohort's closest
 * variables (or point to their names) and make "Not what I meant" the primary action.
 * @param {string | null | undefined} reason the API's hold reason
 * @param {string[]} columns the cohort's variables
 * @returns {{ dropped: boolean, words: string[], suggestions: string[], text: string }}
 */
export function holdNextStep(reason, columns) {
  const match = DROPPED_WORDS.exec(String(reason || ''));
  if (!match) {
    return {
      dropped: false,
      words: [],
      suggestions: [],
      text:
        'If the plan below matches your question, run it. If not, choose “Not what I meant” and ' +
        'ask again in other words.',
    };
  }
  const words = [...match[1].matchAll(/'([^']+)'/g)].map((m) => m[1]);
  const suggestions = suggestColumns(words, columns);
  // The name the engine accepts is shown with its label: "LDL (mg/dL) (ldl_mg_dl)".
  const names = listed(suggestions.map((c) => nameInWords(c)));
  let text;
  if (suggestions.length === 1) {
    text = `This cohort has ${names}. If you meant it, choose “Not what I meant” and ask again using that name.`;
  } else if (suggestions.length) {
    text = `This cohort has ${names}. If you meant one of them, choose “Not what I meant” and ask again using its name.`;
  } else {
    text =
      `If ${words.length === 1 ? 'that word matters' : 'those words matter'}, choose “Not what ` +
      'I meant” and ask again using the cohort’s variable names (see cohort details).';
  }
  return { dropped: true, words, suggestions, text };
}
