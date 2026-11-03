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
      parts.push(`Columns: ${numericColumns.join(', ')}`);
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
 * @returns {{ content: string, result: { intentType: string, title: string | null, summary: string | null, claim?: string, table?: TablePayload } | null }}
 */
export function buildQueryCompletedPresentation(intentRaw, resultPreview, claimArg) {
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
  const content = claim
    ? title || 'Analysis complete.'
    : buildCompletionMessage(intentType, rowCount, hasTable);
  const summary = claim || buildIntentSummary(intentType, resultPreview, rowCount);

  if (!extracted && !title && !summary && !claim) {
    return { content, result: null };
  }

  const result = {
    intentType,
    title,
    summary,
    ...(claim ? { claim } : {}),
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
 * @param {{ plan_line?: string, failure_reason?: string, confidence?: number } | null | undefined} data
 * @param {string} pendingQuery
 * @param {{ datasetId?: string | null, sessionId?: string | null }} [binding] where the plan was held
 * @returns {{ content: string, result: { awaitingConfirmation: true, planLine: string, failureReason: string, pendingQuery: string, confidence: number | null } }}
 */
export function buildConfirmationHoldPresentation(data, pendingQuery, binding = {}) {
  const planLine = typeof data?.plan_line === 'string' ? data.plan_line : '';
  const failureReason =
    typeof data?.failure_reason === 'string' && data.failure_reason
      ? data.failure_reason
      : 'Confirmation required before execution.';
  const confidence = typeof data?.confidence === 'number' ? data.confidence : null;
  // The plan line renders once, inside the card; the message text carries only the reason.
  return {
    content: failureReason,
    result: {
      awaitingConfirmation: true,
      planLine,
      failureReason,
      pendingQuery: pendingQuery || '',
      confidence,
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

export const HIDDEN_CELL_FOOTNOTE =
  `Hidden to protect privacy: counts from 1 to ${SMALL_CELL_THRESHOLD - 1} are not shown, ` +
  "and some other cells are hidden so those counts can't be worked out from totals.";

const STATS_TOKEN =
  /(\bp\s*[<=>]|\bn\s*=|\bCI\b|\bOR\b|\br\s*=|\bSD\b|\bmedian\b|\bexcluded\b|log-rank|t-test|chi-square|ANOVA|Fisher)/;

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
 * Last top-level parenthetical in `sentence` that carries statistics.
 * @param {string} sentence
 * @returns {{ start: number, end: number, inner: string } | null}
 */
function findStatsGroup(sentence) {
  let depth = 0;
  let start = -1;
  let found = null;
  for (let i = 0; i < sentence.length; i += 1) {
    const ch = sentence[i];
    if (ch === '(') {
      if (depth === 0) start = i;
      depth += 1;
    } else if (ch === ')' && depth > 0) {
      depth -= 1;
      if (depth === 0) {
        const inner = sentence.slice(start + 1, i);
        if (STATS_TOKEN.test(inner)) found = { start, end: i, inner };
      }
    }
  }
  return found;
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

/**
 * How a table cell should be shown.
 * @param {unknown} value
 * @returns {{ kind: 'empty' | 'hidden' | 'small' | 'number' | 'text', text: string }}
 */
export function classifyCell(value) {
  if (value === null || value === undefined || value === '') return { kind: 'empty', text: '' };
  if (typeof value === 'number') {
    return Number.isFinite(value)
      ? { kind: 'number', text: String(value) }
      : { kind: 'text', text: String(value) };
  }
  if (typeof value === 'object') return { kind: 'text', text: JSON.stringify(value) };
  const s = String(value).trim();
  if (s.toLowerCase() === HIDDEN_CELL_MARKER) return { kind: 'hidden', text: 'hidden' };
  if (/^<\s*\d+$/.test(s)) return { kind: 'small', text: s.replace(/\s+/g, '') };
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
  filters: 'Patients included',
  scope: 'Scope',
  test: 'Test',
};

/**
 * Readable form of the API's one-line plan ("COMPARE_GROUPS · outcome ldl · …").
 * Display only: Confirm always sends the original line unchanged.
 * @param {string | null | undefined} planLine
 * @returns {{ analysis: string, items: { label: string, value: string }[] }}
 */
export function parsePlanLine(planLine) {
  const parts = String(planLine || '')
    .split(' · ')
    .map((p) => p.trim())
    .filter(Boolean);
  if (parts.length === 0) return { analysis: '', items: [] };
  const [first, ...rest] = parts;
  const analysis =
    INTENT_LABELS[first] ||
    first.charAt(0).toUpperCase() + first.slice(1).toLowerCase().replace(/_/g, ' ');
  const items = rest.map((part) => {
    const space = part.indexOf(' ');
    const key = space > 0 ? part.slice(0, space) : part;
    const raw = space > 0 ? part.slice(space + 1) : '';
    const label = PLAN_KEYS[key];
    if (!label) return { label: '', value: part };
    if (raw === 'none') return { label, value: key === 'filters' ? 'All patients' : 'None' };
    return { label, value: raw };
  });
  return { analysis, items };
}

/**
 * A failed query is either a refusal (the product declining, with a reason) or an error.
 * @param {Record<string, unknown> | null | undefined} data query_failed payload
 * @returns {{ kind: 'refused' | 'error', message: string }}
 */
export function describeQueryFailure(data) {
  const d = asRecord(data);
  const pick = (v) => (typeof v === 'string' && v.trim() ? v.trim() : '');
  const reason = pick(d.failure_reason) || pick(d.error);
  if (d.error_type || !reason) {
    return { kind: 'error', message: reason || 'The analysis stopped before it could answer.' };
  }
  return { kind: 'refused', message: reason };
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
  if (/plan/.test(stage)) return 'Choosing the columns and the test';
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
