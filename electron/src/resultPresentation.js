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
  "and some other cells are hidden so those counts can't be worked out from totals.";

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
  const places = (String(claim || '').match(/\d+\.(\d+)/g) || []).map(
    (n) => n.split('.')[1].length
  );
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
  filters: 'Patients included',
  scope: 'Scope',
  test: 'Test',
};

/** Plan parts whose value is one or more column names. */
const COLUMN_KEYS = new Set(['outcome', 'grouping', 'predictors', 'variables', 'time', 'event']);

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
  const label = columnLabel(column);
  const name =
    label && label.toLowerCase() !== column.toLowerCase()
      ? `${label} (${column})`
      : label || column;
  const value = rawValue.replace(/^\[(.*)\]$/, '$1');
  const sentence = `${name} ${OPERATOR_WORDS[operator]} ${value}`;
  return keepsMissing ? `${sentence}; patients with no value are kept` : sentence;
}

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
    if (key === 'filters') {
      return { label, value: raw.split('; ').map(filterInWords).join(' and ') };
    }
    if (COLUMN_KEYS.has(key)) {
      return { label, value: raw.split(', ').map(columnWithLabel).join(', ') };
    }
    return { label, value: raw };
  });
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
      `This cohort has no column called ${quoteColumn(m[1])}. Check the spelling, or open the cohort details to see its columns.`,
  ],
  [
    /Type mismatch for column '([^']+)'.*got string '([^']*)'/,
    (m) =>
      `${quoteColumn(m[2])} isn't a number, but ${columnLabel(m[1])} holds numbers. Ask again with a number, for example “over 65”.`,
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
      'The question asks for a breakdown but doesn’t say by what. Add “by” and a column, for example “by sex”.',
  ],
  [
    /Cannot group by entity key/,
    () =>
      'That column identifies individual patients, so it can’t form groups. Choose a column such as sex or treatment.',
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
      'The question couldn’t be matched to an analysis. Ask again using a column name from the cohort.',
  ],
];

/** Words that only make sense to the people who built the engine. */
const ENGINE_JARGON =
  /\bLLM\b|\b[A-Z]{3,}_[A-Z_]+\b|\b(primary_variable|group_by|grouping_variable|entity_key)\b|\bintent requires\b|\b[A-Z]+_?[A-Z]*\s+intent\b|QueryPlan|\bscope=|Traceback|Exception|\b\w+Error:|NoneType|\bactive version schema\b|\bvalidation should\b/;

const GENERIC_ENGINE_MESSAGE =
  'The analysis couldn’t use this question as written. Ask again, naming the columns you mean.';

/**
 * Engine wording in plain language. Unknown text with engine jargon gets a generic sentence;
 * anything else is already written for people and is kept.
 * @param {string} text
 * @returns {string}
 */
export function plainFailureMessage(text) {
  for (const [pattern, toPlain] of ENGINE_MESSAGES) {
    const match = pattern.exec(text);
    if (match) return toPlain(match);
  }
  return ENGINE_JARGON.test(text) ? GENERIC_ENGINE_MESSAGE : text;
}

/**
 * A failed query is either a refusal (the product declining, with a reason) or an error.
 * `detail` holds the engine's exact words when the message was reworded.
 * @param {Record<string, unknown> | null | undefined} data query_failed payload
 * @returns {{ kind: 'refused' | 'error', message: string, detail?: string }}
 */
export function describeQueryFailure(data) {
  const d = asRecord(data);
  const pick = (v) => (typeof v === 'string' && v.trim() ? v.trim() : '');
  const reason = pick(d.failure_reason) || pick(d.error);
  const kind = d.error_type || !reason ? 'error' : 'refused';
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
