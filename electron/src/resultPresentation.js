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
