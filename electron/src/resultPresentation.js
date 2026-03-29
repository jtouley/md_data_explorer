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
 * @param {string | null | undefined} intentRaw
 * @param {Record<string, unknown> | null | undefined} resultPreview
 * @returns {{ content: string, result: { intentType: string, title: string | null, table: TablePayload } | null }}
 */
export function buildQueryCompletedPresentation(intentRaw, resultPreview) {
  const intentType = normalizeIntentType(intentRaw);
  const extracted = extractTableFromResultPreview(resultPreview);
  const hasTable = extracted != null;
  const rowCount = extracted ? extracted.rowCount : 0;
  const content = buildCompletionMessage(intentType, rowCount, hasTable);
  const title = resultTitleForIntent(intentType);

  if (!extracted) {
    return { content, result: null };
  }

  return {
    content,
    result: {
      intentType,
      title,
      table: extracted.table,
    },
  };
}
