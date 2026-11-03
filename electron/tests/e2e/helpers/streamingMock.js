/**
 * A clinicalAPI mock that streams SSE-shaped events, for conversation and theme specs.
 * Answers use the API's exact claim and plan-line formats and its hidden-cell markers.
 */

const ANSWER_CLAIM =
  'Mean ldl was 16.14 higher in placebo than drug (95% CI 13.44 to 18.83; Welch t-test p<0.001; ' +
  'n=377; 23 excluded for missing data). Unadjusted comparison by treatment.';
const PLAN_LINE =
  'COMPARE_GROUPS · outcome ldl_mg_dl · grouping treatment · filters age > 65; first_visit >= 2024-01-01 (missing kept) · test Welch t-test';
const ENGINE_FAILURE = 'DESCRIBE intent requires metric (primary_variable)';
const WIDE_COLUMNS = [
  'treatment',
  'n',
  'mean_age',
  'mean_ldl_mg_dl',
  'sd_ldl_mg_dl',
  'median_ldl_mg_dl',
  'mean_hdl_mg_dl',
  'mean_bmi',
  'pct_female',
  'pct_diabetes',
  'mean_sbp',
  'p_value',
];
const REFUSAL =
  'This would list individual patients. I can only report group-level statistics; try asking for counts or averages.';

function installMock({ answerClaim, planLine, refusal, engineFailure, wideColumns }) {
  const calls = [];
  window.__calls = calls;
  window.__streamMs = 150;
  const pending = {};
  const now = new Date().toISOString();
  window.clinicalAPI = {
    healthCheck: async () => ({ status: 'healthy' }),
    listDatasets: async () => ({
      datasets: [
        { dataset_id: 'ds_cohort', name: 'cohort', row_count: 400 },
        ...(window.__sampleLoaded
          ? [{ dataset_id: 'ds_sample', name: 'sample', row_count: 50 }]
          : []),
      ],
      total: window.__sampleLoaded ? 2 : 1,
    }),
    uploadDataset: async () => ({ status: 'failed', message: 'not in this mock' }),
    loadSampleDataset: async () => {
      calls.push(['loadSampleDataset']);
      window.__sampleLoaded = true;
      return { status: 'ready', upload_id: 'ds_sample', dataset_name: 'sample' };
    },
    getDataset: async () => ({
      dataset_id: 'ds_cohort',
      tables: [{ name: 'unified', row_count: 400 }],
      column_schema: { age: 'numeric', ldl: 'numeric', treatment: 'categorical' },
    }),
    previewDataset: async () => ({
      columns: ['patient_id', 'age', 'ldl', 'treatment'],
      rows: [{ patient_id: 'S1', age: 61, ldl: 120.5, treatment: 'drug' }],
      total_rows: 400,
    }),
    listSessions: async () => ({ sessions: window.__sessions || [], total: 0 }),
    createSession: async (datasetId) => {
      calls.push(['createSession', datasetId]);
      const s = {
        session_id: 'sess_created',
        dataset_id: datasetId,
        created_at: now,
        updated_at: now,
      };
      window.__sessions = [s];
      return s;
    },
    deleteSession: async (id) => {
      calls.push(['deleteSession', id]);
      window.__sessions = (window.__sessions || []).filter((s) => s.session_id !== id);
    },
    getPendingEnrichments: async () => ({ suggestions: [], total: 0 }),
    getEnrichmentHistory: async () => ({ patches: [], total: 0 }),
    submitQuery: async (sessionId, datasetId, text, confirmed, plan) => {
      calls.push(['submitQuery', sessionId, datasetId, text, confirmed, plan ?? null]);
      const id = `q${calls.length}`;
      const t = text.toLowerCase();
      pending[id] = confirmed
        ? 'answer'
        : t.includes('older')
          ? 'hold'
          : t.includes('patients over')
            ? 'refuse'
            : t.includes('slow')
              ? 'slow'
              : t.includes('wide')
                ? 'wide'
                : t.includes('nothing')
                  ? 'engine'
                  : t.includes('table')
                    ? 'table'
                    : 'answer';
      return { query_id: id, stream_url: `/stream/${id}` };
    },
    subscribeToQueryStream: (queryId, { onEvent }) => {
      const kind = pending[queryId];
      const ms = window.__streamMs;
      const events = [[0, 'query_progress', { stage: 'parsing', message: 'Analyzing query...' }]];
      if (kind === 'slow') {
        events.push([60000, 'query_completed', { intent_type: 'DESCRIBE', claim: 'Late.' }]);
      } else if (kind === 'hold') {
        events.push([
          ms,
          'query_awaiting_confirmation',
          {
            plan_line: window.__holdPlanLine ?? planLine,
            failure_reason: "I couldn't place the word 'recently', so it is not in the plan.",
            confidence: 0.6,
          },
        ]);
      } else if (kind === 'refuse') {
        events.push([ms, 'query_failed', { failure_reason: refusal, error: refusal }]);
      } else if (kind === 'engine') {
        events.push([ms, 'query_failed', { failure_reason: engineFailure, error: engineFailure }]);
      } else if (kind === 'wide') {
        const row = (name, base) =>
          Object.fromEntries(
            wideColumns.map((c, i) => [c, i === 0 ? name : i === 1 ? base : base / 3 + i / 7])
          );
        const drug = row('drug', 190);
        drug.mean_age = 64.23400000000001;
        drug.p_value = 0.00004;
        events.push([
          ms,
          'query_completed',
          {
            intent_type: 'DESCRIBE',
            claim: 'Mean age was 64.2 in drug (n=190).',
            result_preview: {
              table: { columns: wideColumns, rows: [drug, row('placebo', 187)] },
              row_count: 2,
            },
          },
        ]);
      } else if (kind === 'table') {
        events.push([
          ms,
          'query_completed',
          {
            intent_type: 'COMPARE_GROUPS',
            claim:
              'event = 1 was 31% in drug and 24% in placebo; the difference was not statistically significant (chi-square p=0.120; n=388).',
            result_preview: {
              table: {
                columns: ['event', 'drug', 'placebo'],
                rows: [
                  { event: 'No event', drug: 134, placebo: 148 },
                  { event: 'Event', drug: 61, placebo: '<11' },
                  { event: 'Not recorded', drug: 'suppressed', placebo: 45 },
                ],
              },
              row_count: 3,
            },
          },
        ]);
      } else {
        events.push([
          ms,
          'query_completed',
          {
            intent_type: 'COMPARE_GROUPS',
            claim: answerClaim,
            result_preview: {
              table: {
                columns: ['treatment', 'mean ldl', 'n'],
                rows: [
                  { treatment: 'drug', 'mean ldl': 116.9, n: 190 },
                  { treatment: 'placebo', 'mean ldl': 133.04, n: 187 },
                ],
              },
              row_count: 2,
            },
          },
        ]);
      }
      const timers = events.map(([t, event, data]) =>
        setTimeout(() => onEvent({ event, data }), t)
      );
      return () => timers.forEach(clearTimeout);
    },
  };
}

/**
 * @param {import('@playwright/test').Page} page
 */
async function setupStreamingMock(page) {
  await page.addInitScript(installMock, {
    answerClaim: ANSWER_CLAIM,
    planLine: PLAN_LINE,
    refusal: REFUSAL,
    engineFailure: ENGINE_FAILURE,
    wideColumns: WIDE_COLUMNS,
  });
  await page.goto('/');
  await page.waitForLoadState('networkidle');
}

/**
 * @param {import('@playwright/test').Page} page
 * @param {string} question
 */
async function ask(page, question) {
  await page.locator('#query-input').fill(question);
  await page.locator('#query-input').press('Enter');
}

module.exports = {
  setupStreamingMock,
  ask,
  ANSWER_CLAIM,
  ENGINE_FAILURE,
  PLAN_LINE,
  REFUSAL,
};
