/**
 * "Questions about your file" (ADR017) — renderer + mocked clinicalAPI (synthetic responses, no
 * network). Every open-question kind is shown with its count evidence and answers; an answer
 * waiting on a question says so, links to the panel, and is released once it is answered.
 */
const { test, expect } = require('@playwright/test');

const KINDS = {
  rows_are_patients: '/one-row-per-patient',
  person_role: '/answers',
  missing_code: '/missing-codes',
  date_order: '/date-readings',
  date_century: '/date-readings',
  censored_value: '/answers',
  total_row: '/answers',
  repeated_strata: '/answers',
  aggregate_unsure: '/one-row-per-patient',
};

async function setup(page, kinds) {
  await page.addInitScript(
    ({ kinds }) => {
      const now = new Date().toISOString();
      const answered = new Set();
      window.__answers = [];
      const question = ([kind, route]) => ({
        id: `${kind}:pain_score`,
        kind,
        column: 'pain_score',
        question: `Question of kind ${kind} about pain_score?`,
        evidence: '14 lines of pain_score hold 99.',
        answers: [
          {
            label: 'yes',
            route,
            body: { question_id: `${kind}:pain_score`, answer: 'yes' },
            changes: `Yes: ${kind} settled.`,
          },
          {
            label: 'no',
            route,
            body: { question_id: `${kind}:pain_score`, answer: 'no' },
            changes: 'No: kept as it is.',
          },
        ],
      });
      const open = () => Object.entries(kinds).filter(([k]) => !answered.has(k));
      window.clinicalAPI = {
        healthCheck: async () => ({ status: 'healthy' }),
        listDatasets: async () => ({
          datasets: [{ dataset_id: 'ds_pain', name: 'pain', row_count: 300 }],
          total: 1,
        }),
        getDataset: async () => ({
          dataset_id: 'ds_pain',
          tables: [{ name: 'unified', row_count: 300, columns: ['pain_score'] }],
          metadata: {},
        }),
        previewDataset: async () => ({
          columns: ['pain_score'],
          rows: [{ pain_score: 4 }],
          total_rows: 300,
        }),
        listSessions: async () => ({ sessions: [], total: 0 }),
        createSession: async (datasetId) => ({
          session_id: 'sess_1',
          dataset_id: datasetId,
          created_at: now,
          updated_at: now,
        }),
        getPendingEnrichments: async () => ({ suggestions: [], total: 0 }),
        getEnrichmentHistory: async () => ({ patches: [], total: 0 }),
        getDatasetQuality: async () => ({ status: 'ran', summary: [], checks: [], facts: [] }),
        getFileQuestions: async () => ({ dataset_id: 'ds_pain', questions: open().map(question) }),
        answerFileQuestion: async (datasetId, route, body) => {
          window.__answers.push([datasetId, route, body]);
          answered.add(String(body.question_id).split(':')[0]);
          return { dataset_id: datasetId, facts: [] };
        },
        cancelSubmit: () => {},
        submitQuery: async () => ({ query_id: `q${Date.now()}`, stream_url: '/stream' }),
        subscribeToQueryStream: (_id, { onEvent }) => {
          const waiting = open().map(([k]) => `${k}:pain_score`);
          const event = waiting.length
            ? [
                'query_needs_answers',
                {
                  open_questions: waiting,
                  failure_reason:
                    'This answer waits for you to answer 1 question about this file first.',
                },
              ]
            : [
                'query_completed',
                { intent_type: 'DESCRIBE', claim: 'Mean pain_score was 4.2 (n=286).' },
              ];
          const timer = setTimeout(() => onEvent({ event: event[0], data: event[1] }), 50);
          return () => clearTimeout(timer);
        },
      };
    },
    { kinds }
  );
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.locator('#dataset-select').selectOption('ds_pain');
}

test.describe('Questions about your file', () => {
  test('every kind is shown with its count evidence, answers and what each changes', async ({
    page,
  }) => {
    await setup(page, KINDS);

    await page.getByTestId('dataset-menu-btn').click();
    await page.getByTestId('open-dataset-panel').click();

    const items = page.getByTestId('file-question');
    await expect(items).toHaveCount(Object.keys(KINDS).length);
    for (const kind of Object.keys(KINDS)) {
      const item = page.locator(`[data-testid="file-question"][data-kind="${kind}"]`);
      await expect(item).toContainText(`Question of kind ${kind} about pain_score?`);
      await expect(item.getByTestId('file-question-evidence')).toHaveText(
        '14 lines of pain_score hold 99.'
      );
      await expect(item.getByTestId('file-question-answer')).toHaveText(['yes', 'no']);
      await expect(item).toContainText(`Yes: ${kind} settled.`);
    }
  });

  test('a held answer waits on the question, links to it, and is released once answered', async ({
    page,
  }) => {
    await setup(page, { missing_code: '/missing-codes' });

    await page.locator('#query-input').fill('average pain score');
    await page.locator('#query-input').press('Enter');
    const waiting = page.getByTestId('waiting-on-questions');
    await expect(waiting).toContainText('Waiting on 1 question about this file');
    await waiting.getByTestId('open-file-questions').click();
    const item = page.locator('[data-testid="file-question"][data-kind="missing_code"]');
    await expect(item).toBeVisible();
    await item.getByTestId('file-question-answer').first().click();
    await expect(page.getByTestId('file-questions-hint')).toHaveText(
      'No open questions about this file.'
    );
    await page.locator('#query-input').fill('average pain score');
    await page.locator('#query-input').press('Enter');

    await expect(page.getByTestId('result-claim').last()).toContainText('Mean pain_score was 4.2');
    expect(await page.evaluate(() => window.__answers)).toEqual([
      ['ds_pain', '/missing-codes', { question_id: 'missing_code:pain_score', answer: 'yes' }],
    ]);
  });
});
