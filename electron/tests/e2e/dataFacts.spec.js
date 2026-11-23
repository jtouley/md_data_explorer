/**
 * What the file holds (ADR017) — renderer + mocked clinicalAPI (synthetic responses, no network).
 * The upload status says what one line is; the Data check lists the contract's facts in
 * clinical words, before the check lines.
 */
const { test, expect } = require('@playwright/test');

const FACTS = [
  'Each line is one clinic visit; 60 patients (subj names the patient) over 120 lines.',
  "visit_no numbers each patient's visits in order (1 = first).",
  'Dates like April 03, 2021 in vis_date read as 3 April 2021.',
  '999 in iop_mmhg means not recorded (not measured).',
];

async function setup(page) {
  await page.addInitScript(
    ({ facts }) => {
      let uploaded = false;
      const quality = {
        status: 'ran',
        summary: ['iop_mmhg: 3 values missing'],
        checks: [],
        facts,
      };
      window.clinicalAPI = {
        healthCheck: async () => ({ status: 'healthy' }),
        listDatasets: async () =>
          uploaded
            ? { datasets: [{ dataset_id: 'ds_iop', name: 'iop', row_count: 120 }], total: 1 }
            : { datasets: [], total: 0 },
        getDataset: async () => ({
          dataset_id: 'ds_iop',
          tables: [{ name: 'unified', row_count: 120, columns: ['subj'] }],
          metadata: {},
        }),
        previewDataset: async () => ({
          columns: ['subj'],
          rows: [{ subj: 'G000' }],
          total_rows: 120,
        }),
        listSessions: async () => ({ sessions: [], total: 0 }),
        uploadDataset: async () => {
          uploaded = true;
          return {
            upload_id: 'ds_iop',
            dataset_name: 'iop',
            status: 'ready',
            message: 'ok',
            quality,
            facts,
          };
        },
        getPendingEnrichments: async () => ({ suggestions: [], total: 0 }),
        getEnrichmentHistory: async () => ({ patches: [], total: 0 }),
        getDatasetQuality: async () => quality,
        subscribeToQueryStream: () => () => {},
      };
    },
    { facts: FACTS }
  );
  await page.goto('/');
  await page.waitForLoadState('networkidle');
}

test.describe('What the file holds', () => {
  test('upload says what one line is, and the data check lists the facts in clinical words', async ({
    page,
  }) => {
    await setup(page);

    await page.locator('[data-testid="upload-file-input"]').setInputFiles({
      name: 'iop.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from('subj,visit_no\nG000,1\n'),
    });

    const status = page.getByTestId('upload-status');
    await expect(status).toContainText('The cohort “iop” is ready.');
    await expect(status).toContainText(FACTS[0]);

    await page.getByTestId('dataset-status-action').click();
    await expect(page.getByTestId('data-facts-hint')).toHaveText(
      'What the file holds, read from it and its data dictionary:'
    );
    await expect(page.getByTestId('data-facts-list').locator('li')).toHaveText(FACTS);
    await expect(page.getByTestId('data-check-list').locator('li')).toHaveText([
      'iop_mmhg: 3 values missing',
    ]);
    const shown = (await page.getByTestId('data-facts-list').innerText()).toLowerCase();
    for (const jargon of ['null', 'grain', 'distinct', 'sql']) expect(shown).not.toContain(jargon);
  });
});
