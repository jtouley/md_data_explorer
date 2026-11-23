/**
 * Tables of groups (ADR017) — renderer + mocked clinicalAPI (synthetic responses, no network).
 * "Is each row one patient?" offers a third answer when the API names count columns, and the
 * upload area carries the ADR014 public-estimates box.
 */
const { test, expect } = require('@playwright/test');

async function setup(page) {
  await page.addInitScript(() => {
    let uploaded = false;
    window.__calls = { upload: [], answer: [] };
    window.clinicalAPI = {
      healthCheck: async () => ({ status: 'healthy' }),
      listDatasets: async () =>
        uploaded
          ? { datasets: [{ dataset_id: 'ds_tally', name: 'tally', row_count: 200 }], total: 1 }
          : { datasets: [], total: 0 },
      getDataset: async () => ({
        dataset_id: 'ds_tally',
        tables: [{ name: 'unified', row_count: 200, columns: ['cases'] }],
        metadata: {},
      }),
      previewDataset: async () => ({ columns: ['cases'], rows: [{ cases: 3 }], total_rows: 200 }),
      listSessions: async () => ({ sessions: [], total: 0 }),
      uploadDataset: async (file, name, estimates) => {
        window.__calls.upload.push([file.name, name ?? null, estimates ?? null]);
        uploaded = true;
        return {
          upload_id: 'ds_tally',
          dataset_name: 'tally',
          status: 'ready',
          message: 'ok',
          one_row_per_patient: {
            status: 'pending',
            confirmation: 'a'.repeat(32),
            count_columns: ['cases'],
          },
        };
      },
      answerOneRowPerPatient: async (...args) => {
        window.__calls.answer.push(args);
        return { dataset_id: 'ds_tally', counts: 'cases' };
      },
      getPendingEnrichments: async () => ({ suggestions: [], total: 0 }),
      getEnrichmentHistory: async () => ({ patches: [], total: 0 }),
      getDatasetQuality: async () => null,
      subscribeToQueryStream: () => () => {},
    };
  });
  await page.goto('/');
  await page.waitForLoadState('networkidle');
}

test.describe('Tables of groups', () => {
  test('third answer sends the count column and says what answers count', async ({ page }) => {
    await setup(page);

    await page.locator('[data-testid="upload-file-input"]').setInputFiles({
      name: 'tally.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from('record_code,cases\nR1,3\n'),
    });
    const third = page.getByTestId('patient-rows-cases');
    await expect(third).toHaveText('Each row counts cases for a group (in cases)');
    await third.click();

    await expect(page.getByTestId('upload-status')).toContainText(
      'Answers on “tally” now add up the cases in cases, never count rows.'
    );
    expect(await page.evaluate(() => window.__calls.answer)).toEqual([
      ['ds_tally', 'a'.repeat(32), false, 'cases'],
    ]);
  });

  test('the public-estimates box declares the upload', async ({ page }) => {
    await setup(page);

    await page.getByTestId('dataset-menu-btn').click();
    await page.getByTestId('upload-population-estimates').check();
    await page.locator('[data-testid="upload-file-input"]').setInputFiles({
      name: 'rates.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from('state,rate\nA,1.5\n'),
    });

    await expect(page.getByTestId('upload-status')).toContainText('is ready');
    expect(await page.evaluate(() => window.__calls.upload)).toEqual([['rates.csv', null, true]]);
  });
});
