/**
 * "Load from a link" — renderer + mocked clinicalAPI (no network, synthetic responses).
 */
const { test, expect } = require('@playwright/test');

const SHA = 'b'.repeat(64);
const SOURCE = {
  url: 'https://data.example.test/rows.csv',
  final_url: 'https://data.example.test/rows.csv',
  fetched_at: '2026-11-19T16:20:00+00:00',
  size_bytes: 3042,
  sha256: SHA,
  methods_text: `Data downloaded from https://data.example.test/rows.csv on 19 November 2026 (3,042 bytes; SHA-256 ${SHA}).`,
};

async function setup(page, { fail = '' } = {}) {
  await page.addInitScript(
    ({ source, failWith }) => {
      let loaded = false;
      window.__linkCalls = [];
      window.clinicalAPI = {
        healthCheck: async () => ({ status: 'healthy' }),
        listDatasets: async () =>
          loaded
            ? {
                datasets: [{ dataset_id: 'ds_link', name: 'HCV deaths', row_count: 300 }],
                total: 1,
              }
            : { datasets: [], total: 0 },
        getDataset: async () => ({
          dataset_id: 'ds_link',
          tables: [{ name: 'unified', row_count: 300, columns: ['state'] }],
          metadata: { source },
        }),
        previewDataset: async () => ({
          columns: ['state'],
          rows: [{ state: 'S01' }],
          total_rows: 300,
        }),
        listSessions: async () => ({ sessions: [], total: 0 }),
        uploadDataset: async () => ({}),
        loadDatasetFromUrl: async (url, name) => {
          window.__linkCalls.push([url, name]);
          if (failWith) throw new Error(failWith);
          loaded = true;
          return {
            upload_id: 'ds_link',
            dataset_name: name || 'rows',
            status: 'ready',
            message: 'ok',
            source,
          };
        },
        getPendingEnrichments: async () => ({ suggestions: [], total: 0 }),
        getEnrichmentHistory: async () => ({ patches: [], total: 0 }),
        getDatasetQuality: async () => null,
        subscribeToQueryStream: () => () => {},
      };
    },
    { source: SOURCE, failWith: fail }
  );
  await page.goto('/');
  await page.waitForLoadState('networkidle');
}

test.describe('Load from a link', () => {
  test('menu opens a labelled form; a link loads the cohort and its source is shown', async ({
    page,
  }) => {
    await setup(page);

    await page.getByTestId('dataset-menu-btn').click();
    await page.getByTestId('load-from-link-btn').click();
    const url = page.getByLabel('Link to the data file');
    await expect(url).toBeFocused();

    await url.fill('https://data.example.test/rows.csv');
    await page.getByLabel('Cohort name (optional)').fill('HCV deaths');
    await page.keyboard.press('Enter');

    await expect(page.getByTestId('upload-status')).toContainText(
      'The cohort “HCV deaths” is ready.'
    );
    await expect(page.getByTestId('upload-status')).toContainText(
      'Source saved: data.example.test, 19 November 2026.'
    );
    await expect(page.getByTestId('dataset-link-form')).toHaveCount(0);
    await expect(page.locator('#dataset-select')).toHaveValue('ds_link');
    expect(await page.evaluate(() => window.__linkCalls)).toEqual([
      ['https://data.example.test/rows.csv', 'HCV deaths'],
    ]);

    await page.getByTestId('dataset-status-action').click();
    await expect(page.getByTestId('dataset-source-text')).toHaveText(SOURCE.methods_text);
    await expect(page.getByTestId('copy-dataset-source')).toBeVisible();
  });

  test('an engine refusal is shown in the form and the link is kept', async ({ page }) => {
    await setup(page, {
      fail: '[HTTP 400] That link points to this computer or a private network, not a public website. Only public data on the internet can be loaded from a link.',
    });

    await page.getByTestId('dataset-menu-btn').click();
    await page.getByTestId('load-from-link-btn').click();
    await page.getByLabel('Link to the data file').fill('https://127.0.0.1/a.csv');
    await page.getByTestId('dataset-link-submit').click();

    const error = page.getByTestId('dataset-link-error');
    await expect(error).toHaveText(
      'Couldn’t load that link. That link points to this computer or a private network, not a public website. Only public data on the internet can be loaded from a link.'
    );
    await expect(page.getByLabel('Link to the data file')).toHaveValue('https://127.0.0.1/a.csv');
    await expect(page.getByTestId('dataset-link-submit')).toBeEnabled();
  });

  test('Escape closes the form and returns focus to the cohort menu button', async ({ page }) => {
    await setup(page);

    await page.getByTestId('dataset-menu-btn').click();
    await page.getByTestId('load-from-link-btn').click();
    await page.keyboard.press('Escape');

    await expect(page.getByTestId('dataset-link-form')).toHaveCount(0);
    await expect(page.getByTestId('dataset-menu-btn')).toBeFocused();
  });
});
