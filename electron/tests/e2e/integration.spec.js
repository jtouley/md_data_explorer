/**
 * Integration tests — renderer + mocked clinicalAPI.
 *
 * These tests intercept window.clinicalAPI before the renderer runs,
 * providing controlled responses so we can verify the full wiring:
 * health → datasets → sessions → enrichments → patch history.
 */
const { test, expect } = require('@playwright/test');

const MOCK_DATASETS = {
  datasets: [
    { dataset_id: 'ds_abc', name: 'Patient Records', row_count: 1200 },
    { dataset_id: 'ds_xyz', name: 'Lab Results', row_count: 450 },
  ],
  total: 2,
};

const MOCK_SESSIONS = {
  sessions: [
    {
      session_id: 'sess_001',
      dataset_id: 'ds_abc',
      message_count: 5,
      created_at: '2026-03-27T10:00:00Z',
      updated_at: '2026-03-27T12:00:00Z',
    },
    {
      session_id: 'sess_002',
      dataset_id: 'ds_abc',
      message_count: 0,
      created_at: '2026-03-28T08:00:00Z',
      updated_at: '2026-03-28T08:00:00Z',
    },
  ],
  total: 2,
};

const MOCK_PENDING = {
  suggestions: [
    {
      patch_id: 'p_1',
      operation: 'SET',
      column: 'age_type',
      suggested_value: 'numeric',
      current_value: null,
      confidence: 0.92,
      model_id: 'gpt-4',
    },
  ],
  total: 1,
};

const MOCK_HISTORY = {
  patches: [
    {
      patch_id: 'h_1',
      column: 'gender',
      operation: 'SET',
      value: 'categorical',
      status: 'ACCEPTED',
      created_at: '2026-03-26T09:00:00Z',
      reverted_at: null,
    },
    {
      patch_id: 'h_2',
      column: 'weight',
      operation: 'SET',
      value: 'numeric',
      status: 'REJECTED',
      created_at: '2026-03-25T14:00:00Z',
      reverted_at: null,
    },
  ],
  total: 2,
};

/**
 * Set up page with mocked clinicalAPI injected before renderer loads.
 */
async function setupWithMockAPI(page) {
  await page.addInitScript(
    ({ mockDatasets, mockSessions, mockPending, mockHistory, keys }) => {
      const api = {
        healthCheck: async () => ({ status: 'healthy' }),
        listDatasets: async () => mockDatasets,
        listSessions: async () => mockSessions,
        createSession: async (datasetId) => ({
          session_id: 'sess_new',
          dataset_id: datasetId,
          created_at: new Date().toISOString(),
        }),
        deleteSession: async () => {},
        getPendingEnrichments: async () => mockPending,
        getEnrichmentHistory: async () => mockHistory,
        acceptEnrichment: async () => ({ success: true }),
        rejectEnrichment: async () => ({ success: true }),
        revertEnrichmentPatch: async () => ({ success: true }),
        submitQuery: async () => ({
          query_id: 'q_mock',
          stream_url: null,
          summary: 'Mock result.',
        }),
        subscribeToQueryStream: () => () => {},
        uploadDataset: async (file) => ({
          upload_id: 'ds_mock_upload',
          dataset_name: file.name.replace(/\.[^.]+$/, ''),
          status: 'ready',
          message: 'ok',
        }),
      };
      window.clinicalAPI = api;
    },
    {
      mockDatasets: MOCK_DATASETS,
      mockSessions: MOCK_SESSIONS,
      mockPending: MOCK_PENDING,
      mockHistory: MOCK_HISTORY,
    }
  );
  await page.goto('/');
  await page.waitForLoadState('networkidle');
}

test.describe('Integration: health + datasets on init', () => {
  test('datasets populate selector from mocked API', async ({ page }) => {
    await setupWithMockAPI(page);

    const select = page.locator('#dataset-select');
    const options = select.locator('option');
    await expect(options).toHaveCount(3);
    await expect(options.nth(1)).toHaveText('Patient Records (1200 rows)');
    await expect(options.nth(2)).toHaveText('Lab Results (450 rows)');
  });

  test('connection status shows connected', async ({ page }) => {
    await setupWithMockAPI(page);

    const status = page.locator('#connection-status');
    await expect(status).toHaveClass(/status-connected/);
  });
});

test.describe('Integration: session sidebar', () => {
  test('session list populates from mocked API', async ({ page }) => {
    await setupWithMockAPI(page);

    const items = page.locator('.session-item');
    await expect(items).toHaveCount(2);
  });

  test('active session is highlighted after selecting a session', async ({ page }) => {
    await setupWithMockAPI(page);

    const firstItem = page.locator('.session-item').first();
    await firstItem.click();
    await page.waitForTimeout(200);

    const activeItems = page.locator('.session-item--active');
    await expect(activeItems).toHaveCount(1);
  });

  test('session items show dataset and message count', async ({ page }) => {
    await setupWithMockAPI(page);

    const labels = page.locator('.session-item-label');
    await expect(labels.first()).toContainText('Dataset: ds_');
    const meta = page.locator('.session-item-meta');
    await expect(meta.first()).toContainText('msgs');
    await expect(meta.first()).toContainText('ago');
  });
});

test.describe('Integration: dataset selection triggers enrichment + history', () => {
  test('selecting dataset shows enrichment section with pending suggestions', async ({ page }) => {
    await setupWithMockAPI(page);

    const select = page.locator('#dataset-select');
    await select.selectOption('ds_abc');

    const section = page.locator('#enrichment-section');
    await expect(section).not.toHaveClass(/hidden/);

    const hint = page.getByTestId('enrichment-hint');
    await expect(hint).toContainText('1 pending suggestion.');
  });

  test('patch history table renders with mocked history', async ({ page }) => {
    await setupWithMockAPI(page);

    const select = page.locator('#dataset-select');
    await select.selectOption('ds_abc');

    const histHint = page.getByTestId('patch-history-hint');
    await expect(histHint).toContainText('2 patches in log');

    const rows = page.locator('.patch-history-table tbody tr');
    await expect(rows).toHaveCount(2);
  });

  test('revert button appears only for accepted non-reverted patches', async ({ page }) => {
    await setupWithMockAPI(page);

    await page.locator('#dataset-select').selectOption('ds_abc');

    const revertBtns = page.locator('[data-testid="patch-revert"]');
    await expect(revertBtns).toHaveCount(1);
  });
});

test.describe('Integration: query input enablement', () => {
  test('query input enables after dataset selection', async ({ page }) => {
    await setupWithMockAPI(page);

    const input = page.locator('#query-input');
    await expect(input).toBeDisabled();

    await page.locator('#dataset-select').selectOption('ds_abc');
    await expect(input).toBeEnabled();
  });

  test('query input disables when dataset cleared', async ({ page }) => {
    await setupWithMockAPI(page);

    const select = page.locator('#dataset-select');
    await select.selectOption('ds_abc');

    const input = page.locator('#query-input');
    await expect(input).toBeEnabled();

    await select.selectOption('');
    await expect(input).toBeDisabled();
  });
});

test.describe('Integration: new chat flow', () => {
  test('new chat button resets chat container', async ({ page }) => {
    await setupWithMockAPI(page);

    await page.locator('#dataset-select').selectOption('ds_abc');

    await page.evaluate(() => {
      const chat = document.getElementById('chat-container');
      chat.innerHTML = '<div class="message">Old message</div>';
    });

    const newBtn = page.getByTestId('new-chat-btn');
    await newBtn.click();
    await page.waitForTimeout(300);

    const welcome = page.locator('.welcome-message');
    await expect(welcome).toBeVisible();
  });
});

test.describe('Integration: error resilience', () => {
  test('dataset list shows empty when API throws', async ({ page }) => {
    await page.addInitScript(() => {
      window.clinicalAPI = {
        healthCheck: async () => ({ status: 'healthy' }),
        listDatasets: async () => {
          throw new Error('Network error');
        },
        listSessions: async () => ({ sessions: [], total: 0 }),
        subscribeToQueryStream: () => () => {},
      };
    });
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const select = page.locator('#dataset-select');
    const options = select.locator('option');
    await expect(options).toHaveCount(1);
    await expect(options.first()).toHaveText('Select a dataset...');
  });

  test('enrichment shows error when API throws', async ({ page }) => {
    await page.addInitScript(
      ({ mockDatasets }) => {
        window.clinicalAPI = {
          healthCheck: async () => ({ status: 'healthy' }),
          listDatasets: async () => mockDatasets,
          listSessions: async () => ({ sessions: [], total: 0 }),
          getPendingEnrichments: async () => {
            throw new Error('Enrichment service down');
          },
          getEnrichmentHistory: async () => {
            throw new Error('History service down');
          },
          subscribeToQueryStream: () => () => {},
        };
      },
      { mockDatasets: MOCK_DATASETS }
    );
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await page.locator('#dataset-select').selectOption('ds_abc');

    const hint = page.getByTestId('enrichment-hint');
    await expect(hint).toContainText('Could not load enrichments');
  });
});

test.describe('Integration: dataset upload flow', () => {
  test('upload button is rendered in dataset bar', async ({ page }) => {
    await setupWithMockAPI(page);

    const uploadBtn = page.locator('[data-testid="upload-btn"]');
    await expect(uploadBtn).toBeVisible();
    await expect(uploadBtn).toContainText('Upload dataset');
  });

  test('upload triggers API call and refreshes dataset list', async ({ page }) => {
    await page.addInitScript(
      ({ mockDatasets, mockSessions }) => {
        let uploadCalled = false;
        const extendedDatasets = { ...mockDatasets };
        window.clinicalAPI = {
          healthCheck: async () => ({ status: 'healthy' }),
          listDatasets: async () => {
            if (uploadCalled) {
              return {
                datasets: [
                  ...extendedDatasets.datasets,
                  { dataset_id: 'ds_new', name: 'Uploaded File', row_count: 100 },
                ],
                total: extendedDatasets.total + 1,
              };
            }
            return extendedDatasets;
          },
          listSessions: async () => mockSessions,
          uploadDataset: async (file) => {
            uploadCalled = true;
            return {
              upload_id: 'ds_new',
              dataset_name: file.name.replace(/\.[^.]+$/, ''),
              status: 'ready',
              message: 'ok',
            };
          },
          getPendingEnrichments: async () => ({ suggestions: [], total: 0 }),
          getEnrichmentHistory: async () => ({ patches: [], total: 0 }),
          subscribeToQueryStream: () => () => {},
        };
      },
      { mockDatasets: MOCK_DATASETS, mockSessions: MOCK_SESSIONS }
    );
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const selectBefore = page.locator('#dataset-select option');
    await expect(selectBefore).toHaveCount(3);

    const fileInput = page.locator('[data-testid="upload-file-input"]');
    await fileInput.setInputFiles({
      name: 'new_data.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from('id,val\n1,a\n2,b\n'),
    });

    await page.waitForTimeout(500);

    const uploadStatus = page.locator('[data-testid="upload-status"]');
    await expect(uploadStatus).toContainText('uploaded');

    const selectAfter = page.locator('#dataset-select option');
    await expect(selectAfter).toHaveCount(4);
  });
});

test.describe('Integration: patch revert flow', () => {
  test('clicking revert invokes API and refreshes panel', async ({ page }) => {
    let revertCalled = false;
    await page.addInitScript(
      ({ mockDatasets, mockSessions, mockPending, mockHistory }) => {
        let callCount = 0;
        window.clinicalAPI = {
          healthCheck: async () => ({ status: 'healthy' }),
          listDatasets: async () => mockDatasets,
          listSessions: async () => mockSessions,
          getPendingEnrichments: async () => mockPending,
          getEnrichmentHistory: async () => {
            callCount++;
            if (callCount > 1) {
              return {
                patches: mockHistory.patches.map((p) =>
                  p.patch_id === 'h_1' ? { ...p, reverted_at: new Date().toISOString() } : p
                ),
                total: mockHistory.total,
              };
            }
            return mockHistory;
          },
          revertEnrichmentPatch: async () => ({ success: true }),
          acceptEnrichment: async () => ({ success: true }),
          rejectEnrichment: async () => ({ success: true }),
          subscribeToQueryStream: () => () => {},
        };
      },
      {
        mockDatasets: MOCK_DATASETS,
        mockSessions: MOCK_SESSIONS,
        mockPending: MOCK_PENDING,
        mockHistory: MOCK_HISTORY,
      }
    );
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await page.locator('#dataset-select').selectOption('ds_abc');
    await page.waitForTimeout(500);

    const revertBtn = page.locator('[data-testid="patch-revert"]');
    await expect(revertBtn).toHaveCount(1);

    await revertBtn.click();
    await page.waitForTimeout(500);

    const revertBtnsAfter = page.locator('[data-testid="patch-revert"]');
    await expect(revertBtnsAfter).toHaveCount(0);
  });
});
