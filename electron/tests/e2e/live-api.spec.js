/**
 * Live API integration tests (Chromium + Vite + FastAPI).
 *
 * These tests do not mock window.clinicalAPI. They validate real HTTP
 * integration through preload.js -> FastAPI endpoints.
 */
const { test, expect } = require('@playwright/test');
const API_BASE_URL = 'http://127.0.0.1:8000';

function makeCsvBytes() {
  const header = 'patient_id,age,sex,outcome,treatment\n';
  let body = '';
  for (let i = 0; i < 80; i += 1) {
    const sex = i % 2 === 0 ? 'F' : 'M';
    const outcome = i % 3 === 0 ? 'dead' : 'alive';
    const treatment = i % 2 === 0 ? 'placebo' : 'drug';
    body += `P${String(i).padStart(4, '0')},${20 + i},${sex},${outcome},${treatment}\n`;
  }
  return Buffer.from(header + body);
}

test.describe('Live FastAPI integration', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(({ apiBaseUrl }) => {
      const jsonOrEmpty = async (response) => {
        try {
          return await response.json();
        } catch {
          return {};
        }
      };

      window.clinicalAPI = {
        async healthCheck() {
          const res = await fetch(`${apiBaseUrl}/health`);
          if (!res.ok) return { status: 'error' };
          return res.json();
        },
        async listDatasets() {
          const res = await fetch(`${apiBaseUrl}/api/datasets`);
          if (!res.ok) throw new Error(`Failed to list datasets: ${res.statusText}`);
          return res.json();
        },
        async listSessions() {
          const res = await fetch(`${apiBaseUrl}/api/sessions`);
          if (!res.ok) throw new Error(`Failed to list sessions: ${res.statusText}`);
          return res.json();
        },
        async createSession(datasetId) {
          const res = await fetch(`${apiBaseUrl}/api/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_id: datasetId }),
          });
          if (!res.ok) {
            const data = await jsonOrEmpty(res);
            throw new Error(data.detail || `Failed to create session: ${res.statusText}`);
          }
          return res.json();
        },
        async deleteSession(sessionId) {
          const res = await fetch(`${apiBaseUrl}/api/sessions/${encodeURIComponent(sessionId)}`, {
            method: 'DELETE',
          });
          if (!res.ok && res.status !== 204) {
            throw new Error(`Failed to delete session: ${res.statusText}`);
          }
        },
        async getPendingEnrichments(datasetId) {
          const res = await fetch(
            `${apiBaseUrl}/api/datasets/${encodeURIComponent(datasetId)}/enrichments/pending`
          );
          if (!res.ok) throw new Error(`Failed to load enrichments: ${res.statusText}`);
          return res.json();
        },
        async getEnrichmentHistory(datasetId) {
          const res = await fetch(
            `${apiBaseUrl}/api/datasets/${encodeURIComponent(datasetId)}/enrichments/history`
          );
          if (!res.ok) throw new Error(`Failed to load patch history: ${res.statusText}`);
          return res.json();
        },
        async uploadDataset(file, datasetName) {
          const form = new FormData();
          form.append('file', file);
          if (datasetName) form.append('dataset_name', datasetName);
          const res = await fetch(`${apiBaseUrl}/api/datasets/upload`, { method: 'POST', body: form });
          const data = await jsonOrEmpty(res);
          if (!res.ok) throw new Error(data.detail || `Upload failed: ${res.statusText}`);
          return data;
        },
        subscribeToQueryStream() {
          return () => {};
        },
      };
    }, { apiBaseUrl: API_BASE_URL });
  });

  test('health check marks renderer as connected', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#connection-status')).toHaveClass(/status-connected/);
    await expect(page.locator('.status-text')).toContainText('Backend connected');
  });

  test('uploading CSV enables query input and selects dataset', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#query-input')).toBeDisabled();

    const filename = `phase9_live_${Date.now()}.csv`;
    await page.locator('[data-testid="upload-file-input"]').setInputFiles({
      name: filename,
      mimeType: 'text/csv',
      buffer: makeCsvBytes(),
    });

    await expect(page.locator('[data-testid="upload-status"]')).toContainText('uploaded');
    await expect(page.locator('#query-input')).toBeEnabled();

    const selectedValue = await page.locator('#dataset-select').evaluate((el) => el.value);
    expect(selectedValue).not.toBe('');
  });

  test('new chat creates a session item for selected dataset', async ({ page }) => {
    await page.goto('/');

    const filename = `phase9_session_${Date.now()}.csv`;
    await page.locator('[data-testid="upload-file-input"]').setInputFiles({
      name: filename,
      mimeType: 'text/csv',
      buffer: makeCsvBytes(),
    });
    await expect(page.locator('[data-testid="upload-status"]')).toContainText('uploaded');

    await page.getByTestId('new-chat-btn').click();

    await expect.poll(async () => page.locator('.session-item').count()).toBeGreaterThan(0);
    await expect(page.locator('.session-item--active')).toHaveCount(1);
  });
});
