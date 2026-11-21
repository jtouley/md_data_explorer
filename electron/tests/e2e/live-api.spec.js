/**
 * Live API integration tests (Chromium + Vite + FastAPI).
 *
 * These tests do not mock window.clinicalAPI. They validate real HTTP
 * integration through preload.js -> FastAPI endpoints.
 */
const { test, expect } = require('@playwright/test');
const { installLiveApi } = require('./helpers/liveApi');
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
    await installLiveApi(page, API_BASE_URL);
  });

  test('health check marks renderer as connected', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#connection-status')).toHaveClass(/status-connected/);
    await expect(page.locator('.status-text')).toHaveText('Running on this computer');
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

    await expect(page.locator('[data-testid="upload-status"]')).toContainText('is ready');
    await expect(page.locator('#query-input')).toBeEnabled();

    const selectedValue = await page.locator('#dataset-select').evaluate((el) => el.value);
    expect(selectedValue).not.toBe('');
  });

  test('uploading CSV shows dataset overview summary and preview table', async ({ page }) => {
    await page.goto('/');

    const filename = `phase10_overview_${Date.now()}.csv`;
    await page.locator('[data-testid="upload-file-input"]').setInputFiles({
      name: filename,
      mimeType: 'text/csv',
      buffer: makeCsvBytes(),
    });

    await expect(page.locator('[data-testid="upload-status"]')).toContainText('is ready');
    await expect(page.locator('[data-testid="dataset-overview"]')).not.toHaveClass(/hidden/);
    await expect(page.locator('[data-testid="dataset-overview-summary"]')).toContainText('rows');

    const previewRows = page.locator('[data-testid="dataset-preview-table"] tbody tr');
    await expect(previewRows).toHaveCount(10);
  });

  test('new chat creates a session item for selected dataset', async ({ page }) => {
    await page.goto('/');

    const filename = `phase9_session_${Date.now()}.csv`;
    await page.locator('[data-testid="upload-file-input"]').setInputFiles({
      name: filename,
      mimeType: 'text/csv',
      buffer: makeCsvBytes(),
    });
    await expect(page.locator('[data-testid="upload-status"]')).toContainText('is ready');

    await page.getByTestId('new-chat-btn').click();

    await expect.poll(async () => page.locator('.session-item').count()).toBeGreaterThan(0);
    await expect(page.locator('.session-item--active')).toHaveCount(1);
  });
});
