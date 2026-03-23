/**
 * Renderer smoke tests (Chromium + Vite — same entry as Electron dev shell).
 */
const { test, expect } = require('@playwright/test');

test.describe('Clinical Analytics renderer', () => {
  test('page has correct title', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Clinical Analytics/);
  });

  test('main content area is visible', async ({ page }) => {
    await page.goto('/');
    const main = page.locator('main');
    await expect(main).toBeVisible();
  });

  test('dataset selector is visible', async ({ page }) => {
    await page.goto('/');
    const datasetSelect = page.locator('#dataset-select');
    await expect(datasetSelect).toBeVisible();
  });

  test('connection status is visible', async ({ page }) => {
    await page.goto('/');
    const statusIndicator = page.locator('#connection-status');
    await expect(statusIndicator).toBeVisible();
  });
});
