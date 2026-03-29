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

  test('enrichment panel is hidden until a dataset is selected', async ({ page }) => {
    await page.goto('/');
    const section = page.locator('#enrichment-section');
    await expect(section).toBeAttached();
    await expect(section).toHaveClass(/hidden/);
  });

  test('patch history container is present in DOM', async ({ page }) => {
    await page.goto('/');
    const body = page.getByTestId('patch-history-body');
    await expect(body).toBeAttached();
  });

  test('session sidebar is visible', async ({ page }) => {
    await page.goto('/');
    const sidebar = page.getByTestId('session-sidebar');
    await expect(sidebar).toBeVisible();
  });

  test('new chat button exists in sidebar', async ({ page }) => {
    await page.goto('/');
    const btn = page.getByTestId('new-chat-btn');
    await expect(btn).toBeVisible();
  });

  test('session list container is present', async ({ page }) => {
    await page.goto('/');
    const list = page.getByTestId('session-list');
    await expect(list).toBeAttached();
  });
});
