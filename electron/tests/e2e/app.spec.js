/**
 * E2E tests for Electron app launch.
 *
 * Phase 4 of Electron UI Migration: Verify app skeleton launches correctly.
 */
const { test, expect } = require('./fixtures');

test.describe('Electron App Launch', () => {
  test('app launches with correct title', async ({ window }) => {
    // Assert: Window has correct title
    await expect(window).toHaveTitle(/Clinical Analytics/);
  });

  test('app shows main content area', async ({ window }) => {
    // Assert: Main content area is visible
    const main = window.locator('main');
    await expect(main).toBeVisible();
  });

  test('app shows dataset selector', async ({ window }) => {
    // Assert: Dataset selector is visible
    const datasetSelect = window.locator('#dataset-select');
    await expect(datasetSelect).toBeVisible();
  });

  test('app shows connection status', async ({ window }) => {
    // Assert: Connection status indicator is visible
    const statusIndicator = window.locator('#connection-status');
    await expect(statusIndicator).toBeVisible();
  });
});
