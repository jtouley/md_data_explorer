// @ts-check
const path = require('path');
const { defineConfig, devices } = require('@playwright/test');

const electronRoot = path.resolve(__dirname);
const repoRoot = path.resolve(__dirname, '..');

/**
 * E2E against the Vite renderer (same UI Electron loads in dev).
 *
 * Playwright's Electron harness passes --remote-debugging-port=0, which current
 * Electron macOS binaries reject before the preload loader runs; use Chromium
 * for CI and local gates until upstream aligns (or add test:e2e:electron).
 *
 * @see https://playwright.dev/docs/test-configuration
 */
module.exports = defineConfig({
  testDir: './tests/e2e',
  timeout: 60000,
  expect: {
    timeout: 10000,
  },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: [['html', { open: 'never' }], ['list']],
  use: {
    baseURL: 'http://127.0.0.1:5173',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: [
    {
      command: 'npx vite --config vite.renderer.config.mjs --host 127.0.0.1 --port 5173 --strictPort',
      cwd: electronRoot,
      url: 'http://127.0.0.1:5173',
      reuseExistingServer: true,
      timeout: 120000,
    },
    {
      command: 'uv run uvicorn clinical_analytics.api.main:app --host 127.0.0.1 --port 8000',
      cwd: repoRoot,
      url: 'http://127.0.0.1:8000/health',
      reuseExistingServer: true,
      timeout: 120000,
    },
  ],
});
