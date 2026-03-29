// @ts-check
const path = require('path');
const { defineConfig, devices } = require('@playwright/test');

const electronRoot = path.resolve(__dirname);
const repoRoot = path.resolve(__dirname, '..');

/**
 * E2E against the renderer surface loaded by Electron in development.
 * Native Electron-binary E2E lives in playwright.native.config.js and is
 * enforced separately in the strict quality gate.
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
