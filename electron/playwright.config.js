// @ts-check
const os = require('os');
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
      command: 'uv run python -m clinical_analytics.api.serve --host 127.0.0.1 --port 8000',
      cwd: repoRoot,
      // E2E runs the API in explicit dev mode (no per-launch token); never use with real data.
      env: { ...process.env, CLINICAL_ANALYTICS_DEV_NO_AUTH: '1' },
      url: 'http://127.0.0.1:8000/health',
      reuseExistingServer: true,
      timeout: 120000,
    },
    {
      // model-query.spec.js: a stand-in for the local model server (no Ollama in CI).
      command: 'node tests/e2e/helpers/fakeOllama.js 11999',
      cwd: electronRoot,
      url: 'http://127.0.0.1:11999/api/tags',
      reuseExistingServer: true,
      timeout: 30000,
    },
    {
      // model-query.spec.js: an API of its own (own data home) whose model is the stand-in, so
      // the other specs keep running without a model.
      command: 'uv run python -m clinical_analytics.api.serve --host 127.0.0.1 --port 8001',
      cwd: repoRoot,
      env: {
        ...process.env,
        CLINICAL_ANALYTICS_DEV_NO_AUTH: '1',
        CLINICAL_ANALYTICS_DATA_HOME: path.join(os.tmpdir(), `mdde-e2e-model-query-${process.pid}`),
        OLLAMA_BASE_URL: 'http://127.0.0.1:11999',
        OLLAMA_DEFAULT_MODEL: 'e2e-recorded',
        OLLAMA_FALLBACK_MODEL: 'e2e-recorded',
      },
      url: 'http://127.0.0.1:8001/health',
      reuseExistingServer: true,
      timeout: 120000,
    },
  ],
});
