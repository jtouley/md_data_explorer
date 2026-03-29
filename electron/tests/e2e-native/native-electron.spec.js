const { test, expect, _electron: electron } = require('@playwright/test');
const { spawn } = require('node:child_process');

const REPO_ROOT = require('node:path').resolve(__dirname, '..', '..', '..');
const ELECTRON_ROOT = require('node:path').resolve(REPO_ROOT, 'electron');
const API_URL = 'http://127.0.0.1:8000/health';
const RENDERER_URL = 'http://localhost:5173';

let apiProcess;
let rendererProcess;
let startedApiLocally = false;
let startedRendererLocally = false;

function startProcess(command, args, cwd, name) {
  const child = spawn(command, args, {
    cwd,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: process.env,
  });
  child.stdout.on('data', (chunk) => {
    process.stdout.write(`[${name}] ${chunk}`);
  });
  child.stderr.on('data', (chunk) => {
    process.stderr.write(`[${name}] ${chunk}`);
  });
  return child;
}

async function waitForUrl(url, timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        return;
      }
    } catch {
      // Retry until timeout.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

async function isUrlHealthy(url) {
  try {
    const res = await fetch(url);
    return res.ok;
  } catch {
    return false;
  }
}

function stopProcess(child) {
  if (!child || child.killed) return;
  child.kill('SIGTERM');
}

test.describe.serial('Native Electron binary harness', () => {
  test.beforeAll(async () => {
    const apiAlreadyRunning = await isUrlHealthy(API_URL);
    if (!apiAlreadyRunning) {
      startedApiLocally = true;
      apiProcess = startProcess(
        'uv',
        ['run', 'uvicorn', 'clinical_analytics.api.main:app', '--host', '127.0.0.1', '--port', '8000'],
        REPO_ROOT,
        'api'
      );
    }
    await waitForUrl(API_URL, 60000);

    const rendererAlreadyRunning = await isUrlHealthy(RENDERER_URL);
    if (!rendererAlreadyRunning) {
      startedRendererLocally = true;
      rendererProcess = startProcess(
        'npx',
        ['vite', '--config', 'vite.renderer.config.mjs', '--host', 'localhost', '--port', '5173', '--strictPort'],
        ELECTRON_ROOT,
        'renderer'
      );
    }
    await waitForUrl(RENDERER_URL, 60000);
  });

  test.afterAll(async () => {
    if (startedApiLocally) {
      stopProcess(apiProcess);
    }
    if (startedRendererLocally) {
      stopProcess(rendererProcess);
    }
  });

  test('loads native shell and main UI sections', async () => {
    const env = { ...process.env };
    delete env.ELECTRON_RUN_AS_NODE;

    const app = await electron.launch({
      args: [ELECTRON_ROOT],
      env,
    });

    const page = await app.firstWindow();
    await expect(page).toHaveTitle(/Clinical Analytics/i);
    await expect(page.locator('main')).toBeVisible();
    await expect(page.getByTestId('session-sidebar')).toBeVisible();
    await expect(page.getByTestId('upload-area')).toBeAttached();
    await expect(page.locator('#connection-status')).toContainText(/connected/i);
    await app.close();
  });
});
