const { test, expect, _electron: electron } = require('@playwright/test');
const { spawn, execFileSync } = require('node:child_process');

const REPO_ROOT = require('node:path').resolve(__dirname, '..', '..', '..');
const ELECTRON_ROOT = require('node:path').resolve(REPO_ROOT, 'electron');
const API_URL = 'http://127.0.0.1:8000/health';

// The harness starts a dev-mode API (no token); the app must be told it may use one, and that
// it should use that API instead of starting its own (except in the sidecar test below).
process.env.CLINICAL_ANALYTICS_DEV_NO_AUTH = '1';
process.env.CLINICAL_ANALYTICS_EXTERNAL_BACKEND = '1';
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

function buildForgeMainBundle() {
  const script = `
    const { build } = require('vite');
    const path = require('node:path');
    const root = ${JSON.stringify(ELECTRON_ROOT)};
    const define = {
      MAIN_WINDOW_VITE_DEV_SERVER_URL: JSON.stringify(${JSON.stringify(RENDERER_URL)}),
      MAIN_WINDOW_VITE_NAME: JSON.stringify('main_window'),
    };
    const external = ['electron', 'electron-squirrel-startup'];
    async function bundle(entry, fileName) {
      await build({
        configFile: false,
        root,
        define,
        build: {
          ssr: path.join(root, entry),
          outDir: path.join(root, '.vite/build'),
          emptyOutDir: false,
          rollupOptions: {
            external,
            output: { format: 'cjs', entryFileNames: fileName },
          },
        },
      });
    }
    (async () => {
      await bundle('src/main.js', 'main.js');
      await bundle('src/preload.js', 'preload.js');
    })().catch((error) => {
      console.error(error);
      process.exit(1);
    });
  `;
  execFileSync(process.execPath, ['-e', script], {
    cwd: ELECTRON_ROOT,
    stdio: 'inherit',
  });
}

/** PIDs of backends started by this Electron process (serve.py --exit-with-parent <pid>). */
function sidecarPids(electronPid) {
  const out = execFileSync('ps', ['-eo', 'pid=,args='], { encoding: 'utf8' });
  return out
    .split('\n')
    .filter((line) => line.includes('clinical_analytics.api.serve'))
    .filter((line) => line.includes(`--exit-with-parent ${electronPid}`))
    .map((line) => Number(line.trim().split(/\s+/)[0]));
}

async function healthStatus(baseUrl) {
  try {
    return (await fetch(`${baseUrl}/health`)).status;
  } catch {
    return 'unreachable';
  }
}

test.describe.serial('Native Electron binary harness', () => {
  test.beforeAll(async () => {
    const apiAlreadyRunning = await isUrlHealthy(API_URL);
    if (!apiAlreadyRunning) {
      startedApiLocally = true;
      apiProcess = startProcess(
        'uv',
        [
          'run',
          'python',
          '-m',
          'clinical_analytics.api.serve',
          '--host',
          '127.0.0.1',
          '--port',
          '8000',
        ],
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
        [
          'vite',
          '--config',
          'vite.renderer.config.mjs',
          '--host',
          'localhost',
          '--port',
          '5173',
          '--strictPort',
        ],
        ELECTRON_ROOT,
        'renderer'
      );
    }
    await waitForUrl(RENDERER_URL, 60000);
    buildForgeMainBundle();
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
      args: ['--no-sandbox', ELECTRON_ROOT],
      env: {
        ...env,
        ELECTRON_DISABLE_SANDBOX: '1',
      },
    });

    const page = await app.firstWindow();
    await expect(page).toHaveTitle(/Clinical Analytics/i);
    await expect(page.locator('main')).toBeVisible();
    await expect(page.getByTestId('session-sidebar')).toBeVisible();
    await expect(page.getByTestId('upload-area')).toBeAttached();
    await expect(page.locator('#connection-status')).toHaveClass(/status-connected/);
    await app.close();
  });

  test('theme choice drives nativeTheme, prefers-color-scheme and the saved setting', async () => {
    const env = { ...process.env };
    delete env.ELECTRON_RUN_AS_NODE;
    const app = await electron.launch({
      args: ['--no-sandbox', ELECTRON_ROOT],
      env: { ...env, ELECTRON_DISABLE_SANDBOX: '1' },
      // Let the page see the real nativeTheme instead of Playwright's emulated light scheme.
      colorScheme: 'no-override',
    });
    const page = await app.firstWindow();
    await expect(page.locator('main')).toBeVisible();
    const userData = await app.evaluate(({ app: electronApp }) => electronApp.getPath('userData'));
    const settingsFile = require('node:path').join(userData, 'settings.json');

    try {
      // Assert: the sandboxed preload received the theme main applied before the window opened,
      // so the pre-paint script follows settings.json rather than localStorage.
      const initial = await page.evaluate(() => window.appTheme.initial);
      expect(initial).toBe(await app.evaluate(({ nativeTheme }) => nativeTheme.themeSource));

      // Act: the renderer's Dark choice goes through preload IPC to main.
      await page.getByTestId('theme-switch').getByText('Dark').click();

      // Assert: main switched nativeTheme, which the page sees as a dark system preference.
      await expect
        .poll(() => app.evaluate(({ nativeTheme }) => nativeTheme.themeSource))
        .toBe('dark');
      await expect
        .poll(() => page.evaluate(() => matchMedia('(prefers-color-scheme: dark)').matches))
        .toBe(true);
      const saved = JSON.parse(require('node:fs').readFileSync(settingsFile, 'utf8'));
      expect(saved.theme).toBe('dark');
      await expect
        .poll(() => page.evaluate(() => getComputedStyle(document.body).backgroundColor))
        .toBe('rgb(22, 21, 19)');

      // An untrusted value is refused by the preload bridge.
      const refused = await page.evaluate(() =>
        window.appTheme.set('neon').then(
          () => 'accepted',
          () => 'refused'
        )
      );
      expect(refused).toBe('refused');
    } finally {
      await page.evaluate(() => window.appTheme.set('system'));
      await app.close();
    }
  });
  test('starts its own backend on a free port, reaches /health, and leaves none behind', async () => {
    // Arrange: the harness API already holds port 8000, so the app must pick another one. A
    // throwaway data home keeps the two backends off each other's data store.
    const fs = require('node:fs');
    const os = require('node:os');
    const dataHome = fs.mkdtempSync(require('node:path').join(os.tmpdir(), 'ca-sidecar-'));
    const env = {
      ...process.env,
      ELECTRON_DISABLE_SANDBOX: '1',
      CLINICAL_ANALYTICS_DATA_HOME: dataHome,
    };
    delete env.ELECTRON_RUN_AS_NODE;
    delete env.CLINICAL_ANALYTICS_EXTERNAL_BACKEND;
    delete env.CLINICAL_ANALYTICS_DEV_NO_AUTH;
    delete env.CLINICAL_ANALYTICS_API_TOKEN;
    expect(await isUrlHealthy(API_URL)).toBe(true);

    // Act
    const app = await electron.launch({ args: ['--no-sandbox', ELECTRON_ROOT], env });
    const electronPid = app.process().pid;
    let mainLog = '';
    app.process().stdout.on('data', (chunk) => {
      mainLog += chunk;
    });
    let baseUrl;
    try {
      const page = await app.firstWindow();
      baseUrl = await page.evaluate(() => window.clinicalAPI.getApiBaseUrl());

      // Assert: a loopback address on a port other than the occupied 8000
      expect(baseUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
      expect(baseUrl).not.toBe('http://127.0.0.1:8000');
      // The renderer reaches it through the preload, which verified the token proof first.
      await expect(page.locator('#connection-status')).toHaveClass(/status-connected/, {
        timeout: 90000,
      });
      expect(await healthStatus(baseUrl)).toBe(200);
      // It enforces this launch's token (unlike the dev-mode harness API).
      expect((await fetch(`${baseUrl}/api/datasets`)).status).toBe(401);
      expect(sidecarPids(electronPid)).toHaveLength(1);
    } finally {
      await app.close();
    }

    // Assert: quitting stopped the backend; no orphan is left.
    await expect.poll(() => sidecarPids(electronPid), { timeout: 15000 }).toEqual([]);
    await expect.poll(() => healthStatus(baseUrl), { timeout: 15000 }).toBe('unreachable');
    // Stopped by main on quit with SIGTERM (not by the kill fallback or the parent watchdog).
    expect(mainLog).toContain('Backend stopped (terminated)');
    fs.rmSync(dataHome, { recursive: true, force: true });
  });
  test('a killed app (no quit handler) still leaves no backend: the watchdog stops it', async () => {
    // Arrange
    const fs = require('node:fs');
    const os = require('node:os');
    const dataHome = fs.mkdtempSync(require('node:path').join(os.tmpdir(), 'ca-sidecar-'));
    const env = {
      ...process.env,
      ELECTRON_DISABLE_SANDBOX: '1',
      CLINICAL_ANALYTICS_DATA_HOME: dataHome,
    };
    delete env.ELECTRON_RUN_AS_NODE;
    delete env.CLINICAL_ANALYTICS_EXTERNAL_BACKEND;
    delete env.CLINICAL_ANALYTICS_DEV_NO_AUTH;
    const app = await electron.launch({ args: ['--no-sandbox', ELECTRON_ROOT], env });
    const electronPid = app.process().pid;
    const page = await app.firstWindow();
    const baseUrl = await page.evaluate(() => window.clinicalAPI.getApiBaseUrl());
    expect(await healthStatus(baseUrl)).toBe(200);
    expect(sidecarPids(electronPid)).toHaveLength(1);

    // Act: simulate a crash or force quit
    process.kill(electronPid, 'SIGKILL');

    // Assert
    await expect.poll(() => sidecarPids(electronPid), { timeout: 30000 }).toEqual([]);
    await expect.poll(() => healthStatus(baseUrl), { timeout: 15000 }).toBe('unreachable');
    fs.rmSync(dataHome, { recursive: true, force: true });
  });
});
