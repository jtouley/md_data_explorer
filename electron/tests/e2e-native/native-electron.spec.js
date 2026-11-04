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

/**
 * Launch the app so it starts its own backend: no external-backend flag, no dev opt-out, no
 * launcher token, and a throwaway data home (keeps it off the harness API's store).
 */
async function launchWithSidecar() {
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
  const app = await electron.launch({ args: ['--no-sandbox', ELECTRON_ROOT], env });
  let log = '';
  app.process().stdout.on('data', (chunk) => {
    log += chunk;
  });
  return {
    app,
    electronPid: app.process().pid,
    mainLog: () => log,
    cleanup: () => fs.rmSync(dataHome, { recursive: true, force: true }),
  };
}

/** The app's own page (not the startup window). */
async function appWindow(app, timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const win = app
      .windows()
      .find((w) => !w.isClosed() && !w.url().startsWith('data:') && w.url() !== 'about:blank');
    if (win) {
      await win.waitForLoadState('domcontentloaded');
      return win;
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error('The app window did not open');
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
  test('starts its own backend on a free port behind a startup window, and leaves none behind', async () => {
    // Arrange: the harness API already holds port 8000, so the app must pick another one.
    expect(await isUrlHealthy(API_URL)).toBe(true);
    const { app, electronPid, mainLog, cleanup } = await launchWithSidecar();
    const windowUrls = [];
    app.on('window', (win) => windowUrls.push(win.url()));
    let baseUrl;
    try {
      // Act: the first window is the startup window, shown while the backend boots
      const first = await app.firstWindow();
      const firstUrl = first.url();
      const firstTitle = firstUrl.startsWith('data:')
        ? await first
            .getByTestId('startup-title')
            .textContent()
            .catch(() => 'closed')
        : null;
      const page = await appWindow(app);
      baseUrl = await page.evaluate(() => window.clinicalAPI.getApiBaseUrl());

      // Assert
      expect(firstUrl.startsWith('data:text/html')).toBe(true);
      expect(firstTitle === 'closed' || /Starting Clinical Analytics/.test(firstTitle)).toBe(true);
      expect(baseUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
      expect(baseUrl).not.toBe('http://127.0.0.1:8000');
      // The renderer reaches it through the preload, which verified the token proof first.
      await expect(page.locator('#connection-status')).toHaveClass(/status-connected/, {
        timeout: 90000,
      });
      expect(await healthStatus(baseUrl)).toBe(200);
      expect((await fetch(`${baseUrl}/api/datasets`)).status).toBe(401); // token enforced
      expect(sidecarPids(electronPid)).toHaveLength(1);
      await expect
        .poll(() => app.windows().filter((w) => w.url().startsWith('data:')).length)
        .toBe(0);
    } finally {
      await app.close();
    }

    // Assert: quitting stopped the backend gracefully; no orphan is left.
    await expect.poll(() => sidecarPids(electronPid), { timeout: 20000 }).toEqual([]);
    await expect.poll(() => healthStatus(baseUrl), { timeout: 15000 }).toBe('unreachable');
    expect(mainLog()).toContain('Backend stopped (terminated)');
    cleanup();
  });

  test('page CSP allows only this launch backend port, not other loopback services', async () => {
    // Arrange: the harness API on 8000 stands in for any other local service (CORS allows the
    // dev page's origin, so only the CSP can stop the request)
    const { app, cleanup } = await launchWithSidecar();
    try {
      const page = await appWindow(app);
      const baseUrl = await page.evaluate(() => window.clinicalAPI.getApiBaseUrl());
      await expect(page.locator('#connection-status')).toHaveClass(/status-connected/, {
        timeout: 90000,
      });

      // Act
      const probe = (url) =>
        page.evaluate(
          (u) =>
            fetch(u).then(
              (r) => `status ${r.status}`,
              () => 'blocked'
            ),
          url
        );
      const own = await probe(`${baseUrl}/health`);
      const other = await probe('http://127.0.0.1:8000/health');

      // Assert
      expect(own).toBe('status 200');
      expect(other).toBe('blocked');
    } finally {
      await app.close();
      cleanup();
    }
  });

  test('a backend that dies after startup: dialog offers Restart, which brings a new one', async () => {
    // Arrange
    const { app, electronPid, cleanup } = await launchWithSidecar();
    try {
      const page = await appWindow(app);
      await expect(page.locator('#connection-status')).toHaveClass(/status-connected/, {
        timeout: 90000,
      });
      const [firstPid] = sidecarPids(electronPid);
      // Answer the native dialog with "Restart" and record what it said.
      await app.evaluate(({ dialog }) => {
        globalThis.__crashDialogs = [];
        dialog.showMessageBox = async (opts) => {
          globalThis.__crashDialogs.push({ message: opts.message, buttons: opts.buttons });
          return { response: 0 };
        };
      });

      // Act: the backend crashes
      process.kill(firstPid, 'SIGKILL');

      // Assert: the dialog was shown, a new backend runs and the new window is connected
      await expect
        .poll(() => app.evaluate(() => globalThis.__crashDialogs.length), { timeout: 15000 })
        .toBe(1);
      const shown = await app.evaluate(() => globalThis.__crashDialogs[0]);
      expect(shown.message).toMatch(/stopped unexpectedly/);
      expect(shown.buttons).toEqual(['Restart', 'Quit']);
      await expect
        .poll(() => sidecarPids(electronPid).filter((pid) => pid !== firstPid).length, {
          timeout: 60000,
        })
        .toBe(1);
      const restarted = await appWindow(app);
      await expect(restarted.locator('#connection-status')).toHaveClass(/status-connected/, {
        timeout: 90000,
      });
    } finally {
      await app.close();
    }
    await expect.poll(() => sidecarPids(electronPid), { timeout: 20000 }).toEqual([]);
    cleanup();
  });

  test('a killed app (no quit handler) still leaves no backend', async () => {
    // Arrange
    const { app, electronPid, cleanup } = await launchWithSidecar();
    const page = await appWindow(app);
    const baseUrl = await page.evaluate(() => window.clinicalAPI.getApiBaseUrl());
    expect(await healthStatus(baseUrl)).toBe(200);
    expect(sidecarPids(electronPid)).toHaveLength(1);

    // Act: simulate a crash or force quit
    process.kill(electronPid, 'SIGKILL');

    // Assert: stdin closes and the parent watchdog fires; either stops the backend
    await expect.poll(() => sidecarPids(electronPid), { timeout: 30000 }).toEqual([]);
    await expect.poll(() => healthStatus(baseUrl), { timeout: 15000 }).toBe('unreachable');
    cleanup();
  });
});
