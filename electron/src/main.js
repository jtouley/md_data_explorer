/**
 * Electron main process.
 *
 * Handles window creation, lifecycle events, and backend communication.
 *
 * Security (docs/security/LOCAL_DATA_SECURITY.md):
 * - The packaged page is served from the privileged app:// scheme, so it has an exact origin the
 *   API allows by name (file:// pages send Origin "null", which must never be allowed).
 * - The per-launch API token goes only to that exact page (or the Vite dev server in dev).
 * - The window cannot navigate away from the app or open new windows; the renderer is sandboxed.
 */
import { app, BrowserWindow, dialog, ipcMain, nativeTheme, net, protocol, session } from 'electron';
import os from 'node:os';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import started from 'electron-squirrel-startup';
import {
  API_TOKEN_CHANNEL,
  APP_ENTRY_URL,
  APP_SCHEME,
  contentSecurityPolicy,
  isTrustedRendererUrl,
  resolveApiToken,
  resolveAppFilePath,
  resolveDevNoAuth,
} from './apiToken.js';
import { lockDownWebContents } from './webContentsGuard.js';
import {
  THEME_GET_CHANNEL,
  THEME_SET_CHANNEL,
  WINDOW_BACKGROUND,
  THEME_ARG_PREFIX,
  isThemeSource,
  parseThemeSetting,
  serializeThemeSetting,
} from './theme.js';
import { API_BASE_URL_ARG_PREFIX, DEFAULT_API_BASE_URL } from './apiAuth.js';
import {
  BackendSidecar,
  resolveHealthTimeoutMs,
  resolvePreferredPort,
  resolvePythonExecutable,
  startupFailureMessage,
  unexpectedExitMessage,
  useExternalBackend,
} from './backendSidecar.js';
import { STARTUP_TITLE, STARTUP_WINDOW_SIZE, startupPageUrl } from './startupWindow.js';

// Per-launch API token. scripts/run_app.sh generates it and passes it to both the backend and
// this process (CLINICAL_ANALYTICS_API_TOKEN); started on its own, main generates one. Throws on
// a weak launcher token.
const { token: apiToken, source: apiTokenSource } = resolveApiToken(process.env);
// Explicit developer opt-out: only then may the preload use a backend that does not check tokens.
// Only meaningful for an external dev backend; the backend this app starts always checks tokens.
const devNoAuth = resolveDevNoAuth(process.env) && useExternalBackend(process.env);

// Electron owns the backend (started below) unless the dev flow runs its own API
// (CLINICAL_ANALYTICS_EXTERNAL_BACKEND=1: separate Vite + `make run-api`, or scripts/run_app.sh).
const externalBackend = useExternalBackend(process.env);
/** @type {BackendSidecar | null} */
let sidecar = null;
/** Address of the backend this launch started (null: the preload default / dev override). */
let apiBaseUrl = null;
let backendStopped = false;
/** @type {BrowserWindow | null} shown while the backend starts */
let startupWindow = null;

/** CSP for the app's page: connect-src is exactly the backend this launch uses. */
const pageCsp = () => contentSecurityPolicy(apiBaseUrl || DEFAULT_API_BASE_URL);

const devServerUrl =
  typeof MAIN_WINDOW_VITE_DEV_SERVER_URL === 'string' && MAIN_WINDOW_VITE_DEV_SERVER_URL
    ? MAIN_WINDOW_VITE_DEV_SERVER_URL
    : undefined;

// Must run before the app is ready. standard + secure gives app:// a real origin and lets the page
// use fetch/CORS like an https page.
protocol.registerSchemesAsPrivileged([
  {
    scheme: APP_SCHEME,
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true,
    },
  },
]);

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (started) {
  app.quit();
}

/** Serve the renderer build under app://clinical-analytics/ (no path outside it). */
function registerAppProtocol() {
  const rendererRoot = path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}`);
  protocol.handle(APP_SCHEME, (request) => {
    const file = resolveAppFilePath(request.url, rendererRoot);
    if (!file) return new Response('Not found', { status: 404 });
    return net.fetch(pathToFileURL(file).toString()).then((response) => {
      const headers = new Headers(response.headers);
      headers.set('Content-Security-Policy', pageCsp());
      return new Response(response.body, { status: response.status, headers });
    });
  });
}

/** Dev: the Vite dev server's responses get the same exact-port CSP header. */
function registerDevServerCsp() {
  const origin = new URL(devServerUrl).origin;
  session.defaultSession.webRequest.onHeadersReceived(
    { urls: [`${origin}/*`] },
    (details, callback) => {
      const headers = { ...details.responseHeaders };
      for (const name of Object.keys(headers)) {
        if (name.toLowerCase() === 'content-security-policy') delete headers[name];
      }
      headers['Content-Security-Policy'] = [pageCsp()];
      callback({ responseHeaders: headers });
    }
  );
}

// ============================================================================
// Theme (System / Light / Dark): nativeTheme drives prefers-color-scheme in the page.
// ============================================================================

const settingsPath = () => path.join(app.getPath('userData'), 'settings.json');

/** @returns {string | null} settings file text, or null when there is none yet */
function readSettingsText() {
  try {
    return fs.readFileSync(settingsPath(), 'utf8');
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    console.warn('Could not read settings; using the system theme:', error?.code || error?.message);
    return null;
  }
}

const windowBackground = () =>
  nativeTheme.shouldUseDarkColors ? WINDOW_BACKGROUND.dark : WINDOW_BACKGROUND.light;

function registerThemeHandlers() {
  const assertTrusted = (event) => {
    if (!isTrustedRendererUrl(event.senderFrame?.url, devServerUrl)) {
      throw new Error('Theme change requested by an untrusted page');
    }
  };
  ipcMain.handle(THEME_GET_CHANNEL, (event) => {
    assertTrusted(event);
    return { source: nativeTheme.themeSource, dark: nativeTheme.shouldUseDarkColors };
  });
  ipcMain.handle(THEME_SET_CHANNEL, (event, source) => {
    assertTrusted(event);
    if (!isThemeSource(source)) throw new Error('Unknown theme');
    nativeTheme.themeSource = source;
    fs.mkdirSync(path.dirname(settingsPath()), { recursive: true });
    fs.writeFileSync(settingsPath(), serializeThemeSetting(readSettingsText(), source), 'utf8');
    for (const win of BrowserWindow.getAllWindows()) win.setBackgroundColor(windowBackground());
    return { source, dark: nativeTheme.shouldUseDarkColors };
  });
}

/**
 * Create the main application window.
 * @returns {BrowserWindow} The created window
 */
const createWindow = () => {
  const additionalArguments = [`${THEME_ARG_PREFIX}${nativeTheme.themeSource}`];
  if (apiBaseUrl) additionalArguments.push(`${API_BASE_URL_ARG_PREFIX}${apiBaseUrl}`);
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    title: 'Clinical Analytics',
    // Matches the page's first frame so the window never flashes the other theme.
    backgroundColor: windowBackground(),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true, // preload uses only contextBridge + ipcRenderer (bundled, no Node APIs)
      webviewTag: false,
      preload: path.join(__dirname, 'preload.js'),
      // The saved theme (settings.json), so the page's first paint matches nativeTheme, and the
      // address of the backend this launch started.
      additionalArguments,
    },
  });
  lockDownWebContents(mainWindow.webContents, devServerUrl);

  if (devServerUrl) {
    mainWindow.loadURL(devServerUrl);
  } else {
    mainWindow.loadURL(APP_ENTRY_URL);
  }

  // Open DevTools only when explicitly requested (Cmd+Option+I / Ctrl+Shift+I)
  // Uncomment below line to auto-open DevTools in development:
  // mainWindow.webContents.openDevTools();

  return mainWindow;
};

// ============================================================================
// Application Lifecycle
// ============================================================================

/** Small window while the backend starts (closed once the main window exists). */
function showStartupWindow() {
  startupWindow = new BrowserWindow({
    ...STARTUP_WINDOW_SIZE,
    resizable: false,
    minimizable: false,
    maximizable: false,
    title: STARTUP_TITLE,
    backgroundColor: windowBackground(),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webviewTag: false,
    },
  });
  lockDownWebContents(startupWindow.webContents);
  startupWindow.loadURL(startupPageUrl());
}

function closeStartupWindow() {
  if (startupWindow && !startupWindow.isDestroyed()) startupWindow.destroy();
  startupWindow = null;
}

/** The backend died after it was ready: offer a restart (new backend, new window) or quit. */
async function handleBackendCrash(info) {
  const text = unexpectedExitMessage(info);
  console.error(`${text.message} ${text.detail}`);
  const { response } = await dialog.showMessageBox({
    type: 'error',
    title: text.title,
    message: text.message,
    detail: text.detail,
    buttons: text.buttons,
    defaultId: 0,
    cancelId: 1,
    noLink: true,
  });
  if (response !== 0) {
    app.quit();
    return;
  }
  // The splash first, so closing the old window never leaves zero windows (that would quit).
  showStartupWindow();
  for (const win of BrowserWindow.getAllWindows()) if (win !== startupWindow) win.destroy();
  if (await startBackend()) {
    createWindow();
    closeStartupWindow();
  }
}

/** Show why the backend did not start, then quit (stopping whatever was started). */
function failStartup(failure) {
  const { title, message } = startupFailureMessage(failure || { reason: 'spawn' });
  console.error(`${title}: ${message}`);
  closeStartupWindow();
  dialog.showErrorBox(title, message);
  app.quit();
}

/** Spawn the backend and wait until it is ready; false when it failed (error already shown). */
async function startBackend() {
  try {
    sidecar = new BackendSidecar({
      python: resolvePythonExecutable(process.env, { appRoot: app.getAppPath() }),
      token: apiToken,
      env: process.env,
      preferredPort: resolvePreferredPort(process.env),
      healthTimeoutMs: resolveHealthTimeoutMs(process.env),
      parentPid: process.pid,
      cwd: os.homedir(),
      log: (line) => console.log(line),
      onUnexpectedExit: (info) => void handleBackendCrash(info),
    });
    apiBaseUrl = await sidecar.spawn();
  } catch (failure) {
    failStartup(failure instanceof Error ? { reason: 'spawn', detail: failure.message } : failure);
    return false;
  }
  console.log(`📡 Backend starting at ${apiBaseUrl} (pid ${sidecar.child?.pid})`);
  // The window opens only once the backend answers: the renderer checks the connection on load.
  try {
    await sidecar.waitUntilReady();
  } catch (failure) {
    failStartup(failure);
    return false;
  }
  console.log('✅ Backend ready');
  return true;
}

// Create window when app is ready
app.whenReady().then(async () => {
  console.log('🚀 Clinical Analytics Electron app starting...');
  // Before any window exists, so the startup window's first paint already uses the saved theme.
  nativeTheme.themeSource = parseThemeSetting(readSettingsText());
  if (externalBackend) {
    console.log('📡 Using an already-running backend (CLINICAL_ANALYTICS_EXTERNAL_BACKEND=1)');
  } else {
    showStartupWindow();
    if (!(await startBackend())) return;
  }
  console.log(
    `🔑 API token: ${apiTokenSource === 'env' ? 'from launcher' : 'generated (backend must use the same one)'}` +
      (devNoAuth ? ' — dev mode: an unauthenticated backend is accepted' : '')
  );

  if (devServerUrl) registerDevServerCsp();
  else registerAppProtocol();
  registerThemeHandlers();

  // Hand the token only to the app's own page, never to a page it was navigated to.
  ipcMain.handle(API_TOKEN_CHANNEL, (event) => {
    if (!isTrustedRendererUrl(event.senderFrame?.url, devServerUrl)) {
      throw new Error('API token requested by an untrusted page');
    }
    return { token: apiToken, devNoAuth };
  });

  createWindow();
  closeStartupWindow();

  // macOS: Re-create window when dock icon is clicked
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Stop the backend before quitting: SIGTERM, then a kill after the grace period.
app.on('before-quit', (event) => {
  if (!sidecar || backendStopped) return;
  event.preventDefault();
  backendStopped = true;
  sidecar
    .stop()
    .then((how) => console.log(`🛑 Backend stopped (${how})`))
    .catch((error) => console.error('Could not stop the backend:', error?.message || error))
    .finally(() => app.quit());
});

// Last resort if the process exits some other way: never leave the backend running.
process.on('exit', () => sidecar?.killNow());

// Quit when all windows are closed (except macOS)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error?.message || error);
});
