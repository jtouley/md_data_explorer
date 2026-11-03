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
import { app, BrowserWindow, ipcMain, nativeTheme, net, protocol } from 'electron';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import started from 'electron-squirrel-startup';
import {
  API_TOKEN_CHANNEL,
  APP_ENTRY_URL,
  APP_SCHEME,
  isTrustedRendererUrl,
  resolveApiToken,
  resolveAppFilePath,
  resolveDevNoAuth,
} from './apiToken.js';
import {
  THEME_GET_CHANNEL,
  THEME_SET_CHANNEL,
  WINDOW_BACKGROUND,
  isThemeSource,
  parseThemeSetting,
  serializeThemeSetting,
} from './theme.js';

// Per-launch API token. scripts/run_app.sh generates it and passes it to both the backend and
// this process (CLINICAL_ANALYTICS_API_TOKEN); started on its own, main generates one. Throws on
// a weak launcher token.
const { token: apiToken, source: apiTokenSource } = resolveApiToken(process.env);
// Explicit developer opt-out: only then may the preload use a backend that does not check tokens.
const devNoAuth = resolveDevNoAuth(process.env);

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
    return net.fetch(pathToFileURL(file).toString());
  });
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
 * Keep a window on the app's own page: no navigation elsewhere, no new windows, no webviews.
 * @param {Electron.WebContents} contents
 */
function lockDownWebContents(contents) {
  const blockUntrusted = (event, url) => {
    if (!isTrustedRendererUrl(url, devServerUrl)) {
      event.preventDefault();
      console.warn('Blocked navigation away from the app');
    }
  };
  contents.on('will-navigate', blockUntrusted);
  contents.on('will-redirect', blockUntrusted);
  contents.setWindowOpenHandler(() => ({ action: 'deny' }));
  contents.on('will-attach-webview', (event) => event.preventDefault());
}

/**
 * Create the main application window.
 * @returns {BrowserWindow} The created window
 */
const createWindow = () => {
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
    },
  });
  lockDownWebContents(mainWindow.webContents);

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

// Create window when app is ready
app.whenReady().then(() => {
  console.log('🚀 Clinical Analytics Electron app starting...');
  console.log('📡 Backend API URL: http://127.0.0.1:8000');
  console.log(
    `🔑 API token: ${apiTokenSource === 'env' ? 'from launcher' : 'generated (backend must use the same one)'}` +
      (devNoAuth ? ' — dev mode: an unauthenticated backend is accepted' : '')
  );

  if (!devServerUrl) registerAppProtocol();

  // Before the window exists, so its first paint already uses the saved theme.
  nativeTheme.themeSource = parseThemeSetting(readSettingsText());
  registerThemeHandlers();

  // Hand the token only to the app's own page, never to a page it was navigated to.
  ipcMain.handle(API_TOKEN_CHANNEL, (event) => {
    if (!isTrustedRendererUrl(event.senderFrame?.url, devServerUrl)) {
      throw new Error('API token requested by an untrusted page');
    }
    return { token: apiToken, devNoAuth };
  });

  createWindow();

  // macOS: Re-create window when dock icon is clicked
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

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
