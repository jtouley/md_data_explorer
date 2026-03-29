/**
 * Electron main process.
 *
 * Handles window creation, lifecycle events, and backend communication.
 */
import { app, BrowserWindow } from 'electron';
import path from 'node:path';
import started from 'electron-squirrel-startup';

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (started) {
  app.quit();
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
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false, // Required for preload contextBridge
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // Load the index.html of the app.
  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(
      path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`)
    );
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
  console.log('📡 Backend API URL: http://localhost:8000');

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
  console.error('Uncaught exception:', error);
});
