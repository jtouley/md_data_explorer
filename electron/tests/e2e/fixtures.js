/**
 * Playwright fixtures for Electron E2E tests.
 *
 * Provides reusable test fixtures for launching and interacting with the Electron app.
 */
const { test: base, _electron: electron } = require('@playwright/test');
const path = require('path');

// Get the electron app directory
const electronAppDir = path.resolve(__dirname, '../..');

/**
 * Extended test fixture with Electron app and window.
 */
exports.test = base.extend({
  /**
   * Launch Electron app for each test.
   */
  electronApp: async ({}, use) => {
    // Use electron-forge start to launch in dev mode
    // For tests, we'll use the built version or spawn directly
    const electronApp = await electron.launch({
      args: [electronAppDir],
      cwd: electronAppDir,
    });

    await use(electronApp);
    await electronApp.close();
  },

  /**
   * Get the first window from the Electron app.
   */
  window: async ({ electronApp }, use) => {
    const window = await electronApp.firstWindow();
    await use(window);
  },
});

exports.expect = base.expect;
