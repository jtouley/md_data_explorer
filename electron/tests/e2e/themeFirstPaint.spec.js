/**
 * First paint follows the theme main saved (settings.json, passed through preload as
 * appTheme.initial), not a stale localStorage copy. The renderer script is blocked so only the
 * pre-paint script in index.html runs.
 */
const { test, expect } = require('@playwright/test');

test.describe('Theme before first paint', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/src/renderer.js*', (route) => route.abort());
  });

  for (const [saved, stale, expected] of [
    ['dark', 'light', 'dark'],
    ['light', 'dark', 'light'],
    ['system', 'light', null],
    [null, 'dark', 'dark'],
  ]) {
    test(`saved ${saved} with stale localStorage ${stale} paints ${expected}`, async ({ page }) => {
      // Arrange
      await page.addInitScript(
        ([initial, storage]) => {
          window.localStorage.setItem('ca-theme', storage);
          if (initial) window.appTheme = { initial };
        },
        [saved, stale]
      );

      // Act
      await page.goto('/');

      // Assert
      const theme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
      expect(theme).toBe(expected);
    });
  }
});
