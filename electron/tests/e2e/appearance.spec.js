/**
 * Visual guarantees: dark mode really differs from light, the theme choice persists, and the
 * chat owns the screen once a cohort is selected.
 */
const { test, expect } = require('@playwright/test');
const { setupStreamingMock } = require('./helpers/streamingMock');

const bg = (page, selector) =>
  page.locator(selector).evaluate((n) => getComputedStyle(n).backgroundColor);

test.describe('Appearance', () => {
  test('system dark scheme renders different colours from light', async ({ browser }) => {
    // Arrange
    const light = await browser.newContext({ colorScheme: 'light' });
    const dark = await browser.newContext({ colorScheme: 'dark' });
    const lp = await light.newPage();
    const dp = await dark.newPage();
    await setupStreamingMock(lp);
    await setupStreamingMock(dp);

    // Act
    const values = {
      light: {
        body: await bg(lp, 'body'),
        sidebar: await bg(lp, '.sidebar'),
        composer: await bg(lp, '.query-form'),
      },
      dark: {
        body: await bg(dp, 'body'),
        sidebar: await bg(dp, '.sidebar'),
        composer: await bg(dp, '.query-form'),
      },
    };

    // Assert
    expect(values.light.body).toBe('rgb(247, 246, 243)');
    expect(values.dark.body).toBe('rgb(22, 21, 19)');
    for (const key of ['body', 'sidebar', 'composer']) {
      expect(values.dark[key]).not.toBe(values.light[key]);
    }
    const darkText = await dp.locator('body').evaluate((n) => getComputedStyle(n).color);
    expect(darkText).toBe('rgb(236, 234, 229)');
    await light.close();
    await dark.close();
  });

  test('explicit Dark wins over a light system, persists, and applies on reload', async ({
    browser,
  }) => {
    // Arrange
    const ctx = await browser.newContext({ colorScheme: 'light' });
    const page = await ctx.newPage();
    // Record the theme at the moment <body> is first parsed, i.e. before first paint.
    await page.addInitScript(() => {
      new MutationObserver((_, observer) => {
        if (document.body) {
          window.__themeAtBody = document.documentElement.getAttribute('data-theme');
          observer.disconnect();
        }
      }).observe(document, { childList: true, subtree: true });
    });

    await setupStreamingMock(page);

    // Act
    await page.getByTestId('theme-switch').getByText('Dark').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    await page.reload();

    // Assert: applied by the inline head script, before the body (and the renderer) exist
    await expect.poll(() => page.evaluate(() => window.__themeAtBody)).toBe('dark');
    expect(await bg(page, 'body')).toBe('rgb(22, 21, 19)');
    await expect(page.locator('input[name="theme"][value="dark"]')).toBeChecked();

    // Act: back to System follows the (light) system again
    await page.getByTestId('theme-switch').getByText('System').click();
    await expect(page.locator('html')).not.toHaveAttribute('data-theme', /.+/);
    // The switch cross-fades for 200 ms.
    await expect.poll(() => bg(page, 'body')).toBe('rgb(247, 246, 243)');
    await ctx.close();
  });

  test('theme switch is keyboard operable as a radio group', async ({ page }) => {
    await setupStreamingMock(page);
    await page.locator('input[name="theme"][value="system"]').focus();
    await page.keyboard.press('ArrowRight');
    await expect(page.locator('input[name="theme"][value="light"]')).toBeChecked();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  });

  for (const width of [1280, 900]) {
    test(`chat column has at least 60% of the viewport height after a cohort is selected (${width}px)`, async ({
      browser,
    }) => {
      // Arrange
      const ctx = await browser.newContext({ viewport: { width, height: 820 } });
      const page = await ctx.newPage();
      await setupStreamingMock(page);

      // Act
      await page.locator('#dataset-select').selectOption('ds_cohort');
      await expect(page.getByTestId('dataset-meta')).toBeVisible();

      // Assert
      const ratio = await page.evaluate(
        () =>
          document.getElementById('chat-container').getBoundingClientRect().height /
          window.innerHeight
      );
      expect(ratio).toBeGreaterThanOrEqual(0.6);
      const composer = await page.locator('.query-form').boundingBox();
      expect(composer.y + composer.height).toBeLessThanOrEqual(820);
      await ctx.close();
    });
  }

  test('offline: a banner explains and Try again reconnects', async ({ page }) => {
    // Arrange
    await page.addInitScript(() => {
      let up = false;
      window.clinicalAPI = {
        healthCheck: async () => (up ? { status: 'healthy' } : Promise.reject(new Error('down'))),
        listDatasets: async () => ({ datasets: [], total: 0 }),
        listSessions: async () => ({ sessions: [], total: 0 }),
        subscribeToQueryStream: () => () => {},
      };
      window.__bringUp = () => {
        up = true;
      };
    });
    await page.goto('/');

    // Assert offline
    await expect(page.getByTestId('offline-banner')).toBeVisible();
    await expect(page.getByTestId('welcome-message')).toContainText(
      'Waiting for the analysis engine.'
    );
    await expect(page.locator('#connection-status')).toHaveClass(/status-error/);
    // The banner can't be dismissed: its Try again is the only way to reconnect.
    await expect(page.getByTestId('offline-banner').locator('.banner-dismiss')).toHaveCount(0);
    await expect(page.getByTestId('welcome-message')).not.toContainText('sidebar');

    // Act
    await page.evaluate(() => window.__bringUp());
    await page.locator('#retry-connection').click();

    // Assert
    await expect(page.getByTestId('offline-banner')).toBeHidden();
    await expect(page.locator('.status-text')).toHaveText('Running on this computer');
    await expect(page.getByTestId('welcome-message')).toContainText('Choose a cohort to begin.');
  });
});
