/**
 * Fixes from the independent re-audit: accessible names, wide tables, focus after menu actions
 * and deletes, target sizes, and plain language for engine messages, plans and numbers.
 */
const { test, expect } = require('@playwright/test');
const { setupStreamingMock, ask, ENGINE_FAILURE } = require('./helpers/streamingMock');

test.describe('Re-audit fixes', () => {
  test.beforeEach(async ({ page }) => {
    await setupStreamingMock(page);
    await page.locator('#dataset-select').selectOption('ds_cohort');
  });

  test('theme radios keep their names when the words are hidden under 1000px', async ({ page }) => {
    // Arrange
    await page.setViewportSize({ width: 900, height: 800 });

    // Act
    const dark = page.getByRole('radio', { name: 'Dark' });

    // Assert
    await expect(dark).toHaveCount(1);
    await expect(page.getByRole('radio', { name: 'System' })).toHaveCount(1);
    await expect(page.getByRole('radio', { name: 'Light' })).toHaveCount(1);
    const textWidth = await page
      .locator('.theme-option-text')
      .first()
      .evaluate((n) => n.getBoundingClientRect().width);
    expect(textWidth).toBeLessThanOrEqual(1);
  });

  test('a wide table is a labelled, keyboard-scrollable region with a sticky first column', async ({
    page,
  }) => {
    // Arrange
    await page.setViewportSize({ width: 900, height: 800 });
    await ask(page, 'Describe the wide summary');
    const region = page.getByRole('region', { name: /table/ });
    await expect(region).toBeVisible();
    await expect(page.getByTestId('result-table-hint')).toHaveText(
      'Scroll sideways to see all 12 columns.'
    );
    await expect(page.locator('.result-table-frame')).toHaveClass(/has-more-right/);
    const firstCell = page.locator('.result-table tbody th').first();
    const xBefore = await firstCell.evaluate((n) => n.getBoundingClientRect().left);

    // Act
    await region.focus();
    await page.keyboard.press('End');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');

    // Assert
    await expect(region).toHaveAttribute('tabindex', '0');
    await expect.poll(() => region.evaluate((n) => n.scrollLeft)).toBeGreaterThan(0);
    const xAfter = await firstCell.evaluate((n) => n.getBoundingClientRect().left);
    expect(Math.abs(xAfter - xBefore)).toBeLessThan(1);
  });

  test('table numbers use sensible precision and small p-values never show as 0', async ({
    page,
  }) => {
    // Act
    await ask(page, 'Describe the wide summary');

    // Assert
    const drugRow = page.locator('.result-table tbody tr').first();
    await expect(drugRow).toContainText('64.2');
    await expect(drugRow).not.toContainText('64.234');
    await expect(drugRow).toContainText('<0.001');
    await expect(drugRow.locator('td').first()).toHaveText('190');
  });

  test('engine wording is replaced by a plain sentence; the exact text is only copied, never shown', async ({
    page,
  }) => {
    // Act
    await ask(page, 'Describe nothing in particular');

    // Assert
    const notice = page.getByTestId('refused-notice');
    await expect(notice.locator('.notice-body')).toContainText('didn’t name what to summarise');
    await expect(notice).not.toContainText(/DESCRIBE|primary_variable|LLM/);
    await expect(notice.locator('details')).toHaveCount(0);
    await expect(notice.getByRole('button', { name: 'Copy details' })).toBeVisible();
    expect(ENGINE_FAILURE).toMatch(/primary_variable/);
  });

  test('held plan rows read as words, keep exact names, and the toggle is at least 24px', async ({
    page,
  }) => {
    // Act
    await ask(page, 'Compare LDL by treatment in older patients');
    const card = page.getByTestId('result-confirm-card');
    const list = card.locator('.plan-list');

    // Assert
    await expect(list).toContainText('LDL (mg/dL) — ldl_mg_dl');
    await expect(list).toContainText(
      'Age over 65 and First visit (first_visit) at least 2024-01-01'
    );
    await expect(list).toContainText('patients with no value are kept');
    await expect(list).not.toContainText(/==|>=|missing kept/);
    const height = await card
      .locator('.plan-raw summary')
      .evaluate((n) => n.getBoundingClientRect().height);
    expect(height).toBeGreaterThanOrEqual(24);
  });

  test('Use the sample cohort from the ⋯ menu leaves focus in the composer', async ({ page }) => {
    // Arrange
    await page.getByTestId('dataset-menu-btn').click();

    // Act
    await page.getByTestId('load-sample-btn').click();

    // Assert
    await expect(page.locator('#dataset-select')).toHaveValue('ds_sample');
    await expect(page.locator('#query-input')).toBeFocused();
  });
});

test.describe('Narrow windows (H6)', () => {
  test('the minimum window at 200% zoom: composer reachable, sidebar behind a toggle', async ({
    page,
  }) => {
    // Arrange: 800×600 at 200% zoom is a 400×286 CSS viewport
    await setupStreamingMock(page);
    await page.locator('#dataset-select').selectOption('ds_cohort');
    await page.setViewportSize({ width: 400, height: 286 });

    // Assert: the sidebar is folded and the composer is fully on screen
    await expect(page.getByTestId('session-sidebar')).toBeHidden();
    const box = await page.locator('.query-form').boundingBox();
    expect(box.y + box.height).toBeLessThanOrEqual(286);
    expect(box.width).toBeGreaterThan(300);

    // Act: open the sidebar, then close it with Escape
    const toggle = page.getByTestId('sidebar-toggle');
    await toggle.click();
    await expect(page.getByTestId('session-sidebar')).toBeVisible();
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');
    await page.keyboard.press('Escape');

    // Assert
    await expect(page.getByTestId('session-sidebar')).toBeHidden();
    await expect(toggle).toBeFocused();
  });

  test('the open drawer has a close button, closes on an outside click, and keeps its scroll', async ({
    page,
  }) => {
    // Arrange
    await setupStreamingMock(page);
    await page.locator('#dataset-select').selectOption('ds_cohort');
    await page.setViewportSize({ width: 400, height: 286 });
    const toggle = page.getByTestId('sidebar-toggle');
    const sidebar = page.getByTestId('session-sidebar');

    // Act: open, then the close button inside the drawer
    await toggle.click();
    const close = page.getByRole('button', { name: 'Close cohort and conversations' });
    await expect(close).toBeVisible();
    await expect(sidebar).toHaveCSS('overscroll-behavior-y', 'contain');
    if (process.env.SHOTS_DIR) {
      await page.screenshot({ path: `${process.env.SHOTS_DIR}/narrow-drawer-open.png` });
    }
    await close.click();

    // Assert: closed, focus back on the toggle
    await expect(sidebar).toBeHidden();
    await expect(toggle).toBeFocused();
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');

    // Act: open again, click the conversation beside the drawer
    await toggle.click();
    await expect(sidebar).toBeVisible();
    await page.mouse.click(390, 150);

    // Assert
    await expect(sidebar).toBeHidden();
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');
    if (process.env.SHOTS_DIR) {
      await page.screenshot({ path: `${process.env.SHOTS_DIR}/narrow-drawer-closed.png` });
    }
  });

  test('wide windows do not show the toggle', async ({ page }) => {
    await setupStreamingMock(page);
    await expect(page.getByTestId('sidebar-toggle')).toBeHidden();
    await expect(page.getByTestId('session-sidebar')).toBeVisible();
  });
});

test.describe('Re-audit fixes: deleting a conversation', () => {
  test('deleting a conversation that is not open moves focus to the next row', async ({ page }) => {
    // Arrange
    await page.addInitScript(() => {
      const at = new Date().toISOString();
      window.__sessions = ['sess_a', 'sess_b', 'sess_c'].map((id) => ({
        session_id: id,
        dataset_id: 'ds_cohort',
        created_at: at,
        updated_at: at,
      }));
    });
    await setupStreamingMock(page);
    await page.locator('#dataset-select').selectOption('ds_cohort');
    const rowA = page.locator('.session-item[data-session-id="sess_a"]');
    await expect(rowA).toBeVisible();

    // Act
    await rowA.hover();
    await rowA.getByTestId('session-delete').click();
    await page.getByTestId('session-delete-confirm').click();

    // Assert
    await expect(rowA).toHaveCount(0);
    await expect(
      page.locator('.session-item[data-session-id="sess_b"] [data-testid="session-open"]')
    ).toBeFocused();
  });
});
