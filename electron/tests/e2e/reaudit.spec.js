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

  test('engine wording is replaced by a plain sentence, exact text behind a disclosure', async ({
    page,
  }) => {
    // Act
    await ask(page, 'Describe nothing in particular');

    // Assert
    const notice = page.getByTestId('refused-notice');
    await expect(notice.locator('.notice-body')).toContainText('didn’t name what to summarise');
    await expect(notice.locator('.notice-body')).not.toContainText(/DESCRIBE|primary_variable|LLM/);
    const detail = notice.locator('details');
    await expect(detail.locator('summary')).toHaveText('Exact message');
    await detail.locator('summary').click();
    await expect(detail).toContainText(ENGINE_FAILURE);
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
