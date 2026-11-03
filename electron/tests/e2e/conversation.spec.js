/**
 * Conversation states against a streaming clinicalAPI mock (docs/app/DESIGN.md "Message states").
 */
const { test, expect } = require('@playwright/test');
const { setupStreamingMock, ask, PLAN_LINE, REFUSAL } = require('./helpers/streamingMock');

test.describe('Conversation states', () => {
  test.beforeEach(async ({ page }) => {
    await setupStreamingMock(page);
    await page.locator('#dataset-select').selectOption('ds_cohort');
  });

  test('answer shows the claim as headline, a stats line and caveats, with no "Claim:" prefix', async ({
    page,
  }) => {
    // Act
    await ask(page, 'Compare LDL by treatment');

    // Assert
    await expect(page.getByTestId('result-claim')).toHaveText(
      'Mean ldl was 16.14 higher in placebo than drug.'
    );
    await expect(page.getByTestId('result-stats')).toHaveText(
      '95% CI 13.44 to 18.83 · Welch t-test p<0.001 · n=377 · 23 excluded for missing data'
    );
    await expect(page.locator('.result-notes')).toHaveText('Unadjusted comparison by treatment.');
    await expect(page.getByTestId('result-card-title')).toHaveCount(1);
    await expect(page.locator('.message-assistant').last()).toHaveAttribute(
      'data-state',
      'answered'
    );
  });

  test('first question creates a session titled by that question', async ({ page }) => {
    // Act
    await ask(page, 'Compare LDL by treatment');
    await expect(page.getByTestId('result-claim')).toBeVisible();

    // Assert
    await expect(page.locator('.session-item-label').first()).toHaveText(
      'Compare LDL by treatment'
    );
    const calls = await page.evaluate(() => window.__calls);
    expect(calls.find((c) => c[0] === 'submitQuery')[1]).toBe('sess_created');
  });

  test('stage text renders on one line beside the dots', async ({ page }) => {
    // Arrange
    await page.evaluate(() => {
      window.__streamMs = 5000;
    });

    // Act
    await ask(page, 'Compare LDL by treatment');
    const stage = page.locator('.thinking-stage');
    await expect(stage).toHaveText('Reading your question');

    // Assert: one line high, and not styled as a dot
    const box = await stage.boundingBox();
    expect(box.height).toBeLessThan(30);
    expect(box.width).toBeGreaterThan(100);
  });

  test('table: numbers right-aligned, hidden cells marked with one footnote', async ({ page }) => {
    // Act
    await ask(page, 'Show the event table');

    // Assert
    await expect(page.locator('.cell-hidden--hidden')).toHaveText('hidden†');
    await expect(page.locator('.cell-hidden--small')).toHaveText('<11†');
    await expect(page.getByTestId('hidden-cell-footnote')).toContainText('counts from 1 to 10');
    await expect(page.getByTestId('hidden-cell-footnote')).toHaveCount(1);
    const align = await page
      .locator('.result-table td.num')
      .first()
      .evaluate((td) => getComputedStyle(td).textAlign);
    expect(align).toBe('right');
    const font = await page
      .locator('.result-table td')
      .first()
      .evaluate((td) => getComputedStyle(td).fontFamily);
    expect(font).toContain('Source Sans 3');
  });

  test('held plan: readable plan, Run this analysis sends the exact plan line once', async ({
    page,
  }) => {
    // Act
    await ask(page, 'Compare LDL by treatment in older patients seen recently');
    const card = page.getByTestId('result-confirm-card');
    await expect(card).toContainText('Check this plan before it runs');
    await expect(card.locator('.plan-list')).toContainText('Patients included');
    await card.getByTestId('confirm-query-btn').click();

    // Assert
    await expect(card).toContainText('You approved this plan. The answer is below.');
    await expect(card.getByTestId('confirm-query-btn')).toHaveCount(0);
    const calls = await page.evaluate(() => window.__calls.filter((c) => c[0] === 'submitQuery'));
    expect(calls).toHaveLength(2);
    expect(calls[1]).toEqual([
      'submitQuery',
      'sess_created',
      'ds_cohort',
      'Compare LDL by treatment in older patients seen recently',
      true,
      PLAN_LINE,
    ]);
    const planFont = await card
      .locator('.plan-list')
      .evaluate((n) => parseFloat(getComputedStyle(n).fontSize));
    expect(planFont).toBe(16);
  });

  test('held plan: Not what I meant runs nothing and gives the question back', async ({ page }) => {
    // Act
    await ask(page, 'Compare LDL in older patients');
    const card = page.getByTestId('result-confirm-card');
    await card.getByTestId('reject-query-btn').click();

    // Assert
    await expect(card).toContainText('Not run. Rephrase your question below.');
    await expect(page.locator('#query-input')).toHaveValue('Compare LDL in older patients');
    await expect(page.locator('#query-input')).toBeFocused();
    const submits = await page.evaluate(
      () => window.__calls.filter((c) => c[0] === 'submitQuery').length
    );
    expect(submits).toBe(1);
  });

  test('stop after Run this analysis never offers to send the same plan again', async ({
    page,
  }) => {
    // Arrange: a held plan whose approved answer streams slowly
    await ask(page, 'Compare LDL by treatment in older patients seen recently');
    const card = page.getByTestId('result-confirm-card');
    await expect(card).toContainText('Check this plan before it runs');
    await page.evaluate(() => {
      window.__streamMs = 5000;
    });

    // Act: approve, then stop waiting
    await card.getByTestId('confirm-query-btn').click();
    const submit = page.locator('#submit-btn');
    await expect(submit).toHaveAttribute('aria-label', 'Stop');
    await submit.click();

    // Assert: the approval was sent once and the card cannot send it again
    await expect(page.getByTestId('stopped-notice')).toBeVisible();
    await expect(card).toContainText('You approved this plan and it was sent.');
    await expect(card.getByTestId('confirm-query-btn')).toHaveCount(0);
    const confirmed = await page.evaluate(
      () => window.__calls.filter((c) => c[0] === 'submitQuery' && c[4] === true).length
    );
    expect(confirmed).toBe(1);
  });

  test('a hold that cannot be confirmed keeps Not what I meant usable', async ({ page }) => {
    // Arrange: a hold without a plan line cannot be sent
    await page.evaluate(() => {
      window.__holdPlanLine = '';
    });
    await ask(page, 'Compare LDL by treatment in older patients seen recently');
    const card = page.getByTestId('result-confirm-card');
    await expect(card).toBeVisible();

    // Act
    await card.getByTestId('confirm-query-btn').click();

    // Assert: nothing was sent, and the user can still take the question back
    const reject = card.getByTestId('reject-query-btn');
    await expect(reject).toBeEnabled();
    await reject.click();
    await expect(page.locator('#query-input')).toHaveValue(
      'Compare LDL by treatment in older patients seen recently'
    );
    const confirmed = await page.evaluate(
      () => window.__calls.filter((c) => c[0] === 'submitQuery' && c[4] === true).length
    );
    expect(confirmed).toBe(0);
  });

  test('refusal is a calm notice (not an error) and the question comes back', async ({ page }) => {
    // Act
    await ask(page, 'Show me the patients over 75 with diabetes');

    // Assert
    const notice = page.getByTestId('refused-notice');
    await expect(notice).toContainText('Not answered');
    await expect(notice).toContainText(REFUSAL);
    await expect(page.getByTestId('error-notice')).toHaveCount(0);
    await expect(page.locator('#query-input')).toHaveValue(
      'Show me the patients over 75 with diabetes'
    );
    await expect(page.locator('.message-assistant').last()).toHaveAttribute(
      'data-state',
      'refused'
    );
  });

  test('composer stays editable while streaming and Send becomes Stop', async ({ page }) => {
    // Arrange
    await ask(page, 'This one is slow');
    const submit = page.locator('#submit-btn');
    await expect(submit).toHaveAttribute('aria-label', 'Stop');

    // Act
    await page.locator('#query-input').fill('Drafting the next question');
    await submit.click();

    // Assert
    await expect(page.getByTestId('stopped-notice')).toContainText(
      'Stopped waiting for this answer.'
    );
    await expect(submit).toHaveAttribute('aria-label', 'Ask');
    await expect(page.locator('#query-input')).toHaveValue('Drafting the next question');
    await expect(page.locator('#query-input')).toBeEnabled();
  });

  test('a new answer is scrolled into view', async ({ page }) => {
    // Arrange: fill the column so the next answer would land below the fold
    for (let i = 0; i < 4; i += 1) {
      await ask(page, `Compare LDL by treatment ${i}`);
      await expect(page.getByTestId('result-claim')).toHaveCount(i + 1);
    }

    // Assert: the last headline is inside the chat viewport
    const inView = await page.evaluate(() => {
      const c = document.getElementById('chat-container').getBoundingClientRect();
      const claims = document.querySelectorAll('[data-testid="result-claim"]');
      const r = claims[claims.length - 1].getBoundingClientRect();
      return r.top >= c.top && r.bottom <= c.bottom;
    });
    expect(inView).toBe(true);
  });

  test('example chips fill the box without sending', async ({ page }) => {
    // Act
    const chip = page.locator('.example-chip').first();
    await expect(chip).toHaveText('What is the average age?');
    await chip.click();

    // Assert
    await expect(page.locator('#query-input')).toHaveValue('What is the average age?');
    const submits = await page.evaluate(
      () => window.__calls.filter((c) => c[0] === 'submitQuery').length
    );
    expect(submits).toBe(0);
  });

  test('buttons use the app font, not the browser default', async ({ page }) => {
    await ask(page, 'Compare LDL in older patients');
    const font = await page
      .getByTestId('confirm-query-btn')
      .evaluate((b) => getComputedStyle(b).fontFamily);
    expect(font).toContain('Source Sans 3');
  });
});
