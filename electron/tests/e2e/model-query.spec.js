/**
 * The default question path through the real window: upload a synthetic file, ask a question in
 * words no column matches, and read the answer, its plain-English reading and the SQL.
 *
 * Real renderer (Chromium + Vite), real FastAPI process with its own data home (port 8001,
 * playwright.config.js), and the local model mocked at its HTTP boundary by
 * helpers/fakeOllama.js, so no Ollama is needed. Synthetic data only.
 */
const { test, expect } = require('@playwright/test');
const { installLiveApi } = require('./helpers/liveApi');

const API_BASE_URL = 'http://127.0.0.1:8001';

/** 240 synthetic test visits of 150 patients: rows are visits, so patients are counted distinct. */
function makeVisitsCsv() {
  let body = 'patient_id,encounter_id,visit_year,age_years,sex,result\n';
  for (let i = 0; i < 240; i += 1) {
    const patient = i % 150;
    const age = 40 + ((patient * 7) % 50);
    const result = i % 5 === 0 || i % 7 === 0 ? 'positive' : 'negative';
    const sex = patient % 2 === 0 ? 'female' : 'male';
    body += `V${String(patient).padStart(3, '0')},E${String(i).padStart(4, '0')},${2021 + (i % 4)},${age},${sex},${result}\n`;
  }
  return Buffer.from(body);
}

/** Distinct patients 65 and older with a positive result, per year: the oracle for the table. */
function positivePatientsByYear() {
  const byYear = new Map();
  for (let i = 0; i < 240; i += 1) {
    const patient = i % 150;
    const age = 40 + ((patient * 7) % 50);
    if (age < 65 || !(i % 5 === 0 || i % 7 === 0)) continue;
    const year = 2021 + (i % 4);
    if (!byYear.has(year)) byYear.set(year, new Set());
    byYear.get(year).add(patient);
  }
  return [...byYear.entries()].sort((a, b) => a[0] - b[0]).map(([year, set]) => [year, set.size]);
}

test.describe('Model-written query path (live API, mocked model)', () => {
  test.beforeEach(async ({ page }) => {
    await installLiveApi(page, API_BASE_URL);
  });

  test('upload, ask in plain words, see the answer, its reading and the SQL', async ({ page }) => {
    await page.goto('/');
    await page.locator('[data-testid="upload-file-input"]').setInputFiles({
      name: `model_query_${Date.now()}.csv`,
      mimeType: 'text/csv',
      buffer: makeVisitsCsv(),
    });
    await expect(page.locator('[data-testid="upload-status"]')).toContainText('is ready');
    await expect(page.locator('#query-input')).toBeEnabled();

    // "folks", "came back": words that match no column or value. The question is still answered.
    await page
      .locator('#query-input')
      .fill('How many of our older folks, 65 and up, came back positive each year?');
    await page.locator('#query-input').press('Enter');

    const card = page.locator('[data-testid="result-card"]').last();
    await expect(card).toBeVisible({ timeout: 45000 });
    await expect(page.locator('[data-testid="choice-card"]')).toHaveCount(0);

    // The answer: distinct patients per year, as computed.
    await expect(card.locator('[data-testid="result-claim"]')).toContainText('patients across 4');
    for (const [year, count] of positivePatientsByYear()) {
      const row = card.locator('tbody tr', { hasText: String(year) });
      await expect(row).toContainText(String(count));
    }

    // The plain-English reading, always visible.
    await expect(card.locator('[data-testid="answer-reading"]')).toHaveText(
      "Counted distinct patients (patient_id) where age_years >= 65 AND result = 'positive', by visit_year."
    );

    // The SQL that ran, one click away.
    const sql = card.locator('[data-testid="answer-sql"]');
    await sql.locator('summary').click();
    await expect(sql.locator('pre')).toContainText('COUNT(DISTINCT "patient_id")');
    await expect(sql.locator('pre')).toContainText('FROM unified');
  });
});
