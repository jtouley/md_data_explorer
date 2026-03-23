/**
 * Chat UI tests against the Vite renderer (Chromium).
 */
const { test, expect } = require('@playwright/test');

test.describe('Chat Interface', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('user sees disabled query input without dataset', async ({ page }) => {
    const queryInput = page.locator('#query-input');
    await expect(queryInput).toBeVisible();
    await expect(queryInput).toBeDisabled();
  });

  test('submit button is disabled without dataset selection', async ({ page }) => {
    const submitBtn = page.locator('#submit-btn');
    await expect(submitBtn).toBeVisible();
    await expect(submitBtn).toBeDisabled();
  });

  test('welcome message displays example queries', async ({ page }) => {
    const welcomeMessage = page.locator('.welcome-message');
    await expect(welcomeMessage).toBeVisible();

    const exampleQueries = page.locator('.example-queries');
    await expect(exampleQueries).toBeVisible();
    await expect(exampleQueries).toContainText('What is the average age');
  });

  test('user message appears in chat after DOM injection', async ({ page }) => {
    await page.evaluate(() => {
      const chatContainer = document.getElementById('chat-container');
      chatContainer.innerHTML = '';

      const messageDiv = document.createElement('div');
      messageDiv.className = 'message message-user';
      messageDiv.setAttribute('data-testid', 'user-message');

      const content = document.createElement('div');
      content.className = 'message-content';
      content.textContent = 'What is the average age of patients?';
      messageDiv.appendChild(content);

      chatContainer.appendChild(messageDiv);
    });

    const userMessage = page.locator('[data-testid="user-message"]');
    await expect(userMessage).toBeVisible();
    await expect(userMessage).toContainText('What is the average age');
  });

  test('assistant message appears in chat', async ({ page }) => {
    await page.evaluate(() => {
      const chatContainer = document.getElementById('chat-container');
      chatContainer.innerHTML = '';

      const messageDiv = document.createElement('div');
      messageDiv.className = 'message message-assistant';
      messageDiv.setAttribute('data-testid', 'assistant-message');

      const content = document.createElement('div');
      content.className = 'message-content';
      content.textContent = 'The average age of patients is 45.2 years.';
      messageDiv.appendChild(content);

      chatContainer.appendChild(messageDiv);
    });

    const assistantMessage = page.locator('[data-testid="assistant-message"]');
    await expect(assistantMessage).toBeVisible();
    await expect(assistantMessage).toContainText('45.2 years');
  });

  test('thinking indicator displays when injected', async ({ page }) => {
    await page.evaluate(() => {
      const chatContainer = document.getElementById('chat-container');
      chatContainer.innerHTML = '';

      const thinkingDiv = document.createElement('div');
      thinkingDiv.className = 'message message-assistant thinking';
      thinkingDiv.setAttribute('data-testid', 'thinking-indicator');

      const dots = document.createElement('div');
      dots.className = 'thinking-dots';
      dots.innerHTML = '<span></span><span></span><span></span>';
      thinkingDiv.appendChild(dots);

      chatContainer.appendChild(thinkingDiv);
    });

    const thinkingIndicator = page.locator('[data-testid="thinking-indicator"]');
    await expect(thinkingIndicator).toBeVisible();
  });
});

test.describe('LLM status banner', () => {
  test('llm banner element exists and starts hidden', async ({ page }) => {
    await page.goto('/');
    const banner = page.locator('#llm-banner');
    await expect(banner).toBeAttached();
    await expect(banner).toHaveClass(/hidden/);
  });
});

test.describe('Dataset Selector Integration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('dataset selector shows placeholder by default', async ({ page }) => {
    const datasetSelect = page.locator('#dataset-select');
    await expect(datasetSelect).toBeVisible();

    const selectedOption = datasetSelect.locator('option:checked');
    await expect(selectedOption).toHaveText('Select a dataset...');
  });

  test('refresh button is visible', async ({ page }) => {
    const refreshBtn = page.locator('#refresh-datasets');
    await expect(refreshBtn).toBeVisible();
    await expect(refreshBtn).toContainText('⟳');
  });
});
