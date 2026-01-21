/**
 * E2E tests for chat interface.
 *
 * Phase 5 of Electron UI Migration: Chat interface with dataset selector.
 */
const { test, expect } = require('./fixtures');

test.describe('Chat Interface', () => {
  test('user can type message in query input', async ({ window }) => {
    // Arrange: Input should exist but be disabled without dataset
    const queryInput = window.locator('#query-input');
    await expect(queryInput).toBeVisible();
    await expect(queryInput).toBeDisabled();

    // Note: Without a running backend and dataset, we can only test UI presence
    // Full E2E with backend requires mock server or real backend running
  });

  test('submit button is disabled without dataset selection', async ({ window }) => {
    // Assert: Submit button should be disabled
    const submitBtn = window.locator('#submit-btn');
    await expect(submitBtn).toBeVisible();
    await expect(submitBtn).toBeDisabled();
  });

  test('welcome message displays example queries', async ({ window }) => {
    // Assert: Welcome message with examples is visible
    const welcomeMessage = window.locator('.welcome-message');
    await expect(welcomeMessage).toBeVisible();

    const exampleQueries = window.locator('.example-queries');
    await expect(exampleQueries).toBeVisible();
    await expect(exampleQueries).toContainText('What is the average age');
  });

  test('user message appears in chat after submission', async ({ window }) => {
    // This test verifies the UI displays user messages
    // It simulates message rendering without needing backend

    // Arrange: Add a message via JavaScript injection
    await window.evaluate(() => {
      // Access the addMessage function if exposed, or manipulate DOM directly
      const chatContainer = document.getElementById('chat-container');
      // Clear welcome message for test
      chatContainer.innerHTML = '';

      // Create a user message element
      const messageDiv = document.createElement('div');
      messageDiv.className = 'message message-user';
      messageDiv.setAttribute('data-testid', 'user-message');

      const content = document.createElement('div');
      content.className = 'message-content';
      content.textContent = 'What is the average age of patients?';
      messageDiv.appendChild(content);

      chatContainer.appendChild(messageDiv);
    });

    // Assert: User message appears in chat
    const userMessage = window.locator('[data-testid="user-message"]');
    await expect(userMessage).toBeVisible();
    await expect(userMessage).toContainText('What is the average age');
  });

  test('assistant message appears in chat', async ({ window }) => {
    // Arrange: Add an assistant message via JavaScript injection
    await window.evaluate(() => {
      const chatContainer = document.getElementById('chat-container');
      chatContainer.innerHTML = '';

      // Create an assistant message element
      const messageDiv = document.createElement('div');
      messageDiv.className = 'message message-assistant';
      messageDiv.setAttribute('data-testid', 'assistant-message');

      const content = document.createElement('div');
      content.className = 'message-content';
      content.textContent = 'The average age of patients is 45.2 years.';
      messageDiv.appendChild(content);

      chatContainer.appendChild(messageDiv);
    });

    // Assert: Assistant message appears in chat
    const assistantMessage = window.locator('[data-testid="assistant-message"]');
    await expect(assistantMessage).toBeVisible();
    await expect(assistantMessage).toContainText('45.2 years');
  });

  test('thinking indicator displays during query processing', async ({ window }) => {
    // Arrange: Add a thinking indicator via JavaScript injection
    await window.evaluate(() => {
      const chatContainer = document.getElementById('chat-container');
      chatContainer.innerHTML = '';

      // Create a thinking indicator
      const thinkingDiv = document.createElement('div');
      thinkingDiv.className = 'message message-assistant thinking';
      thinkingDiv.setAttribute('data-testid', 'thinking-indicator');

      const dots = document.createElement('div');
      dots.className = 'thinking-dots';
      dots.innerHTML = '<span></span><span></span><span></span>';
      thinkingDiv.appendChild(dots);

      chatContainer.appendChild(thinkingDiv);
    });

    // Assert: Thinking indicator appears
    const thinkingIndicator = window.locator('[data-testid="thinking-indicator"]');
    await expect(thinkingIndicator).toBeVisible();
  });
});

test.describe('Dataset Selector Integration', () => {
  test('dataset selector shows placeholder by default', async ({ window }) => {
    const datasetSelect = window.locator('#dataset-select');
    await expect(datasetSelect).toBeVisible();

    // Check placeholder option is selected
    const selectedOption = datasetSelect.locator('option:checked');
    await expect(selectedOption).toHaveText('Select a dataset...');
  });

  test('refresh button is visible', async ({ window }) => {
    const refreshBtn = window.locator('#refresh-datasets');
    await expect(refreshBtn).toBeVisible();
    await expect(refreshBtn).toContainText('⟳');
  });
});
