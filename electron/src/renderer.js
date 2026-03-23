/**
 * Clinical Analytics Renderer Process
 *
 * Handles UI interactions, API communication via clinicalAPI (from preload),
 * and chat message management.
 */
import './index.css';

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
  connectionStatus: document.getElementById('connection-status'),
  statusText: document.querySelector('.status-text'),
  llmBanner: document.getElementById('llm-banner'),
  llmBannerText: document.getElementById('llm-banner-text'),
  datasetSelect: document.getElementById('dataset-select'),
  refreshDatasetsBtn: document.getElementById('refresh-datasets'),
  chatContainer: document.getElementById('chat-container'),
  queryForm: document.getElementById('query-form'),
  queryInput: document.getElementById('query-input'),
  submitBtn: document.getElementById('submit-btn'),
  inputHintText: document.getElementById('input-hint-text'),
};

// ============================================================================
// Application State
// ============================================================================

/**
 * @typedef {Object} ChatMessage
 * @property {string} id - Unique message identifier
 * @property {'user' | 'assistant'} role - Message author role
 * @property {string} content - Message text content
 * @property {Object} [result] - Analysis result data (for assistant messages)
 * @property {boolean} [isStreaming] - Whether message is currently streaming
 * @property {Date} timestamp - Message creation time
 */

const state = {
  connected: false,
  currentDatasetId: null,
  sessionId: generateSessionId(),
  /** @type {ChatMessage[]} */
  messages: [],
  /** @type {EventSource | null} */
  activeStream: null,
  isProcessing: false,
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Generate a unique session ID.
 * @returns {string}
 */
function generateSessionId() {
  return `sess_${Date.now().toString(36)}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Update connection status UI.
 * @param {'checking' | 'connected' | 'error'} status
 * @param {string} [message]
 */
/**
 * Show or hide Ollama / local LLM guidance from GET /health payload.
 * @param {Record<string, unknown>} health
 */
function updateLlmBanner(health) {
  const { llmBanner, llmBannerText } = elements;
  if (!llmBanner || !llmBannerText) return;

  llmBanner.classList.remove('llm-banner--error', 'llm-banner--warn');
  llmBanner.classList.add('hidden');

  if (!health || health.status !== 'healthy') {
    return;
  }

  // Older API responses without Ollama probe fields — do not show a banner.
  if (health.ollama_reachable === undefined) {
    return;
  }

  const reachable = health.ollama_reachable === true;
  const defaultOk = health.ollama_default_model_available === true;
  const model = typeof health.ollama_default_model === 'string' ? health.ollama_default_model : 'the configured model';

  if (reachable && defaultOk) {
    return;
  }

  llmBanner.classList.remove('hidden');

  if (!reachable) {
    llmBanner.classList.add('llm-banner--error');
    llmBannerText.textContent =
      `Local LLM (Ollama) is not reachable at ${health.ollama_base_url || 'localhost'}. ` +
      `Tier-3 NL fallback and some enrichments require Ollama running with ${model}.`;
    return;
  }

  llmBanner.classList.add('llm-banner--warn');
  llmBannerText.textContent =
    `Ollama is running but the default model "${model}" was not found. ` +
    'Run `ollama pull` for that model or update config/nl_query.yaml.';
}

function updateConnectionStatus(status, message) {
  const { connectionStatus, statusText } = elements;

  connectionStatus.className = `status-indicator status-${status}`;

  switch (status) {
    case 'connected':
      statusText.textContent = message || 'Connected';
      state.connected = true;
      break;
    case 'error':
      statusText.textContent = message || 'Disconnected';
      state.connected = false;
      break;
    case 'checking':
    default:
      statusText.textContent = message || 'Checking connection...';
      break;
  }
}

/**
 * Enable or disable query input based on dataset selection.
 * @param {boolean} enabled
 */
function setQueryInputEnabled(enabled) {
  const { queryInput, submitBtn, inputHintText } = elements;

  queryInput.disabled = !enabled || state.isProcessing;
  submitBtn.disabled = !enabled || state.isProcessing;

  if (state.isProcessing) {
    inputHintText.textContent = 'Processing your question...';
  } else if (enabled) {
    inputHintText.textContent = 'Press Enter to submit or click the arrow';
    queryInput.focus();
  } else {
    inputHintText.textContent = 'Select a dataset to start asking questions';
  }
}

/**
 * Generate a unique message ID.
 * @returns {string}
 */
function generateMessageId() {
  return `msg_${Date.now().toString(36)}_${Math.random().toString(36).substr(2, 5)}`;
}

/**
 * Escape HTML to prevent XSS.
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Add a message to the chat state and render it.
 * @param {ChatMessage} message
 */
function addMessage(message) {
  state.messages.push(message);
  renderMessage(message);
  scrollToBottom();
}

/**
 * Update an existing message (for streaming).
 * @param {string} messageId
 * @param {Partial<ChatMessage>} updates
 */
function updateMessage(messageId, updates) {
  const messageIndex = state.messages.findIndex((m) => m.id === messageId);
  if (messageIndex === -1) return;

  state.messages[messageIndex] = { ...state.messages[messageIndex], ...updates };
  rerenderMessage(state.messages[messageIndex]);
}

/**
 * Scroll chat container to bottom.
 */
function scrollToBottom() {
  const { chatContainer } = elements;
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Clear the welcome message if present.
 */
function clearWelcomeMessage() {
  const welcomeMessage = elements.chatContainer.querySelector('.welcome-message');
  if (welcomeMessage) {
    welcomeMessage.remove();
  }
}

/**
 * Render a single message to the chat container.
 * @param {ChatMessage} message
 */
function renderMessage(message) {
  const { chatContainer } = elements;

  const messageDiv = document.createElement('div');
  messageDiv.className = `message message-${message.role}`;
  messageDiv.setAttribute('data-message-id', message.id);
  messageDiv.setAttribute('data-testid', `${message.role}-message`);

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';

  if (message.isStreaming) {
    // Show thinking indicator
    contentDiv.innerHTML = `
      <div class="thinking-dots" data-testid="thinking-indicator">
        <span></span><span></span><span></span>
      </div>
    `;
  } else {
    contentDiv.innerHTML = escapeHtml(message.content);

    // Add result card if present
    if (message.result) {
      const resultCard = renderResultCard(message.result);
      contentDiv.appendChild(resultCard);
    }
  }

  messageDiv.appendChild(contentDiv);
  chatContainer.appendChild(messageDiv);
}

/**
 * Re-render an existing message (for updates).
 * @param {ChatMessage} message
 */
function rerenderMessage(message) {
  const existingDiv = elements.chatContainer.querySelector(
    `[data-message-id="${message.id}"]`
  );
  if (!existingDiv) return;

  const contentDiv = existingDiv.querySelector('.message-content');
  if (!contentDiv) return;

  if (message.isStreaming) {
    contentDiv.innerHTML = `
      <div class="thinking-dots" data-testid="thinking-indicator">
        <span></span><span></span><span></span>
      </div>
    `;
  } else {
    contentDiv.innerHTML = escapeHtml(message.content);

    // Add result card if present
    if (message.result) {
      const resultCard = renderResultCard(message.result);
      contentDiv.appendChild(resultCard);
    }
  }
}

/**
 * Render a result card for analysis results.
 * @param {Object} result
 * @returns {HTMLElement}
 */
function renderResultCard(result) {
  const card = document.createElement('div');
  card.className = 'result-card';

  if (result.summary) {
    const summary = document.createElement('p');
    summary.className = 'result-summary';
    summary.textContent = result.summary;
    card.appendChild(summary);
  }

  if (result.table) {
    const tableWrapper = document.createElement('div');
    tableWrapper.className = 'result-table-wrapper';
    tableWrapper.innerHTML = renderTable(result.table);
    card.appendChild(tableWrapper);
  }

  return card;
}

/**
 * Render a simple table from data.
 * @param {Object} tableData - {columns: string[], rows: any[][]}
 * @returns {string} HTML string
 */
function renderTable(tableData) {
  if (!tableData || !tableData.columns || !tableData.rows) {
    return '<p class="result-error">No data available</p>';
  }

  const headerCells = tableData.columns.map((col) => `<th>${escapeHtml(col)}</th>`).join('');
  const rows = tableData.rows
    .map(
      (row) =>
        '<tr>' + row.map((cell) => `<td>${escapeHtml(String(cell ?? ''))}</td>`).join('') + '</tr>'
    )
    .join('');

  return `
    <table class="result-table">
      <thead><tr>${headerCells}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// ============================================================================
// API Communication
// ============================================================================

/**
 * Check backend health and update UI.
 */
async function checkHealth() {
  updateConnectionStatus('checking');

  try {
    // clinicalAPI is exposed via preload script
    if (typeof window.clinicalAPI === 'undefined') {
      throw new Error('clinicalAPI not available - preload script may have failed');
    }

    const health = await window.clinicalAPI.healthCheck();

    if (health.status === 'healthy') {
      updateConnectionStatus('connected', 'Backend connected');
      updateLlmBanner(health);
      console.log('✅ Backend health check passed:', health);
    } else {
      updateLlmBanner({});
      updateConnectionStatus('error', `Backend: ${health.status}`);
      console.warn('⚠️ Backend health check returned:', health);
    }
  } catch (error) {
    updateLlmBanner({});
    updateConnectionStatus('error', 'Backend unavailable');
    console.error('❌ Health check failed:', error);
  }
}

/**
 * Load available datasets into the selector.
 */
async function loadDatasets() {
  const { datasetSelect } = elements;

  try {
    if (typeof window.clinicalAPI === 'undefined') {
      console.warn('clinicalAPI not available');
      return;
    }

    const response = await window.clinicalAPI.listDatasets();
    const { datasets } = response;

    // Clear existing options (except placeholder)
    datasetSelect.innerHTML = '<option value="">Select a dataset...</option>';

    // Add dataset options
    datasets.forEach((dataset) => {
      const option = document.createElement('option');
      option.value = dataset.dataset_id;
      option.textContent = `${dataset.name} (${dataset.row_count} rows)`;
      datasetSelect.appendChild(option);
    });

    console.log(`📊 Loaded ${datasets.length} datasets`);
  } catch (error) {
    console.error('❌ Failed to load datasets:', error);
  }
}

// ============================================================================
// Event Handlers
// ============================================================================

/**
 * Handle dataset selection change.
 */
function handleDatasetChange(event) {
  const datasetId = event.target.value;
  state.currentDatasetId = datasetId;

  if (datasetId) {
    setQueryInputEnabled(true);
    console.log(`📊 Selected dataset: ${datasetId}`);
  } else {
    setQueryInputEnabled(false);
  }
}

/**
 * Handle refresh datasets button click.
 */
async function handleRefreshDatasets() {
  const btn = elements.refreshDatasetsBtn;
  btn.disabled = true;
  btn.textContent = '...';

  await loadDatasets();

  btn.disabled = false;
  btn.textContent = '⟳';
}

/**
 * Handle query form submission.
 * @param {Event} event
 */
async function handleQuerySubmit(event) {
  event.preventDefault();

  const { queryInput } = elements;
  const queryText = queryInput.value.trim();

  if (!queryText || !state.currentDatasetId || state.isProcessing) {
    return;
  }

  // Clear input and update state
  queryInput.value = '';
  handleQueryInputChange(); // Reset textarea height
  state.isProcessing = true;
  setQueryInputEnabled(true); // Will show processing state

  // Clear welcome message on first query
  clearWelcomeMessage();

  // Add user message
  const userMessage = {
    id: generateMessageId(),
    role: 'user',
    content: queryText,
    timestamp: new Date(),
  };
  addMessage(userMessage);

  // Add assistant message with streaming indicator
  const assistantMessageId = generateMessageId();
  const assistantMessage = {
    id: assistantMessageId,
    role: 'assistant',
    content: '',
    isStreaming: true,
    timestamp: new Date(),
  };
  addMessage(assistantMessage);

  console.log('📝 Query submitted:', {
    sessionId: state.sessionId,
    datasetId: state.currentDatasetId,
    query: queryText,
  });

  try {
    // Submit query to backend
    const response = await window.clinicalAPI.submitQuery(
      state.sessionId,
      state.currentDatasetId,
      queryText
    );

    console.log('📡 Query submitted, got response:', response);

    // Start SSE stream if stream URL provided
    if (response.stream_url || response.query_id) {
      await subscribeToQueryStream(response.query_id, assistantMessageId);
    } else {
      // Fallback: use direct result if available
      updateMessage(assistantMessageId, {
        content: response.summary || 'Analysis complete.',
        result: response.result || null,
        isStreaming: false,
      });
    }
  } catch (error) {
    console.error('❌ Query failed:', error);
    updateMessage(assistantMessageId, {
      content: `Sorry, I encountered an error: ${error.message}`,
      isStreaming: false,
    });
  } finally {
    state.isProcessing = false;
    setQueryInputEnabled(!!state.currentDatasetId);
  }
}

/**
 * Subscribe to SSE stream for query results.
 * @param {string} queryId
 * @param {string} assistantMessageId
 */
async function subscribeToQueryStream(queryId, assistantMessageId) {
  return new Promise((resolve, reject) => {
    let contentBuffer = '';
    let resolved = false;
    let cleanupFn = null;

    const handleEvent = ({ event: eventType, data }) => {
      console.log('📨 SSE event:', eventType, data);

      switch (eventType) {
        case 'query_started':
          // Keep streaming indicator
          break;

        case 'query_progress':
          // Update with progress message
          if (data.message || data.stage) {
            updateMessage(assistantMessageId, {
              content: data.message || `Stage: ${data.stage}`,
              isStreaming: true,
            });
          }
          break;

        case 'query_completed': {
          // Final result - extract result_preview for display
          const resultPreview = data.result_preview || {};
          const intentType = data.intent_type || 'analysis';

          // Build response message
          let responseContent = `Analysis complete (${intentType})`;

          // Extract table data if present
          let tableResult = null;
          for (const value of Object.values(resultPreview)) {
            if (value && value.table) {
              tableResult = value.table;
              responseContent = `Found ${value.row_count || tableResult.rows?.length || 0} results`;
              break;
            }
          }

          updateMessage(assistantMessageId, {
            content: responseContent,
            result: tableResult ? { table: tableResult } : null,
            isStreaming: false,
          });

          // Close stream
          if (cleanupFn) cleanupFn();
          resolved = true;
          resolve();
          break;
        }

        case 'query_failed':
          updateMessage(assistantMessageId, {
            content: `Error: ${data.error || 'Unknown error'}`,
            isStreaming: false,
          });
          if (cleanupFn) cleanupFn();
          resolved = true;
          reject(new Error(data.error || 'Query failed'));
          break;

        case 'stream_end':
          console.log('📡 Stream ended');
          if (cleanupFn) cleanupFn();
          if (!resolved) {
            resolved = true;
            resolve();
          }
          break;

        default:
          console.log('Unhandled SSE event:', eventType);
      }
    };

    const handleError = (error) => {
      if (resolved) {
        console.log('📡 SSE connection closed after completion');
        return;
      }

      console.error('❌ SSE stream error:', error);
      if (cleanupFn) cleanupFn();

      const msg = state.messages.find((m) => m.id === assistantMessageId);
      if (msg && msg.isStreaming) {
        updateMessage(assistantMessageId, {
          content: contentBuffer || 'Connection lost. Please try again.',
          isStreaming: false,
        });
      }
      reject(error);
    };

    try {
      // Use callback-based API that works with contextBridge
      cleanupFn = window.clinicalAPI.subscribeToQueryStream(queryId, {
        onEvent: handleEvent,
        onError: handleError,
        onClose: () => {
          console.log('📡 SSE stream closed');
        },
      });
      state.activeStream = { close: cleanupFn };
    } catch (error) {
      console.error('Failed to create SSE stream:', error);
      reject(error);
    }
  });
}

/**
 * Auto-resize textarea as user types.
 */
function handleQueryInputChange() {
  const { queryInput } = elements;
  queryInput.style.height = 'auto';
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 200)}px`;
}

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize the application.
 */
async function init() {
  console.log('🚀 Clinical Analytics Electron app initializing...');

  // Set up event listeners
  elements.datasetSelect.addEventListener('change', handleDatasetChange);
  elements.refreshDatasetsBtn.addEventListener('click', handleRefreshDatasets);
  elements.queryForm.addEventListener('submit', handleQuerySubmit);
  elements.queryInput.addEventListener('input', handleQueryInputChange);

  // Check backend connection
  await checkHealth();

  // Load datasets if connected
  if (state.connected) {
    await loadDatasets();
  }

  console.log('✅ Initialization complete');
}

// Start the app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
