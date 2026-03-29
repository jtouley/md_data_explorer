/**
 * Clinical Analytics Renderer Process
 *
 * Handles UI interactions, API communication via clinicalAPI (from preload),
 * and chat message management.
 */
import './index.css';
import { normalizePendingResponse, renderEnrichmentCards } from './enrichmentPanel.js';
import { normalizeHistoryResponse, renderPatchHistoryTable } from './patchHistoryPanel.js';
import { buildQueryCompletedPresentation } from './resultPresentation.js';
import { normalizeSessionListResponse, renderSessionList } from './sessionSidebar.js';

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
  enrichmentSection: document.getElementById('enrichment-section'),
  enrichmentHint: document.getElementById('enrichment-hint'),
  enrichmentList: document.getElementById('enrichment-list'),
  refreshEnrichmentsBtn: document.getElementById('refresh-enrichments'),
  patchHistoryHint: document.getElementById('patch-history-hint'),
  patchHistoryBody: document.getElementById('patch-history-body'),
  sessionSidebar: document.getElementById('session-sidebar'),
  sessionList: document.getElementById('session-list'),
  newChatBtn: document.getElementById('new-chat-btn'),
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
  const model =
    typeof health.ollama_default_model === 'string'
      ? health.ollama_default_model
      : 'the configured model';

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
  const existingDiv = elements.chatContainer.querySelector(`[data-message-id="${message.id}"]`);
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
  card.setAttribute('data-testid', 'result-card');
  if (result.intentType) {
    card.setAttribute('data-intent', result.intentType);
  }

  if (result.title) {
    const titleEl = document.createElement('div');
    titleEl.className = 'result-card-title';
    titleEl.setAttribute('data-testid', 'result-card-title');
    titleEl.textContent = result.title;
    card.appendChild(titleEl);
  }

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
 * Normalize rows to arrays of scalars (API may send Polars to_dicts() row objects).
 * @param {string[]} columns
 * @param {unknown[]} rows
 * @returns {unknown[][]}
 */
function normalizeTableRows(columns, rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return [];
  }
  return rows.map((row) => {
    if (Array.isArray(row)) {
      return row;
    }
    if (row && typeof row === 'object' && Array.isArray(columns)) {
      return columns.map((col) => {
        const v = /** @type {Record<string, unknown>} */ (row)[col];
        if (v === null || v === undefined) {
          return '';
        }
        if (typeof v === 'object') {
          return JSON.stringify(v);
        }
        return v;
      });
    }
    return [];
  });
}

/**
 * Render a simple table from data.
 * @param {Object} tableData - {columns: string[], rows: any[][] | Record<string, unknown>[]}
 * @returns {string} HTML string
 */
function renderTable(tableData) {
  if (!tableData || !tableData.columns || !tableData.rows) {
    return '<p class="result-error">No data available</p>';
  }

  const cols = tableData.columns;
  const rowArrays = normalizeTableRows(cols, tableData.rows);
  const headerCells = cols.map((col) => `<th>${escapeHtml(col)}</th>`).join('');
  const rows = rowArrays
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

const noopEnrichment = () => {};

/**
 * Load pending metadata enrichments for the current dataset into the panel.
 */
async function loadEnrichmentPanel() {
  const { enrichmentSection, enrichmentHint, enrichmentList, patchHistoryHint, patchHistoryBody } =
    elements;

  if (!enrichmentSection || !enrichmentHint || !enrichmentList) return;

  if (!state.currentDatasetId) {
    enrichmentSection.classList.add('hidden');
    enrichmentHint.textContent = '';
    enrichmentList.replaceChildren();
    if (patchHistoryHint) patchHistoryHint.textContent = '';
    if (patchHistoryBody) patchHistoryBody.replaceChildren();
    return;
  }

  enrichmentSection.classList.remove('hidden');
  enrichmentHint.textContent = 'Loading pending…';
  if (patchHistoryHint) patchHistoryHint.textContent = 'Loading history…';
  if (patchHistoryBody) patchHistoryBody.replaceChildren();

  const hasPending = typeof window.clinicalAPI?.getPendingEnrichments === 'function';
  const hasHistory = typeof window.clinicalAPI?.getEnrichmentHistory === 'function';

  if (!hasPending && !hasHistory) {
    enrichmentHint.textContent = 'Enrichment API not available.';
    renderEnrichmentCards(enrichmentList, [], {
      onAccept: noopEnrichment,
      onReject: noopEnrichment,
    });
    if (patchHistoryHint) patchHistoryHint.textContent = '';
    return;
  }

  try {
    const pendingP = hasPending
      ? window.clinicalAPI.getPendingEnrichments(state.currentDatasetId)
      : Promise.resolve({ suggestions: [], total: 0 });
    const historyP = hasHistory
      ? window.clinicalAPI.getEnrichmentHistory(state.currentDatasetId)
      : Promise.resolve({ patches: [], total: 0 });

    const [pendingRaw, historyRaw] = await Promise.all([pendingP, historyP]);

    const { suggestions, total } = normalizePendingResponse(pendingRaw);
    enrichmentHint.textContent =
      total === 0
        ? 'No pending suggestions.'
        : `${total} pending suggestion${total === 1 ? '' : 's'}.`;
    renderEnrichmentCards(enrichmentList, suggestions, {
      onAccept: (patchId) => handleEnrichmentAccept(patchId),
      onReject: (patchId) => handleEnrichmentReject(patchId),
    });

    const { patches, total: histTotal } = normalizeHistoryResponse(historyRaw);
    if (patchHistoryHint) {
      patchHistoryHint.textContent =
        histTotal === 0
          ? 'No history entries.'
          : `${histTotal} patch${histTotal === 1 ? '' : 'es'} in log (newest first).`;
    }
    if (patchHistoryBody) {
      renderPatchHistoryTable(patchHistoryBody, patches, {
        onRevert: (patchId) => handlePatchRevert(patchId),
      });
    }
  } catch (error) {
    console.error('❌ Failed to load enrichments / history:', error);
    enrichmentHint.textContent = `Could not load enrichments: ${error.message}`;
    renderEnrichmentCards(enrichmentList, [], {
      onAccept: noopEnrichment,
      onReject: noopEnrichment,
    });
    if (patchHistoryHint) {
      patchHistoryHint.textContent = `History: ${error.message}`;
    }
    if (patchHistoryBody) {
      renderPatchHistoryTable(patchHistoryBody, [], { onRevert: noopEnrichment });
    }
  }
}

/**
 * @param {string} patchId
 */
async function handleEnrichmentAccept(patchId) {
  const { enrichmentHint } = elements;
  if (!state.currentDatasetId) return;
  try {
    const res = await window.clinicalAPI.acceptEnrichment(state.currentDatasetId, patchId);
    if (res && res.success === false) {
      throw new Error(res.message || 'Accept failed');
    }
    await loadEnrichmentPanel();
  } catch (error) {
    console.error('Accept enrichment failed:', error);
    if (enrichmentHint) {
      enrichmentHint.textContent = error.message || 'Accept failed';
    }
  }
}

/**
 * @param {string} patchId
 */
async function handleEnrichmentReject(patchId) {
  const { enrichmentHint } = elements;
  if (!state.currentDatasetId) return;
  try {
    const res = await window.clinicalAPI.rejectEnrichment(state.currentDatasetId, patchId);
    if (res && res.success === false) {
      throw new Error(res.message || 'Reject failed');
    }
    await loadEnrichmentPanel();
  } catch (error) {
    console.error('Reject enrichment failed:', error);
    if (enrichmentHint) {
      enrichmentHint.textContent = error.message || 'Reject failed';
    }
  }
}

/**
 * @param {string} patchId
 */
async function handlePatchRevert(patchId) {
  const { patchHistoryHint } = elements;
  if (!state.currentDatasetId) return;
  if (typeof window.clinicalAPI?.revertEnrichmentPatch !== 'function') {
    if (patchHistoryHint) patchHistoryHint.textContent = 'Revert API not available.';
    return;
  }
  try {
    const res = await window.clinicalAPI.revertEnrichmentPatch(state.currentDatasetId, patchId);
    if (res && res.success === false) {
      throw new Error(res.message || 'Revert failed');
    }
    await loadEnrichmentPanel();
  } catch (error) {
    console.error('Revert patch failed:', error);
    if (patchHistoryHint) {
      patchHistoryHint.textContent = error.message || 'Revert failed';
    }
  }
}

// ============================================================================
// Session Sidebar
// ============================================================================

async function loadSessionList() {
  const { sessionList } = elements;
  if (!sessionList) return;
  if (typeof window.clinicalAPI?.listSessions !== 'function') return;

  try {
    const raw = await window.clinicalAPI.listSessions();
    const { sessions } = normalizeSessionListResponse(raw);
    renderSessionList(sessionList, sessions, state.sessionId, {
      onSelect: (id) => handleSessionSelect(id, sessions),
      onDelete: (id) => handleSessionDelete(id),
    });
  } catch (error) {
    console.error('Failed to load sessions:', error);
  }
}

/**
 * @param {string} sessionId
 * @param {object[]} sessions
 */
function handleSessionSelect(sessionId, sessions) {
  if (sessionId === state.sessionId) return;
  const session = sessions.find((s) => s.session_id === sessionId);

  state.sessionId = sessionId;
  state.messages = [];
  state.isProcessing = false;
  if (state.activeStream) {
    state.activeStream.close();
    state.activeStream = null;
  }

  const { chatContainer } = elements;
  chatContainer.innerHTML = '';
  const welcome = document.createElement('div');
  welcome.className = 'welcome-message';
  welcome.innerHTML = `
    <div class="welcome-icon">📊</div>
    <h2>Session restored</h2>
    <p>Switched to session <code>${escapeHtml(sessionId)}</code>. New queries will be associated with this session.</p>
  `;
  chatContainer.appendChild(welcome);

  if (session?.dataset_id && session.dataset_id !== state.currentDatasetId) {
    const { datasetSelect } = elements;
    if (datasetSelect) {
      datasetSelect.value = session.dataset_id;
      state.currentDatasetId = session.dataset_id;
      setQueryInputEnabled(true);
      void loadEnrichmentPanel();
    }
  }

  void loadSessionList();
  console.log('Switched to session:', sessionId);
}

async function handleSessionDelete(sessionId) {
  if (typeof window.clinicalAPI?.deleteSession !== 'function') return;
  try {
    await window.clinicalAPI.deleteSession(sessionId);
    if (sessionId === state.sessionId) {
      await handleNewChat();
    } else {
      await loadSessionList();
    }
  } catch (error) {
    console.error('Delete session failed:', error);
  }
}

async function handleNewChat() {
  if (!state.currentDatasetId) return;
  if (typeof window.clinicalAPI?.createSession !== 'function') {
    state.sessionId = generateSessionId();
    resetChatUI();
    return;
  }
  try {
    const res = await window.clinicalAPI.createSession(state.currentDatasetId);
    state.sessionId = res.session_id;
    resetChatUI();
    await loadSessionList();
  } catch (error) {
    console.error('Create session failed:', error);
    state.sessionId = generateSessionId();
    resetChatUI();
  }
}

function resetChatUI() {
  state.messages = [];
  state.isProcessing = false;
  if (state.activeStream) {
    state.activeStream.close();
    state.activeStream = null;
  }
  const { chatContainer } = elements;
  chatContainer.innerHTML = '';
  const welcome = document.createElement('div');
  welcome.className = 'welcome-message';
  welcome.innerHTML = `
    <div class="welcome-icon">📊</div>
    <h2>Welcome to Clinical Analytics</h2>
    <p>Select a dataset above and ask questions in natural language.</p>
    <div class="example-queries">
      <p class="examples-title">Try asking:</p>
      <ul>
        <li>"What is the average age of patients?"</li>
        <li>"Compare outcomes between treatment groups"</li>
        <li>"Show me the distribution of diagnoses"</li>
      </ul>
    </div>
  `;
  chatContainer.appendChild(welcome);
  setQueryInputEnabled(!!state.currentDatasetId);
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
    void loadEnrichmentPanel();
    void loadSessionList();
  } else {
    setQueryInputEnabled(false);
    void loadEnrichmentPanel();
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
          const pres = buildQueryCompletedPresentation(data.intent_type, data.result_preview || {});

          updateMessage(assistantMessageId, {
            content: pres.content,
            result: pres.result,
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
  if (elements.refreshEnrichmentsBtn) {
    elements.refreshEnrichmentsBtn.addEventListener('click', () => {
      void loadEnrichmentPanel();
    });
  }
  elements.queryForm.addEventListener('submit', handleQuerySubmit);
  elements.queryInput.addEventListener('input', handleQueryInputChange);

  if (elements.newChatBtn) {
    elements.newChatBtn.addEventListener('click', () => void handleNewChat());
  }

  // Check backend connection
  await checkHealth();

  // Load datasets and sessions if connected
  if (state.connected) {
    await loadDatasets();
    await loadSessionList();
  }

  console.log('✅ Initialization complete');
}

// Start the app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
