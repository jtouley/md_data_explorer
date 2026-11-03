/**
 * Clinical Analytics Renderer Process
 *
 * Handles UI interactions, API communication via clinicalAPI (from preload),
 * and chat message management. Layout, states and copy follow docs/app/DESIGN.md.
 */
import './index.css';
import {
  normalizePendingResponse,
  renderEnrichmentCards,
  runEnrichmentDecision,
} from './enrichmentPanel.js';
import { normalizeHistoryResponse, renderPatchHistoryTable } from './patchHistoryPanel.js';
import {
  HIDDEN_CELL_FOOTNOTE,
  buildConfirmationHoldPresentation,
  buildExampleQuestions,
  buildQueryCompletedPresentation,
  claimDecimals,
  classifyCell,
  describeQueryFailure,
  isNumericColumn,
  isPValueColumn,
  parsePlanLine,
  splitClaim,
  stageLabel,
} from './resultPresentation.js';
import { buildConfirmRequest, submitConfirmation } from './confirmHold.js';
import { closeDatasetDeleteConfirm, renderDatasetDeleteConfirm } from './datasetDeletePanel.js';
import {
  normalizeSessionListResponse,
  renderSessionList,
  titleFromQuestion,
} from './sessionSidebar.js';
import { normalizeUploadResponse, renderUploadArea } from './uploadPanel.js';
import { icon } from './icons.js';
import { THEME_STORAGE_KEY, isThemeSource } from './theme.js';

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
  appContainer: document.querySelector('.app-container'),
  connectionStatus: document.getElementById('connection-status'),
  statusText: document.querySelector('.status-text'),
  offlineBanner: document.getElementById('offline-banner'),
  retryConnectionBtn: document.getElementById('retry-connection'),
  llmBanner: document.getElementById('llm-banner'),
  llmBannerText: document.getElementById('llm-banner-text'),
  datasetSelect: document.getElementById('dataset-select'),
  datasetMenuBtn: document.getElementById('dataset-menu-btn'),
  datasetMenu: document.getElementById('dataset-menu'),
  datasetMeta: document.getElementById('dataset-meta'),
  datasetStatus: document.getElementById('dataset-status'),
  openDatasetPanelBtn: document.getElementById('open-dataset-panel'),
  refreshDatasetsBtn: document.getElementById('refresh-datasets'),
  deleteDatasetBtn: document.getElementById('delete-dataset'),
  datasetDeleteArea: document.getElementById('dataset-delete-area'),
  datasetPanel: document.getElementById('dataset-panel'),
  datasetPanelTitle: document.getElementById('dataset-panel-title'),
  closeDatasetPanelBtn: document.getElementById('close-dataset-panel'),
  datasetOverview: document.getElementById('dataset-overview'),
  datasetOverviewSummary: document.getElementById('dataset-overview-summary'),
  datasetPreviewHead: document.getElementById('dataset-preview-head'),
  datasetPreviewBody: document.getElementById('dataset-preview-body'),
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
  uploadArea: document.getElementById('upload-area'),
  themeRadios: document.querySelectorAll('input[name="theme"]'),
};

// ============================================================================
// Application State
// ============================================================================

/**
 * @typedef {'working' | 'answered' | 'held' | 'refused' | 'error' | 'stopped'} MessageKind
 */

/**
 * @typedef {Object} ChatMessage
 * @property {string} id - Unique message identifier
 * @property {'user' | 'assistant'} role - Message author role
 * @property {string} content - Message text content
 * @property {Object} [result] - Analysis result data (for assistant messages)
 * @property {boolean} [isStreaming] - Whether message is currently streaming
 * @property {MessageKind} [kind] - What the assistant message is (see DESIGN.md "Message states")
 * @property {number} [startedAt] - When work on the answer started (ms)
 * @property {string} [detail] - The engine's exact words when the notice reworded them
 * @property {Date} timestamp - Message creation time
 */

const state = {
  connected: false,
  currentDatasetId: null,
  sessionId: generateSessionId(),
  /** True once the session exists in the API (created or picked from the list). */
  sessionOnServer: false,
  /** @type {ChatMessage[]} */
  messages: [],
  /** @type {{ close: () => void, stop?: () => void } | null} */
  activeStream: null,
  isProcessing: false,
  stopRequested: false,
  /** @type {{ dataset_id: string, name: string, row_count?: number }[]} */
  datasets: [],
  /** @type {{ datasetId: string, rowCount: number, columns: string[], rows: object[], schema: Record<string, string> } | null} */
  datasetInfo: null,
  /** First question per session, this run only (never persisted: questions can identify patients). */
  sessionTitles: new Map(),
};

const NEAR_BOTTOM_PX = 80;
const ELAPSED_VISIBLE_AFTER_S = 3;
const LOST_CONNECTION_MESSAGE =
  'The connection to the analysis engine dropped before the answer arrived. Ask again.';
const RESTORED_QUESTION_HINT = 'Your question is back in the box below so you can change it.';

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
 * Create an element with an optional class and text.
 * @param {string} tag
 * @param {string} [className]
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function prefersReducedMotion() {
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
}

/**
 * Announce a status change to screen readers via the persistent live region.
 * @param {string} text
 */
function announce(text) {
  const region = document.getElementById('sr-status');
  if (!region) return;
  region.textContent = '';
  // A fresh text node after clearing makes repeated messages announce again.
  window.setTimeout(() => {
    region.textContent = text;
  }, 50);
}

/**
 * @param {string | null} datasetId
 * @returns {string}
 */
function datasetName(datasetId) {
  if (!datasetId) return '';
  return state.datasets.find((d) => d.dataset_id === datasetId)?.name || datasetId;
}

/**
 * Status line under the cohort switcher (uploads, sample, list errors).
 * @param {string} text
 * @param {'info' | 'ok' | 'error'} [tone]
 */
function setDatasetStatus(text, tone = 'info') {
  const { datasetStatus } = elements;
  if (!datasetStatus) return;
  datasetStatus.textContent = text;
  datasetStatus.dataset.tone = tone;
}

// ============================================================================
// Connection + local language model status
// ============================================================================

/**
 * Show or hide local language model guidance from GET /health payload.
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
      "Some questions need the local language model, which isn't running. " +
      'Simple questions still work.';
    return;
  }

  llmBanner.classList.add('llm-banner--warn');
  llmBannerText.textContent =
    `The local language model "${model}" isn't installed, so some questions won't work. ` +
    'Simple questions still work.';
}

/**
 * Update connection status UI.
 * @param {'checking' | 'connected' | 'error'} status
 */
function updateConnectionStatus(status) {
  const { connectionStatus, statusText, offlineBanner } = elements;

  connectionStatus.className = `status-indicator status-${status}`;

  switch (status) {
    case 'connected':
      statusText.textContent = 'Running on this computer';
      state.connected = true;
      offlineBanner?.classList.add('hidden');
      break;
    case 'error':
      statusText.textContent = 'Not connected';
      state.connected = false;
      offlineBanner?.classList.remove('hidden');
      break;
    case 'checking':
    default:
      statusText.textContent = 'Connecting…';
      break;
  }
}

// ============================================================================
// Composer
// ============================================================================

/**
 * Enable or disable the composer. While an answer is on its way the box stays editable and
 * the send button becomes Stop.
 * @param {boolean} enabled
 */
function setQueryInputEnabled(enabled) {
  const { queryInput, submitBtn, inputHintText } = elements;
  const name = datasetName(state.currentDatasetId);

  queryInput.disabled = !enabled;
  queryInput.placeholder = enabled && name ? `Ask about ${name}…` : 'Choose a cohort to begin';

  if (state.isProcessing) {
    submitBtn.type = 'button';
    submitBtn.disabled = false;
    submitBtn.classList.add('is-stop');
    submitBtn.setAttribute('aria-label', 'Stop');
    submitBtn.title = 'Stop';
    submitBtn.innerHTML = icon('stop');
    inputHintText.textContent = 'Working on your answer. Press Stop to cancel.';
    return;
  }

  submitBtn.type = 'submit';
  submitBtn.disabled = !enabled;
  submitBtn.classList.remove('is-stop');
  submitBtn.setAttribute('aria-label', 'Ask');
  submitBtn.title = 'Ask';
  submitBtn.innerHTML = icon('send');
  inputHintText.textContent = enabled
    ? 'Enter to ask · Shift+Enter for a new line'
    : state.connected
      ? 'Choose a cohort to begin.'
      : "Can't reach the analysis engine on this computer.";

  // Return focus to the box after an answer, but never pull it out of the sidebar or a panel.
  const active = document.activeElement;
  if (enabled && (!active || active === document.body || active === submitBtn)) {
    queryInput.focus();
  }
}

/**
 * Auto-resize textarea as user types.
 */
function handleQueryInputChange() {
  const { queryInput } = elements;
  queryInput.style.height = 'auto';
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 200)}px`;
}

/**
 * Put a question back in the composer (only when the box is empty).
 * @param {string} text
 */
function restoreQuestion(text) {
  const { queryInput } = elements;
  if (!text || queryInput.value) return;
  queryInput.value = text;
  handleQueryInputChange();
}

let elapsedTicker = null;

function startElapsedTicker() {
  if (elapsedTicker) return;
  elapsedTicker = window.setInterval(() => {
    elements.chatContainer.querySelectorAll('.thinking[data-started]').forEach((node) => {
      const seconds = Math.floor((Date.now() - Number(node.dataset.started)) / 1000);
      const out = node.querySelector('.thinking-elapsed');
      if (out) out.textContent = seconds >= ELAPSED_VISIBLE_AFTER_S ? `${seconds} s` : '';
    });
  }, 1000);
}

function stopElapsedTicker() {
  if (elapsedTicker) window.clearInterval(elapsedTicker);
  elapsedTicker = null;
}

function beginProcessing() {
  state.isProcessing = true;
  state.stopRequested = false;
  setQueryInputEnabled(true);
  startElapsedTicker();
}

function endProcessing() {
  state.isProcessing = false;
  state.stopRequested = false;
  state.activeStream = null;
  stopElapsedTicker();
  setQueryInputEnabled(!!state.currentDatasetId);
}

/**
 * Stop waiting for the current answer (Stop button).
 */
function stopActiveQuery() {
  if (!state.isProcessing) return;
  if (state.activeStream?.stop) {
    state.activeStream.stop();
  } else {
    // The question has not reached the stream yet; stop as soon as it does.
    state.stopRequested = true;
  }
}

// ============================================================================
// Chat rendering
// ============================================================================

function isNearBottom() {
  const c = elements.chatContainer;
  return c.scrollHeight - c.scrollTop - c.clientHeight < NEAR_BOTTOM_PX;
}

/**
 * Scroll chat container to bottom.
 */
function scrollToBottom() {
  const { chatContainer } = elements;
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Bring a finished answer into view: the whole answer when it fits, otherwise its top, so the
 * headline is never below the fold.
 * @param {string} messageId
 */
function revealMessage(messageId) {
  const c = elements.chatContainer;
  const node = c.querySelector(`[data-message-id="${messageId}"]`);
  if (!node) return;
  const behavior = prefersReducedMotion() ? 'auto' : 'smooth';
  const cr = c.getBoundingClientRect();
  const nr = node.getBoundingClientRect();
  const top = nr.height > cr.height - 48 ? c.scrollTop + (nr.top - cr.top) - 16 : c.scrollHeight;
  if (typeof c.scrollTo === 'function') c.scrollTo({ top, behavior });
  else c.scrollTop = top;
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

  const wasAtBottom = isNearBottom();
  state.messages[messageIndex] = { ...state.messages[messageIndex], ...updates };
  rerenderMessage(state.messages[messageIndex]);
  if (!wasAtBottom) return;
  if (state.messages[messageIndex].isStreaming) scrollToBottom();
  else revealMessage(messageId);
}

/**
 * @param {ChatMessage} message
 * @returns {MessageKind | 'question'}
 */
function messageKind(message) {
  if (message.role === 'user') return 'question';
  if (message.isStreaming) return 'working';
  if (message.kind) return message.kind;
  return message.result?.awaitingConfirmation ? 'held' : 'answered';
}

/**
 * Render a single message to the chat container.
 * @param {ChatMessage} message
 */
function renderMessage(message) {
  const messageDiv = el('div', `message message-${message.role}`);
  messageDiv.setAttribute('data-message-id', message.id);
  messageDiv.setAttribute('data-testid', `${message.role}-message`);
  messageDiv.dataset.state = messageKind(message);

  const contentDiv = el('div', 'message-content');
  fillMessageContent(contentDiv, message);
  messageDiv.appendChild(contentDiv);
  elements.chatContainer.appendChild(messageDiv);
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

  existingDiv.dataset.state = messageKind(message);
  // Progress updates change the stage text in place, so the indicator never jumps.
  const thinking = contentDiv.querySelector('.thinking');
  if (message.isStreaming && thinking) {
    const stage = thinking.querySelector('.thinking-stage');
    const text = message.content || 'Reading your question';
    if (stage && stage.textContent !== text) stage.textContent = text;
    return;
  }
  fillMessageContent(contentDiv, message);
}

/**
 * @param {HTMLElement} contentDiv
 * @param {ChatMessage} message
 */
function fillMessageContent(contentDiv, message) {
  contentDiv.replaceChildren();
  if (message.role === 'user') {
    contentDiv.textContent = message.content;
    return;
  }
  if (message.isStreaming) {
    contentDiv.appendChild(buildThinking(message));
    return;
  }
  const kind = messageKind(message);
  if (kind === 'refused') {
    contentDiv.appendChild(
      buildNotice('refused', message.content, RESTORED_QUESTION_HINT, message.detail)
    );
  } else if (kind === 'error') {
    contentDiv.appendChild(
      buildNotice('error', message.content, RESTORED_QUESTION_HINT, message.detail)
    );
  } else if (kind === 'stopped') {
    contentDiv.appendChild(buildNotice('stopped', message.content));
  } else if (message.result?.awaitingConfirmation) {
    contentDiv.appendChild(buildHeldCard(message.result));
  } else if (message.result) {
    contentDiv.appendChild(renderResultCard(message.result, message.content));
  } else {
    contentDiv.appendChild(el('p', 'message-text', message.content));
  }
}

/**
 * Working indicator: pulsing dots, then the stage text as a separate, static element.
 * @param {ChatMessage} message
 * @returns {HTMLElement}
 */
function buildThinking(message) {
  const wrap = el('div', 'thinking');
  wrap.dataset.testid = 'thinking-indicator';
  wrap.dataset.started = String(message.startedAt || Date.now());
  const dots = el('span', 'thinking-dots');
  dots.setAttribute('aria-hidden', 'true');
  dots.append(el('span'), el('span'), el('span'));
  wrap.append(
    dots,
    el('span', 'thinking-stage', message.content || 'Reading your question'),
    el('span', 'thinking-elapsed')
  );
  return wrap;
}

const NOTICE_LABELS = {
  refused: { word: 'Not answered', icon: 'info' },
  error: { word: 'Something went wrong', icon: 'alert' },
  stopped: { word: 'Stopped', icon: 'stop' },
};

/**
 * Refusal, error or stopped notice: a word and an icon, never colour alone.
 * @param {'refused' | 'error' | 'stopped'} kind
 * @param {string} text
 * @param {string} [hint]
 * @param {string} [detail] the engine's exact words, behind a disclosure
 * @returns {HTMLElement}
 */
function buildNotice(kind, text, hint, detail) {
  const notice = el('div', `notice notice--${kind}`);
  notice.dataset.testid = `${kind}-notice`;
  const label = el('p', 'state-label');
  label.innerHTML = `${icon(NOTICE_LABELS[kind].icon)}<span>${NOTICE_LABELS[kind].word}</span>`;
  notice.appendChild(label);
  if (text) notice.appendChild(el('p', 'notice-body', text));
  if (hint) notice.appendChild(el('p', 'notice-hint', hint));
  if (detail) {
    const raw = el('details', 'plan-raw notice-detail');
    raw.append(el('summary', '', 'Exact message'), el('p', 'plan-raw-text', detail));
    notice.appendChild(raw);
  }
  return notice;
}

/**
 * Render a result card for analysis results: caption, headline, stats line, caveats, table.
 * @param {Object} result
 * @param {string} [content] the one-line message for results without a claim
 * @returns {HTMLElement}
 */
function renderResultCard(result, content = '') {
  const card = el('div', 'result-card answer');
  card.setAttribute('data-testid', 'result-card');
  if (result.intentType) {
    card.setAttribute('data-intent', result.intentType);
  }

  if (result.title) {
    const titleEl = el('p', 'result-card-title', result.title);
    titleEl.setAttribute('data-testid', 'result-card-title');
    card.appendChild(titleEl);
  }

  if (result.claim) {
    const { headline, stats, notes } = splitClaim(result.claim);
    const claimEl = el('p', 'result-claim', headline);
    claimEl.setAttribute('data-testid', 'result-claim');
    card.appendChild(claimEl);
    if (stats.length) {
      const statsEl = el('p', 'result-stats', stats.join(' · '));
      statsEl.setAttribute('data-testid', 'result-stats');
      card.appendChild(statsEl);
    }
    if (notes) card.appendChild(el('p', 'result-notes', notes));
  } else {
    const lead = content && content !== result.title ? content : '';
    if (lead) card.appendChild(el('p', 'result-lead', lead));
    if (result.summary) {
      card.appendChild(el('p', 'result-stats', String(result.summary).replace(/ \| /g, ' · ')));
    }
  }

  if (result.table) {
    const { wrapper, hasHidden } = buildResultTable(result.table, {
      decimals: claimDecimals(result.claim),
      label: result.title ? `${result.title} table` : 'Results table',
    });
    card.appendChild(wrapper);
    if (hasHidden) {
      const foot = el('p', 'result-footnote');
      foot.dataset.testid = 'hidden-cell-footnote';
      foot.innerHTML = `<span aria-hidden="true">† </span>${escapeHtml(HIDDEN_CELL_FOOTNOTE)}`;
      card.appendChild(foot);
    }
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
      return columns.map((col) => /** @type {Record<string, unknown>} */ (row)[col]);
    }
    return [];
  });
}

/**
 * Mark a table frame while columns are out of view to the right (edge fade and hint).
 * @param {HTMLElement} frame
 * @param {HTMLElement} scroller
 * @param {HTMLElement} hint
 */
function trackTableOverflow(frame, scroller, hint) {
  const update = () => {
    const overflowing = scroller.scrollWidth - scroller.clientWidth > 1;
    const atEnd = scroller.scrollLeft + scroller.clientWidth >= scroller.scrollWidth - 1;
    frame.classList.toggle('has-more-right', overflowing && !atEnd);
    hint.classList.toggle('hidden', !overflowing);
  };
  scroller.addEventListener('scroll', update, { passive: true });
  if (typeof ResizeObserver === 'function') new ResizeObserver(update).observe(scroller);
  window.requestAnimationFrame(update);
}

/**
 * Statistics table: body face, tabular figures, numbers right-aligned, hidden cells marked.
 * The scrolling box is a labelled, focusable region so the keyboard can scroll it; the first
 * column stays in view.
 * @param {{ columns?: string[], rows?: unknown[] }} tableData
 * @param {{ decimals?: number, label?: string }} [opts]
 * @returns {{ wrapper: HTMLElement, hasHidden: boolean }}
 */
function buildResultTable(tableData, opts = {}) {
  const frame = el('div', 'result-table-frame');
  const wrapper = el('div', 'result-table-wrapper');
  frame.appendChild(wrapper);
  if (!tableData || !Array.isArray(tableData.columns) || !Array.isArray(tableData.rows)) {
    wrapper.appendChild(el('p', 'result-error', 'No table was returned for this answer.'));
    return { wrapper: frame, hasHidden: false };
  }
  wrapper.tabIndex = 0;
  wrapper.setAttribute('role', 'region');
  wrapper.setAttribute('aria-label', opts.label || 'Results table');
  wrapper.dataset.testid = 'result-table-region';

  const cols = tableData.columns;
  const rows = normalizeTableRows(cols, tableData.rows);
  const numeric = cols.map((_, i) => i > 0 && isNumericColumn(rows, i));
  const formats = cols.map((col) => ({ decimals: opts.decimals, pValue: isPValueColumn(col) }));
  let hasHidden = false;

  const table = el('table', 'result-table');
  const thead = el('thead');
  const headRow = el('tr');
  cols.forEach((col, i) => {
    const th = el('th', numeric[i] ? 'num' : '', String(col));
    th.scope = 'col';
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = el('tbody');
  rows.forEach((row) => {
    const tr = el('tr');
    cols.forEach((_, i) => {
      const cell = classifyCell(row[i], formats[i]);
      const td = el(i === 0 ? 'th' : 'td', numeric[i] ? 'num' : '');
      if (i === 0) td.scope = 'row';
      if (cell.kind === 'hidden' || cell.kind === 'small') {
        hasHidden = true;
        const mark = el('span', `cell-hidden cell-hidden--${cell.kind}`);
        mark.title = 'Hidden to protect privacy';
        mark.innerHTML = `${escapeHtml(cell.text)}<sup aria-hidden="true">†</sup>`;
        if (cell.kind === 'hidden') mark.setAttribute('aria-label', 'hidden for privacy');
        td.appendChild(mark);
      } else {
        td.textContent = cell.text;
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.append(thead, tbody);
  wrapper.appendChild(table);
  const fade = el('div', 'result-table-fade');
  fade.setAttribute('aria-hidden', 'true');
  const hint = el(
    'p',
    'result-table-hint hidden',
    `Scroll sideways to see all ${cols.length} columns.`
  );
  hint.dataset.testid = 'result-table-hint';
  frame.append(fade, hint);
  trackTableOverflow(frame, wrapper, hint);
  return { wrapper: frame, hasHidden };
}

// ============================================================================
// Held plan (confirmation flow)
// ============================================================================

/**
 * A plan awaiting approval: the reason, the plan as a readable list, the exact plan text on
 * demand, and two plain-language actions.
 * @param {Object} held result from buildConfirmationHoldPresentation
 * @returns {HTMLElement}
 */
function buildHeldCard(held) {
  const card = el('div', 'result-card result-card-confirm');
  card.setAttribute('data-testid', 'result-confirm-card');

  const label = el('p', 'state-label state-label--held');
  label.innerHTML = `${icon('clock')}<span>Check this plan before it runs</span>`;
  card.appendChild(label);

  if (held.failureReason) card.appendChild(el('p', 'held-reason', held.failureReason));

  if (held.planLine) {
    const planEl = el('div', 'result-plan-line');
    planEl.setAttribute('data-testid', 'result-plan-line');
    const { analysis, items } = parsePlanLine(held.planLine);
    if (analysis) planEl.appendChild(el('p', 'plan-analysis', analysis));
    const list = el('dl', 'plan-list');
    items.forEach(({ label: term, value }) => {
      const row = el('div', 'plan-row');
      row.append(el('dt', '', term || 'Also'), el('dd', '', value));
      list.appendChild(row);
    });
    planEl.appendChild(list);
    const raw = el('details', 'plan-raw');
    raw.append(el('summary', '', 'Exact plan text'), el('p', 'plan-raw-text', held.planLine));
    planEl.appendChild(raw);
    card.appendChild(planEl);
  }

  const actions = el('div', 'result-confirm-actions');
  const confirmBtn = el('button', 'btn btn-primary btn-confirm', 'Run this analysis');
  confirmBtn.type = 'button';
  confirmBtn.dataset.testid = 'confirm-query-btn';
  confirmBtn.addEventListener('click', () => {
    if (state.isProcessing) {
      // Another answer is still streaming; leave the hold actionable instead of greying it out.
      setCardNote(card, 'Wait for the current answer to finish, then run this.');
      return;
    }
    // One decision per held plan: a second click must not run it again.
    setCardButtonsDisabled(card, true);
    void handleConfirmPendingQuery(held, card);
  });
  const rejectBtn = el('button', 'btn btn-secondary btn-reject', 'Not what I meant');
  rejectBtn.type = 'button';
  rejectBtn.dataset.testid = 'reject-query-btn';
  rejectBtn.addEventListener('click', () => {
    setCardButtonsDisabled(card, true);
    settleCard(card, 'Not run. Rephrase your question below.');
    restoreQuestion(held.pendingQuery || '');
    elements.queryInput.focus();
  });
  actions.append(confirmBtn, rejectBtn);
  card.appendChild(actions);
  card.appendChild(el('p', 'result-card-note hidden'));

  return card;
}

/**
 * @param {HTMLElement} card
 * @param {boolean} disabled
 */
function setCardButtonsDisabled(card, disabled) {
  card.querySelectorAll('.result-confirm-actions button').forEach((b) => {
    b.disabled = disabled;
  });
}

/**
 * Show a one-line note on a held card.
 * @param {HTMLElement} card
 * @param {string} text
 */
function setCardNote(card, text) {
  const note = card.querySelector('.result-card-note');
  if (!note) return;
  note.textContent = text;
  note.classList.remove('hidden');
  note.setAttribute('role', 'status');
}

/**
 * The plan was decided: replace the buttons with what happened.
 * @param {HTMLElement} card
 * @param {string} text
 */
function settleCard(card, text) {
  card.querySelector('.result-confirm-actions')?.remove();
  card.classList.add('is-settled');
  setCardNote(card, text);
}

/**
 * Run a held plan after the user approved it.
 * @param {Object} held
 * @param {HTMLElement} card
 */
async function handleConfirmPendingQuery(held, card) {
  const built = buildConfirmRequest(held, {
    currentDatasetId: state.currentDatasetId,
    fallbackSessionId: state.sessionId,
  });
  if (!built.ok) {
    setCardNote(card, built.message);
    return;
  }
  const { request } = built;
  beginProcessing();

  const assistantMessageId = addWorkingMessage();
  let outcome = 'error';
  try {
    const response = await submitConfirmation(window.clinicalAPI, request);
    outcome = await followQuery(response, assistantMessageId, request.queryText, {
      datasetId: request.datasetId,
      sessionId: request.sessionId,
    });
  } catch (error) {
    console.error('Confirmed plan failed:', error?.message);
    showFailure(assistantMessageId, 'error', error?.message);
  } finally {
    endProcessing();
  }

  if (outcome === 'completed' || outcome === 'held') {
    settleCard(card, 'You approved this plan. The answer is below.');
  } else {
    // Nothing ran, so the held plan stays confirmable.
    setCardButtonsDisabled(card, false);
  }
}

// ============================================================================
// Welcome / empty states
// ============================================================================

/**
 * @param {'no-dataset' | 'dataset' | 'resumed'} mode
 */
function renderWelcome(mode) {
  const { chatContainer } = elements;
  chatContainer.querySelector('.welcome-message')?.remove();
  const welcome = el('div', `welcome-message welcome-message--${mode}`);
  welcome.dataset.testid = 'welcome-message';

  if (
    mode === 'no-dataset' &&
    !state.connected &&
    elements.connectionStatus?.classList.contains('status-error')
  ) {
    welcome.append(
      el('h2', 'welcome-title', 'Waiting for the analysis engine.'),
      el(
        'p',
        'welcome-text',
        'It runs on this computer. When it can be reached, your cohorts appear in the sidebar.'
      )
    );
  } else if (mode === 'no-dataset') {
    welcome.append(
      el('h2', 'welcome-title', 'Choose a cohort to begin.'),
      el(
        'p',
        'welcome-text',
        'Pick one in the sidebar, or add a file. Answers are worked out on this computer.'
      )
    );
    const actions = el('div', 'welcome-actions');
    if (typeof window.clinicalAPI?.uploadDataset === 'function') {
      const up = el('button', 'btn btn-primary', 'Upload a file…');
      up.type = 'button';
      up.addEventListener('click', () =>
        elements.uploadArea?.querySelector('[data-testid="upload-file-input"]')?.click()
      );
      actions.appendChild(up);
    }
    if (typeof window.clinicalAPI?.loadSampleDataset === 'function') {
      const sample = el('button', 'btn btn-secondary', 'Use the sample cohort');
      sample.type = 'button';
      sample.addEventListener('click', () => void handleLoadSample());
      actions.appendChild(sample);
    }
    if (actions.childElementCount) welcome.appendChild(actions);
  } else if (mode === 'resumed') {
    welcome.append(
      el('h2', 'welcome-title', 'Continuing an earlier conversation'),
      el(
        'p',
        'welcome-text',
        "Earlier answers aren't shown here; new questions continue this conversation."
      )
    );
  } else {
    const name = datasetName(state.currentDatasetId);
    welcome.append(
      el('h2', 'welcome-title', `What would you like to know about ${name}?`),
      el('p', 'welcome-text', 'Each answer names the test, the number of patients and any caveats.')
    );
    const info = state.datasetInfo;
    const examples =
      info && info.datasetId === state.currentDatasetId
        ? buildExampleQuestions(info.columns, info.rows, info.schema)
        : [];
    if (examples.length) {
      const list = el('div', 'example-queries');
      list.setAttribute('role', 'group');
      list.setAttribute('aria-label', 'Example questions');
      examples.forEach((q) => {
        const chip = el('button', 'example-chip', q);
        chip.type = 'button';
        chip.addEventListener('click', () => {
          // Fill the box, never send: the clinician reads the question before asking it.
          elements.queryInput.value = q;
          handleQueryInputChange();
          elements.queryInput.focus();
        });
        list.appendChild(chip);
      });
      welcome.appendChild(list);
    }
  }
  chatContainer.appendChild(welcome);
}

/**
 * Clear the welcome message if present.
 */
function clearWelcomeMessage() {
  elements.chatContainer.querySelector('.welcome-message')?.remove();
}

function resetChatUI() {
  state.messages = [];
  state.isProcessing = false;
  if (state.activeStream) {
    state.activeStream.close();
    state.activeStream = null;
  }
  stopElapsedTicker();
  elements.chatContainer.replaceChildren();
  renderWelcome(state.currentDatasetId ? 'dataset' : 'no-dataset');
  setQueryInputEnabled(!!state.currentDatasetId);
}

// ============================================================================
// Cohort: list, overview, details panel, menu
// ============================================================================

/**
 * Render dataset preview table.
 * @param {string[]} columns
 * @param {Record<string, unknown>[]} rows
 */
function renderDatasetPreview(columns, rows) {
  const { datasetPreviewHead, datasetPreviewBody } = elements;
  if (!datasetPreviewHead || !datasetPreviewBody) return;

  datasetPreviewHead.innerHTML = '';
  datasetPreviewBody.innerHTML = '';

  const headerRow = document.createElement('tr');
  columns.forEach((column) => {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = column;
    headerRow.appendChild(th);
  });
  datasetPreviewHead.appendChild(headerRow);

  rows.forEach((row) => {
    const tr = document.createElement('tr');
    columns.forEach((column) => {
      const td = document.createElement('td');
      const value = row?.[column];
      td.textContent =
        value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
      if (typeof value === 'number') td.className = 'num';
      tr.appendChild(td);
    });
    datasetPreviewBody.appendChild(tr);
  });
}

/**
 * Load dataset summary + preview (P02 parity) into the details panel and the sidebar meta line.
 */
async function loadDatasetOverview() {
  const { datasetOverview, datasetOverviewSummary, datasetMeta } = elements;
  if (!datasetOverview || !datasetOverviewSummary) return;

  if (!state.currentDatasetId) {
    datasetOverview.classList.add('hidden');
    datasetOverviewSummary.textContent = '';
    datasetMeta?.classList.add('hidden');
    renderDatasetPreview([], []);
    return;
  }

  if (
    typeof window.clinicalAPI?.getDataset !== 'function' ||
    typeof window.clinicalAPI?.previewDataset !== 'function'
  ) {
    datasetOverview.classList.add('hidden');
    return;
  }

  const datasetId = state.currentDatasetId;
  datasetOverview.classList.remove('hidden');
  datasetOverviewSummary.textContent = 'Loading…';

  try {
    const [detail, preview] = await Promise.all([
      window.clinicalAPI.getDataset(datasetId),
      window.clinicalAPI.previewDataset(datasetId, 10),
    ]);
    if (datasetId !== state.currentDatasetId) return;

    const table =
      Array.isArray(detail?.tables) && detail.tables.length > 0 ? detail.tables[0] : null;
    const rowCount = table?.row_count ?? preview?.total_rows ?? 0;
    const columns = Array.isArray(preview?.columns) ? preview.columns : [];
    const rows = Array.isArray(preview?.rows) ? preview.rows : [];

    datasetOverviewSummary.textContent = `${rowCount} rows • ${columns.length} columns`;
    renderDatasetPreview(columns, rows);
    if (datasetMeta) {
      datasetMeta.textContent = `${rowCount} rows · ${columns.length} variables`;
      datasetMeta.classList.remove('hidden');
    }
    state.datasetInfo = {
      datasetId,
      rowCount,
      columns,
      rows,
      schema:
        detail?.column_schema && typeof detail.column_schema === 'object'
          ? detail.column_schema
          : {},
    };
    if (
      state.messages.length === 0 &&
      elements.chatContainer.querySelector('.welcome-message--dataset')
    ) {
      renderWelcome('dataset');
    }
  } catch (error) {
    datasetOverviewSummary.textContent = `Couldn't load this cohort's first rows: ${error.message}`;
    renderDatasetPreview([], []);
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
    const datasets = Array.isArray(response?.datasets) ? response.datasets : [];
    state.datasets = datasets;

    // Clear existing options (except placeholder)
    datasetSelect.innerHTML = '<option value="">Choose a cohort…</option>';

    datasets.forEach((dataset) => {
      const option = document.createElement('option');
      option.value = dataset.dataset_id;
      option.textContent = dataset.name;
      datasetSelect.appendChild(option);
    });
    // Keep the current choice across refreshes.
    if (state.currentDatasetId && datasets.some((d) => d.dataset_id === state.currentDatasetId)) {
      datasetSelect.value = state.currentDatasetId;
    }

    console.log(`Loaded ${datasets.length} datasets`);
  } catch (error) {
    console.error('Failed to load datasets:', error?.message);
    setDatasetStatus("Couldn't load the list of cohorts.", 'error');
  }
}

/**
 * @param {boolean} open
 * @param {{ focus?: boolean }} [opts]
 */
function setDatasetMenuOpen(open, opts = {}) {
  const { datasetMenu, datasetMenuBtn } = elements;
  if (!datasetMenu || !datasetMenuBtn) return;
  datasetMenu.classList.toggle('hidden', !open);
  datasetMenuBtn.setAttribute('aria-expanded', String(open));
  if (open) {
    datasetMenu.querySelector('button:not([disabled])')?.focus();
  } else if (opts.focus) {
    datasetMenuBtn.focus();
  }
}

let datasetPanelReturnFocus = null;

/**
 * Open the cohort details sheet (preview, column suggestions, change history).
 * @param {HTMLElement | null} [returnFocusTo]
 */
function openDatasetPanel(returnFocusTo) {
  const { datasetPanel, appContainer, closeDatasetPanelBtn, datasetPanelTitle } = elements;
  if (!datasetPanel || !state.currentDatasetId) return;
  datasetPanelTitle.textContent = datasetName(state.currentDatasetId);
  datasetPanel.classList.remove('hidden');
  appContainer?.classList.add('has-panel');
  datasetPanelReturnFocus = returnFocusTo || elements.datasetMenuBtn;
  closeDatasetPanelBtn?.focus();
}

function closeDatasetPanel() {
  const { datasetPanel, appContainer } = elements;
  if (!datasetPanel || datasetPanel.classList.contains('hidden')) return;
  datasetPanel.classList.add('hidden');
  appContainer?.classList.remove('has-panel');
  datasetPanelReturnFocus?.focus();
  datasetPanelReturnFocus = null;
}

// ============================================================================
// Metadata enrichments + patch history (cohort details panel)
// ============================================================================

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
  enrichmentHint.textContent = 'Loading…';
  if (patchHistoryHint) patchHistoryHint.textContent = 'Loading…';
  if (patchHistoryBody) patchHistoryBody.replaceChildren();

  const hasPending = typeof window.clinicalAPI?.getPendingEnrichments === 'function';
  const hasHistory = typeof window.clinicalAPI?.getEnrichmentHistory === 'function';

  if (!hasPending && !hasHistory) {
    enrichmentHint.textContent = 'Column suggestions are not available.';
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
          ? 'No changes yet.'
          : `${histTotal} patch${histTotal === 1 ? '' : 'es'} in log (newest first).`;
    }
    if (patchHistoryBody) {
      renderPatchHistoryTable(patchHistoryBody, patches, {
        onRevert: (patchId) => handlePatchRevert(patchId),
      });
    }
  } catch (error) {
    console.error('Failed to load enrichments / history:', error?.message);
    enrichmentHint.textContent = `Could not load enrichments: ${error.message}`;
    renderEnrichmentCards(enrichmentList, [], {
      onAccept: noopEnrichment,
      onReject: noopEnrichment,
    });
    if (patchHistoryHint) {
      patchHistoryHint.textContent = `Couldn't load the change history: ${error.message}`;
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
  await runEnrichmentDecision(
    () => window.clinicalAPI.acceptEnrichment(state.currentDatasetId, patchId),
    {
      reload: loadEnrichmentPanel,
      showMessage: (text) => {
        if (enrichmentHint) enrichmentHint.textContent = text;
      },
      fallback: 'Accept failed',
    }
  );
}

/**
 * @param {string} patchId
 */
async function handleEnrichmentReject(patchId) {
  const { enrichmentHint } = elements;
  if (!state.currentDatasetId) return;
  await runEnrichmentDecision(
    () => window.clinicalAPI.rejectEnrichment(state.currentDatasetId, patchId),
    {
      reload: loadEnrichmentPanel,
      showMessage: (text) => {
        if (enrichmentHint) enrichmentHint.textContent = text;
      },
      fallback: 'Reject failed',
    }
  );
}

/**
 * @param {string} patchId
 */
async function handlePatchRevert(patchId) {
  const { patchHistoryHint } = elements;
  if (!state.currentDatasetId) return;
  if (typeof window.clinicalAPI?.revertEnrichmentPatch !== 'function') {
    if (patchHistoryHint) patchHistoryHint.textContent = 'Undo is not available.';
    return;
  }
  await runEnrichmentDecision(
    () => window.clinicalAPI.revertEnrichmentPatch(state.currentDatasetId, patchId),
    {
      reload: loadEnrichmentPanel,
      showMessage: (text) => {
        if (patchHistoryHint) patchHistoryHint.textContent = text;
      },
      fallback: 'Revert failed',
    }
  );
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
    renderSessionList(
      sessionList,
      sessions,
      state.sessionId,
      {
        onSelect: (id) => handleSessionSelect(id, sessions),
        onDelete: (id) => handleSessionDelete(id),
      },
      {
        titles: state.sessionTitles,
        datasetNames: new Map(state.datasets.map((d) => [d.dataset_id, d.name])),
      }
    );
  } catch (error) {
    console.error('Failed to load sessions:', error?.message);
  }
}

/**
 * Make the current conversation a real API session before its first question, so it appears
 * in the sidebar titled by that question.
 */
async function ensureServerSession() {
  if (state.sessionOnServer || typeof window.clinicalAPI?.createSession !== 'function') return;
  try {
    const res = await window.clinicalAPI.createSession(state.currentDatasetId);
    if (res?.session_id) {
      state.sessionId = res.session_id;
      state.sessionOnServer = true;
    }
  } catch (error) {
    console.warn('Could not create a session; continuing without one:', error?.message);
  }
}

/**
 * Point the app at a dataset.
 * @param {string} datasetId
 * @param {{ resetChat: boolean }} opts
 */
function applyDataset(datasetId, { resetChat }) {
  const { datasetSelect, deleteDatasetBtn, openDatasetPanelBtn, newChatBtn, datasetDeleteArea } =
    elements;
  const changed = datasetId !== state.currentDatasetId;
  state.currentDatasetId = datasetId || null;
  if (datasetSelect.value !== (datasetId || '')) datasetSelect.value = datasetId || '';
  if (deleteDatasetBtn) deleteDatasetBtn.disabled = !datasetId;
  if (openDatasetPanelBtn) openDatasetPanelBtn.disabled = !datasetId;
  if (newChatBtn) newChatBtn.disabled = !datasetId;
  if (datasetDeleteArea) closeDatasetDeleteConfirm(datasetDeleteArea);
  if (changed) state.datasetInfo = null;
  if (!datasetId) closeDatasetPanel();
  else if (elements.datasetPanelTitle)
    elements.datasetPanelTitle.textContent = datasetName(datasetId);

  if (resetChat && changed) {
    // A conversation belongs to one cohort: a new cohort starts a new conversation.
    state.sessionId = generateSessionId();
    state.sessionOnServer = false;
    resetChatUI();
  } else {
    setQueryInputEnabled(!!datasetId);
  }

  void loadDatasetOverview();
  void loadEnrichmentPanel();
  if (datasetId) void loadSessionList();
}

/**
 * @param {string} sessionId
 * @param {object[]} sessions
 */
function handleSessionSelect(sessionId, sessions) {
  if (sessionId === state.sessionId) return;
  const session = sessions.find((s) => s.session_id === sessionId);

  state.sessionId = sessionId;
  state.sessionOnServer = true;
  state.messages = [];
  state.isProcessing = false;
  if (state.activeStream) {
    state.activeStream.close();
    state.activeStream = null;
  }
  stopElapsedTicker();

  elements.chatContainer.replaceChildren();
  renderWelcome('resumed');

  if (session?.dataset_id && session.dataset_id !== state.currentDatasetId) {
    applyDataset(session.dataset_id, { resetChat: false });
  } else {
    setQueryInputEnabled(!!state.currentDatasetId);
    void loadSessionList();
  }
}

/**
 * Delete a session (the row already asked for confirmation).
 * @param {string} sessionId
 */
async function handleSessionDelete(sessionId) {
  if (typeof window.clinicalAPI?.deleteSession !== 'function') return;
  try {
    await window.clinicalAPI.deleteSession(sessionId);
    state.sessionTitles.delete(sessionId);
    announce('Conversation deleted.');
    if (sessionId === state.sessionId) {
      await handleNewChat();
      elements.queryInput?.focus();
    } else {
      const rows = [...(elements.sessionList?.querySelectorAll('.session-item') || [])];
      const index = rows.findIndex((row) => row.dataset.sessionId === sessionId);
      await loadSessionList();
      focusSessionRowAt(index);
    }
  } catch (error) {
    console.error('Delete session failed:', error?.message);
    announce("Couldn't delete the conversation.");
    await loadSessionList();
  }
}

/**
 * After a delete, put focus on the row now at `index` (the next one), else the last row, else
 * New conversation, so it never falls to the page.
 * @param {number} index
 */
function focusSessionRowAt(index) {
  const opens = [...(elements.sessionList?.querySelectorAll('[data-testid="session-open"]') || [])];
  const target = opens[Math.max(0, Math.min(index, opens.length - 1))] || elements.newChatBtn;
  target?.focus();
}

async function handleNewChat() {
  if (!state.currentDatasetId) return;
  if (typeof window.clinicalAPI?.createSession !== 'function') {
    state.sessionId = generateSessionId();
    state.sessionOnServer = false;
    resetChatUI();
    return;
  }
  try {
    const res = await window.clinicalAPI.createSession(state.currentDatasetId);
    state.sessionId = res.session_id;
    state.sessionOnServer = true;
    resetChatUI();
    await loadSessionList();
  } catch (error) {
    console.error('Create session failed:', error?.message);
    state.sessionId = generateSessionId();
    state.sessionOnServer = false;
    resetChatUI();
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
  if (datasetId) console.log('Selected dataset:', datasetId);
  applyDataset(datasetId, { resetChat: true });
}

/**
 * Ask for confirmation in the page, then delete the selected dataset.
 */
function handleDeleteDatasetClick() {
  const { datasetSelect, datasetDeleteArea, datasetMenuBtn } = elements;
  const datasetId = state.currentDatasetId;
  if (!datasetId || !datasetDeleteArea) return;
  const name = datasetName(datasetId);
  setDatasetMenuOpen(false);

  const close = () => {
    closeDatasetDeleteConfirm(datasetDeleteArea);
    datasetMenuBtn?.focus();
  };
  const panel = renderDatasetDeleteConfirm(datasetDeleteArea, {
    datasetName: name,
    onCancel: close,
    onConfirm: async () => {
      panel.setBusy(true);
      try {
        await window.clinicalAPI.deleteDataset(datasetId);
      } catch (error) {
        console.error('Delete dataset failed:', error?.message);
        panel.setBusy(false);
        panel.setError(`Could not delete the dataset: ${error?.message || 'unknown error'}`);
        return;
      }
      closeDatasetDeleteConfirm(datasetDeleteArea);
      if (state.activeStream) {
        state.activeStream.close();
        state.activeStream = null;
      }
      datasetSelect.value = '';
      // The chat on screen belonged to the deleted dataset: start a fresh conversation.
      applyDataset('', { resetChat: true });
      await loadDatasets();
      await loadSessionList();
      announce(`Deleted ${name} and its sessions from this computer.`);
      datasetSelect.focus();
    },
  });
}

/**
 * Handle refresh datasets button click.
 */
async function handleRefreshDatasets() {
  const btn = elements.refreshDatasetsBtn;
  btn.disabled = true;
  setDatasetMenuOpen(false, { focus: true });
  setDatasetStatus('Refreshing…');
  await loadDatasets();
  if (elements.datasetStatus?.dataset.tone !== 'error') setDatasetStatus('');
  btn.disabled = false;
}

/**
 * Handle query form submission.
 * @param {Event} event
 */
async function handleQuerySubmit(event) {
  event.preventDefault();

  const { queryInput, inputHintText } = elements;
  const queryText = queryInput.value.trim();

  if (state.isProcessing) {
    if (queryText) inputHintText.textContent = 'Wait for this answer, or press Stop.';
    return;
  }
  if (!queryText || !state.currentDatasetId) {
    return;
  }

  // Clear input and update state
  queryInput.value = '';
  handleQueryInputChange(); // Reset textarea height
  beginProcessing();

  clearWelcomeMessage();

  addMessage({
    id: generateMessageId(),
    role: 'user',
    content: queryText,
    timestamp: new Date(),
  });
  const assistantMessageId = addWorkingMessage();

  await ensureServerSession();
  if (!state.sessionTitles.has(state.sessionId)) {
    state.sessionTitles.set(state.sessionId, titleFromQuestion(queryText));
  }

  // Never log the question text or results: they can identify patients (docs/security).
  console.log('Query submitted:', {
    sessionId: state.sessionId,
    datasetId: state.currentDatasetId,
    queryLength: queryText.length,
  });

  let outcome = 'error';
  try {
    const response = await window.clinicalAPI.submitQuery(
      state.sessionId,
      state.currentDatasetId,
      queryText,
      false
    );
    console.log('Query accepted:', response?.query_id);
    outcome = await followQuery(response, assistantMessageId, queryText, {
      datasetId: state.currentDatasetId,
      sessionId: state.sessionId,
    });
  } catch (error) {
    console.error('Query failed:', error?.message);
    showFailure(assistantMessageId, 'error', error?.message);
  } finally {
    endProcessing();
  }

  if (outcome === 'refused' || outcome === 'error') {
    // Give the question back so it can be edited rather than retyped.
    restoreQuestion(queryText);
  }
  void loadSessionList();
}

/**
 * Placeholder assistant message shown while the answer is worked out.
 * @returns {string} message id
 */
function addWorkingMessage() {
  const id = generateMessageId();
  addMessage({
    id,
    role: 'assistant',
    content: 'Reading your question',
    isStreaming: true,
    startedAt: Date.now(),
    timestamp: new Date(),
  });
  return id;
}

/**
 * Render a refusal or error in place of the working indicator.
 * @param {string} messageId
 * @param {'refused' | 'error'} kind
 * @param {string} [message]
 */
function showFailure(messageId, kind, message, detail) {
  updateMessage(messageId, {
    content: message || 'The analysis stopped before it could answer.',
    isStreaming: false,
    kind,
    ...(detail ? { detail } : {}),
  });
  announce(kind === 'refused' ? 'Not answered.' : 'Something went wrong.');
}

/**
 * Follow a submitted query to its outcome (stream, or the direct result).
 * @param {{ query_id?: string, stream_url?: string, summary?: string, result?: object }} response
 * @param {string} assistantMessageId
 * @param {string} queryText
 * @param {{ datasetId?: string | null, sessionId?: string | null }} binding
 * @returns {Promise<string>} outcome
 */
async function followQuery(response, assistantMessageId, queryText, binding) {
  if (state.stopRequested) {
    markStopped(assistantMessageId);
    return 'stopped';
  }
  if (response?.stream_url || response?.query_id) {
    return subscribeToQueryStream(response.query_id, assistantMessageId, queryText, binding);
  }
  // Fallback: use direct result if available
  updateMessage(assistantMessageId, {
    content: response?.summary || 'Analysis complete.',
    result: response?.result || null,
    isStreaming: false,
    kind: 'answered',
  });
  return 'completed';
}

/**
 * @param {string} messageId
 */
function markStopped(messageId) {
  updateMessage(messageId, {
    content: 'Stopped waiting for this answer.',
    isStreaming: false,
    kind: 'stopped',
  });
  announce('Stopped.');
}

/**
 * Subscribe to SSE stream for query results.
 * @param {string} queryId
 * @param {string} assistantMessageId
 * @param {string} [pendingQuery]
 * @param {{ datasetId?: string | null, sessionId?: string | null }} [binding] dataset/session the query ran on
 * @returns {Promise<'completed' | 'held' | 'refused' | 'error' | 'stopped' | 'ended'>}
 */
async function subscribeToQueryStream(
  queryId,
  assistantMessageId,
  pendingQuery = '',
  binding = {}
) {
  return new Promise((resolve, reject) => {
    let resolved = false;
    let cleanupFn = null;

    const finish = (outcome) => {
      if (cleanupFn) cleanupFn();
      resolved = true;
      resolve(outcome);
    };

    const handleEvent = ({ event: eventType, data }) => {
      if (resolved) return;
      console.log('SSE event:', eventType);

      switch (eventType) {
        case 'query_started':
          // Keep streaming indicator
          break;

        case 'query_progress': {
          const label = stageLabel(data);
          if (label) {
            updateMessage(assistantMessageId, { content: label, isStreaming: true });
          }
          break;
        }

        case 'query_completed': {
          const pres = buildQueryCompletedPresentation(
            data.intent_type,
            data.result_preview || {},
            data.claim
          );

          updateMessage(assistantMessageId, {
            content: pres.content,
            result: pres.result,
            isStreaming: false,
            kind: 'answered',
          });
          announce('Answer ready.');
          finish('completed');
          break;
        }

        case 'query_awaiting_confirmation': {
          const hold = buildConfirmationHoldPresentation(data, pendingQuery, binding);
          updateMessage(assistantMessageId, {
            content: hold.content,
            result: hold.result,
            isStreaming: false,
            kind: 'held',
          });
          announce('The plan needs your confirmation before it runs.');
          finish('held');
          break;
        }

        case 'query_failed': {
          const failure = describeQueryFailure(data);
          showFailure(assistantMessageId, failure.kind, failure.message, failure.detail);
          finish(failure.kind);
          break;
        }

        case 'stream_end': {
          console.log('Stream ended');
          const msg = state.messages.find((m) => m.id === assistantMessageId);
          if (msg?.isStreaming) {
            showFailure(
              assistantMessageId,
              'error',
              'The analysis ended without an answer. Ask again.'
            );
            finish('error');
          } else {
            finish('ended');
          }
          break;
        }

        default:
          console.log('Unhandled SSE event:', eventType);
      }
    };

    const handleError = (error) => {
      if (resolved) {
        console.log('SSE connection closed after completion');
        return;
      }

      console.error('SSE stream error:', error?.message);
      const msg = state.messages.find((m) => m.id === assistantMessageId);
      if (msg && msg.isStreaming) {
        showFailure(assistantMessageId, 'error', LOST_CONNECTION_MESSAGE);
      }
      finish('error');
    };

    const stop = () => {
      if (resolved) return;
      markStopped(assistantMessageId);
      finish('stopped');
    };

    try {
      // Use callback-based API that works with contextBridge
      cleanupFn = window.clinicalAPI.subscribeToQueryStream(queryId, {
        onEvent: handleEvent,
        onError: handleError,
        onClose: () => {
          console.log('SSE stream closed');
        },
      });
      state.activeStream = { close: () => cleanupFn?.(), stop };
    } catch (error) {
      console.error('Failed to create SSE stream:', error?.message);
      reject(error);
    }
  });
}

// ============================================================================
// Dataset Upload (Phase 8)
// ============================================================================

const uploadHandlers = {
  onFileSelect: (file) => {
    void runFromDatasetMenu(() => handleFileUpload(file));
  },
  onLoadSample: () => {
    void runFromDatasetMenu(handleLoadSample);
  },
};

function initUploadArea() {
  const { uploadArea } = elements;
  if (!uploadArea) return;
  if (typeof window.clinicalAPI?.uploadDataset !== 'function') return;
  renderUploadArea(uploadArea, {
    onFileSelect: uploadHandlers.onFileSelect,
    ...(typeof window.clinicalAPI?.loadSampleDataset === 'function'
      ? { onLoadSample: uploadHandlers.onLoadSample }
      : {}),
  });
}

/**
 * Select a just-added dataset.
 * @param {string} datasetId
 */
async function selectNewDataset(datasetId) {
  await loadDatasets();
  if (!datasetId) return;
  elements.datasetSelect.value = datasetId;
  elements.datasetSelect.dispatchEvent(new Event('change'));
}

/**
 * Close the ⋯ menu for a cohort action, keeping focus on the menu button while it runs; once
 * a cohort is ready, move to the composer (unless the user has moved focus elsewhere).
 * @param {() => Promise<void>} action
 */
async function runFromDatasetMenu(action) {
  const fromMenu = !elements.datasetMenu?.classList.contains('hidden');
  if (fromMenu) setDatasetMenuOpen(false, { focus: true });
  const before = state.currentDatasetId;
  await action();
  const active = document.activeElement;
  const focusIsFree = !active || active === document.body || active === elements.datasetMenuBtn;
  if (!focusIsFree) return;
  if (
    state.currentDatasetId &&
    state.currentDatasetId !== before &&
    !elements.queryInput?.disabled
  ) {
    elements.queryInput.focus();
  } else if (fromMenu) {
    elements.datasetMenuBtn?.focus();
  }
}

async function handleLoadSample() {
  if (typeof window.clinicalAPI?.loadSampleDataset !== 'function') return;
  setDatasetStatus('Loading the sample cohort…');

  try {
    const raw = await window.clinicalAPI.loadSampleDataset();
    const result = normalizeUploadResponse(raw);
    if (result.status === 'ready' && result.upload_id) {
      setDatasetStatus(`${result.dataset_name || 'The sample cohort'} is ready.`, 'ok');
      await selectNewDataset(result.upload_id);
    } else {
      setDatasetStatus(`Couldn't load the sample cohort: ${result.message}`, 'error');
    }
  } catch (error) {
    console.error('Sample load failed:', error?.message);
    setDatasetStatus(`Couldn't load the sample cohort: ${error.message}`, 'error');
  }
}

async function handleFileUpload(file) {
  if (typeof window.clinicalAPI?.uploadDataset !== 'function') return;
  setDatasetStatus(`Uploading ${file.name}…`);

  try {
    const raw = await window.clinicalAPI.uploadDataset(file);
    const result = normalizeUploadResponse(raw);

    if (result.status === 'ready') {
      setDatasetStatus(`${result.dataset_name || file.name} is ready.`, 'ok');
      await selectNewDataset(result.upload_id);
    } else {
      setDatasetStatus(`Couldn't upload ${file.name}: ${result.message}`, 'error');
    }
  } catch (error) {
    console.error('Upload failed:', error?.message);
    setDatasetStatus(`Couldn't upload ${file.name}: ${error.message}`, 'error');
  }
}

// ============================================================================
// Appearance (System / Light / Dark)
// ============================================================================

function readStoredTheme() {
  try {
    const value = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isThemeSource(value) ? value : 'system';
  } catch (error) {
    console.warn('Theme storage unavailable; following the system theme:', error?.name);
    return 'system';
  }
}

/**
 * Apply a theme choice to the page (and persist it when the user made it).
 * @param {'system' | 'light' | 'dark'} source
 * @param {{ persist: boolean }} opts
 */
function applyTheme(source, { persist }) {
  const root = document.documentElement;
  if (persist && !prefersReducedMotion()) {
    root.classList.add('theme-changing');
    window.setTimeout(() => root.classList.remove('theme-changing'), 260);
  }
  if (source === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', source);
  elements.themeRadios.forEach((radio) => {
    radio.checked = radio.value === source;
  });
  if (!persist) return;
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, source);
  } catch (error) {
    console.warn('Theme storage unavailable; the choice lasts until the app closes:', error?.name);
  }
  if (typeof window.appTheme?.set === 'function') {
    window.appTheme.set(source).catch((error) => {
      console.warn('Could not save the theme setting:', error?.message);
    });
  }
}

async function initThemeSwitch() {
  let source = readStoredTheme();
  if (typeof window.appTheme?.get === 'function') {
    try {
      const current = await window.appTheme.get();
      if (isThemeSource(current?.source)) source = current.source;
    } catch (error) {
      console.warn('Could not read the theme setting:', error?.message);
    }
  }
  applyTheme(source, { persist: false });
  elements.themeRadios.forEach((radio) => {
    radio.addEventListener('change', () => {
      if (radio.checked && isThemeSource(radio.value)) applyTheme(radio.value, { persist: true });
    });
  });
}

// ============================================================================
// Initialization
// ============================================================================

function paintIcons() {
  document.querySelectorAll('[data-icon]').forEach((node) => {
    node.innerHTML = icon(node.dataset.icon);
  });
  if (elements.datasetMenuBtn) elements.datasetMenuBtn.innerHTML = icon('more');
  if (elements.closeDatasetPanelBtn) elements.closeDatasetPanelBtn.innerHTML = icon('close');
  const plus = elements.newChatBtn?.querySelector('.btn-new-chat-icon');
  if (plus) plus.innerHTML = icon('plus');
  elements.submitBtn.innerHTML = icon('send');
}

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
      updateConnectionStatus('connected');
      updateLlmBanner(health);
      console.log('Backend health check passed:', health.status);
    } else {
      updateLlmBanner({});
      updateConnectionStatus('error');
      console.warn('Backend health check returned:', health.status);
    }
  } catch (error) {
    updateLlmBanner({});
    updateConnectionStatus('error');
    console.error('Health check failed:', error?.message);
  }
}

async function connect() {
  await checkHealth();
  if (state.connected) {
    await loadDatasets();
    await loadSessionList();
  }
  setQueryInputEnabled(!!state.currentDatasetId);
  // The empty state depends on the connection (and its upload buttons on the API).
  if (!state.currentDatasetId && state.messages.length === 0) renderWelcome('no-dataset');
}

function bindEvents() {
  const {
    datasetSelect,
    datasetMenuBtn,
    datasetMenu,
    datasetMeta,
    openDatasetPanelBtn,
    closeDatasetPanelBtn,
    datasetPanel,
    refreshDatasetsBtn,
    deleteDatasetBtn,
    refreshEnrichmentsBtn,
    queryForm,
    queryInput,
    submitBtn,
    newChatBtn,
    retryConnectionBtn,
  } = elements;

  datasetSelect.addEventListener('change', handleDatasetChange);
  refreshDatasetsBtn.addEventListener('click', handleRefreshDatasets);
  deleteDatasetBtn?.addEventListener('click', handleDeleteDatasetClick);
  refreshEnrichmentsBtn?.addEventListener('click', () => void loadEnrichmentPanel());

  datasetMenuBtn?.addEventListener('click', () => {
    setDatasetMenuOpen(datasetMenu.classList.contains('hidden'));
  });
  datasetMenu?.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      event.stopPropagation();
      setDatasetMenuOpen(false, { focus: true });
    }
  });
  document.addEventListener('click', (event) => {
    if (
      datasetMenu &&
      !datasetMenu.classList.contains('hidden') &&
      !datasetMenu.contains(event.target) &&
      !datasetMenuBtn.contains(event.target)
    ) {
      setDatasetMenuOpen(false);
    }
  });

  openDatasetPanelBtn?.addEventListener('click', () => {
    setDatasetMenuOpen(false);
    openDatasetPanel(datasetMenuBtn);
  });
  datasetMeta?.addEventListener('click', () => openDatasetPanel(datasetMeta));
  closeDatasetPanelBtn?.addEventListener('click', closeDatasetPanel);
  datasetPanel?.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeDatasetPanel();
  });

  queryForm.addEventListener('submit', handleQuerySubmit);
  submitBtn.addEventListener('click', (event) => {
    if (state.isProcessing) {
      event.preventDefault();
      stopActiveQuery();
    }
  });
  queryInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (state.isProcessing) {
        if (queryInput.value.trim()) {
          elements.inputHintText.textContent = 'Wait for this answer, or press Stop.';
        }
        return;
      }
      queryForm.requestSubmit();
    }
  });
  queryInput.addEventListener('input', handleQueryInputChange);

  newChatBtn?.addEventListener('click', () => void handleNewChat());
  retryConnectionBtn?.addEventListener('click', () => void connect());
}

/**
 * Initialize the application.
 */
async function init() {
  console.log('Clinical Analytics renderer initializing');

  paintIcons();
  bindEvents();
  void initThemeSwitch();
  renderWelcome('no-dataset');
  initUploadArea();

  await connect();

  console.log('Initialization complete');
}

// Start the app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
