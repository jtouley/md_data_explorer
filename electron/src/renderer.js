// SPDX-License-Identifier: MIT
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
  createEnrichmentPanelGate,
  runEnrichmentDecision,
} from './enrichmentPanel.js';
import { normalizeHistoryResponse, renderPatchHistoryTable } from './patchHistoryPanel.js';
import {
  buildConfirmationHoldPresentation,
  buildExampleQuestions,
  buildQueryCompletedPresentation,
  describeQueryFailure,
  plainFailureMessage,
  stageLabel,
} from './resultPresentation.js';
import {
  answerAnnouncement,
  buildHeldCard,
  buildNotice,
  messageShell,
  renderAnswerCard,
  setCardButtonsDisabled,
  setCardNote,
  settleCard,
} from './answerCard.js';
import { approvedCardNote, buildConfirmRequest, submitConfirmation } from './confirmHold.js';
import { STOPPED, createStopController } from './stopControl.js';
import { bindBannerDismiss, bindSidebarToggle } from './layoutControls.js';
import { closeDatasetDeleteConfirm, renderDatasetDeleteConfirm } from './datasetDeletePanel.js';
import {
  ensureServerSession,
  normalizeSessionListResponse,
  renderSessionList,
  titleFromQuestion,
} from './sessionSidebar.js';
import {
  normalizeUploadResponse,
  readyHeadline,
  renderUploadArea,
  uploadStatusText,
} from './uploadPanel.js';
import {
  dataCheckNeedsReview,
  dataCheckStatusNote,
  normalizeQuality,
  renderDataCheck,
} from './dataCheckPanel.js';
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
  datasetStatusAction: document.getElementById('dataset-status-action'),
  sidebarToggle: document.getElementById('sidebar-toggle'),
  openDatasetPanelBtn: document.getElementById('open-dataset-panel'),
  refreshDatasetsBtn: document.getElementById('refresh-datasets'),
  deleteDatasetBtn: document.getElementById('delete-dataset'),
  datasetDeleteArea: document.getElementById('dataset-delete-area'),
  datasetPanel: document.getElementById('dataset-panel'),
  datasetPanelTitle: document.getElementById('dataset-panel-title'),
  closeDatasetPanelBtn: document.getElementById('close-dataset-panel'),
  datasetOverview: document.getElementById('dataset-overview'),
  datasetOverviewSummary: document.getElementById('dataset-overview-summary'),
  dataCheckSection: document.getElementById('data-check-section'),
  dataCheckHint: document.getElementById('data-check-hint'),
  dataCheckList: document.getElementById('data-check-list'),
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
 * @property {string} [questionId] - The question this assistant message answers
 * @property {Date} timestamp - Message creation time
 */

const state = {
  connected: false,
  currentDatasetId: null,
  sessionId: generateSessionId(),
  /** True once the session exists in the API (created or picked from the list). */
  sessionOnServer: false,
  /** The session creation still in flight, so asking again never creates a second session. */
  /** @type {Promise<void> | null} */
  sessionCreating: null,
  /** @type {ChatMessage[]} */
  messages: [],
  /** @type {{ close: () => void, stop?: () => void } | null} */
  activeStream: null,
  /** Stop for the question in flight, from the moment it is asked (before its request returns). */
  /** @type {ReturnType<typeof createStopController> | null} */
  stopControl: null,
  /** The local language model banner was dismissed; it stays away until its message changes. */
  llmBannerDismissedText: '',
  isProcessing: false,
  /** @type {{ dataset_id: string, name: string, row_count?: number }[]} */
  datasets: [],
  /** @type {{ datasetId: string, rowCount: number, columns: string[], rows: object[], schema: Record<string, string> } | null} */
  datasetInfo: null,
  /** First question per session, this run only (never persisted: questions can identify patients). */
  sessionTitles: new Map(),
};

/** The narrow-window sidebar toggle (see layoutControls.js), once bound. */
/** @type {ReturnType<typeof bindSidebarToggle> | null} */
let sidebar = null;

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
function setDatasetStatus(text, tone = 'info', { showDetails = false } = {}) {
  const { datasetStatus, datasetStatusAction } = elements;
  if (!datasetStatus) return;
  datasetStatus.textContent = text;
  datasetStatus.dataset.tone = tone;
  datasetStatusAction?.classList.toggle('hidden', !showDetails);
}

/**
 * Put text on the clipboard (the "Copy details" buttons). Never shown on screen.
 * @param {string} text
 * @returns {Promise<void>} rejects when nothing was copied, so the button can say so
 */
function copyText(text) {
  if (!navigator.clipboard?.writeText) return Promise.reject(new Error('No clipboard'));
  return navigator.clipboard.writeText(text).catch((error) => {
    console.warn('Could not copy the details:', error?.name);
    throw error;
  });
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

  const text = !reachable
    ? 'Some questions need the local language model, which isn’t running. ' +
      'Simple questions still work.'
    : `The local language model “${model}” isn’t installed, so some questions won’t work. ` +
      'Simple questions still work.';
  // Dismissed: stays away until it has something different to say.
  if (state.llmBannerDismissedText === text) return;
  llmBanner.classList.add(reachable ? 'llm-banner--warn' : 'llm-banner--error');
  llmBannerText.textContent = text;
  llmBanner.classList.remove('hidden');
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
    inputHintText.textContent = 'Wait for this answer, or press Stop.';
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
      : 'Can’t reach the analysis engine on this computer.';

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

/**
 * @param {string} [workingMessageId] the placeholder that Stop turns into "Stopped" at once
 * @returns {ReturnType<typeof createStopController>}
 */
function beginProcessing(workingMessageId) {
  state.isProcessing = true;
  const stopControl = createStopController();
  state.stopControl = stopControl;
  stopControl.onStop(() => {
    // Acknowledge at once, then stop whatever is still in flight.
    if (state.activeStream?.stop) state.activeStream.stop();
    else if (workingMessageId) markStopped(workingMessageId);
    window.clinicalAPI?.cancelSubmit?.();
    if (state.stopControl === stopControl) endProcessing();
  });
  setQueryInputEnabled(true);
  startElapsedTicker();
  return stopControl;
}

function endProcessing() {
  state.isProcessing = false;
  state.stopControl = null;
  state.activeStream = null;
  stopElapsedTicker();
  setQueryInputEnabled(!!state.currentDatasetId);
}

/**
 * Stop waiting for the current answer (Stop button).
 */
function stopActiveQuery() {
  if (!state.isProcessing) return;
  state.stopControl?.stop();
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
  const messageDiv = messageShell(message);
  messageDiv.dataset.state = messageKind(message);
  fillMessageContent(/** @type {HTMLElement} */ (messageDiv.lastElementChild), message);
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
  if (kind === 'refused' || kind === 'error') {
    contentDiv.appendChild(
      buildNotice(kind, message.content, {
        hint: RESTORED_QUESTION_HINT,
        detail: message.detail,
        copy: copyText,
      })
    );
  } else if (kind === 'stopped') {
    contentDiv.appendChild(buildNotice('stopped', message.content));
  } else if (message.result?.awaitingConfirmation) {
    contentDiv.appendChild(buildHeldCard(message.result, heldCardHandlers(message)));
  } else if (message.result) {
    contentDiv.appendChild(renderAnswerCard(message.result, { content: message.content }));
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

// ============================================================================
// Held plan (confirmation flow)
// ============================================================================

/**
 * What the held card's two buttons do.
 * @param {ChatMessage} message the held message
 * @returns {{ columns: string[], onConfirm: (card: HTMLElement) => void, onReject: (card: HTMLElement) => void }}
 */
function heldCardHandlers(message) {
  const held = message.result;
  const info = state.datasetInfo;
  return {
    columns:
      info && info.datasetId === (held.datasetId || state.currentDatasetId) ? info.columns : [],
    onConfirm: (card) => {
      if (state.isProcessing) {
        // Another answer is still streaming; leave the hold actionable instead of greying it out.
        setCardNote(card, 'Wait for the current answer to finish, then run this.');
        return;
      }
      // One decision per held plan: a second click must not run it again.
      setCardButtonsDisabled(card, true);
      void handleConfirmPendingQuery(held, card, message.questionId);
    },
    onReject: (card) => {
      setCardButtonsDisabled(card, true);
      settleCard(card, 'Not run. Rephrase your question below.');
      restoreQuestion(held.pendingQuery || '');
      elements.queryInput.focus();
    },
  };
}

/**
 * Run a held plan after the user approved it.
 * @param {Object} held
 * @param {HTMLElement} card
 */
async function handleConfirmPendingQuery(held, card, questionId) {
  const built = buildConfirmRequest(held, {
    currentDatasetId: state.currentDatasetId,
    fallbackSessionId: state.sessionId,
  });
  if (!built.ok) {
    setCardNote(card, built.message);
    // Nothing was sent: keep "Not what I meant" usable so the question can come back.
    setCardButtonsDisabled(card, false);
    return;
  }
  const { request } = built;
  const assistantMessageId = addWorkingMessage(questionId);
  const stopControl = beginProcessing(assistantMessageId);

  let outcome = 'error';
  let sent = false;
  try {
    const response = await stopControl.race(submitConfirmation(window.clinicalAPI, request));
    // Stopped before the request returned: it may have reached the server, so it counts as sent.
    sent = true;
    outcome =
      response === STOPPED
        ? 'stopped'
        : await followQuery(response, assistantMessageId, request.queryText, {
            datasetId: request.datasetId,
            sessionId: request.sessionId,
          });
  } catch (error) {
    console.error('Confirmed plan failed:', error?.message);
    showFailure(assistantMessageId, 'error', plainFailureMessage(error?.message));
  } finally {
    if (state.stopControl === stopControl) endProcessing();
  }

  if (sent) {
    // Never offer to send it twice, and never say it ran when the answer below is a refusal.
    settleCard(card, approvedCardNote(/** @type {any} */ (outcome)));
  } else {
    // The approval never reached the server, so the held plan stays confirmable.
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
        'It runs on this computer. When it can be reached, you can choose a cohort and ask questions.'
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
        'Earlier answers aren’t shown here; new questions continue this conversation.'
      )
    );
  } else {
    const name = datasetName(state.currentDatasetId);
    welcome.append(
      el('h2', 'welcome-title', `What would you like to know about ${name}?`),
      el(
        'p',
        'welcome-text',
        'Each answer shows how it was worked out: who was included, what was counted and the test.'
      )
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
  state.stopControl = null;
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
    void loadDataCheck(null);
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

    datasetOverviewSummary.textContent = `${rowCount} rows · ${columns.length} variables`;
    renderDatasetPreview(columns, rows);
    void loadDataCheck(datasetId);
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
    datasetOverviewSummary.textContent = `Couldn’t load this cohort’s first rows. ${plainFailureMessage(error?.message, { context: 'panel' })}`;
    renderDatasetPreview([], []);
  }
}

/**
 * Fill the "Data check" section of the cohort details sheet. GET /quality also backfills the
 * check for cohorts uploaded before it existed. A failed request reads "could not run".
 * @param {string | null} datasetId
 */
async function loadDataCheck(datasetId) {
  const { dataCheckSection, dataCheckHint, dataCheckList } = elements;
  if (!dataCheckSection || !dataCheckHint || !dataCheckList) return;
  if (!datasetId || typeof window.clinicalAPI?.getDatasetQuality !== 'function') {
    dataCheckSection.classList.add('hidden');
    return;
  }
  const parts = { hint: dataCheckHint, list: dataCheckList };
  dataCheckSection.classList.remove('hidden');
  renderDataCheck(parts, 'loading');
  let check;
  try {
    check = normalizeQuality(await window.clinicalAPI.getDatasetQuality(datasetId));
  } catch (error) {
    check = {
      status: 'not_run',
      reason: plainFailureMessage(error?.message, { context: 'panel' }),
    };
  }
  if (datasetId !== state.currentDatasetId) return;
  renderDataCheck(parts, check);
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
    setDatasetStatus('Couldn’t load the list of cohorts.', 'error');
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
const enrichmentPanelGate = createEnrichmentPanelGate();

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

  const load = enrichmentPanelGate.startLoad();
  const datasetId = state.currentDatasetId;
  enrichmentSection.classList.remove('hidden');
  enrichmentHint.textContent = 'Loading…';
  if (patchHistoryHint) patchHistoryHint.textContent = 'Loading…';
  if (patchHistoryBody) patchHistoryBody.replaceChildren();

  const hasPending = typeof window.clinicalAPI?.getPendingEnrichments === 'function';
  const hasHistory = typeof window.clinicalAPI?.getEnrichmentHistory === 'function';

  if (!hasPending && !hasHistory) {
    enrichmentHint.textContent = 'Variable suggestions are not available.';
    renderEnrichmentCards(enrichmentList, [], {
      onAccept: noopEnrichment,
      onReject: noopEnrichment,
    });
    if (patchHistoryHint) patchHistoryHint.textContent = '';
    return;
  }

  try {
    const pendingP = hasPending
      ? window.clinicalAPI.getPendingEnrichments(datasetId)
      : Promise.resolve({ suggestions: [], total: 0 });
    const historyP = hasHistory
      ? window.clinicalAPI.getEnrichmentHistory(datasetId)
      : Promise.resolve({ patches: [], total: 0 });

    const [pendingRaw, historyRaw] = await Promise.all([pendingP, historyP]);
    // A newer load (refresh, dataset change, decision reload) owns the panel now.
    if (!enrichmentPanelGate.isLatestLoad(load)) return;

    const { suggestions, total } = normalizePendingResponse(pendingRaw);
    if (enrichmentPanelGate.mayWriteHint(load, 'suggestions')) {
      // With none, the list's own empty state says so (once).
      enrichmentHint.textContent =
        total === 0 ? '' : `${total} pending suggestion${total === 1 ? '' : 's'}.`;
    }
    renderEnrichmentCards(enrichmentList, suggestions, {
      onAccept: (patchId) => handleEnrichmentAccept(patchId),
      onReject: (patchId) => handleEnrichmentReject(patchId),
    });

    const { patches, total: histTotal } = normalizeHistoryResponse(historyRaw);
    if (patchHistoryHint && enrichmentPanelGate.mayWriteHint(load, 'history')) {
      patchHistoryHint.textContent =
        histTotal === 0
          ? ''
          : `${histTotal} patch${histTotal === 1 ? '' : 'es'} in log (newest first).`;
    }
    if (patchHistoryBody) {
      renderPatchHistoryTable(patchHistoryBody, patches, {
        onRevert: (patchId) => handlePatchRevert(patchId),
      });
    }
  } catch (error) {
    console.error('Failed to load enrichments / history:', error?.message);
    if (!enrichmentPanelGate.isLatestLoad(load)) return;
    enrichmentHint.textContent = `Couldn’t load variable suggestions. ${plainFailureMessage(error?.message, { context: 'panel' })}`;
    renderEnrichmentCards(enrichmentList, [], {
      onAccept: noopEnrichment,
      onReject: noopEnrichment,
    });
    if (patchHistoryHint) {
      patchHistoryHint.textContent = `Couldn’t load the change history. ${plainFailureMessage(error?.message, { context: 'panel' })}`;
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
  const datasetId = state.currentDatasetId;
  if (!datasetId) return;
  await runEnrichmentDecision(() => window.clinicalAPI.acceptEnrichment(datasetId, patchId), {
    reload: loadEnrichmentPanel,
    showMessage: (text) => {
      enrichmentPanelGate.noteMessage('suggestions');
      if (enrichmentHint) enrichmentHint.textContent = text;
    },
    isCurrent: () => state.currentDatasetId === datasetId,
    fallback: 'Accept failed',
  });
}

/**
 * @param {string} patchId
 */
async function handleEnrichmentReject(patchId) {
  const { enrichmentHint } = elements;
  const datasetId = state.currentDatasetId;
  if (!datasetId) return;
  await runEnrichmentDecision(() => window.clinicalAPI.rejectEnrichment(datasetId, patchId), {
    reload: loadEnrichmentPanel,
    showMessage: (text) => {
      enrichmentPanelGate.noteMessage('suggestions');
      if (enrichmentHint) enrichmentHint.textContent = text;
    },
    isCurrent: () => state.currentDatasetId === datasetId,
    fallback: 'Reject failed',
  });
}

/**
 * @param {string} patchId
 */
async function handlePatchRevert(patchId) {
  const { patchHistoryHint } = elements;
  const datasetId = state.currentDatasetId;
  if (!datasetId) return;
  if (typeof window.clinicalAPI?.revertEnrichmentPatch !== 'function') {
    if (patchHistoryHint) patchHistoryHint.textContent = 'Undo is not available.';
    return;
  }
  await runEnrichmentDecision(() => window.clinicalAPI.revertEnrichmentPatch(datasetId, patchId), {
    reload: loadEnrichmentPanel,
    showMessage: (text) => {
      enrichmentPanelGate.noteMessage('history');
      if (patchHistoryHint) patchHistoryHint.textContent = text;
    },
    isCurrent: () => state.currentDatasetId === datasetId,
    fallback: 'Revert failed',
  });
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
    state.sessionCreating = null;
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
  state.sessionCreating = null;
  state.messages = [];
  state.isProcessing = false;
  state.stopControl = null;
  sidebar?.close({ focus: false });
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
    announce('Couldn’t delete the conversation.');
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
  // A creation still in flight belonged to the old conversation.
  state.sessionCreating = null;
  if (typeof window.clinicalAPI?.createSession !== 'function') {
    state.sessionId = generateSessionId();
    state.sessionOnServer = false;
    state.sessionCreating = null;
    resetChatUI();
    return;
  }
  try {
    const res = await window.clinicalAPI.createSession(state.currentDatasetId);
    state.sessionId = res.session_id;
    state.sessionOnServer = true;
    state.sessionCreating = null;
    resetChatUI();
    await loadSessionList();
  } catch (error) {
    console.error('Create session failed:', error?.message);
    state.sessionId = generateSessionId();
    state.sessionOnServer = false;
    state.sessionCreating = null;
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
        panel.setError(
          `Couldn’t delete this cohort. ${plainFailureMessage(error?.message, { context: 'delete' })}`
        );
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

  clearWelcomeMessage();

  const questionId = generateMessageId();
  addMessage({
    id: questionId,
    role: 'user',
    content: queryText,
    timestamp: new Date(),
  });
  const assistantMessageId = addWorkingMessage(questionId);
  const stopControl = beginProcessing(assistantMessageId);

  await stopControl.race(ensureServerSession(state, window.clinicalAPI));
  if (!state.sessionTitles.has(state.sessionId)) {
    state.sessionTitles.set(state.sessionId, titleFromQuestion(queryText));
  }

  // Never log the question text or results: they can identify patients (docs/security).
  console.log('Query submitted:', {
    sessionId: state.sessionId,
    datasetId: state.currentDatasetId,
    queryLength: queryText.length,
  });

  let outcome = stopControl.stopped ? 'stopped' : 'error';
  try {
    if (!stopControl.stopped) {
      const response = await stopControl.race(
        window.clinicalAPI.submitQuery(state.sessionId, state.currentDatasetId, queryText, false)
      );
      if (response === STOPPED) {
        outcome = 'stopped';
      } else {
        console.log('Query accepted:', response?.query_id);
        outcome = await followQuery(response, assistantMessageId, queryText, {
          datasetId: state.currentDatasetId,
          sessionId: state.sessionId,
        });
      }
    }
  } catch (error) {
    console.error('Query failed:', error?.message);
    showFailure(assistantMessageId, 'error', plainFailureMessage(error?.message));
  } finally {
    if (state.stopControl === stopControl) endProcessing();
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
function addWorkingMessage(questionId) {
  const id = generateMessageId();
  addMessage({
    id,
    role: 'assistant',
    content: 'Reading your question',
    isStreaming: true,
    startedAt: Date.now(),
    timestamp: new Date(),
    ...(questionId ? { questionId } : {}),
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
  const current = state.messages.find((m) => m.id === messageId);
  if (current && !current.isStreaming) return; // already settled (e.g. Stopped)
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
  const current = state.messages.find((m) => m.id === messageId);
  if (current && !current.isStreaming) return;
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
            data.claim,
            data.plan_line
          );

          updateMessage(assistantMessageId, {
            content: pres.content,
            result: pres.result,
            isStreaming: false,
            kind: 'answered',
          });
          announce(answerAnnouncement(pres.result));
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
      setDatasetStatus(
        uploadStatusText(
          result.dataset_name ? readyHeadline(result.dataset_name) : 'The sample cohort is ready.',
          dataCheckStatusNote(result.quality)
        ),
        'ok',
        { showDetails: dataCheckNeedsReview(result.quality) }
      );
      await selectNewDataset(result.upload_id);
    } else {
      setDatasetStatus(
        `Couldn’t load the sample cohort. ${plainFailureMessage(result.message, { context: 'upload' })}`,
        'error'
      );
    }
  } catch (error) {
    console.error('Sample load failed:', error?.message);
    setDatasetStatus(
      `Couldn’t load the sample cohort. ${plainFailureMessage(error?.message, { context: 'upload' })}`,
      'error'
    );
  }
}

async function handleFileUpload(file) {
  if (typeof window.clinicalAPI?.uploadDataset !== 'function') return;
  setDatasetStatus(`Uploading “${file.name}”…`);

  try {
    const raw = await window.clinicalAPI.uploadDataset(file);
    const result = normalizeUploadResponse(raw);

    if (result.status === 'ready') {
      setDatasetStatus(
        uploadStatusText(
          readyHeadline(result.dataset_name || file.name),
          dataCheckStatusNote(result.quality)
        ),
        'ok',
        { showDetails: dataCheckNeedsReview(result.quality) }
      );
      await selectNewDataset(result.upload_id);
    } else {
      setDatasetStatus(
        `Couldn’t upload “${file.name}”. ${plainFailureMessage(result.message, { context: 'upload' })}`,
        'error'
      );
    }
  } catch (error) {
    console.error('Upload failed:', error?.message);
    setDatasetStatus(
      `Couldn’t upload “${file.name}”. ${plainFailureMessage(error?.message, { context: 'upload' })}`,
      'error'
    );
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
  elements.datasetStatusAction?.addEventListener('click', () =>
    openDatasetPanel(elements.datasetStatusAction)
  );
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

  newChatBtn?.addEventListener('click', () => {
    sidebar?.close({ focus: false });
    void handleNewChat();
  });
  retryConnectionBtn?.addEventListener('click', () => void connect());

  if (elements.appContainer && elements.sidebarToggle && elements.sessionSidebar) {
    sidebar = bindSidebarToggle({
      container: elements.appContainer,
      toggle: /** @type {HTMLButtonElement} */ (elements.sidebarToggle),
      sidebar: elements.sessionSidebar,
      closeButton: document.getElementById('sidebar-close'),
    });
  }
  if (elements.llmBanner) {
    bindBannerDismiss(elements.llmBanner, {
      focusAfter: queryInput,
      onDismiss: () => {
        state.llmBannerDismissedText = elements.llmBannerText?.textContent || '';
      },
    });
  }
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
