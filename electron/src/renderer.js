/**
 * Clinical Analytics Renderer Process
 *
 * Handles UI interactions, API communication via clinicalAPI (from preload),
 * and chat message management.
 *
 * Phase 4: Basic initialization and health check
 * Phase 5: Chat interface implementation
 */
import './index.css';

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
  connectionStatus: document.getElementById('connection-status'),
  statusText: document.querySelector('.status-text'),
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

const state = {
  connected: false,
  currentDatasetId: null,
  sessionId: generateSessionId(),
  messages: [],
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

  queryInput.disabled = !enabled;
  submitBtn.disabled = !enabled;

  if (enabled) {
    inputHintText.textContent = 'Press Enter to submit or click the arrow';
    queryInput.focus();
  } else {
    inputHintText.textContent = 'Select a dataset to start asking questions';
  }
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
      console.log('✅ Backend health check passed:', health);
    } else {
      updateConnectionStatus('error', `Backend: ${health.status}`);
      console.warn('⚠️ Backend health check returned:', health);
    }
  } catch (error) {
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

  if (!queryText || !state.currentDatasetId) {
    return;
  }

  // Clear input
  queryInput.value = '';

  // TODO (Phase 5): Add message to chat, submit query, handle response
  console.log('📝 Query submitted:', {
    sessionId: state.sessionId,
    datasetId: state.currentDatasetId,
    query: queryText,
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
