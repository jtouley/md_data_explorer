// SPDX-License-Identifier: MIT
/**
 * Electron preload script.
 *
 * Provides secure bridge between renderer (browser) and main process.
 * Exposes clinicalAPI object via contextBridge for API communication.
 */
import { contextBridge, ipcRenderer } from 'electron';
import {
  API_TOKEN_CHANNEL,
  DEFAULT_API_BASE_URL,
  INSECURE_BACKEND_MESSAGE,
  apiBaseUrlFromArgv,
  createReplayFilter,
  decideAfterDrop,
  enrichmentDecisionResult,
  readSseStream,
  verifyBackendEnforcesAuth,
  withAuth,
} from './apiAuth.js';
import { THEME_GET_CHANNEL, THEME_SET_CHANNEL, isThemeSource, themeFromArgv } from './theme.js';

// Backend API address: main always passes the one this launch uses (the backend it started, or
// for an external dev backend VITE_API_URL / the default) and builds the CSP header from the same
// value. The fallbacks only matter if the argument is missing.
const API_BASE_URL =
  apiBaseUrlFromArgv(process.argv) || import.meta.env.VITE_API_URL || DEFAULT_API_BASE_URL;

// { token, devNoAuth } from main. Kept in this context, never on window.
let configPromise = null;
function getApiConfig() {
  if (!configPromise) {
    configPromise = ipcRenderer.invoke(API_TOKEN_CHANNEL).catch((error) => {
      console.error('[Preload] Could not get API token:', error?.message);
      configPromise = null;
      return { token: null, devNoAuth: false };
    });
  }
  return configPromise;
}

class InsecureBackendError extends Error {}

// Before any data is sent, check that the backend rejects requests without the token, unless
// this app was itself started in dev mode. A network failure is not cached, so a backend that
// is still starting is checked again on the next call.
let backendCheck = null;
function ensureBackendEnforcesAuth() {
  if (!backendCheck) {
    backendCheck = (async () => {
      const { token, devNoAuth } = await getApiConfig();
      if (devNoAuth) return;
      const verdict = await verifyBackendEnforcesAuth(
        (url, init) => fetch(url, init),
        API_BASE_URL,
        token,
        globalThis.crypto
      );
      if (!verdict.ok) {
        console.error('[Preload] Backend does not enforce auth:', verdict.reason);
        throw new InsecureBackendError(INSECURE_BACKEND_MESSAGE);
      }
    })().catch((error) => {
      if (!(error instanceof InsecureBackendError)) backendCheck = null;
      throw error;
    });
  }
  return backendCheck;
}

/** fetch() with the Authorization header added, after the backend's auth check passed. */
async function apiFetch(url, init = {}) {
  await ensureBackendEnforcesAuth();
  const { token } = await getApiConfig();
  return fetch(url, { ...init, headers: withAuth(token, init.headers) });
}

/**
 * Clinical Analytics API - exposed to renderer process.
 */
const clinicalAPI = {
  /**
   * Check backend health status.
   * @returns {Promise<{status: string}>}
   */
  async healthCheck() {
    try {
      const response = await apiFetch(`${API_BASE_URL}/health`);
      return await response.json();
    } catch (error) {
      return { status: 'error', error: error.message };
    }
  },

  /**
   * List available datasets.
   * @returns {Promise<{datasets: Array, total: number}>}
   */
  async listDatasets() {
    const response = await apiFetch(`${API_BASE_URL}/api/datasets`);
    if (!response.ok) {
      throw new Error(`Failed to list datasets: ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Fetch dataset metadata/details.
   * @param {string} datasetId
   * @returns {Promise<Object>}
   */
  async getDataset(datasetId) {
    const enc = encodeURIComponent(datasetId);
    const response = await apiFetch(`${API_BASE_URL}/api/datasets/${enc}`);
    if (!response.ok) {
      throw new Error(`Failed to load dataset details: ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Data-quality report for a dataset's active version (contract checks run at upload).
   * @param {string} datasetId
   * @returns {Promise<{ summary: string[], passed: number, warnings: number, failed: number, not_run: number, checks: object[] }>}
   */
  async getDatasetQuality(datasetId) {
    const enc = encodeURIComponent(datasetId);
    const response = await apiFetch(`${API_BASE_URL}/api/datasets/${enc}/quality`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data?.detail || `Failed to load data quality: ${response.statusText}`);
    }
    return data;
  },

  /**
   * Fetch dataset preview rows.
   * @param {string} datasetId
   * @param {number} [limit]
   * @returns {Promise<{dataset_id: string, rows: object[], columns: string[], total_rows: number}>}
   */
  async previewDataset(datasetId, limit = 10) {
    const enc = encodeURIComponent(datasetId);
    const response = await apiFetch(`${API_BASE_URL}/api/datasets/${enc}/preview?limit=${limit}`);
    if (!response.ok) {
      throw new Error(`Failed to load dataset preview: ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Submit a natural language query.
   * @param {string} sessionId - Session identifier
   * @param {string} datasetId - Dataset to query
   * @param {string} queryText - Natural language query
   * @param {boolean} [confirmed] - Release a held plan
   * @param {string|null} [confirmPlanLine] - The plan line shown (display only)
   * @param {string|null} [confirmPlanHash] - Hash of the held plan; required to release it
   * @returns {Promise<{query_id: string, status: string, stream_url: string}>}
   */
  async submitQuery(
    sessionId,
    datasetId,
    queryText,
    confirmed = false,
    confirmPlanLine = null,
    confirmPlanHash = null
  ) {
    const response = await apiFetch(`${API_BASE_URL}/api/queries`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        dataset_id: datasetId,
        query_text: queryText,
        confirmed: Boolean(confirmed),
        confirm_plan_line: confirmPlanLine,
        confirm_plan_hash: confirmPlanHash,
      }),
    });
    if (!response.ok) {
      throw new Error(`Failed to submit query: ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Get query result by ID.
   * @param {string} queryId - Query identifier
   * @returns {Promise<Object>}
   */
  async getQueryResult(queryId) {
    const response = await apiFetch(`${API_BASE_URL}/api/queries/${queryId}`);
    if (!response.ok) {
      throw new Error(`Failed to get query result: ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Subscribe to SSE stream for query results.
   * Reads the stream with fetch (so it carries the bearer token) and calls callbacks for events.
   * @param {string} queryId - Query identifier
   * @param {Object} callbacks - Event callbacks
   * @param {Function} callbacks.onEvent - Called for each event with {event, data}
   * @param {Function} callbacks.onError - Called on error
   * @param {Function} callbacks.onClose - Called when stream closes
   * @returns {Function} Cleanup function to close the stream
   */
  subscribeToQueryStream(queryId, callbacks) {
    // fetch streaming, not EventSource: EventSource cannot send the Authorization header.
    const controller = new AbortController();
    const enc = encodeURIComponent(queryId);
    let closed = false;
    let reconnects = 0;
    // A reopened stream replays every event; deliver each one to the renderer only once.
    const replay = createReplayFilter();
    const close = () => {
      if (closed) return;
      closed = true;
      controller.abort();
      if (callbacks.onClose) callbacks.onClose();
    };
    const fail = (error) => {
      if (closed) return;
      console.error('[Preload] SSE error:', error?.message || error);
      if (callbacks.onError) callbacks.onError(error);
    };

    const readOnce = async () => {
      const response = await apiFetch(`${API_BASE_URL}/api/queries/${enc}/stream`, {
        headers: { Accept: 'text/event-stream' },
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        throw new Error(`Stream failed: ${response.status} ${response.statusText}`);
      }
      replay.startConnection();
      await readSseStream(response.body, (raw) => {
        if (closed) return;
        if (!replay.shouldDeliver()) return;
        try {
          const data = JSON.parse(raw);
          if (callbacks.onEvent) callbacks.onEvent({ event: data.event, data });
        } catch (err) {
          console.error('[Preload] Failed to parse SSE data:', err?.message);
        }
      });
    };

    // A stream that ends or breaks before stream_end: ask the server about the query, then
    // reopen the stream (it replays every event, including a result that finished meanwhile).
    const run = async () => {
      for (;;) {
        try {
          await readOnce();
        } catch (error) {
          if (closed || error?.name === 'AbortError') return;
          if (error instanceof InsecureBackendError) return fail(error);
        }
        if (closed) return;
        let status = 0;
        try {
          status = (
            await apiFetch(`${API_BASE_URL}/api/queries/${enc}`, { signal: controller.signal })
          ).status;
        } catch (error) {
          if (closed || error?.name === 'AbortError') return;
        }
        const decision = decideAfterDrop(status, reconnects);
        if (decision === 'reconnect') {
          reconnects += 1;
          continue;
        }
        return fail(
          new Error(
            decision === 'gone'
              ? 'This answer is no longer available. Ask the question again.'
              : 'Lost the connection to the analysis service.'
          )
        );
      }
    };
    void run();

    return close;
  },

  /**
   * Get API base URL (for debugging).
   * @returns {string}
   */
  getApiBaseUrl() {
    return API_BASE_URL;
  },

  /**
   * List pending metadata enrichment suggestions for a dataset.
   * @param {string} datasetId
   * @returns {Promise<{ suggestions: object[], total: number }>}
   */
  async getPendingEnrichments(datasetId) {
    const enc = encodeURIComponent(datasetId);
    const response = await apiFetch(`${API_BASE_URL}/api/datasets/${enc}/enrichments/pending`);
    if (!response.ok) {
      throw new Error(`Failed to load enrichments: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Accept a pending enrichment patch.
   * @param {string} datasetId
   * @param {string} patchId
   * @param {string} [acceptedBy]
   * @returns {Promise<{ success: boolean, message: string, stale?: boolean }>}
   */
  async acceptEnrichment(datasetId, patchId, acceptedBy = 'electron_user') {
    const encD = encodeURIComponent(datasetId);
    const encP = encodeURIComponent(patchId);
    const response = await apiFetch(
      `${API_BASE_URL}/api/datasets/${encD}/enrichments/${encP}/accept`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ accepted_by: acceptedBy }),
      }
    );
    const data = await response.json().catch(() => ({}));
    return enrichmentDecisionResult(response, data, 'Accept');
  },

  /**
   * Reject a pending enrichment patch.
   * @param {string} datasetId
   * @param {string} patchId
   * @param {string} [reason]
   * @returns {Promise<{ success: boolean, message: string, stale?: boolean }>}
   */
  async rejectEnrichment(datasetId, patchId, reason = 'Rejected in Electron') {
    const encD = encodeURIComponent(datasetId);
    const encP = encodeURIComponent(patchId);
    const response = await apiFetch(
      `${API_BASE_URL}/api/datasets/${encD}/enrichments/${encP}/reject`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason }),
      }
    );
    const data = await response.json().catch(() => ({}));
    return enrichmentDecisionResult(response, data, 'Reject');
  },

  /**
   * Patch history for a dataset (all patches from overlay log).
   * @param {string} datasetId
   * @returns {Promise<{ patches: object[], total: number }>}
   */
  async getEnrichmentHistory(datasetId) {
    const enc = encodeURIComponent(datasetId);
    const response = await apiFetch(`${API_BASE_URL}/api/datasets/${enc}/enrichments/history`);
    if (!response.ok) {
      throw new Error(`Failed to load patch history: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Revert an accepted enrichment patch.
   * @param {string} datasetId
   * @param {string} patchId
   * @param {string} [revertedBy]
   * @returns {Promise<{ success: boolean, message: string, stale?: boolean }>}
   */
  async revertEnrichmentPatch(datasetId, patchId, revertedBy = 'electron_user') {
    const encD = encodeURIComponent(datasetId);
    const encP = encodeURIComponent(patchId);
    const response = await apiFetch(
      `${API_BASE_URL}/api/datasets/${encD}/enrichments/${encP}/revert`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reverted_by: revertedBy }),
      }
    );
    const data = await response.json().catch(() => ({}));
    return enrichmentDecisionResult(response, data, 'Revert');
  },

  /**
   * List sessions, optionally filtered by dataset.
   * @param {{ datasetId?: string, limit?: number, offset?: number }} [opts]
   * @returns {Promise<{ sessions: object[], total: number }>}
   */
  async listSessions(opts = {}) {
    const params = new URLSearchParams();
    if (opts.datasetId) params.set('dataset_id', opts.datasetId);
    if (opts.limit != null) params.set('limit', String(opts.limit));
    if (opts.offset != null) params.set('offset', String(opts.offset));
    const qs = params.toString();
    const url = `${API_BASE_URL}/api/sessions${qs ? `?${qs}` : ''}`;
    const response = await apiFetch(url);
    if (!response.ok) {
      throw new Error(`Failed to list sessions: ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Create a new session.
   * @param {string} datasetId
   * @returns {Promise<{ session_id: string, dataset_id: string, created_at: string }>}
   */
  async createSession(datasetId) {
    const response = await apiFetch(`${API_BASE_URL}/api/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id: datasetId }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data?.detail || `Failed to create session: ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Delete a session.
   * @param {string} sessionId
   * @returns {Promise<void>}
   */
  async deleteSession(sessionId) {
    const enc = encodeURIComponent(sessionId);
    const response = await apiFetch(`${API_BASE_URL}/api/sessions/${enc}`, {
      method: 'DELETE',
    });
    if (!response.ok && response.status !== 204) {
      throw new Error(`Failed to delete session: ${response.statusText}`);
    }
  },

  /**
   * Upload a dataset file (CSV, Excel, SPSS).
   * @param {File} file
   * @param {string} [datasetName]
   * @returns {Promise<{ upload_id: string, dataset_name: string, status: string, message: string, quality: object|null }>}
   */
  async uploadDataset(file, datasetName) {
    const form = new FormData();
    form.append('file', file);
    if (datasetName) form.append('dataset_name', datasetName);
    const response = await apiFetch(`${API_BASE_URL}/api/datasets/upload`, {
      method: 'POST',
      body: form,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data?.detail || `Upload failed: ${response.statusText}`);
    }
    return data;
  },

  /**
   * Delete a dataset and everything derived from it (files, tables, sessions, results).
   * @param {string} datasetId
   * @returns {Promise<{ dataset_id: string, removed_paths: number, dropped_tables: number, deleted_sessions: number }>}
   */
  async deleteDataset(datasetId) {
    const enc = encodeURIComponent(datasetId);
    const response = await apiFetch(`${API_BASE_URL}/api/datasets/${enc}`, { method: 'DELETE' });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data?.detail || `Delete failed: ${response.statusText}`);
    }
    return data;
  },

  /**
   * Load the bundled sample cohort into upload storage.
   * @returns {Promise<{ upload_id: string, dataset_name: string, status: string, message: string }>}
   */
  async loadSampleDataset() {
    const response = await apiFetch(`${API_BASE_URL}/api/datasets/sample`, {
      method: 'POST',
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data?.detail || `Sample load failed: ${response.statusText}`);
    }
    return data;
  },
};

// Expose clinicalAPI to renderer process via contextBridge
contextBridge.exposeInMainWorld('clinicalAPI', clinicalAPI);

// Appearance: System / Light / Dark, applied by main through nativeTheme and persisted there.
contextBridge.exposeInMainWorld('appTheme', {
  /** The saved choice main applied before the window opened (null outside Electron). */
  initial: themeFromArgv(process.argv),
  /** @returns {Promise<{ source: string, dark: boolean }>} */
  get: () => ipcRenderer.invoke(THEME_GET_CHANNEL),
  /** @param {'system' | 'light' | 'dark'} source */
  set: (source) => {
    if (!isThemeSource(source)) return Promise.reject(new Error('Unknown theme'));
    return ipcRenderer.invoke(THEME_SET_CHANNEL, source);
  },
});
