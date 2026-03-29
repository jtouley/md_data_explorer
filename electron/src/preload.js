/**
 * Electron preload script.
 *
 * Provides secure bridge between renderer (browser) and main process.
 * Exposes clinicalAPI object via contextBridge for API communication.
 */
import { contextBridge } from 'electron';

// Backend API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
      const response = await fetch(`${API_BASE_URL}/health`);
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
    const response = await fetch(`${API_BASE_URL}/api/datasets`);
    if (!response.ok) {
      throw new Error(`Failed to list datasets: ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Submit a natural language query.
   * @param {string} sessionId - Session identifier
   * @param {string} datasetId - Dataset to query
   * @param {string} queryText - Natural language query
   * @returns {Promise<{query_id: string, status: string, stream_url: string}>}
   */
  async submitQuery(sessionId, datasetId, queryText) {
    const response = await fetch(`${API_BASE_URL}/api/queries`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        dataset_id: datasetId,
        query_text: queryText,
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
    const response = await fetch(`${API_BASE_URL}/api/queries/${queryId}`);
    if (!response.ok) {
      throw new Error(`Failed to get query result: ${response.statusText}`);
    }
    return await response.json();
  },

  /**
   * Subscribe to SSE stream for query results.
   * Handles EventSource internally and calls callbacks for events.
   * @param {string} queryId - Query identifier
   * @param {Object} callbacks - Event callbacks
   * @param {Function} callbacks.onEvent - Called for each event with {event, data}
   * @param {Function} callbacks.onError - Called on error
   * @param {Function} callbacks.onClose - Called when stream closes
   * @returns {Function} Cleanup function to close the stream
   */
  subscribeToQueryStream(queryId, callbacks) {
    const eventSource = new EventSource(`${API_BASE_URL}/api/queries/${queryId}/stream`);

    eventSource.onopen = () => {
      console.log('[Preload] SSE stream opened for:', queryId);
    };

    eventSource.onmessage = (event) => {
      console.log('[Preload] SSE message received:', event.data);
      try {
        const data = JSON.parse(event.data);
        if (callbacks.onEvent) {
          callbacks.onEvent({ event: data.event, data });
        }
      } catch (err) {
        console.error('[Preload] Failed to parse SSE data:', err);
      }
    };

    eventSource.onerror = (error) => {
      console.error('[Preload] SSE error:', error);
      if (callbacks.onError) {
        callbacks.onError(error);
      }
    };

    // Return cleanup function
    return () => {
      console.log('[Preload] Closing SSE stream for:', queryId);
      eventSource.close();
    };
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
    const response = await fetch(`${API_BASE_URL}/api/datasets/${enc}/enrichments/pending`);
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
   * @returns {Promise<{ success: boolean, message: string }>}
   */
  async acceptEnrichment(datasetId, patchId, acceptedBy = 'electron_user') {
    const encD = encodeURIComponent(datasetId);
    const encP = encodeURIComponent(patchId);
    const response = await fetch(
      `${API_BASE_URL}/api/datasets/${encD}/enrichments/${encP}/accept`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ accepted_by: acceptedBy }),
      }
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data?.message || `Accept failed: ${response.statusText}`);
    }
    if (data.success === false) {
      throw new Error(data.message || 'Accept failed');
    }
    return data;
  },

  /**
   * Reject a pending enrichment patch.
   * @param {string} datasetId
   * @param {string} patchId
   * @param {string} [reason]
   * @returns {Promise<{ success: boolean, message: string }>}
   */
  async rejectEnrichment(datasetId, patchId, reason = 'Rejected in Electron') {
    const encD = encodeURIComponent(datasetId);
    const encP = encodeURIComponent(patchId);
    const response = await fetch(
      `${API_BASE_URL}/api/datasets/${encD}/enrichments/${encP}/reject`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason }),
      }
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data?.message || `Reject failed: ${response.statusText}`);
    }
    if (data.success === false) {
      throw new Error(data.message || 'Reject failed');
    }
    return data;
  },

  /**
   * Patch history for a dataset (all patches from overlay log).
   * @param {string} datasetId
   * @returns {Promise<{ patches: object[], total: number }>}
   */
  async getEnrichmentHistory(datasetId) {
    const enc = encodeURIComponent(datasetId);
    const response = await fetch(`${API_BASE_URL}/api/datasets/${enc}/enrichments/history`);
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
   * @returns {Promise<{ success: boolean, message: string }>}
   */
  async revertEnrichmentPatch(datasetId, patchId, revertedBy = 'electron_user') {
    const encD = encodeURIComponent(datasetId);
    const encP = encodeURIComponent(patchId);
    const response = await fetch(
      `${API_BASE_URL}/api/datasets/${encD}/enrichments/${encP}/revert`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reverted_by: revertedBy }),
      }
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data?.message || `Revert failed: ${response.statusText}`);
    }
    if (data.success === false) {
      throw new Error(data.message || 'Revert failed');
    }
    return data;
  },
};

// Expose clinicalAPI to renderer process via contextBridge
contextBridge.exposeInMainWorld('clinicalAPI', clinicalAPI);
