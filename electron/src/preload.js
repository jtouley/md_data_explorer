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
   * Create SSE connection for query streaming.
   * @param {string} queryId - Query identifier
   * @returns {EventSource}
   */
  createQueryStream(queryId) {
    return new EventSource(`${API_BASE_URL}/api/queries/${queryId}/stream`);
  },

  /**
   * Get API base URL (for debugging).
   * @returns {string}
   */
  getApiBaseUrl() {
    return API_BASE_URL;
  },
};

// Expose clinicalAPI to renderer process via contextBridge
contextBridge.exposeInMainWorld('clinicalAPI', clinicalAPI);
