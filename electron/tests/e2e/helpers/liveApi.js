// SPDX-License-Identifier: MIT
/**
 * window.clinicalAPI over plain fetch, for specs that run the renderer in Chromium against a real
 * FastAPI process (no Electron preload there). Same calls and shapes as src/preload.js.
 */

/**
 * @param {import('@playwright/test').Page} page
 * @param {string} apiBaseUrl
 */
async function installLiveApi(page, apiBaseUrl) {
  await page.addInitScript(
    ({ apiBaseUrl }) => {
      const jsonOrEmpty = async (response) => {
        try {
          return await response.json();
        } catch {
          return {};
        }
      };

      window.clinicalAPI = {
        async healthCheck() {
          const res = await fetch(`${apiBaseUrl}/health`);
          if (!res.ok) return { status: 'error' };
          return res.json();
        },
        async listDatasets() {
          const res = await fetch(`${apiBaseUrl}/api/datasets`);
          if (!res.ok) throw new Error(`Failed to list datasets: ${res.statusText}`);
          return res.json();
        },
        async getDataset(datasetId) {
          const res = await fetch(`${apiBaseUrl}/api/datasets/${encodeURIComponent(datasetId)}`);
          if (!res.ok) throw new Error(`Failed to load dataset details: ${res.statusText}`);
          return res.json();
        },
        async previewDataset(datasetId, limit = 10) {
          const res = await fetch(
            `${apiBaseUrl}/api/datasets/${encodeURIComponent(datasetId)}/preview?limit=${limit}`
          );
          if (!res.ok) throw new Error(`Failed to load dataset preview: ${res.statusText}`);
          return res.json();
        },
        async listSessions() {
          const res = await fetch(`${apiBaseUrl}/api/sessions`);
          if (!res.ok) throw new Error(`Failed to list sessions: ${res.statusText}`);
          return res.json();
        },
        async createSession(datasetId) {
          const res = await fetch(`${apiBaseUrl}/api/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_id: datasetId }),
          });
          if (!res.ok) {
            const data = await jsonOrEmpty(res);
            throw new Error(data.detail || `Failed to create session: ${res.statusText}`);
          }
          return res.json();
        },
        async deleteSession(sessionId) {
          const res = await fetch(`${apiBaseUrl}/api/sessions/${encodeURIComponent(sessionId)}`, {
            method: 'DELETE',
          });
          if (!res.ok && res.status !== 204) {
            throw new Error(`Failed to delete session: ${res.statusText}`);
          }
        },
        async getPendingEnrichments(datasetId) {
          const res = await fetch(
            `${apiBaseUrl}/api/datasets/${encodeURIComponent(datasetId)}/enrichments/pending`
          );
          if (!res.ok) throw new Error(`Failed to load enrichments: ${res.statusText}`);
          return res.json();
        },
        async getEnrichmentHistory(datasetId) {
          const res = await fetch(
            `${apiBaseUrl}/api/datasets/${encodeURIComponent(datasetId)}/enrichments/history`
          );
          if (!res.ok) throw new Error(`Failed to load patch history: ${res.statusText}`);
          return res.json();
        },
        async uploadDataset(file, datasetName) {
          const form = new FormData();
          form.append('file', file);
          if (datasetName) form.append('dataset_name', datasetName);
          const res = await fetch(`${apiBaseUrl}/api/datasets/upload`, {
            method: 'POST',
            body: form,
          });
          const data = await jsonOrEmpty(res);
          if (!res.ok) throw new Error(data.detail || `Upload failed: ${res.statusText}`);
          return data;
        },
        async submitQuery(
          sessionId,
          datasetId,
          queryText,
          confirmed = false,
          confirmPlanLine = null,
          confirmPlanHash = null
        ) {
          const res = await fetch(`${apiBaseUrl}/api/queries`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: sessionId,
              dataset_id: datasetId,
              query_text: queryText,
              confirmed: Boolean(confirmed),
              confirm_plan_line: confirmPlanLine,
              confirm_plan_hash: confirmPlanHash,
            }),
          });
          const data = await jsonOrEmpty(res);
          if (!res.ok) throw new Error(data.detail || `Query failed: ${res.statusText}`);
          return data;
        },
        async getDatasetQuality(datasetId) {
          const res = await fetch(
            `${apiBaseUrl}/api/datasets/${encodeURIComponent(datasetId)}/quality`
          );
          if (!res.ok) throw new Error(`Failed to load the data check: ${res.statusText}`);
          return res.json();
        },
        subscribeToQueryStream(queryId, callbacks) {
          const eventSource = new EventSource(`${apiBaseUrl}/api/queries/${queryId}/stream`);
          eventSource.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              if (callbacks && callbacks.onEvent) {
                callbacks.onEvent({ event: data.event, data });
              }
            } catch (err) {
              if (callbacks && callbacks.onError) callbacks.onError(err);
            }
          };
          eventSource.onerror = (error) => {
            if (callbacks && callbacks.onError) callbacks.onError(error);
          };
          return () => eventSource.close();
        },
      };
    },
    { apiBaseUrl }
  );
}

module.exports = { installLiveApi };
