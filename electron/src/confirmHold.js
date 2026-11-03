/**
 * Confirming a held plan, in one place.
 *
 * A held plan may run only on the dataset it was held for, and only as the exact plan the user
 * saw: the backend releases the hold for that plan line alone. When the API starts issuing a hold
 * token, it is carried here (and nowhere else).
 */

/**
 * @typedef {Object} HeldPlan
 * @property {string} [pendingQuery] the question that produced the hold
 * @property {string} [planLine] the exact plan line shown to the user
 * @property {string} [planHash] hash of the exact executable plan, from the hold
 * @property {string | null} [datasetId] dataset the plan was held for
 * @property {string | null} [sessionId] session the plan was held in
 */

/**
 * @typedef {Object} ConfirmRequest
 * @property {string} sessionId
 * @property {string} datasetId
 * @property {string} queryText
 * @property {string} planLine
 * @property {string} planHash
 */

/**
 * Validate a held plan and build the request that releases it.
 * @param {HeldPlan | null | undefined} held
 * @param {{ currentDatasetId: string | null, fallbackSessionId: string }} context
 * @returns {{ ok: true, request: ConfirmRequest } | { ok: false, message: string }}
 */
export function buildConfirmRequest(held, { currentDatasetId, fallbackSessionId }) {
  const queryText = held?.pendingQuery || '';
  const planLine = held?.planLine || '';
  const planHash = typeof held?.planHash === 'string' ? held.planHash.trim() : '';
  if (!queryText || !planLine || !planHash) {
    return { ok: false, message: 'This plan can no longer be run. Ask the question again.' };
  }
  if (held.datasetId && held.datasetId !== currentDatasetId) {
    return {
      ok: false,
      message: 'This plan was for a different cohort. Ask the question again on this one.',
    };
  }
  const datasetId = held.datasetId || currentDatasetId;
  if (!datasetId) {
    return { ok: false, message: 'Choose a cohort, then ask the question again.' };
  }
  return {
    ok: true,
    request: {
      sessionId: held.sessionId || fallbackSessionId,
      datasetId,
      queryText,
      planLine,
      planHash,
    },
  };
}

/**
 * Send the confirmation through the preload API.
 * @param {{ submitQuery: Function }} api window.clinicalAPI
 * @param {ConfirmRequest} request
 * @returns {Promise<{ query_id?: string, stream_url?: string }>}
 */
export function submitConfirmation(api, request) {
  return api.submitQuery(
    request.sessionId,
    request.datasetId,
    request.queryText,
    true,
    request.planLine,
    request.planHash
  );
}
