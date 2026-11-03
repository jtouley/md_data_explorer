/**
 * Authenticated API helpers for the preload script.
 *
 * EventSource cannot send an Authorization header, so the query stream is read with fetch
 * streaming instead. The token stays in the preload context; it is never put in a URL (URLs end
 * up in access logs and history) and is never exposed on window.
 */

/**
 * The API address. An IP literal, never "localhost": localhost may resolve to ::1, where another
 * program could listen and relay /health proofs from the real backend on 127.0.0.1.
 */
export const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

/** IPC channel on which main hands { token, devNoAuth } to the preload script. */
export const API_TOKEN_CHANNEL = 'clinical-api:token';

/** Shown when the backend does not check tokens and this app was not started in dev mode. */
export const INSECURE_BACKEND_MESSAGE =
  'The analysis service on this computer is not the one this app started, or it is running ' +
  'without its access check, so the app will not use it. Close the app and start it again with ' +
  'the launcher (scripts/run_app.sh).';

/** Must match HEALTH_PROOF_CONTEXT in clinical_analytics/api/security.py. */
export const HEALTH_PROOF_CONTEXT = 'clinical-analytics-health-v1:';

/**
 * Random URL-safe nonce for /health.
 * @param {Crypto} cryptoImpl
 * @returns {string}
 */
export function makeNonce(cryptoImpl) {
  const bytes = cryptoImpl.getRandomValues(new Uint8Array(24));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * HMAC-SHA256(token, context + nonce) as lowercase hex (Web Crypto).
 * @param {Crypto} cryptoImpl
 * @param {string} token
 * @param {string} nonce
 * @returns {Promise<string>}
 */
export async function tokenProof(cryptoImpl, token, nonce) {
  const enc = new TextEncoder();
  const key = await cryptoImpl.subtle.importKey(
    'raw',
    enc.encode(token),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  const sig = await cryptoImpl.subtle.sign('HMAC', key, enc.encode(HEALTH_PROOF_CONTEXT + nonce));
  return Array.from(new Uint8Array(sig), (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Before sending the token or any data: the server must prove it knows the token (so a program
 * squatting on the port is refused), report auth_required, and reject a request without it.
 * Neither request carries the token.
 * @param {(url: string, init?: object) => Promise<Response>} fetchFn - plain fetch (no token)
 * @param {string} baseUrl
 * @param {string | null | undefined} token
 * @param {Crypto} cryptoImpl
 * @returns {Promise<{ ok: boolean, reason?: 'no-token' | 'health' | 'proof' | 'probe' }>}
 */
export async function verifyBackendEnforcesAuth(fetchFn, baseUrl, token, cryptoImpl) {
  if (!token) return { ok: false, reason: 'no-token' };
  const nonce = makeNonce(cryptoImpl);
  const health = await fetchFn(`${baseUrl}/health?nonce=${nonce}`);
  const body = await health.json().catch(() => ({}));
  if (body?.auth_required !== true) return { ok: false, reason: 'health' };
  const expected = await tokenProof(cryptoImpl, token, nonce);
  if (typeof body.token_proof !== 'string' || body.token_proof !== expected) {
    return { ok: false, reason: 'proof' };
  }
  const probe = await fetchFn(`${baseUrl}/api/datasets`);
  if (probe.status !== 401) return { ok: false, reason: 'probe' };
  return { ok: true };
}

/**
 * Drop events a reopened stream replays. The server sends a query's events from the start on
 * every connection, in order, so the first ``delivered`` events of a new connection are repeats.
 * @returns {{ startConnection: () => void, shouldDeliver: () => boolean }}
 */
export function createReplayFilter() {
  let delivered = 0;
  let seenThisConnection = 0;
  return {
    startConnection() {
      seenThisConnection = 0;
    },
    shouldDeliver() {
      seenThisConnection += 1;
      if (seenThisConnection <= delivered) return false;
      delivered += 1;
      return true;
    },
  };
}

/** How many times a dropped query stream is reopened before giving up. */
export const MAX_STREAM_RECONNECTS = 3;

/**
 * After a stream drops, decide from GET /api/queries/{id}'s status code what to do. The server
 * replays a query's events from the start, so reopening the stream also delivers a result that
 * finished while the connection was down.
 * @param {number} statusCode
 * @param {number} reconnects - reconnects already made
 * @returns {'reconnect' | 'gone' | 'fail'}
 */
export function decideAfterDrop(statusCode, reconnects) {
  if (statusCode === 404) return 'gone';
  if (statusCode >= 200 && statusCode < 300 && reconnects < MAX_STREAM_RECONNECTS)
    return 'reconnect';
  return 'fail';
}

/**
 * Merge a bearer token into request headers (no header when the token is empty: dev mode).
 * @param {string | null | undefined} token
 * @param {Record<string, string>} [headers]
 * @returns {Record<string, string>}
 */
export function withAuth(token, headers = {}) {
  return token ? { ...headers, Authorization: `Bearer ${token}` } : { ...headers };
}

/**
 * Split buffered SSE text into complete events. Returns parsed `data:` payloads and the
 * unfinished tail to keep for the next chunk.
 * @param {string} buffer
 * @returns {{ events: string[], rest: string }}
 */
export function parseSseBuffer(buffer) {
  const normalized = buffer.replace(/\r\n/g, '\n');
  const blocks = normalized.split('\n\n');
  const rest = blocks.pop() ?? '';
  const events = [];
  for (const block of blocks) {
    const data = block
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).replace(/^ /, ''))
      .join('\n');
    if (data) events.push(data);
  }
  return { events, rest };
}

/**
 * Read an SSE response body to the end, calling onData for each `data:` payload.
 * @param {ReadableStream<Uint8Array>} body
 * @param {(data: string) => void} onData
 * @returns {Promise<void>}
 */
export async function readSseStream(body, onData) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { events, rest } = parseSseBuffer(buffer);
    buffer = rest;
    events.forEach(onData);
  }
  const { events } = parseSseBuffer(`${buffer}${decoder.decode()}\n\n`);
  events.forEach(onData);
}

/**
 * Human-readable message from a FastAPI error body. `detail` is a string for HTTPException and a
 * list of {msg, ...} for request validation (422).
 * @param {unknown} data - Parsed JSON body (or {} when the body was not JSON)
 * @param {string} fallback
 * @returns {string}
 */
export function apiErrorMessage(data, fallback) {
  const detail = data?.detail;
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail)) {
    const msgs = detail.map((d) => d?.msg).filter((m) => typeof m === 'string' && m);
    if (msgs.length) return msgs.join('; ');
  }
  if (typeof data?.message === 'string' && data.message) return data.message;
  return fallback;
}

/** 404 / 409 on accept/reject/revert: the patch was removed or decided elsewhere. */
const STALE_DECISION_STATUSES = new Set([404, 409]);

/**
 * Outcome of an enrichment accept/reject/revert call. A stale patch (404/409) is returned as
 * { success: false, stale: true, message } so the UI can refresh; other failures throw.
 * (A plain object, because custom Error fields do not survive contextBridge.)
 * @param {{ ok: boolean, status: number, statusText?: string }} response
 * @param {unknown} data - Parsed JSON body
 * @param {string} verb - "Accept" | "Reject" | "Revert"
 * @returns {{ success: boolean, message: string, stale?: boolean }}
 */
export function enrichmentDecisionResult(response, data, verb) {
  if (response.ok && data?.success !== false) return data;
  const fallback = `${verb} failed${response.statusText ? `: ${response.statusText}` : ''}`;
  const message = apiErrorMessage(data, fallback);
  if (STALE_DECISION_STATUSES.has(response.status)) {
    return { success: false, stale: true, message };
  }
  throw new Error(message);
}
