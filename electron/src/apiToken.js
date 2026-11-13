// SPDX-License-Identifier: MIT
/**
 * Per-launch API token, app:// page origin and navigation policy (main process only).
 *
 * The FastAPI backend requires `Authorization: Bearer <token>` unless it was started in explicit
 * dev mode. The launcher (scripts/run_app.sh) generates one token per launch and passes it to
 * both processes through CLINICAL_ANALYTICS_API_TOKEN. Started on its own, main generates a fresh
 * token; it only works against a dev-mode backend, which the preload accepts only when this
 * process was also started with CLINICAL_ANALYTICS_DEV_NO_AUTH=1.
 *
 * The packaged page is served from the privileged app:// scheme (not file://): it gives the page
 * a real origin the API can allow by name, and the token goes only to that exact origin.
 */
import { randomBytes } from 'node:crypto';
import path from 'node:path';

import { parseLoopbackApiUrl } from './apiAuth.js';

export { API_TOKEN_CHANNEL } from './apiAuth.js';

export const API_TOKEN_ENV = 'CLINICAL_ANALYTICS_API_TOKEN';
export const DEV_NO_AUTH_ENV = 'CLINICAL_ANALYTICS_DEV_NO_AUTH';
export const APP_SCHEME = 'app';
export const APP_HOST = 'clinical-analytics';
export const APP_ORIGIN = `${APP_SCHEME}://${APP_HOST}`;
export const APP_ENTRY_URL = `${APP_ORIGIN}/index.html`;

/**
 * True only for an explicit developer opt-out (lets the app use an unauthenticated backend).
 * @param {Record<string, string | undefined>} env
 * @returns {boolean}
 */
export function resolveDevNoAuth(env) {
  return (env[DEV_NO_AUTH_ENV] || '').trim() === '1';
}
export const MIN_TOKEN_LENGTH = 32;

/**
 * @param {Record<string, string | undefined>} env
 * @param {(size: number) => Buffer} [random]
 * @returns {{ token: string, source: 'env' | 'generated' }}
 */
export function resolveApiToken(env, random = randomBytes) {
  const fromEnv = (env[API_TOKEN_ENV] || '').trim();
  if (fromEnv) {
    if (fromEnv.length < MIN_TOKEN_LENGTH) {
      throw new Error(`${API_TOKEN_ENV} must be at least ${MIN_TOKEN_LENGTH} characters`);
    }
    return { token: fromEnv, source: 'env' };
  }
  return { token: random(32).toString('base64url'), source: 'generated' };
}

/**
 * Only the app's own page may read the token or be navigated to: the exact app:// origin in a
 * packaged build, the exact Vite dev-server origin in development. Never file:// or a remote page.
 * @param {string | undefined} frameUrl - URL of the frame asking / being navigated to
 * @param {string | undefined} devServerUrl - Vite dev server URL, when in development
 * @returns {boolean}
 */
export function isTrustedRendererUrl(frameUrl, devServerUrl) {
  if (!frameUrl) return false;
  let url;
  try {
    url = new URL(frameUrl);
  } catch {
    return false;
  }
  if (devServerUrl) {
    try {
      return url.origin === new URL(devServerUrl).origin;
    } catch {
      return false;
    }
  }
  // Node reports origin "null" for custom schemes, so compare scheme and host directly.
  return (
    url.protocol === `${APP_SCHEME}:` && url.host === APP_HOST && !url.username && !url.password
  );
}

/**
 * Map an app:// request to a file under the renderer build, or null (wrong host, traversal).
 * @param {string} requestUrl
 * @param {string} rootDir - absolute renderer build directory
 * @returns {string | null}
 */
export function resolveAppFilePath(requestUrl, rootDir) {
  let url;
  try {
    url = new URL(requestUrl);
  } catch {
    return null;
  }
  if (url.protocol !== `${APP_SCHEME}:` || url.host !== APP_HOST) return null;
  let pathname;
  try {
    pathname = decodeURIComponent(url.pathname);
  } catch {
    return null;
  }
  if (pathname.includes('\0')) return null;
  const relative =
    pathname === '/' || pathname === '' ? 'index.html' : pathname.replace(/^\/+/, '');
  const root = path.resolve(rootDir);
  const resolved = path.resolve(root, relative);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) return null;
  return resolved;
}

/**
 * The page's Content-Security-Policy, sent as a response header by main. connect-src names the one
 * backend origin this launch uses (exact port), so page script cannot reach any other loopback
 * service (for example a local model server that could be told to fetch from the internet).
 * index.html keeps a meta CSP limited to 127.0.0.1; both apply, so the header narrows it.
 * @param {string} apiBaseUrl - http://127.0.0.1:<port>
 * @returns {string}
 */
export function contentSecurityPolicy(apiBaseUrl) {
  const parsed = parseLoopbackApiUrl(apiBaseUrl);
  if (!parsed) {
    throw new Error(
      `API address must be http://127.0.0.1:<port>, got ${JSON.stringify(apiBaseUrl)}`
    );
  }
  return [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    `connect-src http://127.0.0.1:${parsed.port}`,
  ].join('; ');
}
