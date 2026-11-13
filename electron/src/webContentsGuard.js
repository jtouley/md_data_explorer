// SPDX-License-Identifier: MIT
/**
 * Keep a window on the app's own page: no navigation elsewhere (main frame or subframe), no new
 * windows, no webviews. Kept out of main.js so it can be unit-tested without Electron.
 */
import { isTrustedRendererUrl } from './apiToken.js';

/**
 * @param {Pick<import('electron').WebContents, 'on' | 'setWindowOpenHandler'>} contents
 * @param {string | undefined} devServerUrl - Vite dev server URL, when in development
 */
// Local documents a subframe may hold without loading anything from elsewhere.
const LOCAL_FRAME_URLS = new Set(['about:blank', 'about:srcdoc']);

export function lockDownWebContents(contents, devServerUrl) {
  const blockUntrusted = (event, url) => {
    if (!isTrustedRendererUrl(url, devServerUrl)) {
      event.preventDefault();
      console.warn('Blocked navigation away from the app');
    }
  };
  contents.on('will-navigate', blockUntrusted);
  contents.on('will-redirect', blockUntrusted);
  // will-navigate covers only the main frame. An iframe gets no preload, but a compromised
  // renderer could still embed a remote document, so subframes face the same check.
  contents.on('will-frame-navigate', (details) => {
    if (LOCAL_FRAME_URLS.has(details.url)) return;
    if (!isTrustedRendererUrl(details.url, devServerUrl)) {
      details.preventDefault();
      console.warn('Blocked a frame from loading a page outside the app');
    }
  });
  contents.setWindowOpenHandler(() => ({ action: 'deny' }));
  contents.on('will-attach-webview', (event) => event.preventDefault());
}
