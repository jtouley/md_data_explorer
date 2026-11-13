// SPDX-License-Identifier: MIT
/**
 * The small window shown while the backend starts (main process). A self-contained data: page:
 * no script, no network (its CSP is default-src 'none'), tokens and type from docs/app/DESIGN.md
 * (--bg, --text-1, --text-2, --accent; --fs-body, --fs-sm; dots pulse on a 1.2 s cycle, none
 * under reduced motion).
 */

export const STARTUP_WINDOW_SIZE = Object.freeze({ width: 420, height: 200 });
export const STARTUP_TITLE = 'Starting Clinical Analytics';
export const STARTUP_DETAIL = 'Getting the analysis engine on this computer ready.';

/** @returns {string} the page's HTML */
export function startupPageHtml() {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>${STARTUP_TITLE}</title>
<style>
  :root { color-scheme: light dark; --bg: #F7F6F3; --text-1: #1C1B19; --text-2: #4A4844; --accent: #3F6E4E; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #161513; --text-1: #ECEAE5; --text-2: #BDB9B1; --accent: #7FB08E; }
  }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text-1); }
  body { display: flex; align-items: center; justify-content: center;
    font-family: "Source Sans 3", "Source Sans Pro", system-ui, sans-serif; }
  main { padding: 24px; max-width: 340px; }
  h1 { margin: 0 0 8px; font-size: 16px; line-height: 25px; font-weight: 600; }
  p { margin: 0; font-size: 13px; line-height: 18px; color: var(--text-2); }
  .dots { display: inline-flex; gap: 4px; margin-left: 8px; vertical-align: middle; }
  .dots span { width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
    animation: pulse 1.2s infinite ease-in-out; }
  .dots span:nth-child(2) { animation-delay: .2s; }
  .dots span:nth-child(3) { animation-delay: .4s; }
  @keyframes pulse { 0%, 80%, 100% { opacity: .25; } 40% { opacity: 1; } }
  @media (prefers-reduced-motion: reduce) { .dots span { animation: none; opacity: .6; } }
</style>
</head>
<body>
<main role="status" aria-live="polite">
  <h1 data-testid="startup-title">${STARTUP_TITLE}<span class="dots" aria-hidden="true"><span></span><span></span><span></span></span></h1>
  <p>${STARTUP_DETAIL}</p>
</main>
</body>
</html>`;
}

/** @returns {string} the page as a data: URL */
export function startupPageUrl() {
  return `data:text/html;charset=utf-8,${encodeURIComponent(startupPageHtml())}`;
}
