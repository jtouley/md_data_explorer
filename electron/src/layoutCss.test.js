// SPDX-License-Identifier: MIT
/**
 * Layout and focus rules that only CSS can carry: checked against index.css as written, and
 * in the rendered app by the Playwright specs.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const css = readFileSync(path.join(here, 'index.css'), 'utf8');
const html = readFileSync(path.join(here, '..', 'index.html'), 'utf8');

/** Body of the first rule whose selector line is exactly `selector`. */
function rule(source, selector) {
  const start = source.indexOf(`${selector} {`);
  if (start === -1) return null;
  return source.slice(start, source.indexOf('}', start));
}

/** Text of an @media block, braces balanced. */
function mediaBlock(query) {
  const start = css.indexOf(`@media ${query} {`);
  if (start === -1) return null;
  let depth = 0;
  for (let i = css.indexOf('{', start); i < css.length; i += 1) {
    if (css[i] === '{') depth += 1;
    if (css[i] === '}') depth -= 1;
    if (depth === 0) return css.slice(start, i + 1);
  }
  return null;
}

describe('composer focus (M4)', () => {
  it('queryForm_focusWithin_hasTheTwoPixelFocusRing', () => {
    // Arrange / Act
    const body = rule(css, '.query-form:focus-within');

    // Assert
    expect(body).toMatch(/outline:\s*2px solid var\(--focus-ring\)/);
  });
});

describe('narrow windows (H6)', () => {
  it('under700px_sidebarCollapsesBehindAToggle', () => {
    // Arrange / Act
    const block = mediaBlock('(max-width: 699px)');

    // Assert
    expect(block).not.toBeNull();
    expect(block).toMatch(/\.app-container\s*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\)/);
    expect(block).toMatch(
      /\.app-container:not\(\.sidebar-open\) \.sidebar\s*\{[^}]*display:\s*none/
    );
    expect(block).toMatch(/\.sidebar-toggle\s*\{[^}]*display:\s*inline-flex/);
  });

  it('wideWindows_hideTheToggle', () => {
    expect(rule(css, '.sidebar-toggle')).toMatch(/display:\s*none/);
  });

  it('html_hasTheToggleAndADismissibleModelBanner', () => {
    expect(html).toMatch(/id="sidebar-toggle"[^>]*aria-controls="session-sidebar"/);
    expect(html.match(/class="btn-icon banner-dismiss"/g)).toHaveLength(1);
  });
});

describe('narrow-window drawer and offline banner (round 2)', () => {
  /** The markup between an element's opening tag and its closing tag. */
  function element(id, tag) {
    const start = html.indexOf(`id="${id}"`);
    return html.slice(start, html.indexOf(`</${tag}>`, start));
  }

  it('offlineBanner_keepsItsRetryButton_andCannotBeDismissed', () => {
    // Arrange / Act
    const banner = element('offline-banner', 'div');

    // Assert
    expect(banner).toMatch(/id="retry-connection"/);
    expect(banner).not.toMatch(/banner-dismiss/);
  });

  it('drawer_hasACloseButtonWithAName', () => {
    // Arrange / Act
    const sidebar = element('session-sidebar', 'aside');

    // Assert
    expect(sidebar).toMatch(
      /<button[^>]*id="sidebar-close"[^>]*aria-label="Close cohort and conversations"/
    );
  });

  it('openDrawer_containsItsOwnScroll', () => {
    // Arrange / Act
    const block = mediaBlock('(max-width: 699px)');

    // Assert
    expect(block).toMatch(
      /\.app-container\.sidebar-open \.sidebar\s*\{[^}]*overscroll-behavior:\s*contain/
    );
    expect(block).toMatch(/\.sidebar-close\s*\{[^}]*display:\s*inline-flex/);
  });

  it('wideWindows_hideTheDrawerCloseButton', () => {
    expect(rule(css, '.sidebar-close')).toMatch(/display:\s*none/);
  });
});

describe('page structure (M3)', () => {
  it('html_hasOneH1', () => {
    expect(html.match(/<h1\b/g)).toHaveLength(1);
  });
});
