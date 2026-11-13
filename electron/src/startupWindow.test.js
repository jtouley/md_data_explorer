import { describe, test, expect } from 'vitest';
import { STARTUP_TITLE, startupPageHtml, startupPageUrl } from './startupWindow.js';

describe('startup window page', () => {
  test('is self-contained: no script, no network (CSP default-src none)', () => {
    // Act
    const html = startupPageHtml();

    // Assert
    expect(html).toContain(`content="default-src 'none'; style-src 'unsafe-inline'"`);
    expect(html).not.toMatch(/<script|src=|href=|url\(/i);
  });

  test.each([
    ['--bg', '#F7F6F3', '#161513'],
    ['--text-1', '#1C1B19', '#ECEAE5'],
    ['--text-2', '#4A4844', '#BDB9B1'],
    ['--accent', '#3F6E4E', '#7FB08E'],
  ])('uses the DESIGN.md %s token in light and dark', (token, light, dark) => {
    const html = startupPageHtml();
    expect(html).toContain(`${token}: ${light}`);
    expect(html.split('prefers-color-scheme: dark')[1]).toContain(`${token}: ${dark}`);
  });

  test('announces progress politely and stops animating under reduced motion', () => {
    const html = startupPageHtml();
    expect(html).toContain('role="status" aria-live="polite"');
    expect(html).toContain('prefers-reduced-motion: reduce');
    expect(html).toContain(STARTUP_TITLE);
  });

  test('round-trips through its data: URL', () => {
    const url = startupPageUrl();
    expect(url.startsWith('data:text/html;charset=utf-8,')).toBe(true);
    expect(decodeURIComponent(url.split(',').slice(1).join(','))).toBe(startupPageHtml());
  });
});
