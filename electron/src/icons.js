// SPDX-License-Identifier: MIT
/**
 * The renderer's one icon set: 16px, 1.5px stroke, round caps, currentColor.
 * Drawn for this app from basic shapes (no third-party icon assets).
 */

const PATHS = {
  plus: '<path d="M8 3.5v9M3.5 8h9"/>',
  more: '<circle cx="3.5" cy="8" r=".9" fill="currentColor"/><circle cx="8" cy="8" r=".9" fill="currentColor"/><circle cx="12.5" cy="8" r=".9" fill="currentColor"/>',
  close: '<path d="M4 4l8 8M12 4l-8 8"/>',
  send: '<path d="M8 13V3.5M4 7.5l4-4 4 4"/>',
  stop: '<rect x="4.5" y="4.5" width="7" height="7" rx="1.2" fill="currentColor" stroke="none"/>',
  info: '<circle cx="8" cy="8" r="6"/><path d="M8 7.2v3.6"/><circle cx="8" cy="5.2" r=".5" fill="currentColor"/>',
  alert:
    '<path d="M8 2.5l6 10.5H2z"/><path d="M8 6.5v3"/><circle cx="8" cy="11.3" r=".5" fill="currentColor"/>',
  clock: '<circle cx="8" cy="8" r="6"/><path d="M8 4.8V8l2.2 1.4"/>',
  check: '<path d="M3.5 8.5l3 3 6-7"/>',
  trash: '<path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5"/>',
  panel: '<rect x="2.5" y="3" width="11" height="10" rx="1.5"/><path d="M9.5 3v10"/>',
  sun: '<circle cx="8" cy="8" r="2.8"/><path d="M8 1.8v1.4M8 12.8v1.4M1.8 8h1.4M12.8 8h1.4M3.6 3.6l1 1M11.4 11.4l1 1M3.6 12.4l1-1M11.4 4.6l1-1"/>',
  moon: '<path d="M12.8 9.6A5.2 5.2 0 0 1 6.4 3.2a5.2 5.2 0 1 0 6.4 6.4z"/>',
  monitor: '<rect x="2" y="3" width="12" height="8" rx="1.2"/><path d="M6 13.5h4M8 11v2.5"/>',
};

/**
 * Inline SVG markup for an icon (decorative: hidden from assistive technology).
 * @param {keyof typeof PATHS} name
 * @param {number} [size]
 * @returns {string}
 */
export function icon(name, size = 16) {
  const body = PATHS[name];
  if (!body) throw new Error(`Unknown icon: ${name}`);
  return (
    `<svg class="icon icon-${name}" width="${size}" height="${size}" viewBox="0 0 16 16" ` +
    'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
    `stroke-linejoin="round" aria-hidden="true" focusable="false">${body}</svg>`
  );
}

export const ICON_NAMES = Object.freeze(Object.keys(PATHS));
