// SPDX-License-Identifier: MIT
/** @vitest-environment jsdom */
import { describe, expect, it } from 'vitest';
import { ICON_NAMES, icon } from './icons.js';

describe('icon', () => {
  it.each(ICON_NAMES)('icon_%s_isDecorativeSvg', (name) => {
    // Arrange
    const host = document.createElement('div');
    // Act
    host.innerHTML = icon(name);
    // Assert
    const svg = host.querySelector('svg');
    expect(svg).toBeTruthy();
    expect(svg.getAttribute('aria-hidden')).toBe('true');
    expect(svg.getAttribute('stroke')).toBe('currentColor');
  });

  it('icon_size_setsWidthAndHeight', () => {
    expect(icon('plus', 20)).toContain('width="20" height="20"');
  });

  it('icon_unknown_throws', () => {
    expect(() => icon('rocket')).toThrow('Unknown icon: rocket');
  });
});
