/**
 * The colour tokens in index.css must keep the contrast recorded in docs/app/DESIGN.md:
 * 4.5:1 for text, 3:1 for UI boundaries, in light and dark.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const css = readFileSync(path.join(here, 'index.css'), 'utf8');

/** Custom properties (hex colours only) declared in the first block matching `selector`. */
function tokens(selector) {
  const start = css.indexOf(`${selector} {`);
  if (start === -1) throw new Error(`No block for ${selector}`);
  const body = css.slice(start, css.indexOf('}', start));
  return Object.fromEntries(
    [...body.matchAll(/--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)].map((m) => [
      m[1],
      m[2].toLowerCase(),
    ])
  );
}

function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const v = parseInt(hex.slice(i, i + 2), 16) / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

const light = tokens(':root');
const darkExplicit = tokens(":root[data-theme='dark']");
const darkSystem = tokens(":root:not([data-theme='light'])");

const TEXT_PAIRS = [
  ['text-1', ['bg', 'surface-1', 'surface-2', 'sunken', 'held-bg', 'danger-bg']],
  ['text-2', ['bg', 'surface-1', 'surface-2', 'sunken', 'held-bg']],
  ['text-3', ['bg', 'surface-1', 'surface-2', 'sunken']],
  ['accent', ['bg', 'surface-1', 'surface-2', 'accent-subtle']],
  ['on-accent', ['accent', 'accent-hover']],
  ['held', ['held-bg']],
  ['notice', ['notice-bg']],
  ['danger', ['bg', 'surface-2', 'danger-bg']],
];
const UI_PAIRS = [
  ['border-strong', ['bg', 'surface-1', 'surface-2']],
  ['held-border', ['held-bg']],
];

const cases = (pairs, min) =>
  ['light', 'dark'].flatMap((theme) =>
    pairs.flatMap(([fg, bgs]) => bgs.map((bg) => [theme, fg, bg, min]))
  );

describe('design tokens', () => {
  it('darkTokens_systemAndExplicitBlocks_identical', () => {
    expect(darkSystem).toEqual(darkExplicit);
  });

  it('tokens_lightAndDark_defineSameColourNames', () => {
    const lightNames = Object.keys(light).sort();
    expect(Object.keys(darkExplicit).sort()).toEqual(lightNames);
  });

  it.each([...cases(TEXT_PAIRS, 4.5), ...cases(UI_PAIRS, 3)])(
    'contrast_%s_%s_on_%s_atLeast_%s',
    (theme, fg, bg, min) => {
      // Arrange
      const set = theme === 'light' ? light : darkExplicit;
      // Act
      const ratio = contrast(set[fg], set[bg]);
      // Assert
      expect(set[fg]).toBeDefined();
      expect(set[bg]).toBeDefined();
      expect(ratio).toBeGreaterThanOrEqual(min);
    }
  );

  it('css_hasNoHardCodedHexOutsideTokenBlocks', () => {
    // Arrange: drop the three token blocks
    const stripped = css
      .replace(/:root \{[\s\S]*?\n\}/, '')
      .replace(/:root\[data-theme='dark'\] \{[\s\S]*?\n\}/, '')
      .replace(/:root:not\(\[data-theme='light'\]\) \{[\s\S]*?\n {2}\}/, '');
    // Act
    const stray = stripped.match(/#[0-9a-fA-F]{3,8}\b/g) || [];
    // Assert
    expect(stray).toEqual([]);
  });
});
