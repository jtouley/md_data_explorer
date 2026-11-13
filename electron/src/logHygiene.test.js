// SPDX-License-Identifier: MIT
import { describe, test, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = (name) => readFileSync(path.join(here, name), 'utf8');

/** Every console.* call in a file, with its argument text (single- or multi-line). */
function consoleCalls(text) {
  return [...text.matchAll(/console\.(log|warn|error|info|debug)\(([\s\S]*?)\);/g)].map(
    (m) => m[2]
  );
}

describe('renderer and preload logs carry no question text or results', () => {
  test.each(['renderer.js', 'preload.js'])(
    '%s console calls never pass query text or SSE payloads',
    (file) => {
      // Arrange
      const calls = consoleCalls(source(file));

      // Act
      const leaking = calls.filter((args) =>
        /\bqueryText\b(?!\.length)|query_text|event\.data|\bdata\)\s*$|,\s*data\s*$|\bresponse\s*$/.test(
          args.trim()
        )
      );

      // Assert
      expect(calls.length).toBeGreaterThan(0);
      expect(leaking).toEqual([]);
    }
  );
});
