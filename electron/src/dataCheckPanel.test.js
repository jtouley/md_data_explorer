// SPDX-License-Identifier: MIT
// @vitest-environment jsdom
import { describe, expect, test } from 'vitest';
import {
  DATA_CHECK_LOADING,
  DATA_CHECK_NOTHING_FLAGGED,
  DATA_CHECK_HINT,
  dataCheckNotRunText,
  dataCheckStatusNote,
  dataCheckNeedsReview,
  normalizeQuality,
  renderDataCheck,
  DATA_FACTS_HINT,
  normalizeFacts,
  renderFacts,
} from './dataCheckPanel.js';

function parts() {
  return { hint: document.createElement('p'), list: document.createElement('ul') };
}

describe('normalizeQuality', () => {
  test.each([
    [null, null],
    ['text', null],
    [{ checks: [] }, null],
    [
      { status: 'not_run', reason: '  disk full ' },
      { status: 'not_run', reason: 'disk full' },
    ],
    [{ status: 'not_run' }, { status: 'not_run', reason: null }],
    [
      { status: 'ran', summary: [] },
      { status: 'ran', lines: [], count: 0 },
    ],
    [{ summary: ['a', '', 3, 'b'] }, { status: 'ran', lines: ['a', 'b'], count: 2 }],
    [
      { summary: ['a', 'b', '…and 4 more'] },
      { status: 'ran', lines: ['a', 'b', '…and 4 more'], count: 6 },
    ],
  ])('normalizes %j', (raw, expected) => {
    expect(normalizeQuality(raw)).toEqual(expected);
  });
});

describe('dataCheckStatusNote', () => {
  test.each([
    [null, null],
    [{ status: 'not_run', reason: 'x' }, 'Data check could not run.'],
    [{ status: 'ran', lines: [], count: 0 }, 'Data check found nothing to flag.'],
    [
      { status: 'ran', lines: ['a'], count: 1 },
      'Data check found 1 thing to review in cohort details.',
    ],
    [
      { status: 'ran', lines: ['a', '…and 4 more'], count: 5 },
      'Data check found 5 things to review in cohort details.',
    ],
  ])('%j -> %s', (check, expected) => {
    expect(dataCheckStatusNote(check)).toBe(expected);
  });
});

describe('renderDataCheck', () => {
  test('lists the API lines as text, with the data-check footnote', () => {
    const p = parts();
    renderDataCheck(p, { status: 'ran', lines: ['ldl: 5.8% missing', '<b>x</b>'], count: 2 });
    expect([...p.list.querySelectorAll('li')].map((li) => li.textContent)).toEqual([
      'ldl: 5.8% missing',
      '<b>x</b>',
    ]);
    expect(p.list.querySelector('b')).toBeNull();
    expect(p.hint.textContent).toBe(DATA_CHECK_HINT);
  });

  test.each([
    ['loading', DATA_CHECK_LOADING],
    [{ status: 'ran', lines: [], count: 0 }, DATA_CHECK_NOTHING_FLAGGED],
    [
      { status: 'not_run', reason: 'The saved data contract for this cohort is damaged.' },
      'Data check could not run: The saved data contract for this cohort is damaged. You can still ask questions.',
    ],
    [null, 'Data check could not run. You can still ask questions.'],
  ])('says in words what happened (%#), clearing earlier lines', (check, text) => {
    const p = parts();
    renderDataCheck(p, { status: 'ran', lines: ['old'], count: 1 });
    renderDataCheck(p, check);
    expect(p.hint.textContent).toBe(text);
    expect(p.list.children).toHaveLength(0);
  });

  test('dataCheckNotRunText without a reason', () => {
    expect(dataCheckNotRunText(null)).toBe(
      'Data check could not run. You can still ask questions.'
    );
  });
});

describe('dataCheckNeedsReview (L1)', () => {
  test.each([
    [null, false],
    [{ status: 'not_run', reason: null }, false],
    [{ status: 'ran', lines: [], count: 0 }, false],
    [{ status: 'ran', lines: ['a'], count: 1 }, true],
  ])('%j -> %s', (check, expected) => {
    expect(dataCheckNeedsReview(check)).toBe(expected);
  });
});

describe('facts from the data contract', () => {
  test.each([
    [null, []],
    ['text', []],
    [
      ['a', '', 2, '  b '],
      ['a', 'b'],
    ],
  ])('normalizeFacts(%j)', (raw, expected) => {
    expect(normalizeFacts(raw)).toEqual(expected);
  });

  test('renders each fact as a list item under its hint, in clinical words', () => {
    const p = parts();
    renderFacts(p, [
      'Each line is one hydrotherapy session; 58 patients (client_ref names the patient) over 913 lines.',
      '999 in end_weight means not recorded.',
    ]);
    expect(p.hint.textContent).toBe(DATA_FACTS_HINT);
    expect([...p.list.children].map((li) => li.textContent)).toEqual([
      'Each line is one hydrotherapy session; 58 patients (client_ref names the patient) over 913 lines.',
      '999 in end_weight means not recorded.',
    ]);
    expect(p.list.classList.contains('hidden')).toBe(false);
  });

  test('hides the facts when there are none', () => {
    const p = parts();
    renderFacts(p, []);
    expect(p.hint.classList.contains('hidden')).toBe(true);
    expect(p.list.children).toHaveLength(0);
  });
});
