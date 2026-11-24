// SPDX-License-Identifier: MIT
// @vitest-environment jsdom
import { describe, expect, test, vi } from 'vitest';
import {
  FILE_QUESTIONS_HINT,
  FILE_QUESTIONS_NONE,
  FILE_QUESTIONS_SAVED,
  buildWaitingCard,
  normalizeFileQuestions,
  renderFileQuestions,
  showAnswerOutcome,
  waitingText,
} from './fileQuestionsPanel.js';

/** One synthetic question per kind, shaped as GET /api/datasets/{id}/questions returns it. */
const KINDS = {
  rows_are_patients: '/one-row-per-patient',
  patient_key: '/answers',
  missing_code: '/missing-codes',
  date_order: '/date-readings',
  date_century: '/date-readings',
  censored_value: '/answers',
  total_row: '/answers',
  repeated_strata: '/answers',
  aggregate_unsure: '/one-row-per-patient',
};

function question(kind, route) {
  return {
    id: `${kind}:col`,
    kind,
    column: 'col',
    question: `Question of kind ${kind}?`,
    evidence: '12 lines of col hold 99.',
    answers: [
      { label: 'yes', route, body: { question_id: `${kind}:col` }, changes: 'Yes changes this.' },
      { label: 'no', route, body: { question_id: `${kind}:col` }, changes: 'No changes that.' },
    ],
  };
}

function parts() {
  return { hint: document.createElement('p'), list: document.createElement('ul') };
}

describe('normalizeFileQuestions', () => {
  test('keeps well-formed questions and only known answer routes', () => {
    const raw = {
      questions: [
        question('missing_code', '/missing-codes'),
        {
          ...question('total_row', '/answers'),
          answers: [{ label: 'x', route: '/delete', body: {} }],
        },
        { id: 1 },
      ],
    };

    const found = normalizeFileQuestions(raw);

    expect(found.map((q) => q.id)).toEqual(['missing_code:col']);
  });

  test.each([null, undefined, {}, { questions: 'x' }])('nothing for %s', (raw) => {
    expect(normalizeFileQuestions(raw)).toEqual([]);
  });
});

describe('renderFileQuestions', () => {
  test.each(Object.entries(KINDS))(
    '%s: question, count evidence, answers and notes',
    (kind, route) => {
      const p = parts();
      const questions = normalizeFileQuestions({ questions: [question(kind, route)] });

      renderFileQuestions(p, questions, { onAnswer: () => {} });

      const item = p.list.querySelector('[data-testid="file-question"]');
      expect(item.dataset.kind).toBe(kind);
      expect(item.textContent).toContain(`Question of kind ${kind}?`);
      expect(item.querySelector('[data-testid="file-question-evidence"]').textContent).toBe(
        '12 lines of col hold 99.'
      );
      expect(
        [...item.querySelectorAll('[data-testid="file-question-answer"]')].map((b) => b.textContent)
      ).toEqual(['yes', 'no']);
      expect(item.textContent).toContain('Yes changes this.');
      expect(p.hint.textContent).toBe(FILE_QUESTIONS_HINT);
    }
  );

  test('no questions says so', () => {
    const p = parts();

    renderFileQuestions(p, [], { onAnswer: () => {} });

    expect(p.hint.textContent).toBe(FILE_QUESTIONS_NONE);
    expect(p.list.children).toHaveLength(0);
  });

  test('an answer disables the buttons and passes the answer; an error enables them again', () => {
    const p = parts();
    const onAnswer = vi.fn();
    const [q] = normalizeFileQuestions({ questions: [question('total_row', '/answers')] });
    renderFileQuestions(p, [q], { onAnswer });
    const [yes] = p.list.querySelectorAll('button');

    yes.click();
    const item = p.list.querySelector('li');
    const disabled = [...item.querySelectorAll('button')].every((b) => b.disabled);
    showAnswerOutcome(item, 'Could not save.');

    expect(onAnswer).toHaveBeenCalledWith(q, q.answers[0], item);
    expect(disabled).toBe(true);
    expect([...item.querySelectorAll('button')].some((b) => b.disabled)).toBe(false);
    expect(item.textContent).toContain('Could not save.');
  });

  test('a saved answer says so', () => {
    const p = parts();
    renderFileQuestions(
      p,
      normalizeFileQuestions({ questions: [question('total_row', '/answers')] }),
      {
        onAnswer: () => {},
      }
    );
    const item = p.list.querySelector('li');

    showAnswerOutcome(item, null);

    expect(item.textContent).toContain(FILE_QUESTIONS_SAVED);
  });
});

describe('waiting on questions', () => {
  test.each([
    [1, 'Waiting on 1 question about this file'],
    [3, 'Waiting on 3 questions about this file'],
    [0, 'Waiting on 1 question about this file'],
  ])('%i -> %s', (count, text) => {
    expect(waitingText(count)).toBe(text);
  });

  test('the card links to the panel', () => {
    const onOpen = vi.fn();

    const card = buildWaitingCard({ count: 2, failureReason: 'Is each line one patient?' }, onOpen);
    card.querySelector('[data-testid="open-file-questions"]').click();

    expect(card.textContent).toContain('Waiting on 2 questions about this file');
    expect(onOpen).toHaveBeenCalledOnce();
  });
});
