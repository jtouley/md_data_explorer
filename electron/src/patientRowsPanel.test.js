// SPDX-License-Identifier: MIT
/** @vitest-environment jsdom */
import { describe, test, expect, vi } from 'vitest';
import {
  PATIENT_ROWS_BODY,
  PATIENT_ROWS_NO,
  PATIENT_ROWS_TITLE,
  PATIENT_ROWS_YES,
  closePatientRowsQuestion,
  normalizePatientRowsQuestion,
  patientRowsAnsweredText,
  renderPatientRowsQuestion,
} from './patientRowsPanel.js';

function setup() {
  const container = document.createElement('div');
  container.className = 'hidden';
  document.body.replaceChildren(container);
  const onAnswer = vi.fn();
  const api = renderPatientRowsQuestion(container, { onAnswer });
  const q = (id) => container.querySelector(`[data-testid="${id}"]`);
  return { container, onAnswer, api, q };
}

describe('renderPatientRowsQuestion', () => {
  test('asks the question, says what each answer changes, and labels the group', () => {
    // Arrange / Act
    const { container, q } = setup();
    const panel = q('patient-rows-question');

    // Assert
    expect(container.classList.contains('hidden')).toBe(false);
    expect(panel.getAttribute('role')).toBe('group');
    expect(document.getElementById(panel.getAttribute('aria-labelledby')).textContent).toBe(
      PATIENT_ROWS_TITLE
    );
    expect(document.getElementById(panel.getAttribute('aria-describedby')).textContent).toBe(
      PATIENT_ROWS_BODY
    );
    expect(q('patient-rows-yes').textContent).toBe(PATIENT_ROWS_YES);
    expect(q('patient-rows-no').textContent).toBe(PATIENT_ROWS_NO);
    expect(PATIENT_ROWS_BODY).toMatch(/count rows/);
    expect(PATIENT_ROWS_BODY).toMatch(/count patients/);
  });

  test('focuses the question, not Yes, so a stray Enter answers nothing', () => {
    // Arrange / Act
    const { onAnswer, q } = setup();
    document.activeElement.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })
    );

    // Assert
    expect(document.activeElement).toBe(q('patient-rows-question'));
    expect(onAnswer).not.toHaveBeenCalled();
  });

  test('neither answer is styled as the default', () => {
    // Arrange / Act
    const { q } = setup();

    // Assert
    expect(q('patient-rows-yes').className).toBe(q('patient-rows-no').className);
  });

  test('without takeFocus, focus stays where the user left it', () => {
    // Arrange
    const input = document.createElement('input');
    const container = document.createElement('div');
    document.body.replaceChildren(input, container);
    input.focus();

    // Act
    const api = renderPatientRowsQuestion(container, { onAnswer: vi.fn(), takeFocus: false });
    const before = document.activeElement;
    api.focus();

    // Assert
    expect(before).toBe(input);
    expect(document.activeElement).toBe(container.firstElementChild);
  });

  test.each([
    ['patient-rows-yes', true],
    ['patient-rows-no', false],
  ])('%s sends its answer once', (testid, expected) => {
    // Arrange
    const { onAnswer, q } = setup();

    // Act
    q(testid).click();

    // Assert
    expect(onAnswer).toHaveBeenCalledTimes(1);
    expect(onAnswer).toHaveBeenCalledWith(expected);
  });

  test('busy disables both buttons; an error is shown as an alert and cleared on the next answer', () => {
    // Arrange
    const { api, container, q } = setup();
    const error = container.querySelector('[role="alert"]');

    // Act
    api.setBusy(true);
    const disabledWhileBusy = [q('patient-rows-yes').disabled, q('patient-rows-no').disabled];
    api.setBusy(false);
    api.setError('Your answer wasn’t saved.');
    const shown = !error.classList.contains('hidden');
    q('patient-rows-no').click();

    // Assert
    expect(disabledWhileBusy).toEqual([true, true]);
    expect(shown).toBe(true);
    expect(error.classList.contains('hidden')).toBe(true);
  });

  test('close empties and hides the container', () => {
    // Arrange
    const { container } = setup();

    // Act
    closePatientRowsQuestion(container);

    // Assert
    expect(container.childElementCount).toBe(0);
    expect(container.classList.contains('hidden')).toBe(true);
  });
});

describe('normalizePatientRowsQuestion', () => {
  test.each([
    [{ status: 'pending', confirmation: ' abc ' }, { confirmation: 'abc' }],
    [{ status: 'pending', confirmation: 7 }, null],
    [{ confirmation: 'abc' }, null],
    [null, null],
  ])('keeps only a pending question with a token (%#)', (raw, expected) => {
    // Arrange / Act
    const result = normalizePatientRowsQuestion(raw);

    // Assert
    expect(result).toEqual(expected);
  });
});

describe('patientRowsAnsweredText', () => {
  test.each([
    ['patients', 'Ward 7', 'Answers on “Ward 7” now count patients, one per row.'],
    ['rows', 'Ward 7', 'Answers on “Ward 7” keep counting rows.'],
    ['patients', '  ', 'Answers on this cohort now count patients, one per row.'],
  ])('%s for %j', (counts, name, expected) => {
    // Arrange / Act
    const text = patientRowsAnsweredText(counts, name);

    // Assert
    expect(text).toBe(expected);
  });
});
