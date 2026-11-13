// SPDX-License-Identifier: MIT
/** @vitest-environment jsdom */
import { describe, expect, it, vi } from 'vitest';
import {
  answerAnnouncement,
  buildHeldCard,
  buildNotice,
  messageShell,
  renderAnswerCard,
} from './answerCard.js';
import { HIDDEN_CELL_FOOTNOTE } from './resultPresentation.js';

// Synthetic answers in the API's exact formats (no real data).
const FILTERED_COUNT = {
  intentType: 'count',
  title: 'Counts',
  claim: 'Total count hidden.',
  planLine:
    'COUNT · outcome none · grouping none · filters age > 83 · counting distinct patient_id · test count',
};
const COMPARE = {
  intentType: 'compare_groups',
  title: 'Group comparison',
  claim:
    'Mean ldl was 16.97 higher in placebo than drug (95% CI 13.96 to 19.99; Welch t-test p<0.001; n=377). Unadjusted.',
  planLine:
    'COMPARE_GROUPS · outcome ldl · grouping treatment · filters none · test group comparison',
  table: {
    columns: ['treatment', 'mean ldl', 'n'],
    rows: [
      { treatment: 'drug', 'mean ldl': 121.41333, n: 206 },
      { treatment: 'placebo', 'mean ldl': 138.38812, n: 171 },
    ],
  },
};
const COHORT_COLUMNS = ['patient_id', 'age', 'sex', 'treatment', 'ldl', 'hdl', 'diabetes'];

const testid = (root, id) => root.querySelector(`[data-testid="${id}"]`);

describe('renderAnswerCard: what was computed (H1)', () => {
  it('renderAnswerCard_filteredCount_showsLimitAndUnitUnderTheClaim', () => {
    // Arrange / Act
    const card = renderAnswerCard(FILTERED_COUNT);

    // Assert
    const plan = testid(card, 'answer-plan');
    expect(plan).not.toBeNull();
    expect(plan.textContent).toContain('Age over 83');
    expect(plan.textContent).toContain('Patients (each Patient ID once)');
    expect(plan.closest('details')).toBeNull();
    const order = [...card.children].map((n) => n.dataset.testid || n.className);
    expect(order.indexOf('result-claim')).toBeLessThan(order.indexOf('answer-plan'));
  });

  it('renderAnswerCard_rowsCounted_saysRows', () => {
    // Arrange / Act
    const card = renderAnswerCard({
      ...FILTERED_COUNT,
      claim: 'Total count: 400.',
      planLine: 'COUNT · outcome none · grouping none · filters none · test count',
    });

    // Assert
    const plan = testid(card, 'answer-plan');
    expect(plan.textContent).toContain('All rows');
    expect(plan.textContent).toContain('Rows');
    expect(plan.textContent).not.toMatch(/Not set|None|Test/);
  });

  it('renderAnswerCard_noPlanLine_drawsNoPlan', () => {
    // Arrange / Act
    const card = renderAnswerCard({ ...COMPARE, planLine: undefined });

    // Assert
    expect(testid(card, 'answer-plan')).toBeNull();
  });
});

describe('renderAnswerCard: hidden counts (H2, M5)', () => {
  it('renderAnswerCard_countHiddenOnlyInTheClaim_showsTheFootnote', () => {
    // Arrange / Act
    const card = renderAnswerCard(FILTERED_COUNT);

    // Assert
    const foot = testid(card, 'hidden-cell-footnote');
    expect(foot).not.toBeNull();
    expect(foot.textContent).toContain(HIDDEN_CELL_FOOTNOTE);
  });

  it('renderAnswerCard_nothingHidden_noFootnote', () => {
    // Arrange / Act
    const card = renderAnswerCard(COMPARE);

    // Assert
    expect(testid(card, 'hidden-cell-footnote')).toBeNull();
  });

  it('renderAnswerCard_everyCellHidden_drawsNoTableButKeepsTheFootnote', () => {
    // Arrange
    const result = {
      intentType: 'count',
      title: 'Counts',
      claim: 'Count by outcome and sex (n hidden).',
      table: {
        columns: ['outcome', 'F', 'M'],
        rows: [
          { outcome: 'outcome=0', F: 'suppressed', M: 'suppressed' },
          { outcome: 'outcome=1', F: '<11', M: 'suppressed' },
        ],
      },
    };

    // Act
    const card = renderAnswerCard(result);

    // Assert
    expect(card.querySelector('table')).toBeNull();
    expect(testid(card, 'hidden-cell-footnote')).not.toBeNull();
  });

  it('renderAnswerCard_tableNumbers_useTheClaimsPrecisionNotThePValues', () => {
    // Arrange / Act
    const card = renderAnswerCard(COMPARE);

    // Assert
    const cells = [...card.querySelectorAll('tbody td')].map((td) => td.textContent);
    expect(cells).toEqual(['121.41', '206', '138.39', '171']);
  });

  it('renderAnswerCard_levelLabels_arePlain', () => {
    // Arrange
    const result = {
      intentType: 'compare_groups',
      title: 'Group comparison',
      claim: 'Outcome differed by sex (chi-square p=0.120; n=388).',
      table: {
        columns: ['outcome', 'F', 'M'],
        rows: [
          { outcome: 'outcome=0', F: 150, M: 140 },
          { outcome: 'outcome=1', F: 50, M: 48 },
        ],
      },
    };

    // Act
    const card = renderAnswerCard(result);

    // Assert
    const rowHeads = [...card.querySelectorAll('tbody th')].map((th) => th.textContent);
    expect(rowHeads).toEqual(['Outcome: 0', 'Outcome: 1']);
  });
});

describe('renderAnswerCard: table guard (B1)', () => {
  it('renderAnswerCard_moreThanTwentyLevels_refusesToDrawTheTable', () => {
    // Arrange: a cross-tab whose 289 column headers are single patients' values
    const columns = ['treatment', ...Array.from({ length: 289 }, (_, i) => `ldl=${80 + i / 10}`)];
    const row = (name) => [name, ...columns.slice(1).map((_, i) => (i % 2 ? 0 : 'suppressed'))];
    const result = {
      intentType: 'compare_groups',
      title: 'Group comparison',
      claim: 'No difference (chi-square p=0.4; n=377).',
      table: { columns, rows: [row('drug'), row('placebo')] },
    };

    // Act
    const card = renderAnswerCard(result);

    // Assert
    expect(card.querySelector('table')).toBeNull();
    expect(card.textContent).not.toMatch(/ldl=|80\.9/);
    const note = testid(card, 'table-withheld');
    expect(note.textContent).toBe(
      'The table isn’t shown because it has 289 groups, and groups that small can describe single patients. Ask again with fewer groups, for example with a measurement put into ranges.'
    );
  });

  it('renderAnswerCard_twentyLevels_isDrawn', () => {
    // Arrange
    const rows = Array.from({ length: 20 }, (_, i) => [`site ${i}`, 20 + i]);
    const result = {
      intentType: 'count',
      title: 'Counts',
      claim: 'n=790.',
      table: { columns: ['site', 'n'], rows },
    };

    // Act
    const card = renderAnswerCard(result);

    // Assert
    expect(card.querySelector('table')).not.toBeNull();
  });
});

describe('answer structure for screen readers (M3)', () => {
  it('renderAnswerCard_claimIsAHeading', () => {
    // Arrange / Act
    const card = renderAnswerCard(COMPARE);

    // Assert
    const claim = testid(card, 'result-claim');
    expect(claim.tagName).toBe('H3');
    expect(claim.textContent).toBe('Mean ldl was 16.97 higher in placebo than drug.');
  });

  it('renderAnswerCard_tableRegionsGetUniqueLabels', () => {
    // Arrange
    const a = renderAnswerCard(COMPARE);
    const b = renderAnswerCard({ ...COMPARE, claim: 'Mean hdl was 2.10 higher in drug (n=377).' });

    // Act
    const labelA = testid(a, 'result-table-region').getAttribute('aria-label');
    const labelB = testid(b, 'result-table-region').getAttribute('aria-label');

    // Assert
    expect(labelA).not.toBe(labelB);
    expect(labelA).toBe('Group comparison table: Mean ldl was 16.97 higher in placebo than drug.');
  });

  it('messageShell_answer_isAnArticleLabelledByItsQuestion', () => {
    // Arrange
    const question = messageShell({ id: 'm1', role: 'user' });
    document.body.appendChild(question);

    // Act
    const answer = messageShell({ id: 'm2', role: 'assistant', questionId: 'm1' });

    // Assert
    expect(answer.tagName).toBe('ARTICLE');
    const labelId = answer.getAttribute('aria-labelledby');
    expect(document.getElementById(labelId)).toBe(question.querySelector('.message-content'));
  });

  it('answerAnnouncement_readsTheClaim', () => {
    expect(answerAnnouncement(COMPARE)).toBe(
      'Answer: Mean ldl was 16.97 higher in placebo than drug.'
    );
    expect(answerAnnouncement({ title: 'Counts' })).toBe('Answer ready.');
  });
});

describe('buildNotice: no raw engine text on screen (H3)', () => {
  it('buildNotice_withDetail_offersCopyDetailsButNeverShowsTheText', () => {
    // Arrange
    const copy = vi.fn();
    const raw = "Execution error: 'StringColumn' object has no attribute 'mean'";

    // Act
    const notice = buildNotice('error', 'The analysis couldn’t finish this question.', {
      hint: 'Your question is back in the box below so you can change it.',
      detail: raw,
      copy,
    });
    const button = notice.querySelector('button');
    button.click();

    // Assert
    expect(notice.textContent).not.toContain('StringColumn');
    expect(button.textContent).toBe('Copy details');
    expect(copy).toHaveBeenCalledWith(raw);
  });
});

describe('buildHeldCard: next step (H5)', () => {
  const held = (failureReason) => ({
    awaitingConfirmation: true,
    planLine:
      'DESCRIBE · outcome none · grouping sex · statistic mean · filters none · limits in order 1) patient_id = S0001 · test count',
    planHash: 'h',
    failureReason,
    pendingQuery: 'average cholesterol by sex',
  });

  it('buildHeldCard_droppedWords_suggestsColumns_andNotWhatIMeantIsPrimary', () => {
    // Arrange
    const reason =
      "I couldn't place the words 'average' and 'cholesterol', so they are not in the plan.";

    // Act
    const card = buildHeldCard(held(reason), {
      columns: COHORT_COLUMNS,
      onConfirm: vi.fn(),
      onReject: vi.fn(),
    });

    // Assert
    expect(testid(card, 'held-next-step').textContent).toContain('This cohort has LDL and HDL.');
    const [first, second] = card.querySelectorAll('.result-confirm-actions button');
    expect(first.dataset.testid).toBe('reject-query-btn');
    expect(first.classList.contains('btn-primary')).toBe(true);
    expect(second.classList.contains('btn-secondary')).toBe(true);
  });

  it('buildHeldCard_planList_hasNoInternalPhrasing', () => {
    // Arrange / Act
    const card = buildHeldCard(held('Confirmation required before execution.'), {
      columns: COHORT_COLUMNS,
      onConfirm: vi.fn(),
      onReject: vi.fn(),
    });

    // Assert
    const list = card.querySelector('.plan-list').textContent;
    expect(list).toContain('Not set');
    expect(list).not.toMatch(/\bNone\b|Test\s*count|Also|statistic mean|limits in order|1\)/);
    const [first] = card.querySelectorAll('.result-confirm-actions button');
    expect(first.dataset.testid).toBe('confirm-query-btn');
    expect(first.classList.contains('btn-primary')).toBe(true);
  });

  it('buildHeldCard_buttons_callTheHandlersWithTheCard', () => {
    // Arrange
    const onConfirm = vi.fn();
    const onReject = vi.fn();
    const card = buildHeldCard(held('Confirmation required before execution.'), {
      columns: [],
      onConfirm,
      onReject,
    });

    // Act
    testid(card, 'confirm-query-btn').click();
    testid(card, 'reject-query-btn').click();

    // Assert
    expect(onConfirm).toHaveBeenCalledWith(card);
    expect(onReject).toHaveBeenCalledWith(card);
  });
});
