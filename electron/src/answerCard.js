// SPDX-License-Identifier: MIT
/**
 * The chat's message shapes (docs/app/DESIGN.md "Message states"), built with createElement
 * only. Each answer shows what was computed under its claim (the plan, with who was included and
 * whether rows or patients were counted), says when anything is hidden, and never draws a table
 * that could describe single patients.
 */
import { icon } from './icons.js';
import {
  HIDDEN_CELL_FOOTNOTE,
  TABLE_LEVEL_LIMIT,
  claimDecimals,
  classifyCell,
  holdNextStep,
  isFullyHiddenTable,
  isNumericColumn,
  isPValueColumn,
  needsHiddenFootnote,
  normalizeTableRows,
  parsePlanLine,
  plainLevelLabel,
  splitClaim,
  tableLevelCount,
  isGroupLevelTable,
  tableWithheldMessage,
} from './resultPresentation.js';

/**
 * @param {string} tag
 * @param {string} [className]
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** The id of a question's text, which labels the answer that follows it. */
export const questionTextId = (messageId) => `question-${messageId}`;

/**
 * The outer element of one chat message. An assistant message is an `article` labelled by the
 * question it answers, so a screen reader can move answer by answer and hear what each answers.
 * @param {{ id: string, role: 'user' | 'assistant', questionId?: string }} message
 * @returns {HTMLElement}
 */
export function messageShell(message) {
  const isAnswer = message.role === 'assistant';
  const shell = el(isAnswer ? 'article' : 'div', `message message-${message.role}`);
  shell.setAttribute('data-message-id', message.id);
  shell.setAttribute('data-testid', `${message.role}-message`);
  const content = el('div', 'message-content');
  if (isAnswer && message.questionId) {
    shell.setAttribute('aria-labelledby', questionTextId(message.questionId));
  } else if (!isAnswer) {
    content.id = questionTextId(message.id);
  }
  shell.appendChild(content);
  return shell;
}

/**
 * What the live region says when an answer lands: its claim, not just "Answer ready".
 * @param {{ claim?: string } | null | undefined} result
 * @returns {string}
 */
export function answerAnnouncement(result) {
  const { headline } = splitClaim(result?.claim);
  return headline ? `Answer: ${headline}` : 'Answer ready.';
}

const NOTICE_LABELS = {
  refused: { word: 'Not answered', icon: 'info' },
  error: { word: 'Something went wrong', icon: 'alert' },
  stopped: { word: 'Stopped', icon: 'stop' },
};

export const COPY_DONE = 'Details copied. Paste them to whoever supports the app.';
export const COPY_FAILED = 'Couldn’t copy the details. Restart the app and try again.';

/**
 * Refusal, error or stopped notice: a word and an icon, never colour alone. The engine's exact
 * words, when the notice reworded them, are offered for copying and never shown.
 * @param {'refused' | 'error' | 'stopped'} kind
 * @param {string} text
 * @param {{ hint?: string, detail?: string, copy?: (text: string) => Promise<unknown> | unknown }} [opts]
 *   `copy` resolves when the text is on the clipboard; a rejection or `false` means it isn't.
 * @returns {HTMLElement}
 */
export function buildNotice(kind, text, opts = {}) {
  const notice = el('div', `notice notice--${kind}`);
  notice.dataset.testid = `${kind}-notice`;
  const label = el('h3', 'state-label');
  label.innerHTML = `${icon(NOTICE_LABELS[kind].icon)}<span>${NOTICE_LABELS[kind].word}</span>`;
  notice.appendChild(label);
  if (text) notice.appendChild(el('p', 'notice-body', text));
  if (opts.hint) notice.appendChild(el('p', 'notice-hint', opts.hint));
  if (opts.detail && opts.copy) {
    const copyBtn = el('button', 'btn btn-secondary btn-sm notice-copy', 'Copy details');
    copyBtn.type = 'button';
    copyBtn.dataset.testid = 'copy-details-btn';
    const done = el('span', 'notice-copied');
    done.setAttribute('role', 'status');
    copyBtn.addEventListener('click', () => {
      done.textContent = '';
      // Say "copied" only once the clipboard has the text; otherwise say it didn't work.
      let pending;
      try {
        pending = Promise.resolve(opts.copy?.(/** @type {string} */ (opts.detail)));
      } catch (error) {
        pending = Promise.reject(error);
      }
      pending.then(
        (copied) => {
          done.textContent = copied === false ? COPY_FAILED : COPY_DONE;
        },
        () => {
          done.textContent = COPY_FAILED;
        }
      );
    });
    const row = el('div', 'notice-actions');
    row.append(copyBtn, done);
    notice.appendChild(row);
  }
  return notice;
}

/**
 * What was computed, under the claim: who was included and whether rows or patients were
 * counted are always shown (never behind a disclosure), with the outcome, groups and test.
 * @param {string} planLine
 * @returns {HTMLElement | null}
 */
export function buildAnswerPlan(planLine) {
  const { items } = parsePlanLine(planLine, { omitUnset: true });
  if (items.length === 0) return null;
  const list = el('dl', 'answer-plan');
  list.dataset.testid = 'answer-plan';
  list.setAttribute('aria-label', 'How this was worked out');
  for (const { key, label, value } of items) {
    const row = el('div', `plan-row plan-row--${key}`);
    row.append(el('dt', '', label), el('dd', '', value));
    list.appendChild(row);
  }
  return list;
}

/**
 * Mark a table frame while columns are out of view to the right (edge fade and hint).
 * @param {HTMLElement} frame
 * @param {HTMLElement} scroller
 * @param {HTMLElement} hint
 */
function trackTableOverflow(frame, scroller, hint) {
  const update = () => {
    const overflowing = scroller.scrollWidth - scroller.clientWidth > 1;
    const atEnd = scroller.scrollLeft + scroller.clientWidth >= scroller.scrollWidth - 1;
    frame.classList.toggle('has-more-right', overflowing && !atEnd);
    hint.classList.toggle('hidden', !overflowing);
  };
  scroller.addEventListener('scroll', update, { passive: true });
  if (typeof ResizeObserver === 'function') new ResizeObserver(update).observe(scroller);
  window.requestAnimationFrame?.(update);
}

/**
 * Statistics table: body face, tabular figures, numbers right-aligned, hidden cells marked.
 * The scrolling box is a labelled, focusable region so the keyboard can scroll it; the first
 * column stays in view. Group labels the engine wrote as "column=value" read as words.
 * @param {{ columns?: string[], rows?: unknown[] }} tableData
 * @param {{ decimals?: number, label?: string }} [opts]
 * @returns {{ wrapper: HTMLElement, hasHidden: boolean }}
 */
export function buildResultTable(tableData, opts = {}) {
  const frame = el('div', 'result-table-frame');
  const wrapper = el('div', 'result-table-wrapper');
  frame.appendChild(wrapper);
  if (!tableData || !Array.isArray(tableData.columns) || !Array.isArray(tableData.rows)) {
    wrapper.appendChild(el('p', 'result-error', 'No table was returned for this answer.'));
    return { wrapper: frame, hasHidden: false };
  }
  wrapper.tabIndex = 0;
  wrapper.setAttribute('role', 'region');
  wrapper.setAttribute('aria-label', opts.label || 'Results table');
  wrapper.dataset.testid = 'result-table-region';

  const cols = tableData.columns;
  const rows = normalizeTableRows(cols, tableData.rows);
  const numeric = cols.map((_, i) => i > 0 && isNumericColumn(rows, i));
  const formats = cols.map((col) => ({ decimals: opts.decimals, pValue: isPValueColumn(col) }));
  let hasHidden = false;

  const table = el('table', 'result-table');
  const thead = el('thead');
  const headRow = el('tr');
  cols.forEach((col, i) => {
    const th = el('th', numeric[i] ? 'num' : '', plainLevelLabel(col));
    th.scope = 'col';
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = el('tbody');
  rows.forEach((row) => {
    const tr = el('tr');
    cols.forEach((_, i) => {
      const cell = classifyCell(row[i], formats[i]);
      const td = el(i === 0 ? 'th' : 'td', numeric[i] ? 'num' : '');
      if (i === 0) td.scope = 'row';
      if (cell.kind === 'hidden' || cell.kind === 'small') {
        hasHidden = true;
        const mark = el('span', `cell-hidden cell-hidden--${cell.kind}`);
        mark.title = 'Hidden to protect privacy';
        mark.append(document.createTextNode(cell.text));
        const dagger = el('sup', '', '†');
        dagger.setAttribute('aria-hidden', 'true');
        mark.appendChild(dagger);
        if (cell.kind === 'hidden') mark.setAttribute('aria-label', 'hidden for privacy');
        td.appendChild(mark);
      } else {
        td.textContent = i === 0 && cell.kind === 'text' ? plainLevelLabel(cell.text) : cell.text;
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.append(thead, tbody);
  wrapper.appendChild(table);
  const fade = el('div', 'result-table-fade');
  fade.setAttribute('aria-hidden', 'true');
  const hint = el(
    'p',
    'result-table-hint hidden',
    `Scroll sideways to see all ${cols.length} columns.`
  );
  hint.dataset.testid = 'result-table-hint';
  frame.append(fade, hint);
  trackTableOverflow(frame, wrapper, hint);
  return { wrapper: frame, hasHidden };
}

/**
 * An answer: caption, claim (the answer's heading), stats line, what was computed, caveats,
 * then the table and the hidden-count footnote.
 * @param {Object} result from buildQueryCompletedPresentation
 * @param {{ content?: string }} [opts] content: the one-line message for results without a claim
 * @returns {HTMLElement}
 */
export function renderAnswerCard(result, opts = {}) {
  const content = opts.content || '';
  const card = el('div', 'result-card answer');
  card.setAttribute('data-testid', 'result-card');
  if (result.intentType) card.setAttribute('data-intent', result.intentType);

  if (result.title) {
    const titleEl = el('p', 'result-card-title', result.title);
    titleEl.setAttribute('data-testid', 'result-card-title');
    card.appendChild(titleEl);
  }

  const { headline, stats, notes } = splitClaim(result.claim);
  if (result.claim) {
    const claimEl = el('h3', 'result-claim', headline);
    claimEl.setAttribute('data-testid', 'result-claim');
    card.appendChild(claimEl);
    if (stats.length) {
      const statsEl = el('p', 'result-stats', stats.join(' · '));
      statsEl.setAttribute('data-testid', 'result-stats');
      card.appendChild(statsEl);
    }
  } else {
    const lead = content && content !== result.title ? content : '';
    if (lead) card.appendChild(el('h3', 'result-lead', lead));
    if (result.summary) {
      card.appendChild(el('p', 'result-stats', String(result.summary).replace(/ \| /g, ' · ')));
    }
  }

  if (result.planLine) {
    const plan = buildAnswerPlan(result.planLine);
    if (plan) card.appendChild(plan);
  }
  if (result.claim && notes) card.appendChild(el('p', 'result-notes', notes));

  let tableHasHidden = false;
  if (result.table) {
    const levels = tableLevelCount(result.table);
    if (levels > TABLE_LEVEL_LIMIT && isGroupLevelTable(result.table, result.analysisType)) {
      // Guard: a table with this many groups can name single patients in its labels.
      const withheld = el('p', 'result-withheld', tableWithheldMessage(levels));
      withheld.dataset.testid = 'table-withheld';
      card.appendChild(withheld);
    } else if (isFullyHiddenTable(result.table)) {
      // Nothing but hidden cells: the claim and the footnote say it all.
      tableHasHidden = true;
    } else {
      const { wrapper, hasHidden } = buildResultTable(result.table, {
        decimals: claimDecimals(result.claim),
        label: `${result.title || 'Results'} table${headline ? `: ${headline}` : ''}`,
      });
      tableHasHidden = hasHidden;
      card.appendChild(wrapper);
    }
  }

  if (needsHiddenFootnote(result.claim, tableHasHidden)) {
    const foot = el('p', 'result-footnote');
    foot.dataset.testid = 'hidden-cell-footnote';
    const dagger = el('span', '', '† ');
    dagger.setAttribute('aria-hidden', 'true');
    foot.append(dagger, document.createTextNode(HIDDEN_CELL_FOOTNOTE));
    card.appendChild(foot);
  }

  return card;
}

/**
 * A plan awaiting approval: the reason, the next step, the plan as a readable list, the exact
 * plan text on demand, and two plain-language actions. When words of the question were left out,
 * "Not what I meant" is the primary action.
 * @param {Object} held result from buildConfirmationHoldPresentation
 * A unit hold (`held.confirmOptions`) shows one button per option instead of "Run this
 * analysis": the doctor picks how the values are read, and a plain run would not pick for them.
 * @param {{ columns?: string[], onConfirm: (card: HTMLElement, planHash?: string) => void,
 *   onReject: (card: HTMLElement) => void }} handlers
 * @returns {HTMLElement}
 */
export function buildHeldCard(held, { columns = [], onConfirm, onReject }) {
  const card = el('div', 'result-card result-card-confirm');
  card.setAttribute('data-testid', 'result-confirm-card');

  const label = el('h3', 'state-label state-label--held');
  label.innerHTML = `${icon('clock')}<span>Check this plan before it runs</span>`;
  card.appendChild(label);

  if (held.failureReason) card.appendChild(el('p', 'held-reason', held.failureReason));
  const next = holdNextStep(held.failureReason, columns);
  const nextEl = el('p', 'held-next', next.text);
  nextEl.dataset.testid = 'held-next-step';
  card.appendChild(nextEl);

  if (held.planLine) {
    const planEl = el('div', 'result-plan-line');
    planEl.setAttribute('data-testid', 'result-plan-line');
    const { analysis, items } = parsePlanLine(held.planLine);
    if (analysis) planEl.appendChild(el('p', 'plan-analysis', analysis));
    const list = el('dl', 'plan-list');
    items.forEach(({ label: term, value }) => {
      const row = el('div', 'plan-row');
      row.append(el('dt', '', term), el('dd', '', value));
      list.appendChild(row);
    });
    planEl.appendChild(list);
    const raw = el('details', 'plan-raw');
    raw.append(el('summary', '', 'Exact plan text'), el('p', 'plan-raw-text', held.planLine));
    planEl.appendChild(raw);
    card.appendChild(planEl);
  }

  const actions = el('div', 'result-confirm-actions');
  const options = Array.isArray(held.confirmOptions) ? held.confirmOptions : [];
  if (options.length) {
    options.forEach((option, index) => {
      const kind = index === 0 ? 'btn-primary' : 'btn-secondary';
      const optionBtn = el('button', `btn ${kind} btn-confirm`, option.label);
      optionBtn.type = 'button';
      optionBtn.dataset.testid = 'confirm-option-btn';
      optionBtn.addEventListener('click', () => onConfirm(card, option.planHash));
      actions.appendChild(optionBtn);
    });
    const rejectOption = el('button', 'btn btn-secondary btn-reject', 'Not what I meant');
    rejectOption.type = 'button';
    rejectOption.dataset.testid = 'reject-query-btn';
    rejectOption.addEventListener('click', () => onReject(card));
    actions.appendChild(rejectOption);
    card.appendChild(actions);
    card.appendChild(el('p', 'result-card-note hidden'));
    return card;
  }
  const confirmBtn = el(
    'button',
    `btn ${next.dropped ? 'btn-secondary' : 'btn-primary'} btn-confirm`,
    'Run this analysis'
  );
  confirmBtn.type = 'button';
  confirmBtn.dataset.testid = 'confirm-query-btn';
  confirmBtn.addEventListener('click', () => onConfirm(card));
  const rejectBtn = el(
    'button',
    `btn ${next.dropped ? 'btn-primary' : 'btn-secondary'} btn-reject`,
    'Not what I meant'
  );
  rejectBtn.type = 'button';
  rejectBtn.dataset.testid = 'reject-query-btn';
  rejectBtn.addEventListener('click', () => onReject(card));
  if (next.dropped) actions.append(rejectBtn, confirmBtn);
  else actions.append(confirmBtn, rejectBtn);
  card.appendChild(actions);
  card.appendChild(el('p', 'result-card-note hidden'));

  return card;
}

/**
 * @param {HTMLElement} card
 * @param {boolean} disabled
 */
export function setCardButtonsDisabled(card, disabled) {
  card.querySelectorAll('.result-confirm-actions button').forEach((b) => {
    /** @type {HTMLButtonElement} */ (b).disabled = disabled;
  });
}

/**
 * Show a one-line note on a held card.
 * @param {HTMLElement} card
 * @param {string} text
 */
export function setCardNote(card, text) {
  const note = card.querySelector('.result-card-note');
  if (!note) return;
  note.textContent = text;
  note.classList.remove('hidden');
  note.setAttribute('role', 'status');
}

/**
 * The plan was decided: replace the buttons with what happened.
 * @param {HTMLElement} card
 * @param {string} text
 */
export function settleCard(card, text) {
  card.querySelector('.result-confirm-actions')?.remove();
  card.classList.add('is-settled');
  setCardNote(card, text);
}
