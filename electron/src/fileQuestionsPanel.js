/**
 * "Questions about your file" (ADR017): every open question the app has about the cohort's file,
 * of any kind, in clinical words with its evidence (counts only, never a row value), one plain
 * button per answer and a short note on what each answer changes. Answers go to the route the API
 * names for each (`GET /api/datasets/{id}/questions`), so this panel knows no question kind.
 *
 * An answer that waits on these questions says "Waiting on N questions about this file", with a
 * button that opens this panel.
 */

export const FILE_QUESTIONS_TITLE = 'Questions about your file';
export const FILE_QUESTIONS_NONE = 'No open questions about this file.';
export const FILE_QUESTIONS_HINT =
  'Answers that depend on these wait until you answer. Your answer is kept with the file.';
export const FILE_QUESTIONS_SAVING = 'Saving your answer…';
export const FILE_QUESTIONS_SAVED = 'Saved. Answers now use it.';
export const OPEN_FILE_QUESTIONS_LABEL = 'Answer them';

/** The routes an answer may be posted to (under /api/datasets/{id}). */
export const ANSWER_ROUTES = Object.freeze([
  '/answers',
  '/missing-codes',
  '/date-readings',
  '/one-row-per-patient',
]);

/**
 * @typedef {{ label: string, route: string, body: Record<string, unknown>, changes: string }} FileAnswer
 * @typedef {{ id: string, kind: string, column: string | null, question: string, evidence: string,
 *   answers: FileAnswer[] }} FileQuestion
 */

/**
 * The API's questions, keeping only well-formed ones whose answers post to a known route.
 * @param {unknown} raw response of getFileQuestions
 * @returns {FileQuestion[]}
 */
export function normalizeFileQuestions(raw) {
  const list = Array.isArray(raw?.questions) ? raw.questions : [];
  const text = (v) => (typeof v === 'string' ? v : '');
  return list
    .filter((q) => q && typeof q.id === 'string' && typeof q.question === 'string')
    .map((q) => ({
      id: q.id,
      kind: text(q.kind),
      column: typeof q.column === 'string' ? q.column : null,
      question: q.question,
      evidence: text(q.evidence),
      answers: (Array.isArray(q.answers) ? q.answers : [])
        .filter(
          (a) =>
            a &&
            typeof a.label === 'string' &&
            ANSWER_ROUTES.includes(a.route) &&
            a.body &&
            typeof a.body === 'object'
        )
        .map((a) => ({ label: a.label, route: a.route, body: a.body, changes: text(a.changes) })),
    }))
    .filter((q) => q.answers.length > 0);
}

/**
 * "Waiting on 2 questions about this file".
 * @param {number} count
 * @returns {string}
 */
export function waitingText(count) {
  const n = Number.isInteger(count) && count > 0 ? count : 1;
  return `Waiting on ${n} question${n === 1 ? '' : 's'} about this file`;
}

/**
 * @param {string} tag
 * @param {string} [className]
 * @param {string} [text]
 */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Render the questions into `list`; `hint` says what they are for (or that there are none).
 * Clicking an answer disables the question's buttons and calls `onAnswer`.
 * @param {{ hint: HTMLElement, list: HTMLElement }} parts
 * @param {FileQuestion[]} questions
 * @param {{ onAnswer: (question: FileQuestion, answer: FileAnswer, item: HTMLElement) => void }} handlers
 */
export function renderFileQuestions({ hint, list }, questions, { onAnswer }) {
  list.replaceChildren();
  hint.textContent = questions.length ? FILE_QUESTIONS_HINT : FILE_QUESTIONS_NONE;
  for (const question of questions) {
    const item = el('li', 'file-question');
    item.dataset.testid = 'file-question';
    item.dataset.kind = question.kind;
    item.appendChild(el('p', 'file-question-text', question.question));
    if (question.evidence) {
      const evidence = el('p', 'file-question-evidence', question.evidence);
      evidence.dataset.testid = 'file-question-evidence';
      item.appendChild(evidence);
    }
    const answers = el('div', 'file-question-answers');
    for (const answer of question.answers) {
      const row = el('div', 'file-question-answer');
      const button = el('button', 'btn-secondary', answer.label);
      button.type = 'button';
      button.dataset.testid = 'file-question-answer';
      button.addEventListener('click', () => {
        for (const b of answers.querySelectorAll('button')) b.disabled = true;
        onAnswer(question, answer, item);
      });
      row.appendChild(button);
      if (answer.changes) row.appendChild(el('span', 'file-question-changes', answer.changes));
      answers.appendChild(row);
    }
    item.appendChild(answers);
    const status = el('p', 'file-question-status');
    status.dataset.testid = 'file-question-status';
    status.setAttribute('role', 'status');
    item.appendChild(status);
    list.appendChild(item);
  }
}

/**
 * Show the outcome of an answer on its question: saved, or the error with the buttons enabled.
 * @param {HTMLElement} item
 * @param {string | null} error
 */
export function showAnswerOutcome(item, error) {
  const status = item.querySelector('[data-testid="file-question-status"]');
  if (status) status.textContent = error || FILE_QUESTIONS_SAVED;
  if (error) for (const b of item.querySelectorAll('button')) b.disabled = false;
}

/**
 * The answer area of a question that waits on questions about the file.
 * @param {{ count: number, failureReason?: string }} waiting
 * @param {() => void} onOpen opens the panel
 * @returns {HTMLElement}
 */
export function buildWaitingCard(waiting, onOpen) {
  const card = el('div', 'result-card result-card-waiting');
  card.dataset.testid = 'waiting-on-questions';
  card.appendChild(el('h3', 'state-label state-label--held', waitingText(waiting.count)));
  if (waiting.failureReason) card.appendChild(el('p', 'held-reason', waiting.failureReason));
  const open = el('button', 'btn-primary', OPEN_FILE_QUESTIONS_LABEL);
  open.type = 'button';
  open.dataset.testid = 'open-file-questions';
  open.addEventListener('click', () => onOpen());
  card.appendChild(open);
  return card;
}
