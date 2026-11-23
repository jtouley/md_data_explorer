// SPDX-License-Identifier: MIT
/**
 * "Is each row one patient?": asked once after an upload whose rows are told apart by a column
 * not named as a patient ID (or by nothing), when no ID-like column repeats. The API decides
 * when to ask (`one_row_per_patient` on the upload response); this panel never assumes an
 * answer. Until the user picks one, answers count rows.
 */

export const PATIENT_ROWS_TITLE = 'Is each row one patient?';
export const PATIENT_ROWS_BODY =
  'Until you choose, answers count rows. Yes: they count patients, one per row. No: they keep counting rows.';
export const PATIENT_ROWS_YES = 'Yes, one patient per row';
export const PATIENT_ROWS_NO = 'No, rows are visits or tests';
export const PATIENT_ROWS_CASES = 'Each row counts cases for a group';
export const PATIENT_ROWS_CASES_COLUMN = 'Column with the counts';

/**
 * The pending question on an upload response, or null (not asked, or already answered yes).
 * @param {unknown} raw `one_row_per_patient` from the upload response
 * @returns {{ confirmation: string, countColumns: string[] } | null}
 */
export function normalizePatientRowsQuestion(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const {
    status,
    confirmation,
    count_columns: counts,
  } = /** @type {Record<string, unknown>} */ (raw);
  if (status !== 'pending' || typeof confirmation !== 'string' || !confirmation.trim()) {
    return null;
  }
  const countColumns = Array.isArray(counts)
    ? counts.filter((c) => typeof c === 'string' && c.trim()).map((c) => c.trim())
    : [];
  return { confirmation: confirmation.trim(), countColumns };
}

/**
 * What the status line says once the answer is saved.
 * @param {'patients' | 'rows' | 'cases'} counts from the API
 * @param {string} name cohort name
 * @param {string} [countColumn] the column named by the third answer
 * @returns {string}
 */
export function patientRowsAnsweredText(counts, name, countColumn) {
  const shown = String(name || '').trim();
  const cohort = shown ? `“${shown}”` : 'this cohort';
  if (counts === 'cases') {
    return `Answers on ${cohort} now add up the cases in ${countColumn || 'the count column'}, never count rows.`;
  }
  return counts === 'patients'
    ? `Answers on ${cohort} now count patients, one per row.`
    : `Answers on ${cohort} keep counting rows.`;
}

/**
 * Render the question into a container. With `takeFocus`, focus moves to the panel itself, not
 * to a button, so a stray Enter answers nothing; Tab reaches Yes, then No. Both answers look the
 * same: neither is the default.
 * @param {HTMLElement} container
 * The third answer ("Each row counts cases for a group") is offered when the API names count
 * columns; with several, the person chooses which one holds the counts.
 * @param {{ onAnswer: (oneRowPerPatient: boolean, countsCasesIn?: string) => (void | Promise<void>), takeFocus?: boolean, countColumns?: string[] }} opts
 * @returns {{ setBusy: (busy: boolean) => void, setError: (message: string) => void, focus: () => void }}
 */
export function renderPatientRowsQuestion(
  container,
  { onAnswer, takeFocus = true, countColumns = [] }
) {
  container.replaceChildren();
  container.classList.remove('hidden');

  const panel = document.createElement('section');
  panel.className = 'patient-rows-question';
  panel.setAttribute('role', 'group');
  panel.setAttribute('aria-labelledby', 'patient-rows-title');
  panel.setAttribute('aria-describedby', 'patient-rows-body');
  panel.tabIndex = -1;
  panel.dataset.testid = 'patient-rows-question';

  const title = document.createElement('p');
  title.id = 'patient-rows-title';
  title.className = 'patient-rows-title';
  title.textContent = PATIENT_ROWS_TITLE;

  const body = document.createElement('p');
  body.id = 'patient-rows-body';
  body.className = 'patient-rows-body';
  body.textContent = PATIENT_ROWS_BODY;

  const error = document.createElement('p');
  error.className = 'patient-rows-error hidden';
  error.setAttribute('role', 'alert');

  const actions = document.createElement('div');
  actions.className = 'result-confirm-actions';

  const yesBtn = document.createElement('button');
  yesBtn.type = 'button';
  yesBtn.className = 'btn btn-secondary';
  yesBtn.dataset.testid = 'patient-rows-yes';
  yesBtn.textContent = PATIENT_ROWS_YES;

  const noBtn = document.createElement('button');
  noBtn.type = 'button';
  noBtn.className = 'btn btn-secondary';
  noBtn.dataset.testid = 'patient-rows-no';
  noBtn.textContent = PATIENT_ROWS_NO;

  let casesBtn = null;
  let columnSelect = null;
  if (countColumns.length) {
    casesBtn = document.createElement('button');
    casesBtn.type = 'button';
    casesBtn.className = 'btn btn-secondary';
    casesBtn.dataset.testid = 'patient-rows-cases';
    casesBtn.textContent =
      countColumns.length === 1
        ? `${PATIENT_ROWS_CASES} (in ${countColumns[0]})`
        : PATIENT_ROWS_CASES;
    if (countColumns.length > 1) {
      columnSelect = document.createElement('select');
      columnSelect.dataset.testid = 'patient-rows-count-column';
      columnSelect.setAttribute('aria-label', PATIENT_ROWS_CASES_COLUMN);
      for (const column of countColumns) {
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        columnSelect.appendChild(option);
      }
    }
  }

  const setBusy = (busy) => {
    yesBtn.disabled = busy;
    noBtn.disabled = busy;
    if (casesBtn) casesBtn.disabled = busy;
    panel.setAttribute('aria-busy', busy ? 'true' : 'false');
  };
  const setError = (message) => {
    error.textContent = message;
    error.classList.toggle('hidden', !message);
  };
  const answer = (oneRowPerPatient) => {
    setError('');
    void onAnswer(oneRowPerPatient);
  };

  yesBtn.addEventListener('click', () => answer(true));
  noBtn.addEventListener('click', () => answer(false));
  if (casesBtn) {
    casesBtn.addEventListener('click', () => {
      setError('');
      void onAnswer(false, columnSelect ? columnSelect.value : countColumns[0]);
    });
  }

  actions.append(yesBtn, noBtn);
  if (columnSelect) actions.append(columnSelect);
  if (casesBtn) actions.append(casesBtn);
  panel.append(title, body, error, actions);
  container.appendChild(panel);
  const focus = () => panel.focus();
  if (takeFocus) focus();

  return { setBusy, setError, focus };
}

/**
 * Hide and empty the question's container.
 * @param {HTMLElement} container
 */
export function closePatientRowsQuestion(container) {
  container.replaceChildren();
  container.classList.add('hidden');
}
