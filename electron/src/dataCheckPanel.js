/**
 * Data check (contract quality report) for a cohort: pure helpers + the details-sheet section.
 * The API sends plain-language lines with small counts already hidden; nothing is recomputed.
 * Meaning is carried by words, never by colour alone (docs/app/DESIGN.md).
 */

export const DATA_CHECK_STILL_USABLE = 'You can still ask questions.';
export const DATA_CHECK_NOTHING_FLAGGED = 'Nothing flagged.';
export const DATA_CHECK_PRIVACY_HINT =
  "Hidden to protect privacy: counts from 1 to 10 are not shown, and some other counts are hidden so those counts can't be worked out from totals.";
export const DATA_CHECK_LOADING = 'Checking…';

const MORE_LINE = /^…and (\d+) more$/;

/**
 * @typedef {{ status: 'ran', lines: string[], count: number } | { status: 'not_run', reason: string | null }} DataCheck
 */

/**
 * Normalise the API `quality` value (a report, `{ status: 'not_run' }`, or nothing).
 * @param {unknown} raw
 * @returns {DataCheck | null} null when the API sent nothing (an older server)
 */
export function normalizeQuality(raw) {
  if (!raw || typeof raw !== 'object') return null;
  if (raw.status === 'not_run') {
    const reason = typeof raw.reason === 'string' && raw.reason.trim() ? raw.reason.trim() : null;
    return { status: 'not_run', reason };
  }
  if (!Array.isArray(raw.summary)) return null;
  const lines = raw.summary.filter((l) => typeof l === 'string' && l.length > 0);
  const last = lines.length ? MORE_LINE.exec(lines[lines.length - 1]) : null;
  const count = last ? lines.length - 1 + Number(last[1]) : lines.length;
  return { status: 'ran', lines, count };
}

/**
 * One sentence for the upload status line.
 * @param {DataCheck | null} check
 * @returns {string | null}
 */
export function dataCheckStatusNote(check) {
  if (!check) return null;
  if (check.status === 'not_run') return 'Data check could not run.';
  if (check.count === 0) return 'Data check found nothing to flag.';
  const things = check.count === 1 ? '1 thing' : `${check.count} things`;
  return `Data check found ${things} to review in cohort details.`;
}

/**
 * Sentence for a check that did not run, with the API's reason when it gave one.
 * @param {string | null} reason
 * @returns {string}
 */
export function dataCheckNotRunText(reason) {
  const why = reason ? `: ${reason.replace(/\.$/, '')}` : '';
  return `Data check could not run${why}. ${DATA_CHECK_STILL_USABLE}`;
}

/**
 * Fill the "Data check" section of the cohort details sheet.
 * @param {{ hint: HTMLElement, list: HTMLElement }} parts
 * @param {DataCheck | null | 'loading'} check
 */
export function renderDataCheck({ hint, list }, check) {
  list.replaceChildren();
  if (check === 'loading') {
    hint.textContent = DATA_CHECK_LOADING;
    return;
  }
  if (!check || check.status === 'not_run') {
    hint.textContent = dataCheckNotRunText(check?.reason ?? null);
    return;
  }
  if (check.lines.length === 0) {
    hint.textContent = DATA_CHECK_NOTHING_FLAGGED;
    return;
  }
  hint.textContent = DATA_CHECK_PRIVACY_HINT;
  for (const line of check.lines) {
    const item = document.createElement('li');
    item.textContent = line;
    list.appendChild(item);
  }
}
