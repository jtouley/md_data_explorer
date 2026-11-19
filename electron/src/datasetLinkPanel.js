// SPDX-License-Identifier: MIT
/**
 * "Load from a link": an in-page form for the link to a public data file, and the helpers that
 * show where a cohort came from (link, date, checksum) for a methods section.
 */

/**
 * @typedef {{ url: string, finalUrl: string, fetchedAt: string, sizeBytes: number,
 *   sha256: string, methodsText: string }} DatasetSource
 */

/**
 * The app's check before sending: a link the engine will at least try. The engine still
 * checks everything (public address, HTTPS, size); this only catches typing slips early.
 * @param {string} text
 * @returns {string} a sentence saying what to fix, or '' when the link can be sent
 */
export function linkProblem(text) {
  const value = String(text ?? '').trim();
  if (!value) return 'Paste a link first.';
  if (/^http:\/\//i.test(value)) {
    return 'Only secure links (https://) can be loaded. Try the same link with https://.';
  }
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    return 'That doesn’t look like a web link. Copy the download link again.';
  }
  if (parsed.protocol !== 'https:') return 'Paste a web link that starts with https://.';
  return '';
}

/**
 * @param {unknown} raw `source` from the API (the from-url response, or dataset metadata)
 * @returns {DatasetSource | null}
 */
export function normalizeSource(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const r = /** @type {Record<string, unknown>} */ (raw);
  const url = typeof r.url === 'string' ? r.url : '';
  const sha256 = typeof r.sha256 === 'string' ? r.sha256 : '';
  const fetchedAt = typeof r.fetched_at === 'string' ? r.fetched_at : '';
  if (!url || !sha256 || !fetchedAt) return null;
  const finalUrl = typeof r.final_url === 'string' && r.final_url ? r.final_url : url;
  const sizeBytes = Number.isFinite(Number(r.size_bytes)) ? Number(r.size_bytes) : 0;
  const methodsText =
    typeof r.methods_text === 'string' && r.methods_text
      ? r.methods_text
      : methodsSentence({ finalUrl, fetchedAt, sizeBytes, sha256 });
  return { url, finalUrl, fetchedAt, sizeBytes, sha256, methodsText };
}

/**
 * "19 November 2026" (UTC, like the stored time), or the raw text if it cannot be read.
 * @param {string} iso
 * @returns {string}
 */
export function formatFetchDate(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  });
}

/**
 * The methods-section sentence, matching the engine's wording (used when only stored fields exist).
 * @param {{ finalUrl: string, fetchedAt: string, sizeBytes: number, sha256: string }} s
 * @returns {string}
 */
export function methodsSentence(s) {
  return (
    `Data downloaded from ${s.finalUrl} on ${formatFetchDate(s.fetchedAt)} ` +
    `(${s.sizeBytes.toLocaleString('en-US')} bytes; SHA-256 ${s.sha256}).`
  );
}

/**
 * Short line for the status area: "Source saved: data.cdc.gov, 19 November 2026."
 * @param {DatasetSource} source
 * @returns {string}
 */
export function sourceStatusNote(source) {
  let host = source.finalUrl;
  try {
    host = new URL(source.finalUrl).hostname;
  } catch {
    // keep the full link
  }
  return `Source saved: ${host}, ${formatFetchDate(source.fetchedAt)}.`;
}

/**
 * Render the link form into a container (shown below the cohort switcher).
 * @param {HTMLElement} container
 * @param {{ onSubmit: (url: string, name: string) => (void | Promise<void>), onCancel: () => void }} handlers
 * @returns {{ setBusy: (busy: boolean) => void, setError: (message: string) => void }}
 */
export function renderDatasetLinkForm(container, { onSubmit, onCancel }) {
  container.replaceChildren();
  container.classList.remove('hidden');

  const form = document.createElement('form');
  form.className = 'dataset-link-form';
  form.noValidate = true;
  form.setAttribute('aria-labelledby', 'dataset-link-title');
  form.dataset.testid = 'dataset-link-form';

  const title = document.createElement('p');
  title.id = 'dataset-link-title';
  title.className = 'dataset-link-title';
  title.textContent = 'Load public data from a link';

  const urlLabel = document.createElement('label');
  urlLabel.className = 'dataset-link-label';
  urlLabel.htmlFor = 'dataset-link-url';
  urlLabel.textContent = 'Link to the data file';

  const urlInput = document.createElement('input');
  urlInput.type = 'url';
  urlInput.id = 'dataset-link-url';
  urlInput.className = 'dataset-link-input';
  urlInput.required = true;
  urlInput.autocomplete = 'off';
  urlInput.spellcheck = false;
  urlInput.placeholder = 'https://';
  urlInput.dataset.testid = 'dataset-link-url';
  urlInput.setAttribute('aria-describedby', 'dataset-link-hint dataset-link-error');

  const hint = document.createElement('p');
  hint.id = 'dataset-link-hint';
  hint.className = 'dataset-link-hint';
  hint.textContent =
    'The download link for a CSV, Excel or SPSS file on a public site, such as CDC, CMS or ' +
    'WHO. The app only downloads the file; nothing on this computer is sent. The link and ' +
    'date are saved with the cohort for your methods section.';

  const nameLabel = document.createElement('label');
  nameLabel.className = 'dataset-link-label';
  nameLabel.htmlFor = 'dataset-link-name';
  nameLabel.textContent = 'Cohort name (optional)';

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.id = 'dataset-link-name';
  nameInput.className = 'dataset-link-input';
  nameInput.autocomplete = 'off';
  nameInput.maxLength = 200;
  nameInput.dataset.testid = 'dataset-link-name';

  const error = document.createElement('p');
  error.id = 'dataset-link-error';
  error.className = 'dataset-link-error hidden';
  error.setAttribute('role', 'alert');
  error.dataset.testid = 'dataset-link-error';

  const actions = document.createElement('div');
  actions.className = 'result-confirm-actions';

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'btn-reject';
  cancelBtn.dataset.testid = 'dataset-link-cancel';
  cancelBtn.textContent = 'Cancel';

  const submitBtn = document.createElement('button');
  submitBtn.type = 'submit';
  submitBtn.className = 'btn btn-primary';
  submitBtn.dataset.testid = 'dataset-link-submit';
  submitBtn.textContent = 'Load data';

  let busy = false;
  const setBusy = (value) => {
    busy = value;
    urlInput.readOnly = value;
    nameInput.readOnly = value;
    submitBtn.disabled = value;
    cancelBtn.disabled = value;
    submitBtn.textContent = value ? 'Downloading…' : 'Load data';
    form.setAttribute('aria-busy', String(value));
  };
  const setError = (message) => {
    error.textContent = message;
    error.classList.toggle('hidden', !message);
    if (message) urlInput.setAttribute('aria-invalid', 'true');
    else urlInput.removeAttribute('aria-invalid');
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    if (busy) return;
    const problem = linkProblem(urlInput.value);
    if (problem) {
      setError(problem);
      urlInput.focus();
      return;
    }
    setError('');
    void onSubmit(urlInput.value.trim(), nameInput.value.trim());
  });
  cancelBtn.addEventListener('click', () => onCancel());
  form.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !busy) {
      event.stopPropagation();
      onCancel();
    }
  });

  actions.append(cancelBtn, submitBtn);
  form.append(title, urlLabel, urlInput, hint, nameLabel, nameInput, error, actions);
  container.appendChild(form);
  urlInput.focus();

  return { setBusy, setError };
}

/**
 * Hide and empty the form container.
 * @param {HTMLElement} container
 */
export function closeDatasetLinkForm(container) {
  container.replaceChildren();
  container.classList.add('hidden');
}
