/**
 * Upload panel helpers (Phase 8) — pure functions + DOM rendering.
 */

const ALLOWED_EXTENSIONS = new Set(['.csv', '.xlsx', '.xls', '.sav']);

export const QUALITY_NOT_RUN_NOTE = 'Data check could not run';

/**
 * One plain-language data-quality note from a report or a not-run status (already suppressed).
 * @param {unknown} quality API `quality` value: a report, `{ status: 'not_run' }`, or null
 * @returns {string|null} null when there is nothing to say
 */
export function qualityNote(quality) {
  if (!quality || typeof quality !== 'object') return null;
  if (quality.status === 'not_run') return QUALITY_NOT_RUN_NOTE;
  const lines = Array.isArray(quality.summary)
    ? quality.summary.filter((l) => typeof l === 'string')
    : [];
  return lines.length ? `Data check: ${lines.join('; ')}` : null;
}

/**
 * @param {Record<string, unknown>|null|undefined} raw
 * @returns {{ upload_id: string, dataset_name: string, status: string, message: string, quality_note: string|null }}
 */
export function normalizeUploadResponse(raw) {
  if (!raw || typeof raw !== 'object') {
    return { upload_id: '', dataset_name: '', status: 'failed', message: '', quality_note: null };
  }
  return {
    upload_id: String(raw.upload_id ?? ''),
    dataset_name: String(raw.dataset_name ?? ''),
    status: String(raw.status ?? 'failed'),
    message: String(raw.message ?? ''),
    quality_note: qualityNote(raw.quality),
  };
}

/**
 * Status line with the data-quality note appended.
 * @param {string} headline
 * @param {string|null} note from qualityNote
 * @returns {string}
 */
export function uploadStatusText(headline, note) {
  return note ? `${headline}. ${note}` : headline;
}

/**
 * @param {number} bytes
 * @returns {string}
 */
export function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * @param {string} filename
 * @returns {boolean}
 */
export function isAllowedExtension(filename) {
  const dot = filename.lastIndexOf('.');
  if (dot === -1) return false;
  return ALLOWED_EXTENSIONS.has(filename.slice(dot).toLowerCase());
}

/**
 * Render the upload area UI into a container.
 * @param {HTMLElement} container
 * @param {{ onFileSelect: (file: File) => void, onLoadSample?: () => void, statusText?: string }} handlers
 */
export function renderUploadArea(container, handlers) {
  container.replaceChildren();

  const wrapper = document.createElement('div');
  wrapper.className = 'upload-area';

  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.csv,.xlsx,.xls,.sav';
  input.className = 'upload-file-input';
  input.dataset.testid = 'upload-file-input';
  input.addEventListener('change', () => {
    if (input.files && input.files[0]) {
      handlers.onFileSelect(input.files[0]);
    }
  });

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'btn-upload';
  btn.dataset.testid = 'upload-btn';
  btn.textContent = 'Upload a file…';
  btn.addEventListener('click', () => input.click());

  wrapper.appendChild(input);
  wrapper.appendChild(btn);

  if (typeof handlers.onLoadSample === 'function') {
    const sampleBtn = document.createElement('button');
    sampleBtn.type = 'button';
    sampleBtn.className = 'btn-upload btn-load-sample';
    sampleBtn.dataset.testid = 'load-sample-btn';
    sampleBtn.textContent = 'Use the sample cohort';
    sampleBtn.addEventListener('click', () => handlers.onLoadSample());
    wrapper.appendChild(sampleBtn);
  }

  if (handlers.statusText) {
    const status = document.createElement('span');
    status.className = 'upload-status';
    status.dataset.testid = 'upload-status';
    status.textContent = handlers.statusText;
    wrapper.appendChild(status);
  }

  container.appendChild(wrapper);
}
