/**
 * Upload panel helpers (Phase 8) — pure functions + DOM rendering.
 */

const ALLOWED_EXTENSIONS = new Set(['.csv', '.xlsx', '.xls', '.sav']);

/**
 * Plain-language data-quality lines from an upload response (already small-cell suppressed).
 * @param {unknown} quality
 * @returns {string[]}
 */
function qualitySummary(quality) {
  const lines = quality && typeof quality === 'object' ? quality.summary : null;
  return Array.isArray(lines) ? lines.filter((l) => typeof l === 'string') : [];
}

/**
 * @param {Record<string, unknown>|null|undefined} raw
 * @returns {{ upload_id: string, dataset_name: string, status: string, message: string, quality_summary: string[] }}
 */
export function normalizeUploadResponse(raw) {
  if (!raw || typeof raw !== 'object') {
    return { upload_id: '', dataset_name: '', status: 'failed', message: '', quality_summary: [] };
  }
  return {
    upload_id: String(raw.upload_id ?? ''),
    dataset_name: String(raw.dataset_name ?? ''),
    status: String(raw.status ?? 'failed'),
    message: String(raw.message ?? ''),
    quality_summary: qualitySummary(raw.quality),
  };
}

/**
 * Status line after a successful upload, with the data-quality notes appended.
 * @param {string} headline
 * @param {string[]} qualityLines
 * @returns {string}
 */
export function uploadStatusText(headline, qualityLines) {
  return qualityLines.length ? `${headline}. Data check: ${qualityLines.join('; ')}` : headline;
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
