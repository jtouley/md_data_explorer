// SPDX-License-Identifier: MIT
/**
 * Upload panel helpers (Phase 8) — pure functions + DOM rendering.
 */

import { normalizeQuality } from './dataCheckPanel.js';

const ALLOWED_EXTENSIONS = new Set(['.csv', '.xlsx', '.xls', '.sav']);

/**
 * Upload states the API sends. 'needs_review': the cohort is stored and can be asked, but an
 * error-level data check failed; the API message names the check and what to do next.
 * @typedef {'ready' | 'needs_review' | 'failed'} UploadStatus
 */

/**
 * @param {Record<string, unknown>|null|undefined} raw
 * @returns {{ upload_id: string, dataset_name: string, status: string, message: string, failed_checks: string[], quality: import('./dataCheckPanel.js').DataCheck | null }}
 */
export function normalizeUploadResponse(raw) {
  if (!raw || typeof raw !== 'object') {
    return {
      upload_id: '',
      dataset_name: '',
      status: 'failed',
      message: '',
      failed_checks: [],
      quality: null,
    };
  }
  return {
    upload_id: String(raw.upload_id ?? ''),
    dataset_name: String(raw.dataset_name ?? ''),
    status: String(raw.status ?? 'failed'),
    message: String(raw.message ?? ''),
    failed_checks: Array.isArray(raw.failed_checks)
      ? raw.failed_checks.filter((id) => typeof id === 'string')
      : [],
    quality: normalizeQuality(raw.quality),
  };
}

/**
 * True when the upload stored a cohort the user can select: ready, or stored with a failed check.
 * @param {{ status: string, upload_id: string }} result
 * @returns {boolean}
 */
export function uploadIsUsable(result) {
  return (
    (result.status === 'ready' || result.status === 'needs_review') && Boolean(result.upload_id)
  );
}

/**
 * Status line for an upload: "ready" with the data-check note, or, for 'needs_review', the
 * API's sentence naming the failed check and the next step.
 * @param {{ status: string, message: string }} result
 * @param {string} name cohort or file name
 * @param {string|null} note from dataCheckStatusNote
 * @param {string} [unnamedHeadline] the ready sentence when there is no name
 * @returns {string}
 */
export function uploadResultText(result, name, note, unnamedHeadline = readyHeadline('')) {
  const shown = String(name || '').trim();
  if (result.status !== 'needs_review') {
    return uploadStatusText(shown ? readyHeadline(shown) : unnamedHeadline, note);
  }
  const headline = shown ? `The cohort “${shown}” needs review.` : 'The cohort needs review.';
  const detail = result.message.trim() || 'A data check failed. Open the cohort details to see it.';
  return `${headline} ${detail}`;
}

/**
 * Status line with the data-check sentence appended ("{name} is ready. Data check found …").
 * @param {string} headline a full sentence
 * @param {string|null} note from dataCheckStatusNote
 * @returns {string}
 */
export function uploadStatusText(headline, note) {
  return note ? `${headline} ${note}` : headline;
}

/**
 * "The cohort “ward 7” is ready.": the name is quoted as typed, so a lowercase file name never
 * starts the sentence.
 * @param {string} name
 * @returns {string}
 */
export function readyHeadline(name) {
  const shown = String(name || '').trim();
  return shown ? `The cohort “${shown}” is ready.` : 'The cohort is ready.';
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
