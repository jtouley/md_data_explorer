// SPDX-License-Identifier: MIT
/** @vitest-environment jsdom */
import { describe, test, expect, vi } from 'vitest';
import {
  normalizeUploadResponse,
  uploadStatusText,
  readyHeadline,
  formatFileSize,
  isAllowedExtension,
  renderUploadArea,
  uploadIsUsable,
  uploadResultText,
} from './uploadPanel.js';
import { dataCheckStatusNote } from './dataCheckPanel.js';

describe('normalizeUploadResponse', () => {
  test('extracts upload_id and status from valid response', () => {
    const raw = { upload_id: 'u_123', dataset_name: 'Test', status: 'ready', message: 'ok' };
    const result = normalizeUploadResponse(raw);
    expect(result.upload_id).toBe('u_123');
    expect(result.status).toBe('ready');
    expect(result.dataset_name).toBe('Test');
  });

  test('handles null/undefined input', () => {
    expect(normalizeUploadResponse(null)).toEqual({
      upload_id: '',
      dataset_name: '',
      status: 'failed',
      message: '',
      failed_checks: [],
      quality: null,
      sheet_note: '',
    });
    expect(normalizeUploadResponse(undefined)).toEqual({
      upload_id: '',
      dataset_name: '',
      status: 'failed',
      message: '',
      failed_checks: [],
      quality: null,
      sheet_note: '',
    });
  });
});

describe('upload sheet note', () => {
  const note =
    'Read 1 of 2 sheets: “Deaths”. Not read: “Notes”. To use another sheet, save it as its own file and upload it.';

  test.each([
    [{ sheet_note: note }, note],
    [{ sheet_note: `  ${note}  ` }, note],
    [{ sheet_note: '' }, ''],
    [{ sheet_note: null }, ''],
    [{ sheet_note: 3 }, ''],
    [{}, ''],
  ])('normalizeUploadResponse keeps only a text sheet note (%#)', (raw, expected) => {
    expect(normalizeUploadResponse(raw).sheet_note).toBe(expected);
  });

  test('the sheet note is shown in the status line before the data check', () => {
    const { sheet_note: sheetNote, quality } = normalizeUploadResponse({
      sheet_note: note,
      quality: { status: 'not_run', reason: 'x' },
    });
    expect(uploadStatusText('A is ready.', sheetNote, dataCheckStatusNote(quality))).toBe(
      `A is ready. ${note} Data check could not run.`
    );
  });

  test('a one-sheet workbook leaves the headline alone', () => {
    expect(uploadStatusText('A is ready.', '', null)).toBe('A is ready.');
  });
});

describe('upload data check', () => {
  test.each([
    [
      { quality: { status: 'ran', summary: ['ldl: 5.8% missing', 7] } },
      { status: 'ran', lines: ['ldl: 5.8% missing'], count: 1 },
    ],
    [
      { quality: { status: 'not_run', reason: 'the check failed (OSError)', summary: [] } },
      { status: 'not_run', reason: 'the check failed (OSError)' },
    ],
    [{ quality: null }, null],
    [{}, null],
  ])('normalizeUploadResponse keeps the normalized check (%#)', (raw, expected) => {
    expect(normalizeUploadResponse(raw).quality).toEqual(expected);
  });

  test('a failed check is visible in the upload status line', () => {
    const { quality } = normalizeUploadResponse({ quality: { status: 'not_run', reason: 'x' } });
    expect(uploadStatusText('A is ready.', dataCheckStatusNote(quality))).toBe(
      'A is ready. Data check could not run.'
    );
  });

  test('a report adds one sentence pointing to cohort details', () => {
    const { quality } = normalizeUploadResponse({ quality: { summary: ['ldl: 5.8% missing'] } });
    expect(uploadStatusText('A is ready.', dataCheckStatusNote(quality))).toBe(
      'A is ready. Data check found 1 thing to review in cohort details.'
    );
  });

  test('an older server without quality leaves the headline alone', () => {
    expect(
      uploadStatusText('A is ready.', dataCheckStatusNote(normalizeUploadResponse({}).quality))
    ).toBe('A is ready.');
  });
});

describe('formatFileSize', () => {
  test('formats bytes', () => {
    expect(formatFileSize(512)).toBe('512 B');
  });

  test('formats kilobytes', () => {
    expect(formatFileSize(2048)).toBe('2.0 KB');
  });

  test('formats megabytes', () => {
    expect(formatFileSize(5 * 1024 * 1024)).toBe('5.0 MB');
  });

  test('handles zero', () => {
    expect(formatFileSize(0)).toBe('0 B');
  });
});

describe('isAllowedExtension', () => {
  test('allows .csv', () => {
    expect(isAllowedExtension('data.csv')).toBe(true);
  });

  test('allows .xlsx', () => {
    expect(isAllowedExtension('report.xlsx')).toBe(true);
  });

  test('allows .xls', () => {
    expect(isAllowedExtension('old.xls')).toBe(true);
  });

  test('allows .sav', () => {
    expect(isAllowedExtension('spss.sav')).toBe(true);
  });

  test('rejects .exe', () => {
    expect(isAllowedExtension('bad.exe')).toBe(false);
  });

  test('rejects no extension', () => {
    expect(isAllowedExtension('noext')).toBe(false);
  });

  test('case-insensitive', () => {
    expect(isAllowedExtension('DATA.CSV')).toBe(true);
  });
});

describe('renderUploadArea', () => {
  test('renders file input and upload button', () => {
    const container = document.createElement('div');
    renderUploadArea(container, { onFileSelect: vi.fn() });

    const input = container.querySelector('input[type="file"]');
    expect(input).not.toBeNull();
    expect(input.accept).toContain('.csv');

    const btn = container.querySelector('[data-testid="upload-btn"]');
    expect(btn).not.toBeNull();
  });

  test('triggers onFileSelect when file chosen', () => {
    const container = document.createElement('div');
    const onFileSelect = vi.fn();
    renderUploadArea(container, { onFileSelect });

    const input = container.querySelector('input[type="file"]');
    const file = new File(['col1,col2\na,b'], 'test.csv', { type: 'text/csv' });
    Object.defineProperty(input, 'files', { value: [file] });
    input.dispatchEvent(new Event('change'));

    expect(onFileSelect).toHaveBeenCalledWith(file);
  });

  test('shows upload status text', () => {
    const container = document.createElement('div');
    renderUploadArea(container, { onFileSelect: vi.fn(), statusText: 'Uploading…' });

    const status = container.querySelector('[data-testid="upload-status"]');
    expect(status).not.toBeNull();
    expect(status.textContent).toBe('Uploading…');
  });

  test('renders Load sample when onLoadSample provided', () => {
    const container = document.createElement('div');
    const onLoadSample = vi.fn();
    renderUploadArea(container, { onFileSelect: vi.fn(), onLoadSample });

    const btn = container.querySelector('[data-testid="load-sample-btn"]');
    expect(btn).not.toBeNull();
    btn.click();
    expect(onLoadSample).toHaveBeenCalledOnce();
  });

  test('renderUploadArea_labels_plainLanguageActions', () => {
    // Arrange
    const container = document.createElement('div');
    // Act
    renderUploadArea(container, { onFileSelect: vi.fn(), onLoadSample: vi.fn() });
    // Assert
    expect(container.querySelector('[data-testid="upload-btn"]').textContent).toBe(
      'Upload a file…'
    );
    expect(container.querySelector('[data-testid="load-sample-btn"]').textContent).toBe(
      'Use the sample cohort'
    );
  });
});

describe('readyHeadline (L1)', () => {
  test.each([
    ['cohort', 'The cohort “cohort” is ready.'],
    ['Trial A', 'The cohort “Trial A” is ready.'],
    ['', 'The cohort is ready.'],
  ])('readyHeadline(%s) never starts with a lowercase file name', (name, expected) => {
    // Act
    const text = readyHeadline(name);

    // Assert
    expect(text).toBe(expected);
    expect(text).toMatch(/^[A-Z]/);
  });
});

describe('needs_review uploads', () => {
  const review = {
    upload_id: 'u_1',
    status: 'needs_review',
    message:
      'Uploaded, but a data check failed: patient_id: 127 repeated values. Questions that count patients will first ask you to confirm that rows are visits.',
  };

  test.each([
    [{ upload_id: 'u_1', status: 'ready' }, true],
    [{ upload_id: 'u_1', status: 'needs_review' }, true],
    [{ upload_id: '', status: 'needs_review' }, false],
    [{ upload_id: 'u_1', status: 'failed' }, false],
  ])('uploadIsUsable(%o) is %s', (result, expected) => {
    expect(uploadIsUsable(result)).toBe(expected);
  });

  test('normalizeUploadResponse keeps failed check ids and drops non-strings', () => {
    const result = normalizeUploadResponse({
      ...review,
      failed_checks: ['patient_id_no_duplicates', 3],
    });
    expect(result.failed_checks).toEqual(['patient_id_no_duplicates']);
  });

  test('the status line names the cohort and the failed check, never "ready"', () => {
    const text = uploadResultText(
      review,
      'deaths',
      'Data check found 2 things to review in cohort details.'
    );
    expect(text).toBe(`The cohort “deaths” needs review. ${review.message}`);
    expect(text).not.toMatch(/ready/);
  });

  test('a needs_review upload without a message still says how to see the check', () => {
    expect(uploadResultText({ status: 'needs_review', message: ' ' }, '', null)).toBe(
      'The cohort needs review. A data check failed. Open the cohort details to see it.'
    );
  });

  test('a ready upload without a name uses the given sentence', () => {
    expect(
      uploadResultText({ status: 'ready', message: '' }, '', null, 'The sample cohort is ready.')
    ).toBe('The sample cohort is ready.');
  });

  const sheetNote =
    'Read 1 of 2 sheets: “Deaths”. Not read: “Notes”. To use another sheet, save it as its own file and upload it.';

  test('a ready upload shows the sheet note before the data check', () => {
    const result = normalizeUploadResponse({
      upload_id: 'u_1',
      status: 'ready',
      sheet_note: sheetNote,
      quality: { status: 'not_run', reason: 'x' },
    });
    expect(uploadResultText(result, 'deaths', dataCheckStatusNote(result.quality))).toBe(
      `${readyHeadline('deaths')} ${sheetNote} Data check could not run.`
    );
  });

  test('a needs_review upload keeps the failed check and the sheet note', () => {
    const result = normalizeUploadResponse({ ...review, sheet_note: sheetNote });
    const text = uploadResultText(result, 'deaths', null);
    expect(text).toBe(`The cohort “deaths” needs review. ${review.message} ${sheetNote}`);
    expect(text).not.toMatch(/ready/);
  });

  test('a needs_review upload from a one-sheet file adds nothing after the check', () => {
    const result = normalizeUploadResponse({ ...review, sheet_note: '' });
    expect(uploadResultText(result, 'deaths', null)).toBe(
      `The cohort “deaths” needs review. ${review.message}`
    );
  });
});
