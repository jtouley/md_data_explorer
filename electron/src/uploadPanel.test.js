/** @vitest-environment jsdom */
import { describe, test, expect, vi } from 'vitest';
import {
  normalizeUploadResponse,
  formatFileSize,
  isAllowedExtension,
  renderUploadArea,
} from './uploadPanel.js';

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
    });
    expect(normalizeUploadResponse(undefined)).toEqual({
      upload_id: '',
      dataset_name: '',
      status: 'failed',
      message: '',
    });
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
});
