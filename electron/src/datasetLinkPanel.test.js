// SPDX-License-Identifier: MIT
/** @vitest-environment jsdom */
import { describe, test, expect, vi } from 'vitest';
import {
  closeDatasetLinkForm,
  formatFetchDate,
  linkProblem,
  methodsSentence,
  normalizeSource,
  renderDatasetLinkForm,
  sourceStatusNote,
} from './datasetLinkPanel.js';
import { plainFailureMessage } from './resultPresentation.js';

const SHA = 'a'.repeat(64);
const RAW_SOURCE = {
  url: 'https://data.example.test/rows.csv',
  final_url: 'https://mirror.example.test/rows.csv',
  fetched_at: '2026-11-19T16:20:00+00:00',
  size_bytes: 12345,
  sha256: SHA,
  methods_text: 'Data downloaded from https://mirror.example.test/rows.csv on 19 November 2026.',
};

function setup() {
  const container = document.createElement('div');
  container.className = 'hidden';
  document.body.appendChild(container);
  const onSubmit = vi.fn();
  const onCancel = vi.fn();
  const api = renderDatasetLinkForm(container, { onSubmit, onCancel });
  const q = (id) => container.querySelector(`[data-testid="${id}"]`);
  return { container, onSubmit, onCancel, api, q };
}

describe('linkProblem', () => {
  test.each([
    ['', 'Paste a link first.'],
    ['   ', 'Paste a link first.'],
    [null, 'Paste a link first.'],
    ['http://data.example.test/a.csv', 'https://'],
    ['not a link', 'doesn’t look like a web link'],
    ['ftp://data.example.test/a.csv', 'starts with https://'],
  ])('%j → names what to fix', (text, words) => {
    // Act
    const problem = linkProblem(text);

    // Assert
    expect(problem).toContain(words);
  });

  test('a plain https link has no problem', () => {
    // Act / Assert
    expect(linkProblem('  https://data.example.test/rows.csv?accessType=DOWNLOAD ')).toBe('');
  });
});

describe('normalizeSource', () => {
  test('reads the API fields', () => {
    // Act
    const source = normalizeSource(RAW_SOURCE);

    // Assert
    expect(source).toEqual({
      url: RAW_SOURCE.url,
      finalUrl: RAW_SOURCE.final_url,
      fetchedAt: RAW_SOURCE.fetched_at,
      sizeBytes: 12345,
      sha256: SHA,
      methodsText: RAW_SOURCE.methods_text,
    });
  });

  test.each([
    null,
    undefined,
    'text',
    {},
    { url: 'https://x.example.test' },
    { ...RAW_SOURCE, sha256: '' },
  ])('%j (not a recorded source) → null', (raw) => {
    // Act / Assert
    expect(normalizeSource(raw)).toBeNull();
  });

  test('stored metadata without methods_text gets the same sentence the engine writes', () => {
    // Arrange
    const stored = { ...RAW_SOURCE };
    delete stored.methods_text;
    delete stored.final_url;

    // Act
    const source = normalizeSource(stored);

    // Assert
    expect(source?.finalUrl).toBe(RAW_SOURCE.url);
    expect(source?.methodsText).toBe(
      `Data downloaded from ${RAW_SOURCE.url} on 19 November 2026 (12,345 bytes; SHA-256 ${SHA}).`
    );
  });
});

describe('source wording', () => {
  test('formatFetchDate uses the UTC day and falls back to the raw text', () => {
    // Act / Assert
    expect(formatFetchDate('2026-11-19T23:59:00+00:00')).toBe('19 November 2026');
    expect(formatFetchDate('yesterday')).toBe('yesterday');
  });

  test('sourceStatusNote names the site and the date', () => {
    // Act
    const note = sourceStatusNote(normalizeSource(RAW_SOURCE));

    // Assert
    expect(note).toBe('Source saved: mirror.example.test, 19 November 2026.');
  });

  test('methodsSentence groups the byte count', () => {
    // Act
    const text = methodsSentence({
      finalUrl: 'https://a.example.test/x.csv',
      fetchedAt: '2026-01-02T00:00:00+00:00',
      sizeBytes: 1048576,
      sha256: SHA,
    });

    // Assert
    expect(text).toContain('on 2 January 2026 (1,048,576 bytes;');
  });

  test.each([
    'That link points to this computer or a private network, not a public website. Only public data on the internet can be loaded from a link.',
    'The website says there is no file at that link (error 404). It may have moved: open the page in your browser and copy the download link again.',
    'That link opens a web page, not a data file. On the page, find the Download or Export button, right-click it, choose Copy Link, and paste that link here.',
    'That file is larger than 100 MB, the most this app loads. Choose fewer years, places or columns on the website and download a smaller file.',
  ])('engine sentence reaches the user unchanged: %s', (sentence) => {
    // Arrange: how preload hands a failed response over
    const status = sentence.includes('404') ? 502 : 400;

    // Act
    const shown = plainFailureMessage(`[HTTP ${status}] ${sentence}`, { context: 'upload' });

    // Assert
    expect(shown).toBe(sentence);
  });
});

describe('renderDatasetLinkForm', () => {
  test('labels both fields, explains what is sent, and focuses the link field', () => {
    // Act
    const { container, q } = setup();

    // Assert
    const url = q('dataset-link-url');
    expect(container.classList.contains('hidden')).toBe(false);
    expect(container.querySelector('label[for="dataset-link-url"]')?.textContent).toBe(
      'Link to the data file'
    );
    expect(container.querySelector('label[for="dataset-link-name"]')?.textContent).toContain(
      'optional'
    );
    expect(url.getAttribute('aria-describedby')).toContain('dataset-link-hint');
    expect(container.textContent).toContain('nothing on this computer is sent');
    expect(document.activeElement).toBe(url);
  });

  test('submit with a valid link passes the trimmed link and name', () => {
    // Arrange
    const { q, onSubmit } = setup();
    q('dataset-link-url').value = '  https://data.example.test/rows.csv ';
    q('dataset-link-name').value = ' HCV deaths ';

    // Act
    q('dataset-link-submit').click();

    // Assert
    expect(onSubmit).toHaveBeenCalledWith('https://data.example.test/rows.csv', 'HCV deaths');
  });

  test('submit with an http link shows the problem and does not send', () => {
    // Arrange
    const { q, onSubmit } = setup();
    q('dataset-link-url').value = 'http://data.example.test/rows.csv';

    // Act
    q('dataset-link-submit').click();

    // Assert
    expect(onSubmit).not.toHaveBeenCalled();
    expect(q('dataset-link-error').textContent).toContain('https://');
    expect(q('dataset-link-error').getAttribute('role')).toBe('alert');
    expect(q('dataset-link-url').getAttribute('aria-invalid')).toBe('true');
  });

  test('busy state blocks a second submit and says it is downloading', () => {
    // Arrange
    const { q, api, onSubmit } = setup();
    q('dataset-link-url').value = 'https://data.example.test/rows.csv';

    // Act
    api.setBusy(true);
    q('dataset-link-url').closest('form').requestSubmit();

    // Assert
    expect(onSubmit).not.toHaveBeenCalled();
    expect(q('dataset-link-submit').textContent).toBe('Downloading…');
    expect(q('dataset-link-cancel').disabled).toBe(true);
  });

  test('setError clears with an empty message', () => {
    // Arrange
    const { q, api } = setup();
    api.setError('The website refused the download.');

    // Act
    api.setError('');

    // Assert
    expect(q('dataset-link-error').classList.contains('hidden')).toBe(true);
    expect(q('dataset-link-url').hasAttribute('aria-invalid')).toBe(false);
  });

  test('Escape and Cancel call onCancel; Escape is ignored while busy', () => {
    // Arrange
    const { q, api, onCancel } = setup();
    const escape = () =>
      q('dataset-link-url').dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })
      );

    // Act
    escape();
    q('dataset-link-cancel').click();
    api.setBusy(true);
    escape();

    // Assert
    expect(onCancel).toHaveBeenCalledTimes(2);
  });

  test('closeDatasetLinkForm empties and hides the container', () => {
    // Arrange
    const { container } = setup();

    // Act
    closeDatasetLinkForm(container);

    // Assert
    expect(container.childElementCount).toBe(0);
    expect(container.classList.contains('hidden')).toBe(true);
  });
});
