import { describe, test, expect } from 'vitest';
import path from 'node:path';
import {
  API_TOKEN_ENV,
  APP_ENTRY_URL,
  DEV_NO_AUTH_ENV,
  isTrustedRendererUrl,
  resolveApiToken,
  resolveAppFilePath,
  resolveDevNoAuth,
} from './apiToken.js';

describe('resolveApiToken', () => {
  test('uses the launcher token when one is set', () => {
    // Arrange
    const token = 'x'.repeat(43);

    // Act
    const result = resolveApiToken({ [API_TOKEN_ENV]: token });

    // Assert
    expect(result).toEqual({ token, source: 'env' });
  });

  test.each([undefined, '', '   '])('generates a fresh token when env is %j', (value) => {
    // Arrange
    const env = { [API_TOKEN_ENV]: value };

    // Act
    const first = resolveApiToken(env);
    const second = resolveApiToken(env);

    // Assert
    expect(first.source).toBe('generated');
    expect(first.token.length).toBeGreaterThanOrEqual(32);
    expect(first.token).not.toBe(second.token);
  });

  test('rejects a launcher token too short to resist guessing', () => {
    // Arrange
    const env = { [API_TOKEN_ENV]: 'short' };

    // Act / Assert
    expect(() => resolveApiToken(env)).toThrow(API_TOKEN_ENV);
  });
});

describe('isTrustedRendererUrl', () => {
  test.each([
    [APP_ENTRY_URL, undefined, true],
    ['app://clinical-analytics/assets/x.js', undefined, true],
    ['file:///app/renderer/main_window/index.html', undefined, false],
    ['file:///tmp/evil.html', undefined, false],
    ['app://other-host/index.html', undefined, false],
    ['app://user:pw@clinical-analytics/index.html', undefined, false],
    ['http://localhost:5173/', 'http://localhost:5173', true],
    ['http://localhost:5173/index.html', 'http://localhost:5173/', true],
    [APP_ENTRY_URL, 'http://localhost:5173', false],
    ['https://evil.example/', 'http://localhost:5173', false],
    ['http://localhost:5173/', undefined, false],
    [undefined, 'http://localhost:5173', false],
    ['not a url', 'http://localhost:5173', false],
  ])('%s with dev server %s -> %s', (frameUrl, devUrl, expected) => {
    // Arrange / Act
    const trusted = isTrustedRendererUrl(frameUrl, devUrl);

    // Assert
    expect(trusted).toBe(expected);
  });
});

describe('resolveAppFilePath', () => {
  const root = path.resolve('/opt/app/renderer/main_window');

  test.each([
    ['app://clinical-analytics/', path.join(root, 'index.html')],
    ['app://clinical-analytics/index.html', path.join(root, 'index.html')],
    ['app://clinical-analytics/assets/index-abc.js', path.join(root, 'assets', 'index-abc.js')],
    ['app://clinical-analytics/a%20b.css', path.join(root, 'a b.css')],
    // The URL parser collapses dot segments, so these stay inside the build.
    ['app://clinical-analytics/../../etc/passwd', path.join(root, 'etc', 'passwd')],
    ['app://clinical-analytics/%2e%2e/%2e%2e/etc/passwd', path.join(root, 'etc', 'passwd')],
  ])('maps %s inside the renderer build', (url, expected) => {
    // Arrange / Act
    const file = resolveAppFilePath(url, root);

    // Assert
    expect(file).toBe(expected);
  });

  test.each([
    'app://clinical-analytics/..%2f..%2fsecret',
    'app://other-host/index.html',
    'file:///etc/passwd',
    'app://clinical-analytics/%E0%A4%A',
    'nonsense',
  ])('refuses %s', (url) => {
    // Arrange / Act
    const file = resolveAppFilePath(url, root);

    // Assert
    expect(file).toBeNull();
  });
});

describe('resolveDevNoAuth', () => {
  test.each([
    [undefined, false],
    ['', false],
    ['0', false],
    ['true', false],
    ['1', true],
    [' 1 ', true],
  ])('%j -> %s', (value, expected) => {
    // Arrange / Act
    const result = resolveDevNoAuth({ [DEV_NO_AUTH_ENV]: value });

    // Assert
    expect(result).toBe(expected);
  });
});
