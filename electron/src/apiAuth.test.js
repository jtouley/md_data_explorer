import { describe, test, expect } from 'vitest';
import { webcrypto } from 'node:crypto';
import {
  DEFAULT_API_BASE_URL,
  MAX_STREAM_RECONNECTS,
  apiBaseUrlFromArgv,
  apiErrorMessage,
  enrichmentDecisionResult,
  createReplayFilter,
  decideAfterDrop,
  tokenProof,
  parseSseBuffer,
  readSseStream,
  verifyBackendEnforcesAuth,
  withAuth,
} from './apiAuth.js';

function streamOf(chunks) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(encoder.encode(c)));
      controller.close();
    },
  });
}

describe('withAuth', () => {
  test('adds a bearer header when a token is present', () => {
    // Arrange / Act
    const headers = withAuth('tok', { 'Content-Type': 'application/json' });

    // Assert
    expect(headers).toEqual({ 'Content-Type': 'application/json', Authorization: 'Bearer tok' });
  });

  test.each([null, undefined, ''])('adds no header in dev mode (token %j)', (token) => {
    // Arrange / Act
    const headers = withAuth(token, { A: 'b' });

    // Assert
    expect(headers).toEqual({ A: 'b' });
  });
});

describe('parseSseBuffer', () => {
  test.each([
    ['', [], ''],
    ['data: {"a":1}\n\n', ['{"a":1}'], ''],
    ['data: {"a":1}\n\ndata: {"b"', ['{"a":1}'], 'data: {"b"'],
    ['data: one\r\n\r\n', ['one'], ''],
    [': keepalive\n\n', [], ''],
    ['data: line1\ndata: line2\n\n', ['line1\nline2'], ''],
  ])('parses %j', (buffer, events, rest) => {
    // Arrange / Act
    const result = parseSseBuffer(buffer);

    // Assert
    expect(result).toEqual({ events, rest });
  });
});

describe('readSseStream', () => {
  test('emits events split across chunk boundaries in order', async () => {
    // Arrange
    const body = streamOf([
      'data: {"event":"query_st',
      'arted"}\n\ndata: {"event":"stream_end"}',
      '\n\n',
    ]);
    const seen = [];

    // Act
    await readSseStream(body, (d) => seen.push(JSON.parse(d).event));

    // Assert
    expect(seen).toEqual(['query_started', 'stream_end']);
  });

  test('flushes a final event that lacks the trailing blank line', async () => {
    // Arrange
    const body = streamOf(['data: last']);
    const seen = [];

    // Act
    await readSseStream(body, (d) => seen.push(d));

    // Assert
    expect(seen).toEqual(['last']);
  });
});

const TOKEN = 't'.repeat(43);

/** Fake backend: /health answers with a proof computed from `serverToken` (null = no proof). */
function fakeBackend({
  authRequired = true,
  serverToken = TOKEN,
  probeStatus = 401,
  healthJson = true,
} = {}) {
  const calls = [];
  const fn = async (url, init) => {
    calls.push({ url, init });
    const u = new URL(url);
    if (u.pathname === '/health') {
      const nonce = u.searchParams.get('nonce');
      const body = { status: 'healthy', auth_required: authRequired };
      if (serverToken && nonce) body.token_proof = await tokenProof(webcrypto, serverToken, nonce);
      return {
        status: 200,
        json: async () => (healthJson ? body : Promise.reject(new Error('html'))),
      };
    }
    return { status: probeStatus, json: async () => ({}) };
  };
  fn.calls = calls;
  return fn;
}

describe('tokenProof', () => {
  test('matches the Python token_proof for the same token and nonce', async () => {
    // Arrange: vector from clinical_analytics.api.security.token_proof("t" * 43, "0123456789abcdef0123")
    const expected = '380a120a89abe44c000ad0bb1911d93ef57eedc745576e41325f83ba005b5103';

    // Act
    const proof = await tokenProof(webcrypto, TOKEN, '0123456789abcdef0123');

    // Assert
    expect(proof).toBe(expected);
  });
});

describe('verifyBackendEnforcesAuth', () => {
  test('accepts the backend that proves it knows the token, never sending the token', async () => {
    // Arrange
    const fetchFn = fakeBackend();

    // Act
    const verdict = await verifyBackendEnforcesAuth(
      fetchFn,
      'http://localhost:8000',
      TOKEN,
      webcrypto
    );

    // Assert
    expect(verdict).toEqual({ ok: true });
    expect(fetchFn.calls.every((c) => !c.init?.headers?.Authorization)).toBe(true);
    expect(fetchFn.calls.every((c) => !c.url.includes(TOKEN))).toBe(true);
    expect(new URL(fetchFn.calls[0].url).searchParams.get('nonce')).toMatch(/^[0-9a-f]{48}$/);
  });

  test.each([
    ['a squatter that says auth_required and 401s', { serverToken: 'x'.repeat(43) }, 'proof'],
    ['a squatter with no proof at all', { serverToken: null }, 'proof'],
    ['health says no auth', { authRequired: false }, 'health'],
    ['health not JSON', { healthJson: false }, 'health'],
    ['probe answered without token', { probeStatus: 200 }, 'probe'],
  ])('refuses %s', async (_name, backend, reason) => {
    // Arrange
    const fetchFn = fakeBackend(backend);

    // Act
    const verdict = await verifyBackendEnforcesAuth(
      fetchFn,
      'http://localhost:8000',
      TOKEN,
      webcrypto
    );

    // Assert
    expect(verdict).toEqual({ ok: false, reason });
  });

  test.each([null, undefined, ''])(
    'refuses without a token (%j), contacting nothing',
    async (token) => {
      // Arrange
      const fetchFn = fakeBackend();

      // Act
      const verdict = await verifyBackendEnforcesAuth(
        fetchFn,
        'http://localhost:8000',
        token,
        webcrypto
      );

      // Assert
      expect(verdict).toEqual({ ok: false, reason: 'no-token' });
      expect(fetchFn.calls).toEqual([]);
    }
  );
});

describe('createReplayFilter', () => {
  test('delivers each event once across a reconnect that replays from the start', () => {
    // Arrange
    const filter = createReplayFilter();
    const delivered = [];
    const connection = (events) => {
      filter.startConnection();
      events.forEach((e) => filter.shouldDeliver() && delivered.push(e));
    };

    // Act
    connection(['started', 'progress']);
    connection(['started', 'progress', 'completed', 'end']);
    connection(['started', 'progress', 'completed', 'end']);

    // Assert
    expect(delivered).toEqual(['started', 'progress', 'completed', 'end']);
  });
});

describe('decideAfterDrop', () => {
  test.each([
    [200, 0, 'reconnect'],
    [200, MAX_STREAM_RECONNECTS - 1, 'reconnect'],
    [200, MAX_STREAM_RECONNECTS, 'fail'],
    [404, 0, 'gone'],
    [0, 0, 'fail'],
    [500, 0, 'fail'],
    [401, 0, 'fail'],
  ])('status %s after %s reconnects -> %s', (status, reconnects, expected) => {
    // Arrange / Act
    const decision = decideAfterDrop(status, reconnects);

    // Assert
    expect(decision).toBe(expected);
  });
});

describe('API address', () => {
  test('defaults to the IPv4 loopback literal, never localhost', () => {
    // Arrange / Act / Assert
    expect(DEFAULT_API_BASE_URL).toBe('http://127.0.0.1:8000');
  });

  test('preload and CSP use no localhost address for the API', async () => {
    // Arrange
    const { readFileSync } = await import('node:fs');
    const preload = readFileSync(new URL('./preload.js', import.meta.url), 'utf8');
    const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');

    // Act
    const csp = html.match(/connect-src ([^;"]*)/)[1];

    // Assert
    expect(preload).not.toMatch(/localhost:8000/);
    // Outer bound for any context (the sidecar's port varies); main's header narrows it to the
    // exact port, see contentSecurityPolicy.
    expect(csp.trim()).toBe('http://127.0.0.1:*');
  });
});

describe('contentSecurityPolicy (header from main)', () => {
  test('connect-src is exactly the backend origin of this launch', async () => {
    // Arrange
    const { contentSecurityPolicy } = await import('./apiToken.js');

    // Act
    const csp = contentSecurityPolicy('http://127.0.0.1:51234');

    // Assert
    const connect = csp.match(/connect-src ([^;]*)/)[1].trim();
    expect(connect).toBe('http://127.0.0.1:51234');
    expect(csp).not.toMatch(/\*/);
    expect(csp).toContain("default-src 'self'");
  });

  test.each([
    'http://127.0.0.1:*',
    'http://localhost:8000',
    'http://127.0.0.1:0',
    'http://127.0.0.1:65536',
    'https://127.0.0.1:8000',
    '',
    null,
  ])('refuses %j', async (url) => {
    const { contentSecurityPolicy } = await import('./apiToken.js');
    expect(() => contentSecurityPolicy(url)).toThrow(/127\.0\.0\.1/);
  });
});

describe('apiBaseUrlFromArgv', () => {
  test.each([
    [
      'sidecar port',
      ['electron', '--clinical-api-base-url=http://127.0.0.1:54321'],
      'http://127.0.0.1:54321',
    ],
    ['lowest port', ['--clinical-api-base-url=http://127.0.0.1:1'], 'http://127.0.0.1:1'],
    ['highest port', ['--clinical-api-base-url=http://127.0.0.1:65535'], 'http://127.0.0.1:65535'],
    ['no argument', ['electron', '.'], null],
    ['undefined argv', undefined, null],
    ['port zero', ['--clinical-api-base-url=http://127.0.0.1:0'], null],
    ['port too large', ['--clinical-api-base-url=http://127.0.0.1:65536'], null],
    ['localhost', ['--clinical-api-base-url=http://localhost:8000'], null],
    ['remote host', ['--clinical-api-base-url=http://evil.example:8000'], null],
    ['https', ['--clinical-api-base-url=https://127.0.0.1:8000'], null],
    ['path suffix', ['--clinical-api-base-url=http://127.0.0.1:8000/x'], null],
    ['userinfo', ['--clinical-api-base-url=http://a@127.0.0.1:8000'], null],
    ['empty value', ['--clinical-api-base-url='], null],
  ])('%s', (_name, argv, expected) => {
    // Act
    const url = apiBaseUrlFromArgv(argv);

    // Assert
    expect(url).toBe(expected);
  });
});

describe('apiErrorMessage', () => {
  test.each([
    ['string detail', { detail: 'Patch not found' }, 'Patch not found'],
    [
      'validation detail list',
      { detail: [{ msg: 'Input should be a valid string' }, { msg: 'Field required' }] },
      'Input should be a valid string; Field required',
    ],
    ['legacy message', { message: 'old style' }, 'old style'],
    ['empty detail list', { detail: [] }, 'fallback'],
    ['list without msg', { detail: [{ loc: ['body'] }] }, 'fallback'],
    ['non-JSON body', {}, 'fallback'],
    ['null body', null, 'fallback'],
  ])('%s', (_name, data, expected) => {
    expect(apiErrorMessage(data, 'fallback')).toBe(expected);
  });
});

describe('enrichmentDecisionResult', () => {
  test('returns the body on success', () => {
    const body = { success: true, message: 'Suggestion accepted' };
    expect(enrichmentDecisionResult({ ok: true, status: 200 }, body, 'Accept')).toBe(body);
  });

  test.each([404, 409])('marks %i as stale instead of throwing', (status) => {
    const result = enrichmentDecisionResult(
      { ok: false, status, statusText: 'x' },
      { detail: 'Only a pending patch can be accepted' },
      'Accept'
    );
    expect(result).toEqual({
      success: false,
      stale: true,
      message: 'Only a pending patch can be accepted',
    });
  });

  test('throws the joined validation messages for 422', () => {
    expect(() =>
      enrichmentDecisionResult(
        { ok: false, status: 422, statusText: 'Unprocessable Entity' },
        { detail: [{ msg: 'Input should be a valid string' }] },
        'Reject'
      )
    ).toThrow('Input should be a valid string');
  });

  test('throws with the status text when a 500 has no body', () => {
    expect(() =>
      enrichmentDecisionResult(
        { ok: false, status: 500, statusText: 'Internal Server Error' },
        {},
        'Revert'
      )
    ).toThrow('Revert failed: Internal Server Error');
  });

  test('throws when a 200 body reports success false', () => {
    expect(() =>
      enrichmentDecisionResult(
        { ok: true, status: 200 },
        { success: false, message: 'nope' },
        'Accept'
      )
    ).toThrow('nope');
  });
});
