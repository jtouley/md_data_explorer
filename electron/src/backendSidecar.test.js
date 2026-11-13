import { describe, test, expect, vi } from 'vitest';
import { execFileSync } from 'node:child_process';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { webcrypto } from 'node:crypto';
import { tokenProof } from './apiAuth.js';
import {
  BACKEND_HOST,
  BackendSidecar,
  DEFAULT_PREFERRED_PORT,
  STORE_CHECK_FAILED_EXIT,
  BACKEND_CWD_NAME,
  buildBackendArgs,
  buildBackendEnv,
  ensureBackendCwd,
  resolveSidecarSettings,
  cannotStartLine,
  choosePort,
  ephemeralPort,
  isPortFree,
  resolveHealthTimeoutMs,
  resolvePreferredPort,
  resolvePythonExecutable,
  startupFailureMessage,
  stopProcessTree,
  unexpectedExitMessage,
  useExternalBackend,
  waitForHealth,
} from './backendSidecar.js';

const TOKEN = 'a'.repeat(43);
const STORE_REASON = 'The data store is open in another program. Close it and start again.';
const STORE_LINE = `Clinical Analytics cannot start: ${STORE_REASON}`;

/** A child process stand-in: emits stdout/stderr data and exit like ChildProcess. */
function makeFakeChild({ pid = 4321, exitOnSignal = 'SIGTERM' } = {}) {
  const child = new EventEmitter();
  child.pid = pid;
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.exitOnSignal = exitOnSignal;
  child.signals = [];
  child.receive = (signal) => {
    child.signals.push(signal);
    if (signal === 'SIGKILL' || signal === child.exitOnSignal) {
      setImmediate(() => child.emit('exit', null, signal));
    }
  };
  return child;
}

/** fetch stand-in for a real backend that knows TOKEN (health proof + 401 probe). */
function makeBackendFetch({ refuseFirst = 0, token = TOKEN } = {}) {
  let calls = 0;
  return vi.fn(async (url) => {
    calls += 1;
    if (calls <= refuseFirst) throw new TypeError('fetch failed: ECONNREFUSED');
    const u = new URL(url);
    if (u.pathname === '/health') {
      const nonce = u.searchParams.get('nonce');
      const body = { status: 'healthy', auth_required: true };
      if (nonce) body.token_proof = await tokenProof(webcrypto, token, nonce);
      return { ok: true, status: 200, json: async () => body };
    }
    return { ok: false, status: 401, json: async () => ({}) };
  });
}

function makeSidecar(overrides = {}) {
  const child = overrides.child || makeFakeChild();
  const spawnFn = vi.fn(() => child);
  const killFn = vi.fn((pid, signal) => {
    expect(pid).toBe(-child.pid);
    child.receive(signal);
  });
  const sidecar = new BackendSidecar({
    python: '/opt/app/python/bin/python3',
    token: TOKEN,
    env: { PATH: '/usr/bin', CLINICAL_ANALYTICS_DEV_NO_AUTH: '1' },
    preferredPort: 8000,
    healthTimeoutMs: 2000,
    parentPid: 999,
    cwd: '/home/doctor',
    platform: 'linux',
    spawnFn,
    killFn,
    fetchFn: makeBackendFetch(),
    cryptoImpl: webcrypto,
    choosePortFn: async () => 51234,
    stopGraceMs: 50,
    forceKillWaitMs: 50,
    stdioDrainMs: 5,
    pollMs: 1,
    ...overrides,
  });
  return { sidecar, child, spawnFn, killFn };
}

describe('configuration', () => {
  test.each([
    [{}, false],
    [{ CLINICAL_ANALYTICS_EXTERNAL_BACKEND: '1' }, true],
    [{ CLINICAL_ANALYTICS_EXTERNAL_BACKEND: ' 1 ' }, true],
    [{ CLINICAL_ANALYTICS_EXTERNAL_BACKEND: 'true' }, false],
    [{ CLINICAL_ANALYTICS_EXTERNAL_BACKEND: '' }, false],
  ])('useExternalBackend(%o) -> %s', (env, expected) => {
    expect(useExternalBackend(env)).toBe(expected);
  });

  test('python: explicit CLINICAL_ANALYTICS_PYTHON wins (step 2 points it at the bundled runtime)', () => {
    // Act
    const python = resolvePythonExecutable(
      { CLINICAL_ANALYTICS_PYTHON: '/Applications/CA.app/python/bin/python3' },
      { appRoot: '/repo/electron' }
    );

    // Assert
    expect(python).toBe('/Applications/CA.app/python/bin/python3');
  });

  test.each([
    ['linux', path.join('/repo', '.venv', 'bin', 'python')],
    ['darwin', path.join('/repo', '.venv', 'bin', 'python')],
    ['win32', path.join('/repo', '.venv', 'Scripts', 'python.exe')],
  ])('python: defaults to the dev venv on %s', (platform, expected) => {
    expect(resolvePythonExecutable({}, { appRoot: '/repo/electron', platform })).toBe(expected);
  });

  test.each([
    ['', DEFAULT_PREFERRED_PORT],
    ['8123', 8123],
    ['1', 1],
    ['65535', 65535],
  ])('preferred port "%s" -> %d', (raw, expected) => {
    expect(resolvePreferredPort({ CLINICAL_ANALYTICS_API_PORT: raw })).toBe(expected);
  });

  test.each(['0', '65536', 'abc', '80.5', '-1'])('preferred port "%s" is rejected', (raw) => {
    expect(() => resolvePreferredPort({ CLINICAL_ANALYTICS_API_PORT: raw })).toThrow(/port number/);
  });

  test.each([
    ['', 60000],
    ['1500', 1500],
  ])('health timeout "%s" -> %d', (raw, expected) => {
    expect(resolveHealthTimeoutMs({ CLINICAL_ANALYTICS_BACKEND_TIMEOUT_MS: raw })).toBe(expected);
  });

  test.each(['0', '-5', 'soon'])('health timeout "%s" is rejected', (raw) => {
    expect(() => resolveHealthTimeoutMs({ CLINICAL_ANALYTICS_BACKEND_TIMEOUT_MS: raw })).toThrow();
  });

  test.each([
    [{}, { preferredPort: DEFAULT_PREFERRED_PORT, healthTimeoutMs: 60000 }],
    [
      { CLINICAL_ANALYTICS_API_PORT: '8123', CLINICAL_ANALYTICS_BACKEND_TIMEOUT_MS: '1500' },
      { preferredPort: 8123, healthTimeoutMs: 1500 },
    ],
  ])('sidecar settings from %o', (env, expected) => {
    expect(resolveSidecarSettings(env)).toEqual(expected);
  });

  test.each([
    [{ CLINICAL_ANALYTICS_API_PORT: 'abc' }, /CLINICAL_ANALYTICS_API_PORT must be a port number/],
    [{ CLINICAL_ANALYTICS_API_PORT: '0' }, /CLINICAL_ANALYTICS_API_PORT/],
    [{ CLINICAL_ANALYTICS_API_PORT: '70000' }, /CLINICAL_ANALYTICS_API_PORT/],
    [{ CLINICAL_ANALYTICS_BACKEND_TIMEOUT_MS: 'soon' }, /CLINICAL_ANALYTICS_BACKEND_TIMEOUT_MS/],
    [{ CLINICAL_ANALYTICS_BACKEND_TIMEOUT_MS: '-5' }, /CLINICAL_ANALYTICS_BACKEND_TIMEOUT_MS/],
  ])('a bad setting %o is a config failure: correct the setting, not reinstall', (env, pattern) => {
    // Act
    let failure;
    try {
      resolveSidecarSettings(env);
    } catch (thrown) {
      failure = thrown;
    }
    const { message } = startupFailureMessage(failure);

    // Assert
    expect(failure).toMatchObject({ reason: 'config', detail: expect.stringMatching(pattern) });
    expect(message).toMatch(pattern);
    expect(message).toMatch(/\. Correct the setting, then start Clinical Analytics again\.$/);
    expect(message).not.toMatch(/[Rr]einstall/);
  });

  test('backend cwd is a folder of the app, created owner-only; never the home directory', () => {
    // Arrange
    const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'ca-userdata-'));
    try {
      // Act
      const cwd = ensureBackendCwd(userData);
      const again = ensureBackendCwd(userData); // a second launch finds it there

      // Assert
      expect(cwd).toBe(path.join(userData, BACKEND_CWD_NAME));
      expect(again).toBe(cwd);
      expect(fs.statSync(cwd).isDirectory()).toBe(true);
      if (process.platform !== 'win32') expect(fs.statSync(cwd).mode & 0o777).toBe(0o700);
      expect(cwd).not.toBe(os.homedir());
    } finally {
      fs.rmSync(userData, { recursive: true, force: true });
    }
  });

  test.each([[''], [undefined], ['relative/dir']])(
    'backend cwd: user-data dir %j is refused',
    (dir) => {
      const mkdirSync = vi.fn();
      expect(() => ensureBackendCwd(dir, { mkdirSync })).toThrow(/absolute path/);
      expect(mkdirSync).not.toHaveBeenCalled();
    }
  );

  test.each([['EACCES'], ['ENOSPC'], ['EROFS']])(
    'backend cwd: a folder that cannot be created (%s) fails as a folder problem naming it',
    (code) => {
      // Arrange
      const mkdirSync = () => {
        throw Object.assign(new Error(`${code}: cannot create`), { code });
      };

      // Act
      let failure;
      try {
        ensureBackendCwd('/var/app-data', { mkdirSync });
      } catch (error) {
        failure = error;
      }

      // Assert: its own reason, with the folder, so the message is not "reinstall"
      expect(failure).toEqual({
        reason: 'folder',
        folder: path.join('/var/app-data', BACKEND_CWD_NAME),
        detail: code,
      });
    }
  );

  // Where each case plants its file: the working directory, or the user site-packages of a
  // home directory made for the test (found by asking the interpreter itself).
  const userSite = (python, env) =>
    execFileSync(python, ['-c', 'import site; print(site.getusersitepackages())'], { env })
      .toString()
      .trim();
  test.each([
    ['json.py', 'the working directory'],
    ['argparse.py', 'the working directory'],
    ['usercustomize.py', 'user site-packages'],
  ])('a planted %s in %s is not imported by the interpreter the app starts', (file, where) => {
    // Arrange: a real interpreter, started with the app's flags, with a shadow module in reach
    const python = process.platform === 'win32' ? 'python' : 'python3';
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ca-shadow-'));
    const cwd = path.join(root, 'cwd');
    const home = path.join(root, 'home');
    fs.mkdirSync(cwd);
    fs.mkdirSync(home);
    const env = { ...process.env, HOME: home, APPDATA: home, USERPROFILE: home };
    delete env.PYTHONUSERBASE;
    delete env.PYTHONNOUSERSITE;
    const dir = where === 'user site-packages' ? userSite(python, env) : cwd;
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, file), 'raise SystemExit("planted module ran")\n');
    const args = buildBackendArgs(51234, 999);
    const interpreterFlags = args.slice(0, args.indexOf('-m'));
    // json.tool imports json and argparse; it adds nothing to sys.path itself
    const probe = ['-m', 'json.tool'];
    const run = (flags) => {
      try {
        execFileSync(python, [...flags, ...probe], { cwd, env, input: '{}', stdio: 'pipe' });
        return 'real module';
      } catch (error) {
        return String(error.stderr).includes('planted module ran') ? 'planted' : `failed: ${error}`;
      }
    };
    try {
      // Act
      const withoutFlags = run([]);
      const withAppFlags = run(interpreterFlags);

      // Assert: a plain interpreter imports the planted file; the app's isolated one does not
      expect(withoutFlags).toBe('planted');
      expect(withAppFlags).toBe('real module');
      expect(interpreterFlags).toEqual(['-I', '-u']); // -I: no cwd, user site or PYTHON* vars
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  test('backend args: isolated interpreter first, bind loopback, pass the port and the parent pid', () => {
    expect(buildBackendArgs(51234, 999)).toEqual([
      '-I',
      '-u',
      '-m',
      'clinical_analytics.api.serve',
      '--host',
      '127.0.0.1',
      '--port',
      '51234',
      '--exit-with-parent',
      '999',
      '--stop-on-stdin-close',
    ]);
  });

  test('backend env carries the token and drops the dev opt-out and Ollama auto-download', () => {
    // Arrange
    const base = {
      PATH: '/usr/bin',
      PYTHONPATH: '/home/doctor/scripts',
      CLINICAL_ANALYTICS_DEV_NO_AUTH: '1',
      CLINICAL_ANALYTICS_OLLAMA_AUTOINIT: '1',
      ELECTRON_RUN_AS_NODE: '1',
    };

    // Act
    const env = buildBackendEnv(base, TOKEN);

    // Assert
    expect(env.CLINICAL_ANALYTICS_API_TOKEN).toBe(TOKEN);
    expect(env.PATH).toBe('/usr/bin');
    expect(env).not.toHaveProperty('CLINICAL_ANALYTICS_DEV_NO_AUTH');
    expect(env).not.toHaveProperty('CLINICAL_ANALYTICS_OLLAMA_AUTOINIT');
    expect(env).not.toHaveProperty('ELECTRON_RUN_AS_NODE');
    expect(env).not.toHaveProperty('PYTHONPATH'); // no import path from the user's shell
    expect(base.CLINICAL_ANALYTICS_DEV_NO_AUTH).toBe('1'); // input not mutated
  });
});

describe('port selection', () => {
  test('uses the preferred port when it is free', async () => {
    const pickFree = vi.fn(async () => 50000);
    await expect(choosePort(8000, { isFree: async () => true, pickFree })).resolves.toBe(8000);
    expect(pickFree).not.toHaveBeenCalled();
  });

  test('falls back to a free port when the preferred one is taken', async () => {
    await expect(
      choosePort(8000, { isFree: async () => false, pickFree: async () => 50001 })
    ).resolves.toBe(50001);
  });

  test('isPortFree sees a real listener on 127.0.0.1, and ephemeralPort finds a free port', async () => {
    // Arrange: occupy a port
    const server = net.createServer();
    await new Promise((resolve) => server.listen({ port: 0, host: BACKEND_HOST }, resolve));
    const busy = server.address().port;

    try {
      // Act / Assert
      await expect(isPortFree(busy)).resolves.toBe(false);
      const free = await ephemeralPort();
      expect(free).not.toBe(busy);
      await expect(isPortFree(free)).resolves.toBe(true);
      await expect(choosePort(busy)).resolves.not.toBe(busy);
    } finally {
      await new Promise((resolve) => server.close(resolve));
    }
  });
});

describe('startup failure messages', () => {
  test('a folder that cannot be created names the folder and does not say reinstall', () => {
    // Act
    const { message } = startupFailureMessage({
      reason: 'folder',
      folder: '/Users/dr/Library/Application Support/Clinical Analytics/backend',
      detail: 'EACCES',
    });

    // Assert
    expect(message).toBe(
      'Clinical Analytics could not create its working folder ' +
        '"/Users/dr/Library/Application Support/Clinical Analytics/backend" (EACCES). ' +
        'Check that this folder, and the folders around it, can be written by your user ' +
        'account and that the disk is not full, then start Clinical Analytics again.'
    );
    expect(message).not.toMatch(/reinstall/i);
  });

  test('exit code 2 shows the one-line reason serve.py printed, without repeating the title', () => {
    // Act
    const { title, message } = startupFailureMessage({
      reason: 'exited',
      code: STORE_CHECK_FAILED_EXIT,
      signal: null,
      stderrText: `INFO something\n${STORE_LINE}\n`,
    });

    // Assert
    expect(title).toBe('Clinical Analytics could not start');
    expect(message).toBe(STORE_REASON);
  });

  test.each([
    [
      { reason: 'exited', code: 1, signal: null, stderrText: 'Traceback\nImportError: boom' },
      /stopped before it was ready \(exit code 1\)\. Start Clinical Analytics again\.[\s\S]*ImportError: boom/,
    ],
    [
      { reason: 'exited', code: 2, signal: null, stderrText: 'usage: serve.py [-h]' },
      /\(exit code 2\)/,
    ],
    [
      { reason: 'exited', code: null, signal: 'SIGKILL', stderrText: '' },
      /stopped before it was ready \(signal SIGKILL\)\. Start Clinical Analytics again\. If this keeps happening, reinstall the app\.$/,
    ],
    [
      { reason: 'timeout', timeoutMs: 60000 },
      /^The analysis engine was not ready after 60 seconds\. Start Clinical Analytics again\.$/,
    ],
    [
      { reason: 'untrusted', detail: 'proof' },
      /^Another program is using the port .* Start Clinical Analytics again\. It will choose a free port\.$/,
    ],
    [
      { reason: 'spawn', detail: 'spawn /x/python ENOENT' },
      /could not be started \(spawn \/x\/python ENOENT\)\. Reinstall Clinical Analytics\.$/,
    ],
    [{ reason: 'spawn' }, /\(unknown error\)/],
    [
      { reason: 'config', detail: 'VITE_API_URL must be http://127.0.0.1:<port>' },
      /^VITE_API_URL must be/,
    ],
    [
      { reason: 'config' },
      /^A startup setting is not valid\. Correct the setting, then start Clinical Analytics again\.$/,
    ],
  ])('%o', (failure, pattern) => {
    expect(startupFailureMessage(failure).message).toMatch(pattern);
  });

  test.each([
    [{ reason: 'exited', code: 1, signal: null, stderrText: 'x' }],
    [{ reason: 'timeout', timeoutMs: 1000 }],
    [{ reason: 'untrusted' }],
    [{ reason: 'spawn', detail: 'EACCES' }],
  ])("%o: says what to do next, in the copy deck's words", (failure) => {
    // Act
    const { message } = startupFailureMessage(failure);

    // Assert: "analysis engine" (DESIGN.md), a next step, no "service", no exclamation
    expect(message).toMatch(/analysis engine/);
    expect(message).toMatch(/Start Clinical Analytics again|Reinstall Clinical Analytics/);
    expect(message).not.toMatch(/service|!|Oops|Please/);
  });

  test.each([
    ['', null],
    ['no reason here', null],
    [`${STORE_LINE}\r\n`, STORE_LINE],
    [
      'Clinical Analytics cannot start: first\nClinical Analytics cannot start: last',
      'Clinical Analytics cannot start: last',
    ],
  ])('cannotStartLine(%j)', (text, expected) => {
    expect(cannotStartLine(text)).toBe(expected);
  });
});

describe('waitForHealth', () => {
  const base = {
    baseUrl: 'http://127.0.0.1:51234',
    token: TOKEN,
    cryptoImpl: webcrypto,
    pollMs: 1,
  };

  test('retries while the port is closed, then accepts a backend that proves the token', async () => {
    // Arrange
    const fetchFn = makeBackendFetch({ refuseFirst: 3 });

    // Act
    const result = await waitForHealth({ ...base, fetchFn, timeoutMs: 5000, exited: () => null });

    // Assert
    expect(result).toBe('ready');
    expect(fetchFn.mock.calls.some(([url]) => url.includes('nonce='))).toBe(true);
  });

  test('a squatter that cannot prove the token is refused', async () => {
    const fetchFn = makeBackendFetch({ token: 'b'.repeat(43) });
    await expect(
      waitForHealth({ ...base, fetchFn, timeoutMs: 5000, exited: () => null })
    ).rejects.toMatchObject({
      reason: 'untrusted',
      detail: 'proof',
    });
  });

  test.each([[404], [500], [503], [301]])(
    'a program on the port that answers /health with %i goes to the token check and is refused at once',
    async (status) => {
      // Arrange: not this app's backend (no auth_required, no proof); a long deadline
      const fetchFn = vi.fn(async () => ({ ok: false, status, json: async () => ({}) }));
      const started = Date.now();

      // Act
      const failure = await waitForHealth({
        ...base,
        fetchFn,
        timeoutMs: 60000,
        exited: () => null,
      }).catch((f) => f);

      // Assert: 'untrusted' promptly, not a 60 s wait ending in 'timeout'
      expect(failure).toMatchObject({ reason: 'untrusted', detail: 'health' });
      expect(Date.now() - started).toBeLessThan(2000);
      expect(fetchFn).toHaveBeenCalledTimes(2); // /health, then /health?nonce=
      expect(startupFailureMessage(failure).message).toMatch(/Another program is using the port/);
    }
  );

  test('stopped by the app while waiting: rejects with "stopped" before any other reason', async () => {
    const fetchFn = makeBackendFetch({ refuseFirst: 1000 });
    const exit = { code: null, signal: 'SIGTERM', stderrText: '' };
    await expect(
      waitForHealth({
        ...base,
        fetchFn,
        timeoutMs: 5000,
        exited: () => exit,
        stopped: () => true,
      })
    ).rejects.toEqual({ reason: 'stopped' });
  });

  test('stops waiting as soon as the backend exits', async () => {
    const fetchFn = makeBackendFetch({ refuseFirst: 1000 });
    const exit = { code: 2, signal: null, stderrText: STORE_LINE };
    await expect(
      waitForHealth({ ...base, fetchFn, timeoutMs: 5000, exited: () => exit })
    ).rejects.toMatchObject({
      reason: 'exited',
      code: 2,
    });
    expect(fetchFn).not.toHaveBeenCalled();
  });

  test('times out', async () => {
    // Arrange: a fake clock that jumps 400 ms per poll
    let t = 0;
    const fetchFn = makeBackendFetch({ refuseFirst: 1000 });

    // Act / Assert
    await expect(
      waitForHealth({
        ...base,
        fetchFn,
        timeoutMs: 1000,
        exited: () => null,
        now: () => t,
        wait: async () => {
          t += 400;
        },
      })
    ).rejects.toMatchObject({ reason: 'timeout', timeoutMs: 1000 });
    expect(fetchFn).toHaveBeenCalledTimes(3); // t = 0, 400, 800; none after the deadline
  });
});

describe('BackendSidecar', () => {
  test('spawns the configured python on the chosen port, detached, with the token', async () => {
    // Arrange
    const { sidecar, spawnFn } = makeSidecar();

    // Act
    const url = await sidecar.spawn();

    // Assert
    expect(url).toBe('http://127.0.0.1:51234');
    const [python, args, options] = spawnFn.mock.calls[0];
    expect(python).toBe('/opt/app/python/bin/python3');
    expect(args).toEqual(buildBackendArgs(51234, 999));
    expect(options.detached).toBe(true);
    expect(options.cwd).toBe('/home/doctor');
    expect(options.stdio).toEqual(['pipe', 'pipe', 'pipe']); // stdin: the stop lifeline
    expect(options.env.CLINICAL_ANALYTICS_API_TOKEN).toBe(TOKEN);
    expect(options.env).not.toHaveProperty('CLINICAL_ANALYTICS_DEV_NO_AUTH');
  });

  test('is not detached on Windows (taskkill /T stops the tree there)', async () => {
    const { sidecar, spawnFn } = makeSidecar({ platform: 'win32' });
    await sidecar.spawn();
    expect(spawnFn.mock.calls[0][2].detached).toBe(false);
  });

  test('becomes ready when the backend answers /health with the token proof', async () => {
    const { sidecar } = makeSidecar({ fetchFn: makeBackendFetch({ refuseFirst: 2 }) });
    await sidecar.spawn();
    await expect(sidecar.waitUntilReady()).resolves.toBe('ready');
  });

  test('exit code 2 before /health: the failure carries the store-check line for the error window', async () => {
    // Arrange
    const { sidecar, child } = makeSidecar({ fetchFn: makeBackendFetch({ refuseFirst: 1e9 }) });
    await sidecar.spawn();

    // Act
    child.stderr.emit('data', Buffer.from(`starting\n${STORE_LINE}\n`));
    child.emit('exit', STORE_CHECK_FAILED_EXIT, null);
    const failure = await sidecar.waitUntilReady().catch((f) => f);

    // Assert
    expect(failure).toMatchObject({ reason: 'exited', code: 2 });
    expect(startupFailureMessage(failure).message).toBe(STORE_REASON);
  });

  test("'exit' before the last stderr chunk: the reason that arrives before 'close' is kept", async () => {
    // Arrange: Node may emit 'exit' while stderr still holds the final line
    const { sidecar, child } = makeSidecar({
      fetchFn: makeBackendFetch({ refuseFirst: 1e9 }),
      stdioDrainMs: 5000,
    });
    await sidecar.spawn();
    const waiting = sidecar.waitUntilReady().catch((f) => f);

    // Act
    child.emit('exit', STORE_CHECK_FAILED_EXIT, null);
    await new Promise((resolve) => setTimeout(resolve, 20)); // polls run; the exit is not final yet
    child.stderr.emit('data', Buffer.from(`${STORE_LINE}\n`));
    child.emit('close', STORE_CHECK_FAILED_EXIT, null);
    const failure = await waiting;

    // Assert
    expect(failure).toMatchObject({ reason: 'exited', code: 2, stderrText: STORE_LINE });
    expect(startupFailureMessage(failure).message).toBe(STORE_REASON);
  });

  test("pipes held open after 'exit' (no 'close'): the exit is still reported after the drain wait", async () => {
    // Arrange
    const { sidecar, child } = makeSidecar({ stdioDrainMs: 10 });
    await sidecar.spawn();

    // Act
    child.emit('exit', 1, null);
    const info = await sidecar.exitPromise;

    // Assert
    expect(info).toMatchObject({ code: 1, signal: null });
  });

  test('stop() while still starting: the wait ends as "stopped", not as a failure to report', async () => {
    // Arrange: the backend never answers; the user quits during the wait
    const { sidecar, killFn } = makeSidecar({
      fetchFn: makeBackendFetch({ refuseFirst: 1e9 }),
      healthTimeoutMs: 5000,
    });
    await sidecar.spawn();
    const waiting = sidecar.waitUntilReady().catch((f) => f);

    // Act
    const how = await sidecar.stop();
    const failure = await waiting;

    // Assert: SIGTERM was the app's own request, so there is nothing to tell the user
    expect(how).toBe('terminated');
    expect(killFn).toHaveBeenCalledWith(-4321, 'SIGTERM');
    expect(failure).toEqual({ reason: 'stopped' });
  });

  test('stop() while the port is still being chosen: nothing is spawned', async () => {
    // Arrange
    let releasePort;
    const { sidecar, spawnFn } = makeSidecar({
      choosePortFn: () => new Promise((resolve) => (releasePort = () => resolve(51234))),
    });
    const spawning = sidecar.spawn().catch((f) => f);

    // Act
    await expect(sidecar.stop()).resolves.toBe('not-started');
    releasePort();

    // Assert
    expect(await spawning).toEqual({ reason: 'stopped' });
    expect(spawnFn).not.toHaveBeenCalled();
  });

  test('a missing python (spawn error event) is reported as a spawn failure', async () => {
    const { sidecar, child } = makeSidecar({ fetchFn: makeBackendFetch({ refuseFirst: 1e9 }) });
    await sidecar.spawn();
    child.emit(
      'error',
      Object.assign(new Error('spawn /opt/app/python/bin/python3 ENOENT'), { code: 'ENOENT' })
    );
    await expect(sidecar.waitUntilReady()).rejects.toMatchObject({
      reason: 'spawn',
      detail: expect.stringMatching(/ENOENT/),
    });
  });

  test('a synchronous spawn throw is reported as a spawn failure', async () => {
    const { sidecar } = makeSidecar({
      spawnFn: () => {
        throw new Error('EACCES');
      },
    });
    await expect(sidecar.spawn()).rejects.toMatchObject({ reason: 'spawn', detail: 'EACCES' });
  });

  test('keeps only the last 40 stderr lines', async () => {
    const { sidecar, child } = makeSidecar();
    await sidecar.spawn();
    child.stderr.emit('data', Array.from({ length: 100 }, (_, i) => `line ${i}`).join('\n'));
    expect(sidecar.stderrLines).toHaveLength(40);
    expect(sidecar.stderrLines.at(-1)).toBe('line 99');
  });

  test('stop sends SIGTERM to the process group and resolves when it exits', async () => {
    // Arrange
    const { sidecar, child, killFn } = makeSidecar();
    await sidecar.spawn();

    // Act
    const how = await sidecar.stop();

    // Assert
    expect(how).toBe('terminated');
    expect(killFn).toHaveBeenCalledTimes(1);
    expect(killFn).toHaveBeenCalledWith(-child.pid, 'SIGTERM');
  });

  test('stop kills the group after the grace period when SIGTERM is ignored', async () => {
    // Arrange
    const child = makeFakeChild({ exitOnSignal: 'NONE' });
    const { sidecar, killFn } = makeSidecar({ child });
    await sidecar.spawn();

    // Act
    const how = await sidecar.stop();

    // Assert
    expect(how).toBe('killed');
    expect(killFn.mock.calls.map((c) => c[1])).toEqual(['SIGTERM', 'SIGKILL']);
  });

  test('stop after the backend already exited sends nothing', async () => {
    const { sidecar, child, killFn } = makeSidecar();
    await sidecar.spawn();
    child.emit('exit', 0, null);
    await sidecar.exitPromise;
    await expect(sidecar.stop()).resolves.toBe('already-exited');
    expect(killFn).not.toHaveBeenCalled();
  });

  test('stop before spawn is a no-op', async () => {
    const { sidecar } = makeSidecar();
    await expect(sidecar.stop()).resolves.toBe('not-started');
  });

  test('killNow SIGKILLs the group synchronously (process exit handler)', async () => {
    const { sidecar, child, killFn } = makeSidecar();
    await sidecar.spawn();
    sidecar.killNow();
    expect(killFn).toHaveBeenCalledWith(-child.pid, 'SIGKILL');
  });
});

describe('stopProcessTree on Windows', () => {
  function windowsChild({ exitsOnStdinClose }) {
    const child = makeFakeChild();
    child.stdin = {
      end: vi.fn(() => exitsOnStdinClose && setImmediate(() => child.emit('exit', 0, null))),
    };
    let exited = false;
    const onExit = new Promise((resolve) => child.once('exit', resolve)).then(() => {
      exited = true;
    });
    const execFileFn = vi.fn((cmd, args, cb) => {
      if (args.includes('/F')) setImmediate(() => child.emit('exit', 1, null));
      cb(null);
    });
    return { child, onExit, execFileFn, isExited: () => exited };
  }

  test('graceful: closes stdin (serve.py --stop-on-stdin-close) and never runs taskkill', async () => {
    // Arrange
    const { child, onExit, execFileFn, isExited } = windowsChild({ exitsOnStdinClose: true });

    // Act
    const how = await stopProcessTree(child, {
      platform: 'win32',
      graceMs: 1000,
      execFileFn,
      isExited,
      onExit,
    });

    // Assert
    expect(how).toBe('terminated');
    expect(child.stdin.end).toHaveBeenCalledTimes(1);
    expect(execFileFn).not.toHaveBeenCalled();
  });

  test('still running after the grace period: taskkill /T /F ends the tree', async () => {
    // Arrange
    const { child, onExit, execFileFn, isExited } = windowsChild({ exitsOnStdinClose: false });

    // Act
    const how = await stopProcessTree(child, {
      platform: 'win32',
      graceMs: 20,
      execFileFn,
      isExited,
      onExit,
    });

    // Assert
    expect(how).toBe('killed');
    expect(child.stdin.end).toHaveBeenCalledTimes(1);
    expect(execFileFn.mock.calls.map((c) => [c[0], c[1]])).toEqual([
      ['taskkill', ['/pid', '4321', '/T', '/F']],
    ]);
  });

  test.each([
    ['taskkill reports an error', new Error('ERROR: Access is denied.')],
    ['taskkill reports success but the process stays', null],
  ])('%s: stop gives up after the bounded wait instead of hanging quit', async (_name, error) => {
    // Arrange: nothing ever ends this child
    const child = makeFakeChild({ exitOnSignal: 'NONE' });
    child.stdin = { end: vi.fn() };
    const execFileFn = vi.fn((cmd, args, cb) => cb(error));
    const log = vi.fn();
    const started = Date.now();

    // Act
    const how = await stopProcessTree(child, {
      platform: 'win32',
      graceMs: 10,
      forceWaitMs: 30,
      execFileFn,
      isExited: () => false,
      onExit: new Promise(() => {}),
      log,
    });

    // Assert
    expect(how).toBe('not-stopped');
    expect(Date.now() - started).toBeLessThan(2000);
    expect(execFileFn).toHaveBeenCalledTimes(1);
    const logged = log.mock.calls.map((c) => c[0]).join('\n');
    expect(logged).toMatch(/still running 30 ms after the forced kill/);
    if (error) expect(logged).toMatch(/taskkill failed: ERROR: Access is denied\./);
  });

  test('POSIX: a process that survives SIGKILL does not hang stop either', async () => {
    // Arrange
    const child = makeFakeChild({ exitOnSignal: 'NONE' });
    const killFn = vi.fn();

    // Act
    const how = await stopProcessTree(child, {
      platform: 'linux',
      graceMs: 5,
      forceWaitMs: 20,
      killFn,
      isExited: () => false,
      onExit: new Promise(() => {}),
    });

    // Assert
    expect(how).toBe('not-stopped');
    expect(killFn.mock.calls.map((c) => c[1])).toEqual(['SIGTERM', 'SIGKILL']);
  });

  test('POSIX never touches stdin: SIGTERM to the group is the graceful request', async () => {
    // Arrange
    const child = makeFakeChild();
    child.stdin = { end: vi.fn() };
    const onExit = new Promise((resolve) => child.once('exit', resolve));
    const killFn = vi.fn((pid, sig) => child.receive(sig));

    // Act
    const how = await stopProcessTree(child, {
      platform: 'darwin',
      graceMs: 1000,
      killFn,
      isExited: () => false,
      onExit,
    });

    // Assert
    expect(how).toBe('terminated');
    expect(killFn).toHaveBeenCalledWith(-4321, 'SIGTERM');
    expect(child.stdin.end).not.toHaveBeenCalled();
  });

  test('a process that is already gone (ESRCH) is not an error', async () => {
    const child = makeFakeChild();
    const onExit = Promise.resolve();
    const killFn = () => {
      throw Object.assign(new Error('kill ESRCH'), { code: 'ESRCH' });
    };
    await expect(
      stopProcessTree(child, {
        platform: 'linux',
        graceMs: 5,
        killFn,
        isExited: () => false,
        onExit,
      })
    ).resolves.toBe('terminated');
  });
});

describe('health wait timeouts', () => {
  const base = {
    baseUrl: 'http://127.0.0.1:51234',
    token: TOKEN,
    cryptoImpl: webcrypto,
    pollMs: 1,
  };
  const never = () => new Promise(() => {});

  test('a squatter that accepts and never answers ends in a timeout, not a stall', async () => {
    // Arrange
    const fetchFn = vi.fn(never);
    const started = Date.now();

    // Act
    const failure = await waitForHealth({
      ...base,
      fetchFn,
      timeoutMs: 300,
      attemptMs: 50,
      exited: () => null,
    }).catch((f) => f);

    // Assert
    expect(failure).toMatchObject({ reason: 'timeout', timeoutMs: 300 });
    expect(Date.now() - started).toBeLessThan(2000);
    expect(fetchFn.mock.calls.length).toBeGreaterThan(1); // retried after each attempt timed out
  });

  test('health answers but the token check hangs: refused as untrusted within the deadline', async () => {
    // Arrange: /health is fine, the nonce request never returns
    const fetchFn = vi.fn(async (url) => {
      if (url.includes('nonce=')) return never();
      return { ok: true, status: 200, json: async () => ({}) };
    });

    // Act
    const failure = await waitForHealth({
      ...base,
      fetchFn,
      timeoutMs: 2000,
      attemptMs: 50,
      exited: () => null,
    }).catch((f) => f);

    // Assert
    expect(failure).toMatchObject({ reason: 'untrusted', detail: 'no-answer' });
  });

  test.each([
    ['the deadline passes before the proof is sent', 'deadline'],
    ['the proof request is cut short by the deadline', 'cut-short'],
  ])(
    'health answers just before the deadline and %s: reported as a timeout, not untrusted',
    async (_label, mode) => {
      // Arrange: a fake clock; /health answers at 59.8 s of a 60 s wait
      let clock = 0;
      const fetchFn = vi.fn(async (url) => {
        if (url.endsWith('/health')) {
          clock = mode === 'deadline' ? 60000 : 59800;
          return { ok: true, status: 200, json: async () => ({}) };
        }
        // the proof request runs until its 200 ms (what is left of the deadline) are used up
        clock = 60000;
        return never();
      });

      // Act
      const failure = await waitForHealth({
        ...base,
        fetchFn,
        timeoutMs: 60000,
        attemptMs: 3000,
        now: () => clock,
        exited: () => null,
      }).catch((f) => f);

      // Assert: a slow start of the app's own backend, not another program on the port
      expect(failure).toMatchObject({ reason: 'timeout', timeoutMs: 60000 });
    }
  );

  test('each attempt carries an abort signal', async () => {
    const fetchFn = makeBackendFetch();
    await waitForHealth({ ...base, fetchFn, timeoutMs: 1000, exited: () => null });
    expect(fetchFn.mock.calls.every(([, init]) => init?.signal instanceof AbortSignal)).toBe(true);
  });
});

describe('unexpected exit after startup', () => {
  test('calls onUnexpectedExit when a ready backend dies on its own', async () => {
    // Arrange
    const onUnexpectedExit = vi.fn();
    const { sidecar, child } = makeSidecar({ onUnexpectedExit });
    await sidecar.spawn();
    await sidecar.waitUntilReady();

    // Act
    child.stderr.emit('data', 'MemoryError\n');
    child.emit('exit', null, 'SIGKILL');
    await sidecar.exitPromise;

    // Assert
    expect(onUnexpectedExit).toHaveBeenCalledTimes(1);
    const text = unexpectedExitMessage(onUnexpectedExit.mock.calls[0][0]);
    expect(text.message).toBe('The analysis engine stopped.');
    expect(text.buttons).toEqual(['Restart engine', 'Quit']);
    expect(text.detail).toMatch(
      /^Restart it to keep working\. Your cohorts are still on this computer\.\n\nDetails: signal SIGKILL\.[\s\S]*MemoryError/
    );
  });

  test.each([
    ['during startup (the error window covers it)', false],
    ['after the app asked it to stop', true],
  ])('is not called %s', async (_name, readyThenStop) => {
    // Arrange
    const onUnexpectedExit = vi.fn();
    const { sidecar, child } = makeSidecar({
      onUnexpectedExit,
      fetchFn: makeBackendFetch({ refuseFirst: readyThenStop ? 0 : 1e9 }),
    });
    await sidecar.spawn();

    // Act
    if (readyThenStop) {
      await sidecar.waitUntilReady();
      await sidecar.stop();
    } else {
      child.emit('exit', 1, null);
      await sidecar.exitPromise;
    }

    // Assert
    expect(onUnexpectedExit).not.toHaveBeenCalled();
  });
});
