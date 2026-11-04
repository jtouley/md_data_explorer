/**
 * The Python backend as an Electron-owned sidecar (main process only; no `electron` import, so
 * every dependency is injectable and unit-tested with a fake child).
 *
 * Lifecycle:
 * 1. Pick a port on 127.0.0.1: the preferred one if free, otherwise one the OS hands out.
 * 2. Spawn `<python> -m clinical_analytics.api.serve --host 127.0.0.1 --port <port>
 *    --exit-with-parent <electron pid>` with this launch's API token in its environment
 *    (never the dev no-auth opt-out, never Ollama auto-download).
 * 3. Wait for /health, then check the server proves it knows the token (apiAuth.js), so a
 *    program squatting on the port is never used.
 * 4. On quit: SIGTERM to the process group, SIGKILL after a grace period (Windows: taskkill /T,
 *    then /T /F). If Electron itself dies, serve.py's parent watchdog exits the backend.
 *
 * The dev flow (separate Vite and `make run-api`, or scripts/run_app.sh) sets
 * CLINICAL_ANALYTICS_EXTERNAL_BACKEND=1 and keeps using an already-running API.
 */
import { spawn as nodeSpawn, execFile as nodeExecFile } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';
import { verifyBackendEnforcesAuth } from './apiAuth.js';

export const BACKEND_HOST = '127.0.0.1';
export const EXTERNAL_BACKEND_ENV = 'CLINICAL_ANALYTICS_EXTERNAL_BACKEND';
export const PYTHON_ENV = 'CLINICAL_ANALYTICS_PYTHON';
export const PORT_ENV = 'CLINICAL_ANALYTICS_API_PORT';
export const HEALTH_TIMEOUT_ENV = 'CLINICAL_ANALYTICS_BACKEND_TIMEOUT_MS';
export const DEFAULT_PREFERRED_PORT = 8000;
export const DEFAULT_HEALTH_TIMEOUT_MS = 60000;
export const DEFAULT_STOP_GRACE_MS = 5000;
/** serve.py exits with this code after printing one "cannot start" line (store check, model). */
export const STORE_CHECK_FAILED_EXIT = 2;
const STDERR_TAIL_LINES = 40;
// Never handed to the sidecar: the dev opt-out would let the renderer accept an unauthenticated
// backend, and auto-init downloads Ollama models.
const STRIPPED_ENV = ['CLINICAL_ANALYTICS_DEV_NO_AUTH', 'CLINICAL_ANALYTICS_OLLAMA_AUTOINIT'];

/** @param {Record<string, string | undefined>} env */
export function useExternalBackend(env) {
  return (env[EXTERNAL_BACKEND_ENV] || '').trim() === '1';
}

/**
 * Python that runs the backend: CLINICAL_ANALYTICS_PYTHON, else the dev venv next to electron/.
 * (Packaging step 2 points the env var, or this default, at the bundled runtime.)
 * @param {Record<string, string | undefined>} env
 * @param {{ appRoot: string, platform?: string }} opts - appRoot: the electron/ directory
 */
export function resolvePythonExecutable(env, { appRoot, platform = process.platform }) {
  const configured = (env[PYTHON_ENV] || '').trim();
  if (configured) return configured;
  const repoRoot = path.resolve(appRoot, '..');
  return platform === 'win32'
    ? path.join(repoRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(repoRoot, '.venv', 'bin', 'python');
}

/** @param {Record<string, string | undefined>} env */
export function resolvePreferredPort(env) {
  const raw = (env[PORT_ENV] || '').trim();
  if (!raw) return DEFAULT_PREFERRED_PORT;
  const port = Number(raw);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`${PORT_ENV} must be a port number (1-65535), got "${raw}"`);
  }
  return port;
}

/** @param {Record<string, string | undefined>} env */
export function resolveHealthTimeoutMs(env) {
  const raw = (env[HEALTH_TIMEOUT_ENV] || '').trim();
  const value = raw ? Number(raw) : DEFAULT_HEALTH_TIMEOUT_MS;
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(
      `${HEALTH_TIMEOUT_ENV} must be a positive number of milliseconds, got "${raw}"`
    );
  }
  return value;
}

/** @param {number} port @param {number} parentPid */
export function buildBackendArgs(port, parentPid) {
  return [
    '-m',
    'clinical_analytics.api.serve',
    '--host',
    BACKEND_HOST,
    '--port',
    String(port),
    '--exit-with-parent',
    String(parentPid),
  ];
}

/**
 * @param {Record<string, string | undefined>} baseEnv
 * @param {string} token
 */
export function buildBackendEnv(baseEnv, token) {
  const env = { ...baseEnv, CLINICAL_ANALYTICS_API_TOKEN: token, PYTHONUNBUFFERED: '1' };
  for (const name of STRIPPED_ENV) delete env[name];
  delete env.ELECTRON_RUN_AS_NODE;
  return env;
}

/**
 * True if nothing listens on host:port (we can bind it ourselves).
 * @param {number} port
 * @param {string} [host]
 * @returns {Promise<boolean>}
 */
export function isPortFree(port, host = BACKEND_HOST) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.listen({ port, host, exclusive: true }, () => server.close(() => resolve(true)));
  });
}

/** @returns {Promise<number>} a port the OS reports free on 127.0.0.1 */
export function ephemeralPort(host = BACKEND_HOST) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen({ port: 0, host, exclusive: true }, () => {
      const { port } = /** @type {import('node:net').AddressInfo} */ (server.address());
      server.close(() => resolve(port));
    });
  });
}

/**
 * The preferred port when free, otherwise a free one.
 * @param {number} preferred
 * @param {{ isFree?: (port: number) => Promise<boolean>, pickFree?: () => Promise<number> }} [deps]
 */
export async function choosePort(preferred, deps = {}) {
  const isFree = deps.isFree || ((p) => isPortFree(p));
  const pickFree = deps.pickFree || (() => ephemeralPort());
  if (await isFree(preferred)) return preferred;
  return pickFree();
}

/**
 * The one-line reason serve.py printed before exiting with STORE_CHECK_FAILED_EXIT, if any.
 * @param {string} stderrText
 * @returns {string | null}
 */
export function cannotStartLine(stderrText) {
  const lines = String(stderrText || '')
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.startsWith('Clinical Analytics cannot start:'));
  return lines.length ? lines[lines.length - 1] : null;
}

/**
 * Text for the error window when the backend did not start.
 * @param {{ reason: 'exited' | 'timeout' | 'untrusted' | 'spawn', code?: number | null,
 *   signal?: string | null, stderrText?: string, detail?: string, timeoutMs?: number }} failure
 * @returns {{ title: string, message: string }}
 */
export function startupFailureMessage(failure) {
  const title = 'Clinical Analytics could not start';
  if (failure.reason === 'exited') {
    const line = cannotStartLine(failure.stderrText || '');
    if (failure.code === STORE_CHECK_FAILED_EXIT && line) return { title, message: line };
    const how = failure.signal
      ? `was stopped (${failure.signal})`
      : `exited with code ${failure.code}`;
    const tail = lastLines(failure.stderrText || '', 5);
    return {
      title,
      message: `The analysis service ${how} while starting.${tail ? `\n\nLast output:\n${tail}` : ''}`,
    };
  }
  if (failure.reason === 'timeout') {
    return {
      title,
      message: `The analysis service did not become ready within ${Math.round((failure.timeoutMs || 0) / 1000)} seconds.`,
    };
  }
  if (failure.reason === 'untrusted') {
    return {
      title,
      message:
        'Another program answered on the analysis service port and could not prove it is this ' +
        "app's service, so the app will not use it.",
    };
  }
  return {
    title,
    message: `The analysis service could not be started: ${failure.detail || 'unknown error'}`,
  };
}

function lastLines(text, n) {
  return String(text)
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .slice(-n)
    .join('\n');
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Poll /health until it answers, then require the token proof. Resolves 'ready', or rejects
 * with a failure object for startupFailureMessage.
 * @param {{ baseUrl: string, token: string, fetchFn: typeof fetch, cryptoImpl: Crypto,
 *   timeoutMs: number, pollMs?: number, exited: () => object | null,
 *   now?: () => number, wait?: (ms: number) => Promise<void> }} opts
 */
export async function waitForHealth(opts) {
  const { baseUrl, token, fetchFn, cryptoImpl, timeoutMs } = opts;
  const pollMs = opts.pollMs ?? 250;
  const now = opts.now || Date.now;
  const wait = opts.wait || sleep;
  const deadline = now() + timeoutMs;
  for (;;) {
    const exit = opts.exited();
    if (exit) throw { reason: 'exited', ...exit };
    let answered = false;
    try {
      const res = await fetchFn(`${baseUrl}/health`);
      answered = res.ok;
    } catch {
      answered = false; // not listening yet
    }
    if (answered) {
      const verdict = await verifyBackendEnforcesAuth(fetchFn, baseUrl, token, cryptoImpl);
      if (verdict.ok) return 'ready';
      const exitAfter = opts.exited();
      if (exitAfter) throw { reason: 'exited', ...exitAfter };
      throw { reason: 'untrusted', detail: verdict.reason };
    }
    if (now() >= deadline) throw { reason: 'timeout', timeoutMs };
    await wait(pollMs);
  }
}

/**
 * Stop the backend and everything it started. Resolves once the child has exited.
 * @param {import('node:child_process').ChildProcess} child
 * @param {{ platform?: string, graceMs?: number, killFn?: (pid: number, sig: string) => void,
 *   execFileFn?: typeof nodeExecFile, isExited: () => boolean, onExit: Promise<unknown> }} opts
 */
export async function stopProcessTree(child, opts) {
  const platform = opts.platform || process.platform;
  const graceMs = opts.graceMs ?? DEFAULT_STOP_GRACE_MS;
  const killFn = opts.killFn || ((pid, sig) => process.kill(pid, sig));
  const execFileFn = opts.execFileFn || nodeExecFile;
  if (opts.isExited() || !child.pid) return 'already-exited';

  const signalTree = (force) => {
    try {
      if (platform === 'win32') {
        const args = ['/pid', String(child.pid), '/T'];
        if (force) args.push('/F');
        execFileFn('taskkill', args, () => {});
      } else {
        // Negative pid: the whole process group (the child was spawned detached as its leader).
        killFn(-child.pid, force ? 'SIGKILL' : 'SIGTERM');
      }
    } catch (error) {
      if (error?.code !== 'ESRCH') throw error; // ESRCH: already gone
    }
  };

  signalTree(false);
  const exitedInTime = await Promise.race([
    opts.onExit.then(() => true),
    sleep(graceMs).then(() => false),
  ]);
  if (exitedInTime) return 'terminated';
  signalTree(true);
  await opts.onExit;
  return 'killed';
}

/**
 * One backend process per app launch.
 */
export class BackendSidecar {
  /**
   * @param {{ python: string, token: string, env: Record<string, string | undefined>,
   *   preferredPort: number, healthTimeoutMs: number, parentPid: number, cwd: string,
   *   spawnFn?: typeof nodeSpawn, fetchFn?: typeof fetch, cryptoImpl?: Crypto,
   *   choosePortFn?: (preferred: number) => Promise<number>, platform?: string,
   *   stopGraceMs?: number, killFn?: (pid: number, sig: string) => void,
   *   execFileFn?: typeof nodeExecFile, log?: (line: string) => void, pollMs?: number }} opts
   */
  constructor(opts) {
    this.opts = opts;
    this.child = null;
    this.port = null;
    this.exitInfo = null;
    this.stderrLines = [];
    this.exitPromise = null;
  }

  get baseUrl() {
    return this.port == null ? null : `http://${BACKEND_HOST}:${this.port}`;
  }

  /** Choose the port and spawn; returns the base URL right away (health is awaited separately). */
  async spawn() {
    const o = this.opts;
    const platform = o.platform || process.platform;
    this.port = await (o.choosePortFn || ((p) => choosePort(p)))(o.preferredPort);
    const spawnFn = o.spawnFn || nodeSpawn;
    const log = o.log || (() => {});
    try {
      this.child = spawnFn(o.python, buildBackendArgs(this.port, o.parentPid), {
        cwd: o.cwd,
        env: buildBackendEnv(o.env, o.token),
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: platform !== 'win32', // own process group, so stop() can signal the whole tree
        windowsHide: true,
      });
    } catch (error) {
      throw { reason: 'spawn', detail: error?.message || String(error) };
    }
    this.exitPromise = new Promise((resolve) => {
      this.child.once('exit', (code, signal) => {
        this.exitInfo = { code, signal: signal || null, stderrText: this.stderrLines.join('\n') };
        resolve(this.exitInfo);
      });
      this.child.once('error', (error) => {
        // ENOENT etc.: the process never started.
        this.exitInfo = {
          code: null,
          signal: null,
          stderrText: this.stderrLines.join('\n'),
          spawnError: error?.message || String(error),
        };
        resolve(this.exitInfo);
      });
    });
    const keep = (chunk) => {
      for (const line of String(chunk).split(/\r?\n/)) {
        if (!line) continue;
        this.stderrLines.push(line);
        if (this.stderrLines.length > STDERR_TAIL_LINES) this.stderrLines.shift();
      }
    };
    this.child.stderr?.on('data', (chunk) => {
      keep(chunk);
      log(`[backend] ${String(chunk).trimEnd()}`);
    });
    this.child.stdout?.on('data', (chunk) => log(`[backend] ${String(chunk).trimEnd()}`));
    return this.baseUrl;
  }

  /** Resolves when the backend is healthy and proved it holds the token; rejects with a failure. */
  async waitUntilReady() {
    const o = this.opts;
    try {
      return await waitForHealth({
        baseUrl: this.baseUrl,
        token: o.token,
        fetchFn: o.fetchFn || ((url, init) => fetch(url, init)),
        cryptoImpl: o.cryptoImpl || globalThis.crypto,
        timeoutMs: o.healthTimeoutMs,
        pollMs: o.pollMs,
        exited: () => this.exitInfo,
      });
    } catch (failure) {
      if (failure?.reason === 'exited' && this.exitInfo?.spawnError) {
        throw { reason: 'spawn', detail: this.exitInfo.spawnError };
      }
      throw failure;
    }
  }

  /** Stop the backend (idempotent). */
  async stop() {
    if (!this.child) return 'not-started';
    return stopProcessTree(this.child, {
      platform: this.opts.platform,
      graceMs: this.opts.stopGraceMs,
      killFn: this.opts.killFn,
      execFileFn: this.opts.execFileFn,
      isExited: () => this.exitInfo !== null,
      onExit: this.exitPromise,
    });
  }

  /** Last-chance synchronous kill (process 'exit' handler): never leave the backend behind. */
  killNow() {
    if (!this.child?.pid || this.exitInfo) return;
    const platform = this.opts.platform || process.platform;
    try {
      if (platform === 'win32') {
        (this.opts.execFileFn || nodeExecFile)(
          'taskkill',
          ['/pid', String(this.child.pid), '/T', '/F'],
          () => {}
        );
      } else {
        (this.opts.killFn || ((pid, sig) => process.kill(pid, sig)))(-this.child.pid, 'SIGKILL');
      }
    } catch {
      // Already gone.
    }
  }
}
