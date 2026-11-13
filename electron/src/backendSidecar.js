// SPDX-License-Identifier: MIT
/**
 * The Python backend as an Electron-owned sidecar (main process only; no `electron` import, so
 * every dependency is injectable and unit-tested with a fake child).
 *
 * Lifecycle:
 * 1. Pick a port on 127.0.0.1: the preferred one if free, otherwise one the OS hands out.
 * 2. Spawn `<python> -I -u -m clinical_analytics.api.serve --host 127.0.0.1 --port <port>
 *    --exit-with-parent <electron pid> --stop-on-stdin-close` with this launch's API token in its
 *    environment (never the dev no-auth opt-out, never Ollama auto-download), from a directory
 *    of the app's own. `-I` (isolated mode) keeps the working directory and the user's
 *    site-packages off Python's import path and ignores every PYTHON* variable (PYTHONPATH,
 *    PYTHONHOME, PYTHONSTARTUP, ...), so a stray json.py or a .pth file in the user's files is
 *    never imported by the process that holds the token. A venv still finds its own packages
 *    (pyvenv.cfg, not the environment). `-u` keeps output unbuffered, as PYTHONUNBUFFERED would.
 * 3. Wait until anything answers /health, then check the server proves it knows the token
 *    (apiAuth.js), so a program squatting on the port is never used.
 * 4. On quit: SIGTERM to the process group (Windows: close its stdin), then a forced kill of the
 *    tree after a grace period, and a bounded wait for that kill (quit never hangs on it). If
 *    Electron itself dies, its stdin closes and serve.py's parent watchdog notices too, so the
 *    backend shuts down either way.
 *
 * The dev flow (separate Vite and `make run-api`, or scripts/run_app.sh) sets
 * CLINICAL_ANALYTICS_EXTERNAL_BACKEND=1 and keeps using an already-running API.
 */
import { spawn as nodeSpawn, execFile as nodeExecFile } from 'node:child_process';
import fs from 'node:fs';
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
export const DEFAULT_STOP_GRACE_MS = 10000; // uvicorn drains in-flight requests (an upload) first
/** How long stop() waits for the exit after the forced kill before giving up (quit goes on). */
export const DEFAULT_FORCE_KILL_WAIT_MS = 5000;
/**
 * After 'exit', how long to wait for 'close' (stdout/stderr drained) before reporting the exit
 * anyway: a grandchild that inherited the pipes can hold them open after the backend is gone.
 */
export const DEFAULT_STDIO_DRAIN_MS = 1000;
/** serve.py exits with this code after printing one "cannot start" line (store check, model). */
export const STORE_CHECK_FAILED_EXIT = 2;
const STDERR_TAIL_LINES = 40;
// Never handed to the sidecar: the dev opt-out would let the renderer accept an unauthenticated
// backend, and auto-init downloads Ollama models.
// PYTHONPATH is ignored under -I anyway; dropping it as well keeps it out of anything the
// backend itself starts.
const STRIPPED_ENV = [
  'CLINICAL_ANALYTICS_DEV_NO_AUTH',
  'CLINICAL_ANALYTICS_OLLAMA_AUTOINIT',
  'PYTHONPATH',
];
/** Folder under the app's user-data directory that the backend runs in. */
export const BACKEND_CWD_NAME = 'backend';

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
    throw new Error(`${PORT_ENV} must be a port number (1-65535), got "${raw}".`);
  }
  return port;
}

/** @param {Record<string, string | undefined>} env */
export function resolveHealthTimeoutMs(env) {
  const raw = (env[HEALTH_TIMEOUT_ENV] || '').trim();
  const value = raw ? Number(raw) : DEFAULT_HEALTH_TIMEOUT_MS;
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(
      `${HEALTH_TIMEOUT_ENV} must be a positive number of milliseconds, got "${raw}".`
    );
  }
  return value;
}

/**
 * Port and health timeout from the environment. A bad value is a setting to correct, not a
 * broken install, so it is reported as { reason: 'config' } (the message says which variable).
 * @param {Record<string, string | undefined>} env
 * @returns {{ preferredPort: number, healthTimeoutMs: number }}
 */
export function resolveSidecarSettings(env) {
  try {
    return {
      preferredPort: resolvePreferredPort(env),
      healthTimeoutMs: resolveHealthTimeoutMs(env),
    };
  } catch (error) {
    throw { reason: 'config', detail: error.message };
  }
}

/**
 * The directory the backend runs in: a folder of the app's own, created owner-only. Never the
 * home directory, where the user's scratch files live.
 * @param {string} userDataDir - app.getPath('userData')
 * @param {{ mkdirSync: typeof fs.mkdirSync }} [fsImpl]
 * @returns {string}
 */
export function ensureBackendCwd(userDataDir, fsImpl = fs) {
  if (!userDataDir || !path.isAbsolute(userDataDir)) {
    throw new Error(`the app's data directory must be an absolute path, got "${userDataDir}"`);
  }
  const dir = path.join(userDataDir, BACKEND_CWD_NAME);
  try {
    fsImpl.mkdirSync(dir, { recursive: true, mode: 0o700 });
  } catch (error) {
    // A filesystem refusal (EACCES after a sudo run, ENOSPC, EROFS): a problem with this folder
    // that reinstalling does not fix, so it gets its own failure naming the folder. Anything
    // without a system error code is a bug and is thrown as it is.
    if (typeof error?.code !== 'string') throw error;
    throw { reason: 'folder', folder: dir, detail: error.code };
  }
  return dir;
}

/** @param {number} port @param {number} parentPid */
export function buildBackendArgs(port, parentPid) {
  return [
    '-I', // isolated: no cwd or user site-packages on sys.path, PYTHON* variables ignored
    '-u', // unbuffered stdout/stderr (PYTHONUNBUFFERED is ignored under -I)
    '-m',
    'clinical_analytics.api.serve',
    '--host',
    BACKEND_HOST,
    '--port',
    String(port),
    '--exit-with-parent',
    String(parentPid),
    '--stop-on-stdin-close',
  ];
}

/**
 * @param {Record<string, string | undefined>} baseEnv
 * @param {string} token
 */
export function buildBackendEnv(baseEnv, token) {
  const env = { ...baseEnv, CLINICAL_ANALYTICS_API_TOKEN: token };
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

const CANNOT_START_PREFIX = 'Clinical Analytics cannot start:';

/**
 * The one-line reason serve.py printed before exiting with STORE_CHECK_FAILED_EXIT, if any.
 * @param {string} stderrText
 * @returns {string | null}
 */
export function cannotStartLine(stderrText) {
  const lines = String(stderrText || '')
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.startsWith(CANNOT_START_PREFIX));
  return lines.length ? lines[lines.length - 1] : null;
}

const START_AGAIN = 'Start Clinical Analytics again.';

/** "exit code 1" / "signal SIGKILL": how the process ended, for the detail line of a message. */
function exitDescription({ code, signal }) {
  return signal ? `signal ${signal}` : `exit code ${code}`;
}

/**
 * Developer-only sentences in a serve.py reason: tracebacks, log lines and build commands. They go
 * to "Copy details"; everything else (paths and folder names included) is the recovery step the
 * clinician needs and stays in the message.
 */
const DEVELOPER_SENTENCE =
  /`|\bmake\s+[a-z][\w-]*|Traceback|^\s*File "|^(INFO|WARNING|ERROR|DEBUG|CRITICAL)\b/;

/** The bundled model is missing or damaged: the one reason reinstalling fixes. */
const MISSING_BUNDLED_FILE =
  /^The embedding model\b.*\b(is missing from|does not match its pinned)\b/;

/**
 * serve.py's one-line reason as the message, with only developer sentences moved to details.
 * @param {string} reason
 * @returns {{ message: string, details: string }}
 */
function storeCheckReason(reason) {
  if (MISSING_BUNDLED_FILE.test(reason)) {
    return {
      message:
        'A file the analysis engine needs is missing or can’t be read. Reinstall Clinical Analytics.',
      details: reason,
    };
  }
  const sentences = reason.split(/(?<=\.)\s+(?=\S)/);
  const kept = sentences.filter((sentence) => !DEVELOPER_SENTENCE.test(sentence));
  const moved = sentences.filter((sentence) => DEVELOPER_SENTENCE.test(sentence));
  if (!kept.length) {
    return {
      message: `The analysis engine stopped before it was ready. ${START_AGAIN}`,
      details: reason,
    };
  }
  return { message: kept.join(' '), details: moved.join('\n') };
}

/** Exit status and the backend's last output, for "Copy details". */
function exitDetails(failure, lines) {
  const tail = lastLines(failure.stderrText || '', lines);
  return `${exitDescription(failure)}.` + (tail ? `\n\nLast output:\n${tail}` : '');
}

/**
 * Text for the error window when the backend did not start. `message` is one or two plain
 * sentences that say what happened and what to do next, in the copy deck's words
 * (docs/app/DESIGN.md: "analysis engine", "on this computer"). Paths, exit codes, signals and log
 * lines go in `details`, which the window offers behind "Copy details".
 * @param {{ reason: 'exited' | 'timeout' | 'untrusted' | 'spawn' | 'config' | 'folder',
 *   code?: number | null, signal?: string | null, stderrText?: string, detail?: string,
 *   timeoutMs?: number, folder?: string }} failure
 * @returns {{ title: string, message: string, details: string }}
 */
export function startupFailureMessage(failure) {
  const title = 'Clinical Analytics could not start';
  if (failure.reason === 'exited') {
    const line = cannotStartLine(failure.stderrText || '');
    if (failure.code === STORE_CHECK_FAILED_EXIT && line) {
      // serve.py's own one-line reason (it already says what to do); the title carries the rest.
      const reason = line.slice(CANNOT_START_PREFIX.length).trim() || line;
      return { title, ...storeCheckReason(reason) };
    }
    return {
      title,
      message:
        `The analysis engine stopped before it was ready. ${START_AGAIN} ` +
        'If this keeps happening, reinstall the app.',
      details: exitDetails(failure, 5),
    };
  }
  if (failure.reason === 'timeout') {
    return {
      title,
      message:
        `The analysis engine was not ready after ${Math.round((failure.timeoutMs || 0) / 1000)} ` +
        `seconds. ${START_AGAIN}`,
      details: '',
    };
  }
  if (failure.reason === 'untrusted') {
    return {
      title,
      message:
        'Another program is using the port Clinical Analytics chose for its analysis engine, ' +
        `so the app did not connect to it. ${START_AGAIN} It will choose a free port.`,
      details: failure.detail || '',
    };
  }
  if (failure.reason === 'folder') {
    return {
      title,
      // The folder is what the user has to fix, so it stays in the sentence; the OS error doesn't.
      message:
        `Clinical Analytics couldn’t create its working folder “${failure.folder || ''}”. Check ` +
        'that your user account can write to it and that the disk isn’t full, then start ' +
        'Clinical Analytics again.',
      details: failure.detail || '',
    };
  }
  if (failure.reason === 'config') {
    // Only developer launch settings get here; the setting's name is the way to fix it.
    return {
      title,
      message:
        `${failure.detail || 'A startup setting is not valid.'} ` +
        'Correct the setting, then start Clinical Analytics again.',
      details: '',
    };
  }
  return {
    title,
    message: 'The analysis engine could not be started. Reinstall Clinical Analytics.',
    details: failure.detail || 'unknown error',
  };
}

/**
 * Text for the dialog when the backend dies after it was ready. How it ended and its last
 * output go in `details` (behind "Copy details"), never in the dialog's text.
 * @param {{ code?: number | null, signal?: string | null, stderrText?: string }} info
 * @returns {{ title: string, message: string, detail: string, details: string, buttons: string[] }}
 */
export function unexpectedExitMessage(info) {
  return {
    title: 'Clinical Analytics',
    message: 'The analysis engine stopped.',
    detail: 'Restart it to keep working. Your cohorts are still on this computer.',
    details: exitDetails(info, 3),
    buttons: ['Restart engine', 'Quit'],
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

/** Per-request limit while waiting for the backend: a squatter that never answers cannot stall us. */
export const HEALTH_ATTEMPT_TIMEOUT_MS = 3000;

class DeadlineExceeded extends Error {}

/**
 * fetch with a per-request timeout (AbortSignal), never past the overall deadline.
 * @param {typeof fetch} fetchFn
 * @param {number} attemptMs
 * @param {() => number} remainingMs
 */
export function timedFetch(fetchFn, attemptMs, remainingMs) {
  return (url, init = {}) => {
    const left = remainingMs();
    if (left <= 0) return Promise.reject(new DeadlineExceeded('deadline'));
    const controller = new AbortController();
    const limit = Math.max(1, Math.min(attemptMs, left));
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => {
        controller.abort();
        reject(new DOMException('timed out', 'TimeoutError'));
      }, limit);
    });
    // Race as well as abort: a fetch stand-in (or a stuck socket) may ignore the signal.
    return Promise.race([fetchFn(url, { ...init, signal: controller.signal }), timeout]).finally(
      () => clearTimeout(timer)
    );
  };
}

/**
 * Poll /health until anything answers, then require the token proof. Any HTTP response counts as
 * an answer, whatever its status: something is listening, and only the token check can say
 * whether it is this app's backend (a program on the port that answers 404 is refused at once as
 * 'untrusted', not polled until the deadline). Every request has its own timeout and the whole
 * wait has one deadline (a port squatter that accepts the connection and never answers ends in
 * a 'timeout' or 'untrusted' failure, not a stall). Resolves 'ready', or rejects with a failure
 * object for startupFailureMessage; { reason: 'stopped' } means the app itself asked the backend
 * to stop (the user quit), which is not a failure to report.
 * @param {{ baseUrl: string, token: string, fetchFn: typeof fetch, cryptoImpl: Crypto,
 *   timeoutMs: number, pollMs?: number, attemptMs?: number, exited: () => object | null,
 *   stopped?: () => boolean, now?: () => number, wait?: (ms: number) => Promise<void> }} opts
 */
export async function waitForHealth(opts) {
  const { baseUrl, token, cryptoImpl, timeoutMs } = opts;
  const pollMs = opts.pollMs ?? 250;
  const now = opts.now || Date.now;
  const wait = opts.wait || sleep;
  const deadline = now() + timeoutMs;
  const fetchFn = timedFetch(
    opts.fetchFn,
    opts.attemptMs ?? HEALTH_ATTEMPT_TIMEOUT_MS,
    () => deadline - now()
  );
  const stopped = opts.stopped || (() => false);
  const failIfGone = () => {
    if (stopped()) throw { reason: 'stopped' };
    const exit = opts.exited();
    if (exit) throw { reason: 'exited', ...exit };
  };
  for (;;) {
    failIfGone();
    let answered;
    try {
      await fetchFn(`${baseUrl}/health`);
      answered = true; // any status: something listens here; the token check decides what it is
    } catch {
      answered = false; // not listening yet, or no answer within the attempt timeout
    }
    if (answered) {
      let verdict;
      let outOfTime = false;
      try {
        verdict = await verifyBackendEnforcesAuth(fetchFn, baseUrl, token, cryptoImpl);
      } catch (error) {
        // The overall deadline, not the server, ended the proof (it was not sent, or its request
        // only had what was left of the wait): a slow start, not another program on the port.
        outOfTime = error instanceof DeadlineExceeded || now() >= deadline;
        verdict = {
          ok: false,
          reason: error?.name === 'TimeoutError' ? 'no-answer' : 'probe-failed',
        };
      }
      failIfGone();
      if (outOfTime) throw { reason: 'timeout', timeoutMs };
      if (verdict.ok) return 'ready';
      throw { reason: 'untrusted', detail: verdict.reason };
    }
    if (now() >= deadline) throw { reason: 'timeout', timeoutMs };
    await wait(Math.min(pollMs, Math.max(0, deadline - now())));
  }
}

/**
 * Stop the backend and everything it started. Resolves once the child has exited, or with
 * 'not-stopped' when it is still running forceWaitMs after the forced kill (taskkill refused,
 * a process in another session): the caller must be able to quit either way.
 *
 * POSIX: SIGTERM to the process group (uvicorn drains requests), SIGKILL after the grace period.
 * Windows has no SIGTERM for a windowless console child and `taskkill /T` without /F cannot close
 * one, so the graceful request is closing the backend's stdin (serve.py --stop-on-stdin-close);
 * only after the grace period does `taskkill /T /F` end the tree.
 * @param {import('node:child_process').ChildProcess} child
 * @param {{ platform?: string, graceMs?: number, forceWaitMs?: number,
 *   killFn?: (pid: number, sig: string) => void, execFileFn?: typeof nodeExecFile,
 *   isExited: () => boolean, onExit: Promise<unknown>, log?: (line: string) => void }} opts
 * @returns {Promise<'already-exited' | 'terminated' | 'killed' | 'not-stopped'>}
 */
export async function stopProcessTree(child, opts) {
  const platform = opts.platform || process.platform;
  const graceMs = opts.graceMs ?? DEFAULT_STOP_GRACE_MS;
  const forceWaitMs = opts.forceWaitMs ?? DEFAULT_FORCE_KILL_WAIT_MS;
  const log = opts.log || (() => {});
  const killFn = opts.killFn || ((pid, sig) => process.kill(pid, sig));
  const execFileFn = opts.execFileFn || nodeExecFile;
  if (opts.isExited() || !child.pid) return 'already-exited';

  const gracefulRequest = () => {
    if (platform === 'win32') {
      child.stdin?.end();
      return;
    }
    signalGroup(killFn, child.pid, 'SIGTERM');
  };
  const forceKill = () => {
    if (platform === 'win32') {
      execFileFn('taskkill', ['/pid', String(child.pid), '/T', '/F'], (error) => {
        if (error) log(`[backend] taskkill failed: ${error.message || error}`);
      });
      return;
    }
    signalGroup(killFn, child.pid, 'SIGKILL');
  };
  const exitsWithin = (ms) => {
    let timer;
    const timeout = new Promise((resolve) => {
      timer = setTimeout(() => resolve(false), ms);
    });
    return Promise.race([opts.onExit.then(() => true), timeout]).finally(() => clearTimeout(timer));
  };

  gracefulRequest();
  if (await exitsWithin(graceMs)) return 'terminated';
  forceKill();
  if (await exitsWithin(forceWaitMs)) return 'killed';
  log(`[backend] still running ${forceWaitMs} ms after the forced kill (pid ${child.pid})`);
  return 'not-stopped';
}

function signalGroup(killFn, pid, signal) {
  try {
    // Negative pid: the whole process group (the child was spawned detached as its leader).
    killFn(-pid, signal);
  } catch (error) {
    if (error?.code !== 'ESRCH') throw error; // ESRCH: already gone
  }
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
   *   stopGraceMs?: number, forceKillWaitMs?: number, stdioDrainMs?: number, killFn?: (pid: number, sig: string) => void,
   *   execFileFn?: typeof nodeExecFile, log?: (line: string) => void, pollMs?: number,
   *   attemptMs?: number, onUnexpectedExit?: (info: object) => void }} opts
   */
  constructor(opts) {
    this.opts = opts;
    this.ready = false;
    this.stopping = false;
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
    // stop() ran while the port was being chosen (the user quit): do not start anything.
    if (this.stopping) throw { reason: 'stopped' };
    const spawnFn = o.spawnFn || nodeSpawn;
    const log = o.log || (() => {});
    try {
      this.child = spawnFn(o.python, buildBackendArgs(this.port, o.parentPid), {
        cwd: o.cwd,
        env: buildBackendEnv(o.env, o.token),
        // stdin stays open for the backend's lifetime; closing it is the graceful stop request.
        stdio: ['pipe', 'pipe', 'pipe'],
        detached: platform !== 'win32', // own process group, so stop() can signal the whole tree
        windowsHide: true,
      });
    } catch (error) {
      throw { reason: 'spawn', detail: error?.message || String(error) };
    }
    this.exitPromise = new Promise((resolve) => {
      // 'exit' can fire before the last stderr chunk is delivered (the pipes drain afterwards),
      // and that chunk is where serve.py's "cannot start" line is. So the exit is reported on
      // 'close' (stdio drained), or stdioDrainMs after 'exit' if the pipes never close.
      let drainTimer = null;
      const report = (code, signal) => {
        clearTimeout(drainTimer);
        if (this.exitInfo) return;
        this.exitInfo = { code, signal: signal || null, stderrText: this.stderrLines.join('\n') };
        resolve(this.exitInfo);
        // Died after it was ready, and nobody asked it to stop: the app offers a restart.
        if (this.ready && !this.stopping) o.onUnexpectedExit?.(this.exitInfo);
      };
      this.child.once('exit', (code, signal) => {
        drainTimer = setTimeout(
          () => report(code, signal),
          o.stdioDrainMs ?? DEFAULT_STDIO_DRAIN_MS
        );
      });
      this.child.once('close', (code, signal) => report(code, signal));
      this.child.once('error', (error) => {
        clearTimeout(drainTimer);
        if (this.exitInfo) return;
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
    // Writing is never done; an EPIPE after the backend exits must not crash main.
    this.child.stdin?.on('error', () => {});
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
      const result = await waitForHealth({
        baseUrl: this.baseUrl,
        token: o.token,
        fetchFn: o.fetchFn || ((url, init) => fetch(url, init)),
        cryptoImpl: o.cryptoImpl || globalThis.crypto,
        timeoutMs: o.healthTimeoutMs,
        pollMs: o.pollMs,
        attemptMs: o.attemptMs,
        exited: () => this.exitInfo,
        stopped: () => this.stopping,
      });
      this.ready = true;
      return result;
    } catch (failure) {
      if (failure?.reason === 'exited' && this.exitInfo?.spawnError) {
        throw { reason: 'spawn', detail: this.exitInfo.spawnError };
      }
      throw failure;
    }
  }

  /** Stop the backend (idempotent). Also ends a spawn() or waitUntilReady() still in progress. */
  async stop() {
    this.stopping = true;
    if (!this.child) return 'not-started';
    return stopProcessTree(this.child, {
      platform: this.opts.platform,
      graceMs: this.opts.stopGraceMs,
      forceWaitMs: this.opts.forceKillWaitMs,
      log: this.opts.log,
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
