# Protecting patient data on the clinician's computer

The app runs entirely on one machine, but it holds patient-level research data. This page lists
what the app does to keep that data contained and what the person installing it must do.

## Logs never contain questions or values

Questions typed by a clinician, filter values and row values can identify a patient, so no log
line at any level carries them.

- Call sites log a 12-character keyed hash (`query_hash`, HMAC-SHA256 with a random key kept
  in `<data home>/.log-key`, mode 0600) and a length or count (`query_len`, `filter_count`,
  `response_length`) instead of text. The key stops anyone with the log from confirming a short
  value such as a diagnosis code by hashing guesses. Column names may be logged; values may not.
- Exceptions are logged as `error_type` plus a keyed `error_hash`, never their message (database
  and parsing errors often quote cell values). Rendered tracebacks keep frames and exception
  types; the message after the type is replaced with `[message redacted]`. When analysis fails,
  the user sees a plain-language message, not the exception text.
- One handler formats everything: structlog, the standard library `logging` used by older
  modules, uvicorn's error and access loggers, and Python warnings all pass through the same
  redaction before they are written to stderr. Tracebacks are rebuilt from the exception objects:
  file, line, function and exception type only, including exception-group members and chained
  causes; messages, notes and source lines are never printed.
- Standard-library messages are treated as unsafe: a `%`-style template is kept with its
  arguments replaced, and any other message (f-strings) is replaced by `[message redacted]` plus
  the logger name and line number. Only uvicorn's own startup lines are allowlisted. Access log
  lines keep the method, path and status but drop the query string. Warnings keep their file,
  line and category.
- uvicorn is always started through `python -m clinical_analytics.api.serve`, which passes
  `log_config=None` so uvicorn does not install its own handlers; app startup re-applies the
  redacting setup anyway, in case uvicorn was launched some other way.
- Keys that can carry free text (`error`, `sql`, `message`, `detail`, `plan`, `context`, ...)
  are hashed or sized wherever they appear, including inside nested dicts and lists.
- If the log key cannot be created (read-only data home, unusual filesystem), hashing switches
  off with one warning and only sizes are logged; it never falls back to an unkeyed hash.
- The LLM prompt overlay is read from `<data home>/nl_query_learning/`, not the world-writable
  `/tmp/nl_query_learning` used before.
- `clinical_analytics.core.log_redaction.redact_sensitive_fields` is a structlog processor that
  backs this up: an event key such as `query`, `question`, `filters`, `value`, `rows` or
  `response` is replaced by `<key>_hash` / `<key>_size` before the line is written.
- The API's log level defaults to `INFO`. Set `CLINICAL_ANALYTICS_LOG_LEVEL=DEBUG` only while
  debugging; debug lines follow the same rule.
- SQL debugging (`MDDE_SQL_DEBUG`) logs the SQL length and hash, not the SQL text, because the
  SQL carries filter values.
- The Electron renderer and preload log query ids, event types and lengths, never the question or
  a result payload.
- `tests/core/test_log_redaction.py` runs a full ask at DEBUG with the redacting processor turned
  off and fails if the question text or a filter value reaches any log event.

## Only the app can call the local API

The API listens on `127.0.0.1:8000`, but every process on the computer, and any web page open
in a browser, can send requests to a loopback port. These checks close that
(`clinical_analytics/api/security.py`, `electron/src/main.js`):

- **Per-launch bearer token.** `scripts/run_app.sh` generates a random token for each launch
  (`secrets.token_urlsafe(32)`) and passes it in `CLINICAL_ANALYTICS_API_TOKEN` to the backend and
  to Electron. Electron main gives it to the preload script over IPC, and only to the app's own
  page; the preload adds `Authorization: Bearer <token>` to every request. The token is never put
  on `window` or in a URL. The API answers `401` to any request without the right token, except
  `GET /health` (the launcher polls it before the window opens) and CORS preflights. Tokens
  shorter than 32 characters stop the API at startup.
- **No silent open mode.** Without a token the API refuses to start unless
  `CLINICAL_ANALYTICS_DEV_NO_AUTH=1` is set (`make run-api` and the test suite set it). If it is
  started anyway, every request except `/health` gets `503`. `/health` reports `auth_required`.
- **The app checks it is talking to its own backend.** Before sending the token or any data,
  the preload sends `GET /health?nonce=<random>` and requires `token_proof =
  HMAC-SHA256(token, "clinical-analytics-health-v1:" + nonce)`, `auth_required: true`, and a
  `401` for a request without the token. A program squatting on port 8000 cannot produce the
  proof, so it never receives the token. If a check fails, the app shows a plain message asking
  the user to restart with the launcher (unless Electron itself was started with
  `CLINICAL_ANALYTICS_DEV_NO_AUTH=1`).
- **One address.** The app, its CSP and the launcher use `http://127.0.0.1:8000`, never
  `localhost`: if `localhost` resolved to `::1`, a program listening on `[::1]:8000` could relay
  `/health` proofs from the real backend and receive the token.
- **The launcher checks too.** `scripts/run_app.sh` refuses to start if something already
  answers on port 8000 (IPv4 or IPv6 loopback), checks that the backend it started is still running once `/health`
  answers, and runs `python -m clinical_analytics.api.security verify-health` (the same proof)
  before opening the app.
- **Exact page origin.** The packaged page is served from a privileged `app://clinical-analytics`
  scheme (`protocol.registerSchemesAsPrivileged` + `protocol.handle`), not `file://`. CORS allows
  exactly that origin plus the Vite dev server; `"null"` (what `file://` pages send) and `*` are
  rejected at startup. CORS does not allow credentials (the token is a header, not a cookie).
- **Locked-down window.** The renderer is sandboxed (`sandbox: true`, context isolation, no Node).
  Navigation away from the app's own origin and `window.open` are blocked, and webviews are not
  allowed, so a link or injected script cannot load another page that would receive the token.
- **Loopback Host only.** `TrustedHostMiddleware` rejects requests whose `Host` header is not
  `127.0.0.1` or `localhost`, which defeats DNS-rebinding pages
  (`CLINICAL_ANALYTICS_ALLOWED_HOSTS` overrides the list; tests add `testserver`).

**Query stream.** `EventSource` cannot send headers, so the preload reads
`/api/queries/{id}/stream` with `fetch` streaming instead (same SSE wire format). The
alternative, a short-lived token in the query string, was rejected because URLs land in access
logs, and it would need a second token type. If the stream drops before it ends, the preload asks
`GET /api/queries/{id}` and, if the query still exists, reopens the stream (up to three times);
the server replays the query's events, so an answer that finished meanwhile still arrives, and
the preload drops the replayed events the renderer has already seen.

**Development.** `make run-api` runs the API with `CLINICAL_ANALYTICS_DEV_NO_AUTH=1`. To use the
Electron app against it, start Electron with the same variable (`CLINICAL_ANALYTICS_DEV_NO_AUTH=1
npm start`). Never use dev mode with real patient data.

## Deleting a dataset removes all of it

**Delete dataset** in the dataset bar (or `DELETE /api/datasets/{id}`) asks for confirmation in
the page, then removes, for that dataset id:

- the uploaded file, the unified CSV and per-table CSVs (`uploads/raw/`)
- its metadata and enrichment overlays (`uploads/metadata/{id}.json`, `uploads/metadata/overlays/{id}/`)
- its Parquet exports (`parquet/{id}_*.parquet`)
- its DuckDB tables; the store is then rewritten into a fresh file, because DuckDB leaves the
  bytes of dropped tables in free blocks
- any copy of the store left by a failed DuckDB upgrade check (`analytics.duckdb.backup-*`, see
  [DuckDB upgrades and rollbacks](#duckdb-upgrades-and-rollbacks))
- its sessions and their messages (SQLite runs with `secure_delete=ON`, so deleted rows are
  zeroed rather than left in free pages)
- in-memory query results, events and conversation history

Metadata goes last, so a delete that fails part-way leaves the dataset listed and can be retried.
A durable `analytics.duckdb.needs-compaction` marker is written before any table is dropped and
removed only after the rewritten store has replaced the old file, so a failed rewrite is redone
on retry instead of being reported as success. The purge holds a process-wide store lock and
refuses (HTTP 409) while any connection to the store is open. Opening a connection while a
purge holds the lock waits at most 15 seconds, then answers HTTP 503 "The data store is busy";
store and file work runs in worker threads, so a long rewrite never stalls other requests. A
`DataStore` that is dropped without `close()` releases its registration when garbage-collected.
A marker left by a crash is picked up at the next API start and the rewrite finished then. While a delete runs, queries,
previews and semantic-layer loads for that dataset get HTTP 409, and caches are dropped again
when it ends. Upload ids are random (`user_upload_<uuid4 hex>`), so two uploads of the same file
can never share an id. A one-time `VACUUM` at startup clears free pages left by deletes made
before `secure_delete` was turned on.
`tests/api/test_dataset_delete.py` uploads a dataset, adds a session, a message, an overlay and a
query service, deletes it, and checks that neither the dataset id nor a cell value appears in
any file name or file content under the data home or in the session database.

## DuckDB upgrades and rollbacks

The DuckDB store is `analytics.duckdb` in the data home's `data/` folder.

**File format.** The app creates and rewrites the store in DuckDB storage format `v1.0.0`
(`STORE_STORAGE_VERSION` in `clinical_analytics.storage.store_format`). Both DuckDB 1.x and
2.x can read that format. Without this setting, DuckDB 2.0 would create files in its own
format, which 1.x cannot open. With it, rolling back to an app release that uses DuckDB 1.4
keeps working. DuckDB keeps a file's existing format when it writes to it, so stores created
before the upgrade are not converted. Raise the pinned format only once rollback to the older
DuckDB line is no longer supported.

**Engine check at startup.** `analytics.duckdb.engine` records the DuckDB version that last
opened the store. It is written when the store is first created. At startup, the app compares
it with the running DuckDB. A different patch release, such as 2.0.0 to 2.0.1, needs nothing
more. A different major or minor line, such as 1.4 to 2.0, triggers a check:

1. The app first makes sure no other program has the store open. It then copies the store,
   and its `.wal` file if present, to `analytics.duckdb.backup-<previous version>-<UTC time>-<random>`
   in the same folder. While the copy is made and checked, the store is locked against other
   connections.
2. The running DuckDB opens the store read-write and reads its catalog.
3. If that works, the app records the new version and **deletes the backup**.

The backup exists only for those few seconds: it is a full copy of patient data, and keeping
it has no value because the file format does not change. If a previous start passed the check
but stopped before deleting the backup, the next start deletes it. A backup with no
`analytics.duckdb` next to it is also deleted at startup.

**If the check fails**, the app does not start. `python -m clinical_analytics.api.serve` prints
one line saying why and exits with status 2:

- *Unreadable store* (written by a newer DuckDB, or damaged). The error names the backup path,
  and the backup is kept. The backup is the store exactly as it was just before the check, and
  the check normally leaves the store unchanged. So first run the app release you used before,
  and copy the backup over `analytics.duckdb` only if that release cannot open the store either.
  Then **delete the backup yourself**. It holds every dataset as it was
  before the upgrade, and an older app release does not know about it, so datasets deleted
  there remain in the backup until you delete it. Deleting a dataset in this release also
  deletes any backup, and a later successful start deletes it too.
- *Store open in another program* (for example, the DuckDB command-line tool). The message is
  "The data store is open in another program." Close that program and start the app again. This
  is checked before anything is copied, so no backup is made.
- *No room for the backup* (or another file error during the copy). Any partial copy is deleted
  and the app does not start.

The app never keeps more than one backup. When a start finds a backup left by a failed check,
it copies the store again and deletes the old backup, since the store may have changed since then.

## Disk encryption is required

Deleting a file does not overwrite the disk blocks it used, and on SSDs nothing in software can
guarantee that it does. The data home (see `clinical_analytics.core.data_home`) holds patient
data at rest, so **the computer running this app must have full-disk encryption turned on**:

- **macOS:** FileVault (System Settings → Privacy & Security → FileVault).
- **Windows:** BitLocker (Settings → Privacy & security → Device encryption, or Control Panel →
  BitLocker Drive Encryption). Windows Home editions offer Device encryption only on some hardware.
- **Linux:** LUKS full-disk encryption, chosen at install time.

Backups (Time Machine, File History, cloud sync) copy the data home too: encrypt them, or exclude
the data home. Moving the data home with `CLINICAL_ANALYTICS_DATA_HOME` to an unencrypted
external or network drive defeats this requirement.
