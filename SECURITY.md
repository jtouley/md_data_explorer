# Security policy

Clinical Analytics is a local, offline desktop app for clinicians. Its two promises are that
**patient data never leaves the machine** and that **patient counts from 1 to 10 are never shown
or recoverable**. A way to break either promise is a security vulnerability here, and we treat it
as one.

## Reporting a vulnerability

Report privately through GitHub Security Advisories:

1. Go to <https://github.com/jtouley/md_data_explorer/security/advisories/new>
   (the repository's **Security** tab, then **Report a vulnerability**).
2. Describe the issue, the steps to reproduce it and the impact.
3. Use synthetic data only. **Never attach real patient data**, screenshots of it, or files
   derived from it. The synthetic kit in `tests/fixtures/synthetic/` and the app's **Load sample**
   cohort are enough to reproduce almost anything; if they are not, describe the shape of the data
   (column types, group sizes) instead.

Do not open a public issue, pull request or discussion for a vulnerability. If the
**Report a vulnerability** button is not available, open a public issue that says only that you
have a security report and need a private channel, with no details, and the maintainer will
follow up.

## What to expect

- An acknowledgement within 14 days. This is a single-maintainer project
  (see [MAINTAINERS.md](MAINTAINERS.md)), so a fix can take longer; you will get status updates
  in the advisory.
- We will agree a disclosure date with you. The fix and a published GitHub Security Advisory go
  out together, and you are credited unless you ask not to be.

## Supported versions

There are no tagged releases yet. Fixes land on `main`; run the latest `main`.

## Scope

In scope:

- **Small counts shown or recoverable.** Any answer, chart, table, export, log line, error
  message, API response or chat history that shows a patient count from 1 to 10, or lets one be
  worked out: for example by subtracting two suppressed totals, from percentages and a known
  denominator, from complementary filters, or from repeated queries. The threshold is
  `small_cell_threshold` in `src/clinical_analytics/core/config_loader.py`.
- **Patient data leaving the machine.** Any path by which uploaded data, column names, category
  labels or results reach a network host other than loopback: telemetry, update checks, crash
  reports, remote fonts or scripts in the Electron window, a non-local LLM endpoint used without
  the user setting it, or the API listening beyond `127.0.0.1`.
- **Data written where it should not be.** Patient-level data in logs, temp files outside the
  data home, or files that `make check-data` would let into git.
- **Local API and desktop shell.** Bypassing the per-launch API token, cross-origin access to the
  local API from a web page, Electron renderer escapes (`nodeIntegration`, unsafe IPC,
  navigation to remote content), and path traversal in uploads or dataset deletion.
- **Supply chain.** Tampering with the pinned embedding model or its SHA-256 checks
  (`scripts/fetch_embedding_model.py`), or with CI workflows.

Out of scope:

- Attacks that need an attacker already running code as the user on the same machine, or
  physical access to an unlocked machine. The app does not encrypt data at rest; the README asks
  users to turn on full-disk encryption.
- Pointing `OLLAMA_BASE_URL` at a remote host on purpose. The README warns against it; a way the
  app does this *without* the user choosing it is in scope.
- Whether a statistical method is appropriate for a dataset. Report wrong results as a bug.

The [threat model](docs/security/THREAT_MODEL.md) and
[local data security notes](docs/security/LOCAL_DATA_SECURITY.md) describe the design these
promises rest on.
