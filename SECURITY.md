# Security policy

Clinical Analytics is a local, offline desktop app for clinicians. Its two promises are that
**patient data never leaves the machine** and that **anything the app writes out of the machine
(an export, a results packet, a published log) shows no patient count from 1 to 10**: such counts
are written as "<11". On screen, to the person who already holds the data, every answer shows its
true numbers ([ADR015](docs/implementation/ADR/ADR015.md)). A way to break either promise is a
security vulnerability here, and we treat it as one.

## Reporting a vulnerability

Report privately through GitHub Security Advisories:

1. Go to <https://github.com/jtouley/md_data_explorer/security/advisories/new>
   (the repository's **Security** tab, then **Report a vulnerability**).
2. Describe the issue, the steps to reproduce it and the impact.
3. Use synthetic data only. **Never attach real patient data**, screenshots of it, or files
   derived from it. The synthetic kit in `tests/fixtures/synthetic/` and the app's **Use the sample
   cohort** button are enough to reproduce almost anything; if they are not, describe the shape of
   the data (column types, group sizes) instead. The only exception is acceptance testing on
   public, openly licensed, de-identified datasets: fetched at test time from their public source,
   checked against a pinned checksum, used under their licence, never committed, and never shown
   row by row. Aggregate results (counts, statistics, grades) may appear in logs, issues and pull
   requests. Everything else, unit tests included, uses synthetic data.

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

- **Small counts leaving the machine.** Any export, results packet, published log or other file
  written out of the app that shows a patient count from 1 to 10 as a number, i.e. a path that
  writes an answer out without `core/export_boundary.py`. Masking is per cell: a masked count
  that can be worked out from the other cells of an exported answer is by design, not a
  vulnerability (ADR015). Counts shown on screen are not in scope. The threshold is
  `small_cell_threshold` in `src/clinical_analytics/resources/config/nl_query.yaml`.
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
