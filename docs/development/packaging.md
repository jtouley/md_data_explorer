# Packaging the desktop app

Step 1 of the installer work: the backend runs from an installed, read-only, relocated location
with no network, and Electron owns its lifecycle. Steps 2-6 (bundled Python runtime, signing,
`.dmg`/`.msi`) build on this.

## Supported platforms

| Platform | Minimum | Why |
|---|---|---|
| macOS | 13 (Ventura), Apple silicon and Intel | `onnxruntime==1.23.2` ships `macosx_13_0_arm64` and `macosx_13_0_x86_64` wheels for cp311-cp313. 1.24+ dropped Intel and needs macOS 14; macOS 12 has no cp313 wheel. |
| Windows | 10/11 x64 (arm64 also has wheels) | |
| Linux | x86_64 / aarch64, glibc 2.28 (`manylinux_2_28`) | |

`onnxruntime` is pinned exactly in `pyproject.toml`; move the pin only to a release at least two
weeks old that still covers the platforms above.

## What the installed backend reads, and from where

- Read-only files ship inside the package, under `src/clinical_analytics/resources/`
  (`config/*.yaml`, the synthetic demo cohort `demo/sample_cohort.csv`, and the embedding model
  under `models/`). They are found with `importlib.resources` through
  `clinical_analytics.core.resources`, whatever the cwd. Nothing is written there.
- Everything written (uploads, DuckDB store, session DB, caches) goes to the per-user data home
  (`clinical_analytics.core.data_home`).

## Embedding model

- `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`,
  its official ONNX export, run with ONNX Runtime and HF `tokenizers` (`core/embeddings.py`).
  Apache-2.0; see `THIRD_PARTY_NOTICES.md`.
- Build step: `make fetch-embedding-model` downloads the pinned files once and checks each
  SHA-256 (`scripts/fetch_embedding_model.py`). Run it before `uv build`; the sdist and the wheel
  both include the model (`[tool.hatch.build] artifacts`; a plain `uv build` builds the wheel from
  the sdist, so it has to be in both).
- At start, `serve.py` checks the model is present and its SHA-256 is the pinned one (cached by
  size and mtime in the data home), and otherwise exits with code 2 and one line. It is never
  downloaded at runtime. `CLINICAL_ANALYTICS_EMBEDDING_MODEL_DIR` points at a model elsewhere.

## The backend as an Electron sidecar

Electron main (`electron/src/backendSidecar.js`) starts, checks and stops the backend:

| Setting | Default | Notes |
|---|---|---|
| `CLINICAL_ANALYTICS_PYTHON` | `../.venv/bin/python` (Windows `..\.venv\Scripts\python.exe`) | Step 2 points it at the bundled runtime |
| `CLINICAL_ANALYTICS_API_PORT` | 8000 | Used if free, otherwise an OS-assigned port on 127.0.0.1 |
| `CLINICAL_ANALYTICS_BACKEND_TIMEOUT_MS` | 60000 | Whole startup wait; each request also times out after 3 s |
| `CLINICAL_ANALYTICS_EXTERNAL_BACKEND=1` | off | Dev flow: use an already-running API (`make run-api`, `scripts/run_app.sh`) |

- Spawned as `python -I -u -m clinical_analytics.api.serve --host 127.0.0.1 --port P
  --exit-with-parent <electron pid> --stop-on-stdin-close`, with this launch's API token, from
  `<userData>/backend` (a folder of the app's own, not the home directory). `-I` (isolated
  mode) keeps the working directory and user site-packages off Python's import path and ignores
  every `PYTHON*` variable, so a `json.py` or `.pth` file in the user's files is never imported
  by the backend; the venv still finds its packages. `-u` keeps the output unbuffered.
- A bad `CLINICAL_ANALYTICS_API_PORT` or `CLINICAL_ANALYTICS_BACKEND_TIMEOUT_MS` stops startup
  with a message naming the variable.
- A startup window shows while it boots; the main window opens once `/health` answers and the
  server proves it knows the token. Failures show an error window (on exit code 2, serve.py's
  one-line reason). A backend that dies later gets a Restart / Quit dialog.
- The page's CSP header allows `connect-src` to exactly that backend's origin.
- Stop on quit: POSIX SIGTERM to the process group, SIGKILL after 10 s; Windows closes the
  backend's stdin (graceful), then `taskkill /T /F` after 10 s. If Electron dies, the closed stdin
  and the parent watchdog shut the backend down gracefully; it hard-exits only if that has not
  finished after 120 s. The watchdog checks that Electron's PID is still running (on Linux also
  that it is the same process, by start time), not that it is the direct parent, so a launcher
  between Electron and the interpreter is fine. Both lifelines set uvicorn's `should_exit` once
  the server has started; neither raises a signal.

## Checks

- `make smoke-installed`: runs the default `uv build` (sdist, then the wheel from it), checks the
  wheel contains the model, installs it in a fresh venv outside the repo, starts
  the API from an unrelated cwd with a temp HOME/XDG data home (in a loopback-only network
  namespace when available), uploads the synthetic cohort and runs a query. Also in CI.
- `cd electron && xvfb-run -a npx playwright test -c playwright.native.config.js`: the app starts
  its own backend, the CSP blocks other loopback ports, a crashed backend can be restarted, and
  no backend is left after a quit or a SIGKILL of Electron.

## Known gap for the storage owner

A graceful stop lets in-flight requests finish. If the process is killed hard mid-upload (power
loss, or the 120 s fallback), `UserDatasetStorage.save_upload` can leave a partial raw CSV and its
metadata JSON is rewritten in place under a lock rather than written to a temp file and renamed.
