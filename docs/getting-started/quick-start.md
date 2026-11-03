# Quick start

Assumes you have done the [Installation](installation.md) steps.

1. Start the app. On macOS or Linux:

   ```bash
   ./scripts/run_app.sh
   ```

   On Windows (or if you prefer two terminals), start the API and then the desktop app:

   ```bash
   uv run python -m clinical_analytics.api.serve --port 8000
   cd electron && npm start        # second terminal
   ```

2. In the desktop app, click **Load sample**. This loads the bundled synthetic sample cohort (`data/demo/sample_cohort.csv`; no real patients).
3. Ask `compare outcome by diabetes`, then `what predicts outcome`.
4. To use your own data, click **Upload dataset**. See [Uploading data](uploading-data.md).

Patient counts from 1 to 10 are shown as `<11` in answers.
