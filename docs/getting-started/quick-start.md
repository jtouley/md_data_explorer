# Quick start

Assumes you have done the [Installation](installation.md) steps.

1. Start the app. On macOS or Linux:

   ```bash
   ./scripts/run_app.sh
   ```

   On any OS, including Windows, start the desktop app; it starts its own backend:

   ```bash
   cd electron && npm start
   ```

2. In the desktop app, click **Use the sample cohort**. This loads the bundled synthetic sample cohort (`src/clinical_analytics/resources/demo/sample_cohort.csv`; no real patients).
3. Ask `compare outcome by diabetes`, then `compare age by outcome`.
4. To use your own data, click **Upload a file…**. See [Uploading data](uploading-data.md).

Patient counts from 1 to 10 are shown as `<11` in answers.
