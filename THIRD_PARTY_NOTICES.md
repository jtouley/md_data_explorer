# Third-party notices

Components redistributed inside the Clinical Analytics app (not ordinary package dependencies,
whose licences ship with their own wheels and npm packages).

## all-MiniLM-L6-v2 sentence-embedding model

- Source: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Files bundled: `onnx/model.onnx` (the repository's official ONNX export, SHA-256
  `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452`), `tokenizer.json` and the
  model's config JSON files; every file is pinned by SHA-256 in
  `src/clinical_analytics/core/embeddings.py` (`MODEL_FILES`) and fetched at build time by
  `scripts/fetch_embedding_model.py`.
- Licence: Apache License 2.0. Cited from the model card at that revision
  (`README.md`, front matter `license: apache-2.0`):
  <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md>.
  The repository has no separate LICENSE file; the licence text is in
  [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) (from <https://www.apache.org/licenses/LICENSE-2.0.txt>).
- Base model: `nreimers/MiniLM-L6-H384-uncased` (Microsoft MiniLM), fine-tuned by the
  sentence-transformers project. The bundled files are unmodified.

## Fonts

Source Sans 3 and Source Serif 4 (SIL Open Font License 1.1); see `electron/src/fonts/*-OFL.md`.
