# Purpose and steel thread

This page says what the product is for and the order in which it gets built. It is the current direction. The [master plan](../implementation/MASTER_PLAN.md) lists older plans and the Electron cutover.

## Purpose

This is a local data analyst for anyone in medicine who has a dataset and a question: medical students, residents, fellows, attendings and faculty, in any specialty.

- **Any question.** They ask in their own words, about anything a capable data tool such as TextQL could answer.
- **Right answer, method shown.** They get the right answer and see the method, from a local model (Ollama).
- **Privacy.** Data never leaves the machine, and no count of 1–10 patients is ever shown.
- **Any data.** Public datasets, registry exports, and hospital extracts with their dictionaries.

Use cases such as the ID-fellow research guide are examples, not the scope. The product is not narrowed to one specialty, one workflow or a fixed list of question types.

## What "works" means

Right answers on real public datasets, through the real upload API, with the local model running, from a fresh app state. Passing unit tests, a green CI or a safe hold do not count as "works".

## Steel thread, in order

The first unfinished step is the one being worked on. Work that does not move it waits, except for the privacy, security and data-loss fixes described under [Merge gate](#merge-gate).

1. **The model tier runs on uploaded datasets.** This came with #101.
2. **An acceptance harness on COVID-MS** (GDSI, PhysioNet). It reports RIGHT, WRONG, HELD, ERROR and LEAK for each question, which gives the baseline.
   - It arrives with PR #104 as `make accept DATASET=covid_ms MODEL=...`. The target is not on `main` yet.
   - It reuses `presented_answer` from [`src/clinical_analytics/eval/headless.py`](../../src/clinical_analytics/eval/headless.py) and the presented-answer graders in [`tests/eval/test_presented_answers.py`](../../tests/eval/test_presented_answers.py).
3. **The data contract becomes the model's only context:** dictionary, codebooks, units, grain and keys.
4. **A governed SQL path for one table.** The model writes SQL and it is checked against the contract. Small cells are suppressed in the output, and the SQL is shown.
5. **Statistics as tools.** The model picks the test, and the existing core computes it.
6. **MVP.** Widen to the corpus: Sepsis 2019, the MIMIC demos, population sets with ranges, and patient-level sets across specialties. Compare local Ollama with a hosted stand-in. The bar is 0 WRONG and 0 LEAK.

The design reference is [ADR013](../implementation/ADR/ADR013.md). Take from it only what the current step needs.

## Merge gate

A change merges only if it moves the acceptance table, or fixes a WRONG, LEAK or ERROR that the table shows. Hardening for cases the corpus doesn't hit waits.

**Exception:** fixes for privacy, security or data loss merge on their own merits, whether or not the corpus triggers the issue.

## Real data

Synthetic data is the rule. The only exception is acceptance testing on public, openly licensed, de-identified datasets: fetched at test time from their public source, checked against a pinned checksum, used under their licence, never committed, and never shown row by row. Aggregate results (counts, statistics, grades) may appear in logs, issues and pull requests. Everything else, unit tests included, uses synthetic data.

The same rule appears in [CONTRIBUTING.md](../../CONTRIBUTING.md) and [SECURITY.md](../../SECURITY.md).
