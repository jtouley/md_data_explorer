# Purpose and steel thread

This page says what the product is for and the order in which it gets built. It is the current direction. The [master plan](../implementation/MASTER_PLAN.md) lists older plans and the Electron cutover.

## Purpose

This is a local data analyst for anyone in medicine who has a dataset and a question: medical students, residents, fellows, attendings and faculty, in any specialty.

- **Any question.** They ask in their own words, about anything a capable data tool such as TextQL could answer.
- **Right answer, method shown.** They get the right answer and see the method, from a local model (Ollama).
- **Privacy.** Data never leaves the machine. No answer shows a count of 1–10 patients, or a count that can be worked out from that answer or by dropping one of its filters. See [What privacy does not cover yet](#what-privacy-does-not-cover-yet).
- **Any data.** Public datasets, registry exports, and hospital extracts with their dictionaries.

Use cases such as the ID-fellow research guide are examples, not the scope. The product is not narrowed to one specialty, one workflow or a fixed list of question types.

## What "works" means

Right answers on real public datasets, through the real upload API, with the local model running, from a fresh app state. Passing unit tests, a green CI or a safe hold do not count as "works".

The acceptance table reports two rates and never adds them together:

- **Answered right:** questions that expect an answer and got the right one. This is the usefulness number.
- **Correctly held:** questions that must be refused (a patient listing, for example) and were. This is a safety number, with WRONG and LEAK.

## Steel thread, in order

The first unfinished step is the one being worked on. Work that does not move it waits, except for the privacy, security and data-loss fixes described under [Merge gate](#merge-gate).

1. **The model tier runs on uploaded datasets.** This came with #101.
2. **An acceptance harness on COVID-MS** (GDSI, PhysioNet). It reports RIGHT, WRONG, HELD, ERROR and LEAK for each question, which gives the baseline.
   - It came with #104: `make accept DATASET=covid_ms MODEL=...`.
   - It reuses `presented_answer` from [`src/clinical_analytics/eval/headless.py`](../../src/clinical_analytics/eval/headless.py) and the presented-answer graders in [`tests/eval/test_presented_answers.py`](../../tests/eval/test_presented_answers.py).
   - Baseline: of 14 questions that expect an answer, 1 is answered right and 13 are held. The one question that must be held is held. 0 WRONG, 0 LEAK.
3. **The data contract becomes the model's only context:** dictionary, codebooks, units, grain and keys.
   - **Done when** the COVID-MS answered-right count rises with the dictionary uploaded, at 0 WRONG and 0 LEAK, on the held-out questions. Better context alone does not count.
   - **The contract feeds word accounting.** Its synonyms and descriptions count a question's words as used ("MS type" is in `ms_type2`'s description). This moves forward from ADR013's PR 2.
   - **A hold offers a choice of column,** not "rephrase". It lists the 2–3 contract columns that best match the word, plus "none of these". The choice is saved to the contract as a user synonym, so each word is asked about once per dataset.
   - **A confirmation re-runs the stored plan.** It never asks the model again, so a confirmed plan can't change under the user.
   - **The harness** reports answered right and correctly held separately, and an answer that needed one column choice separately from both. It gets a permanent held-out question set: paraphrases written by someone who has not read the code, graded by the same oracles. Hand-written vocabulary rules are capped (see [ADR013 §5.1](../implementation/ADR/ADR013.md#51-using-answer_checks-on-sql)).
   - **Merge order:** #102, then corpus formats and ZIP upload, then #107, then a re-run of `make accept` with and without the model.
4. **A governed SQL path for one table.** The model writes SQL and it is checked against the contract. Small cells are suppressed in the output, and the SQL is shown.
   - **Done when** model-written SQL answers the questions that the word-accounting gate would hold, on one table. #106 (user-typed SQL) is the validator and sandbox it runs in.
   - **Every filter is checked within each answer, on both the question path and the SQL path.** If dropping any one filter would change the count by 1–10 patients, the answer is refused. This stops "count A, then count A and B" without remembering earlier questions.
   - **Every answer shows the step-by-step patient count:** "1,141 patients → 612 women → 341 with RRMS". Steps of 1–10 patients are hidden.
   - The design for this step is cut down to what #106 built. See [ADR013](../implementation/ADR/ADR013.md).
5. **Statistics as tools.** The model picks the test, and the existing core computes it.
6. **MVP.** Widen to the corpus: Sepsis 2019, the MIMIC demos, population sets with ranges, and patient-level sets across specialties. Compare local Ollama with a hosted stand-in. The bar is 0 WRONG and 0 LEAK.
   - **A dataset-version log of shown counts** lands here. It refuses a count that differs from one already shown by a single changed filter.

The design reference is [ADR013](../implementation/ADR/ADR013.md). Take from it only what the current step needs.

### What waits

These don't move the acceptance table, so they wait until a step needs them:

- packaging and signed installers;
- contract revisions and lineage;
- charts;
- encryption at rest (full-disk encryption covers it; see [local data security](../security/LOCAL_DATA_SECURITY.md)).

## What privacy does not cover yet

The per-answer filter check sees one answer at a time. Two unrelated questions can still be subtracted: for example, "patients over 65" and later "patients over 66". Until the dataset-version log lands in step 6, the app does not detect this. After it lands, the log refuses a count one filter change away from a count already shown, and notes other overlaps; it cannot catch every combination of earlier questions.

Mature systems have the same limit. [DataSHIELD](https://github.com/datashield/dsBase) answers it with a warning, a session log and sanctions from a data custodian. This app has no custodian, so it promises only what its code enforces.

## Merge gate

A change merges only if it moves the acceptance table, or fixes a WRONG, LEAK or ERROR that the table shows. Hardening for cases the corpus doesn't hit waits.

**Exception:** fixes for privacy, security or data loss merge on their own merits, whether or not the corpus triggers the issue.

**Issues follow the table.** Each reproducible WRONG, LEAK, ERROR or unacceptable HOLD gets a GitHub issue. One issue covers one cause and lists every question it affects. It links the PR that fixes it and the oracle the fix must pass.

- An unacceptable HOLD is a hold on a question that expects an answer, when the hold gives the clinician nothing to act on, or when it happens on a held-out paraphrase.
- Issues are labelled `acceptance` plus one grade label: `grade:WRONG`, `grade:LEAK`, `grade:ERROR` or `grade:HOLD`.

## What we borrow from other tools

We adopt none of these; each needs a server, a database, a licence or a workflow that doesn't fit a local Polars app. We borrow their mechanisms.

| Tool | What we borrow | Where it lands |
|---|---|---|
| [Microsoft Data Formulator](https://github.com/microsoft/data-formulator) | Clarify by choosing, not rephrasing: a hold offers columns to pick. Later, follow-ups that name the answer they build on. | Step 3; step 6 |
| [DataSHIELD](https://github.com/datashield/dsBase) | Its subset rule (a filter must remove no one or enough people), made strict. Its model-saturation check (`nfilter.glm`). | Step 4; step 5 |
| [OHDSI ATLAS](https://github.com/OHDSI/Atlas) | Inclusion-rule attrition: count every combination of filters, then show the step-by-step count ([Book of OHDSI, ch. 10](https://book.ohdsi.org/Cohorts.html)). | Step 4 |
| [jamovi](https://github.com/jamovi/jamovi) and [JASP](https://github.com/jasp-stats/jasp-desktop) | Assumption checks, and a rank-based alternative next to each parametric test. | Step 5 |
| [ACRO](https://github.com/AI-SDC/ACRO) | A residual degrees-of-freedom floor for regressions, and dominance checks on sums and means. | Steps 4–5 |

**Nearest competitor:** [Domino Clinical Data Explorer](https://github.com/dominodatalab/clinical-data-explorer) (Apache-2.0). A chat agent calls aggregate data tools, can run on a local Ollama, and never shows the model a hidden result. It has **no small-cell suppression**, and it is built to run on the Domino platform. This app's edge is small-cell control, grain, units and statistical refusals, not "local" alone. Re-check it at MVP.

## Real data

Synthetic data is the rule. The only exception is acceptance testing on public, openly licensed, de-identified datasets: fetched at test time from their public source, checked against a pinned checksum, used under their licence, never committed, and never shown row by row. Aggregate results (counts, statistics, grades) may appear in logs, issues and pull requests. Everything else, unit tests included, uses synthetic data.

The same rule appears in [CONTRIBUTING.md](../../CONTRIBUTING.md) and [SECURITY.md](../../SECURITY.md).
