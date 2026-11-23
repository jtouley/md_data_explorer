# Accuracy direction: where we are and what we decided (2026-11-23)

This records the diagnosis and the plan so they survive outside any one working session. It contains no sealed test questions or answers.

## The goal

A physician with no data skills asks a plain-English question about their own data and gets a correct answer, with the method shown in plain words. It runs on their own computer with a local model. Wrong numbers must be zero: when unsure, the app asks back or says so.

## Diagnosis

Five independent reviews (failure forensics, a live model probe, architecture, evaluation process, ingestion) agreed:

- **The plateau is the app, not the model.** On 12 held-out questions the app got wrong, llama3.1:8b's first plan from the app's own prompt computed the right answer on 9. The app showed 5.
- **The files were misread at upload.**
  - The patient key was wrong on 3 of 16 datasets.
  - 5 of 21 date columns were never parsed.
  - Missing-value codes (99, 999) were stored as real values.
  - Code labels were corrupted, and no units were recorded.
  - Correct visit-level files were flagged for review.
- **Question-time guessing piled up instead.** `model_query` grew from 3,763 to 5,630 lines. Its regex calls went from 25 to 107, and its word lists from 23 to 98. The wrong-answer rate stayed flat at 21–25% across four sealed runs.
- **The coverage check worked like a coin flip.** It read the model's description of its plan, not what the plan computed. It held about as many right plans (69) as wrong ones (62), and passed meaning inversions such as "under 65" compiled as `>= 65`.
- **Public summary tables went down the patient path.**
  - A fresh CDC frequency table scored 0 of 18 right, with every number counting rows instead of summing the case-count column.
  - The URL loader hard-coded "not an estimates table".
  - Structure detection gave up silently when a few strata repeated.
  - No test put an aggregate table through the model path.
  - The held answers hid errors rather than catching them.

## Decisions

1. **The data contract is the semantic model and the single source of truth** (PR #155, ADR017). It records the grain, the patient key with evidence, sequences, roles, and per-column dates, missing codes, labels, units and pairs. Aggregate tables are marked as frequency or estimate, with their total rows. Facts are inferred once at upload, every one has a source, and the user can correct them. This is standard semantic-layer practice, as in dbt MetricFlow, Cube, LookML and WrenAI MDL.
2. **When unsure, ask; never silently change a number.**
   - Undocumented codes, two-digit centuries, an open day/month order and unrecognised aggregate shapes all become clinical questions.
   - While such a question is open, answers that depend on it hold.
3. **The model path reads the contract** (PR #154 rework).
   - The model fills fixed slots, constrained by the contract.
   - Checks look at what the compiled plan computes: direction, negation, thresholds, unit counted and timepoint.
   - Plans can count by each patient's first or latest value.
   - Frequency tables sum their count column, and estimate tables read the matching row.
   - Filter values are checked against real values.
   - The reading card is built from templates in clinical words.
   - Guessing code is deleted.
4. **Evaluation:**
   - Several sealed sets are authored separately and blind to the app, and at least one is a URL-loaded public summary table.
   - Fresh public datasets (CDC, NHANES) are measured before and after.
   - Every held answer is scored as if released.
   - We report stage-wise metrics and a confidence range rather than a pass/fail on small sets. The realistic target is under 2% wrong.
   - Builders never see sealed content.

## Regression guard (contract inference on 13 files the builder never saw)

| Fact | main | #155 (third check) |
|---|---|---|
| Patient key | 8/13 | 13/13 |
| What one line is | 4/13 | 13/13 |
| Dates parsed right | 10/15 | 15/15 |
| Missing codes handled | 0/9 | 9/9 |
| Real values turned into missing | 0 | 0 |
| Upload failures | 0 | 0 |
| Dates silently misread | 0 | 0 |

The three zero rows are merge conditions for every later change to contract inference.

## Next

- Close #155's fourth review round.
- Rework #154 against the contract, folding in techniques from a hands-on teardown of open-source text-to-SQL tools.
- Re-run the fresh public datasets and the new sealed sets once both PRs land.
