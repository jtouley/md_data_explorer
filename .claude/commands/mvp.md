# MVP Command

Drive the repo to the v0.1.0 MVP in one session.

## What This Does

Executes the one-shot MVP prompt maintained in `docs/review/MVP_ONE_SHOT_PROMPT.md`. The work lands on the `mvp/v0.1` integration branch and ends with one PR into `main`. It never merges, tags, or rewrites history.

## Usage

```
/mvp
```

## Implementation

1. Read `docs/review/MVP_ONE_SHOT_PROMPT.md`.
2. Treat the text between the two `=====` lines as your task. Execute it step by step, honouring its "Stop and ask" list.
3. Read `docs/review/2026-10-26-repo-review.md` for the reasoning behind each step. If the repo state no longer matches its assumptions (for example, PR #59, #60 or #61 already merged), skip the steps that are already done and say so in the PR body.
