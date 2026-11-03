#!/usr/bin/env python3
"""Generate the synthetic cohorts used for golden tests and demos.

Every file is fully synthetic (seeded numpy draws, no real patients) and deterministic, so
expected answers can be recomputed by the tests from the same CSVs. Each dataset targets a
case where clinical data commonly breaks naive statistics:

- cohort.csv: 400 patients, continuous labs with a known treatment effect and ~5% missing LDL,
  0/1 diabetes, 1/2-coded statin use, a coded 3-level DEXA result with labels in the header,
  a 3-site factor, a binary outcome from a known logistic model, and right-censored follow-up.
- unequal_groups.csv: n=5 vs n=40 with very different variances (Welch vs pooled t-test).
- sparse_cells.csv: a 2x2 table with expected counts below 5 (Fisher's exact test).
- separation.csv: a predictor that perfectly separates the outcome (logistic regression is
  not estimable).

Run from the repo root:  uv run python scripts/make_synthetic_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic"
SEED = 20261103

DEXA_COLUMN = "DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis"
STATIN_COLUMN = "Statin use 1: Yes 2: No"


def make_cohort(rng: np.random.Generator, n: int = 400) -> pl.DataFrame:
    age = rng.integers(35, 85, n)
    sex = rng.choice(["F", "M"], n)
    treated = rng.random(n) < 0.5
    site = rng.choice(["North", "South", "West"], n, p=[0.4, 0.35, 0.25])
    site_shift = np.select([site == "North", site == "South"], [0.0, 4.0], 9.0)
    ldl = 135.0 - 18.0 * treated + site_shift + rng.normal(0, 15, n)
    ldl_missing = rng.random(n) < 0.05
    hdl = 52.0 + 3.0 * (sex == "F") + rng.normal(0, 8, n)
    bmi = 27.0 + 0.04 * (age - 60) + rng.normal(0, 4, n)
    diabetes = (rng.random(n) < 0.12 + 0.004 * (age - 35)).astype(np.int64)
    statin = np.where(rng.random(n) < 0.35 + 0.3 * diabetes, 1, 2)
    dexa = np.select(
        [age < 55, age < 70],
        [rng.choice([1, 2, 3], n, p=[0.7, 0.25, 0.05]), rng.choice([1, 2, 3], n, p=[0.45, 0.4, 0.15])],
        rng.choice([1, 2, 3], n, p=[0.25, 0.45, 0.30]),
    )
    logit = -5.0 + 0.06 * age - 0.9 * treated + 0.8 * diabetes
    outcome = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    hazard = 0.0015 * np.exp(0.02 * (age - 60) - 0.5 * treated)
    event_time = rng.exponential(1 / hazard)
    censor_time = rng.uniform(180, 1460, n)
    followup_days = np.minimum(event_time, censor_time).round().astype(np.int64) + 1
    event = (event_time <= censor_time).astype(np.int64)
    return pl.DataFrame(
        {
            "patient_id": [f"S{i:04d}" for i in range(1, n + 1)],
            "age": age,
            "sex": sex,
            "treatment": np.where(treated, "drug", "placebo"),
            "site": site,
            "ldl": [None if m else round(float(v), 1) for v, m in zip(ldl, ldl_missing, strict=True)],
            "hdl": hdl.round(1),
            "bmi": bmi.round(1),
            "diabetes": diabetes,
            STATIN_COLUMN: statin,
            DEXA_COLUMN: dexa,
            "outcome": outcome,
            "followup_days": followup_days,
            "event": event,
        }
    )


def make_unequal_groups(rng: np.random.Generator) -> pl.DataFrame:
    small = rng.normal(140, 25, 5)
    large = rng.normal(132, 4, 40)
    return pl.DataFrame(
        {
            "patient_id": [f"U{i:03d}" for i in range(1, 46)],
            "arm": ["rare"] * 5 + ["common"] * 40,
            "sbp": np.concatenate([small, large]).round(1),
        }
    )


def make_sparse_cells() -> pl.DataFrame:
    # 2x2: exposed 3/4 with outcome, unexposed 1/16 with outcome -> expected counts < 5.
    rows = [("yes", 1)] * 3 + [("yes", 0)] * 1 + [("no", 1)] * 1 + [("no", 0)] * 15
    return pl.DataFrame(
        {
            "patient_id": [f"F{i:03d}" for i in range(1, len(rows) + 1)],
            "exposed": [r[0] for r in rows],
            "infection": [r[1] for r in rows],
        }
    )


def make_separation(rng: np.random.Generator) -> pl.DataFrame:
    marker = np.concatenate([rng.uniform(0, 4.9, 30), rng.uniform(5.1, 10, 30)]).round(2)
    return pl.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(1, 61)],
            "marker": marker,
            "age": rng.integers(30, 80, 60),
            "relapse": [0] * 30 + [1] * 30,
        }
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "cohort.csv": make_cohort(rng),
        "unequal_groups.csv": make_unequal_groups(rng),
        "sparse_cells.csv": make_sparse_cells(),
        "separation.csv": make_separation(rng),
    }
    for name, frame in datasets.items():
        frame.write_csv(OUT_DIR / name)
        print(f"wrote {OUT_DIR / name} ({frame.height} rows)")


if __name__ == "__main__":
    main()
