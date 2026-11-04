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
- data/demo/sample_cohort.csv: the 80-row "Load sample" cohort the app ships, built from
  arithmetic patterns (no random draws) with two planted exceptions so the outcome is not a
  deterministic function of the predictors.

Paths are relative to the repo root; the files written are exactly the synthetic kit that
scripts/check_no_data_files.py allows to be tracked.

Usage (from the repo root):
    uv run python scripts/make_synthetic_data.py                # rewrite the committed files
    uv run python scripts/make_synthetic_data.py --out-dir DIR  # write the kit under DIR
    uv run python scripts/make_synthetic_data.py --check        # regenerate in a temp dir and
                                                                # fail if any committed file differs
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = Path("tests") / "fixtures" / "synthetic"
DEMO_PATH = Path("data") / "demo" / "sample_cohort.csv"
SEED = 20261103
# Demo rows whose outcome breaks the "age >= 55 or diabetes" rule; without them a logistic
# model on the demo cohort would face perfect separation.
DEMO_OUTCOME_EXCEPTIONS = frozenset({11, 20})

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


def make_demo_cohort(n: int = 80) -> pl.DataFrame:
    i = np.arange(1, n + 1)
    age = 25 + (7 * i) % 55
    diabetes = (i % 3 == 0).astype(np.int64)
    rule = (age >= 55) | (diabetes == 1)
    outcome = np.where(np.isin(i, list(DEMO_OUTCOME_EXCEPTIONS)), ~rule, rule).astype(np.int64)
    return pl.DataFrame(
        {
            "patient_id": [f"P{k:03d}" for k in i],
            "outcome": outcome,
            "age": age,
            "sex": np.where(i % 2 == 1, "M", "F"),
            "diabetes": diabetes,
            "hypertension": (i % 4 == 0).astype(np.int64),
        }
    )


def build_kit() -> dict[Path, pl.DataFrame]:
    """Every synthetic file, keyed by its path relative to the repo root.

    Draw order matters: all seeded datasets share one generator, so append new ones at the end.
    """
    rng = np.random.default_rng(SEED)
    return {
        SYNTHETIC_DIR / "cohort.csv": make_cohort(rng),
        SYNTHETIC_DIR / "unequal_groups.csv": make_unequal_groups(rng),
        SYNTHETIC_DIR / "sparse_cells.csv": make_sparse_cells(),
        SYNTHETIC_DIR / "separation.csv": make_separation(rng),
        DEMO_PATH: make_demo_cohort(),
    }


def write_kit(root: Path) -> list[Path]:
    """Write the kit under root (the repo root to refresh the committed files)."""
    written = []
    for rel, frame in build_kit().items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(target)
        written.append(rel)
    return written


def mismatches(generated_root: Path, committed_root: Path = REPO_ROOT) -> list[Path]:
    """Kit files whose committed bytes differ from a fresh generation (or are missing)."""
    bad = []
    for rel in build_kit():
        committed = committed_root / rel
        if not committed.is_file() or committed.read_bytes() != (generated_root / rel).read_bytes():
            bad.append(rel)
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--out-dir", type=Path, help="Write the kit under this directory instead of the repo")
    mode.add_argument("--check", action="store_true", help="Fail if the committed kit differs from a regeneration")
    args = parser.parse_args(argv)

    if args.check:
        with tempfile.TemporaryDirectory(prefix="synthetic-kit-") as tmp:
            write_kit(Path(tmp))
            bad = mismatches(Path(tmp))
        if bad:
            print("Committed synthetic files differ from scripts/make_synthetic_data.py output:", file=sys.stderr)
            for rel in bad:
                print(f"  {rel}", file=sys.stderr)
            print(
                "Regenerate with `uv run python scripts/make_synthetic_data.py` and review the diff.", file=sys.stderr
            )
            return 1
        print(f"Synthetic kit matches the generator byte for byte ({len(build_kit())} files).")
        return 0

    root = args.out_dir or REPO_ROOT
    for rel in write_kit(root):
        print(f"wrote {root / rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
