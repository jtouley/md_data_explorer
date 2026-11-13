"""Reverse-accounting probes: plan elements the question never asked for.

A generated frame (seeded, synthetic) whose numeric lab values share digits with words in
the questions: Creatinine holds 2.2 and 3.3, so a "2" in "type 2 diabetes" or a "3" in
"stage 3 kidney disease" must never turn into a Creatinine limit. Truth is computed here
with Polars from each question's plain meaning; None means the question must be held.
"""

from __future__ import annotations

import random
from typing import Any

import polars as pl

c = pl.col


def reverse_frame(size: int = 400) -> pl.DataFrame:
    """Synthetic labs cohort, reproduced exactly from its seed."""
    rng = random.Random(11)

    def pick(options: list[Any]) -> list[Any]:
        return [rng.choice(options) for _ in range(size)]

    return pl.DataFrame(
        {
            "patient_id": [f"R{i:04d}" for i in range(size)],
            "age": [rng.randint(20, 90) for _ in range(size)],
            "sex": pick(["Male", "Female"]),
            "Creatinine": pick([0.8, 1.1, 1.4, 2.2, 3.3, 2.0]),
            "diabetes": pick(["Type 1", "Type 2", "None"]),
            "kidney disease": pick(["Stage 3", "Stage 4", "None"]),
            "BMI": pick([22.0, 25.5, 28.4, 30.0, 31.2, 35.0]),
            "dialysis_years": [rng.randint(0, 10) for _ in range(size)],
            "hba1c": [round(rng.uniform(4.5, 11), 1) for _ in range(size)],
        }
    )


def _count(frame: pl.DataFrame, where: pl.Expr) -> int:
    return frame.filter(where).height


def _mean(frame: pl.DataFrame, column: str, where: pl.Expr) -> Any:
    return frame.filter(where)[column].mean()


def reverse_probes(frame: pl.DataFrame) -> list[tuple[str, Any]]:
    """(question, truth) pairs. A truth of None means the plan must be held, never run."""
    return [
        ("how many patients with type 2 diabetes?", _count(frame, c("diabetes") == "Type 2")),
        ("average hba1c in type 2 diabetes", _mean(frame, "hba1c", c("diabetes") == "Type 2")),
        ("how many patients with stage 3 kidney disease?", _count(frame, c("kidney disease") == "Stage 3")),
        (
            "average age of patients with stage 3 kidney disease",
            _mean(frame, "age", c("kidney disease") == "Stage 3"),
        ),
        ("how many patients on dialysis for over 2 years?", _count(frame, c("dialysis_years") > 2)),
        ("how many patients with BMI 30 plus?", _count(frame, c("BMI") >= 30)),
        ("how many patients with creatinine over 2?", _count(frame, c("Creatinine") > 2)),
        # "over 2 years on dialysis": the 2 is dialysis years, not age. Must not run as age > 2.
        ("how many patients over 2 years on dialysis?", None),
    ]
