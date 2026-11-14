"""Ground truth for tests/eval/corpus/questions/sepsis_survival.yaml (UCI Sepsis Survival
Minimal Clinical Records, primary cohort: one row per sepsis episode, no patient ID).

Written from the UCI variable list, not from the app's code. The codes are in the headers:
sex_0male_1female, and hospital_outcome_1alive_0dead, where death is 0 (the inverse of the
usual death = 1). With no patient ID a row cannot be linked to another, so counts are of
episodes (the app's "visits"), and each episode counts as one person for the small-cell check.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import patient_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

SEX = "sex_0male_1female"
OUTCOME = "hospital_outcome_1alive_0dead"
DIED = kit.text(OUTCOME) == "0"
AGE = kit.number("age_years")

ORACLES: dict[str, kit.Oracle] = {
    "sepsis_survival-total-001": kit.count(unit="visits"),
    "sepsis_survival-episode-002": kit.count_by("episode_number", unit="visits"),
    "sepsis_survival-sexdeath-003": kit.rates(DIED, SEX),
    "sepsis_survival-episodedeath-004": kit.rates(DIED, "episode_number"),
    "sepsis_survival-age-005": kit.stat_by("median", "age_years", OUTCOME),
    "sepsis_survival-meanage-006": kit.stat_by("mean", "age_years", SEX),
    "sepsis_survival-deaths-007": kit.count(DIED & (AGE >= 65), unit="visits"),
    "sepsis_survival-or-008": kit.odds_ratio(AGE >= 65, DIED),
    "sepsis_survival-repeat-009": kit.held(),
    "sepsis_survival-after2020-010": kit.held(),
}


def answers(path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(pl.scan_csv(path, infer_schema=False), oracles or ORACLES)
