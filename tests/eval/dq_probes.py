"""Data-quality probe files from the independent DQ review, as synthetic frames.

Each file is rebuilt here from the review's description (scratchpad reviews/tmp-dq), with
invented values only. Every question below has no single correct answer until the doctor
settles something (grain, spellings, implausible values) or the data is fixed (values recorded
as a limit, a column with no values), so each must be held or refused. On main they were
answered: "Total count: 170." beside "25.0% of patients ... (n=200)", "Mean age was 53.54",
"Mean crp was 5.57 higher in B than A ... p<0.001".
"""

from __future__ import annotations

import polars as pl

from clinical_analytics.core.query_plan import QueryPlan

N = 200
IDS = [f"P{i:03d}" for i in range(N)]


def dq_frames() -> dict[str, pl.DataFrame]:
    """Probe name -> frame."""
    ages = [30 + (i * 7) % 50 for i in range(N)]
    ages[0], ages[1] = 400, -3
    glucose = [round(4.0 + (i % 50) / 10, 1) for i in range(N)]
    glucose[0] = -5.0
    crp = [str(20 + i % 7) for i in range(N)]
    for i in range(0, 60, 2):
        crp[i] = "<5"
    for i in (1, 3, 5):
        crp[i] = "N/A"
    text_crp = [str(round(1 + (i % 80) / 10, 1)) for i in range(N)]
    text_crp[0], text_crp[1], text_crp[2], text_crp[4] = "<5", "N/A", ".", ">100"
    text_glucose = [str(round(4.0 + (i % 50) / 10, 1)) for i in range(N)]
    text_glucose[0], text_glucose[1] = "-5", "n/a"
    return {
        "dupids": pl.DataFrame(
            {"patient_id": IDS[:170] + IDS[:30], "age": [40] * 170 + [80] * 30, "died": [0] * 150 + [1] * 50}
        ),
        "range": pl.DataFrame({"patient_id": IDS, "age": ages, "glucose": glucose}),
        "crp_arm": pl.DataFrame({"patient_id": IDS, "arm": ["A" if i < 100 else "B" for i in range(N)], "crp": crp}),
        "textnum": pl.DataFrame({"patient_id": IDS, "crp": text_crp, "age": [50] * N}),
        "allnull": pl.DataFrame(
            {"patient_id": IDS, "hba1c": pl.Series([None] * N, dtype=pl.Float64), "sex": ["F", "M"] * (N // 2)}
        ),
        "sex": pl.DataFrame(
            {"patient_id": IDS, "sex": ["Male"] * 60 + ["male"] * 50 + ["M"] * 40 + ["Female"] * 50, "age": [50] * N}
        ),
        # Glucose stored as text ("n/a" keeps the column text): -5 must be held as it is for floats.
        "textglucose": pl.DataFrame({"patient_id": IDS, "glucose": text_glucose}),
        # A spelling held by 3 rows: a hold that names it shows the label of a 3-patient level.
        "sex_rare": pl.DataFrame(
            {"patient_id": IDS, "sex": ["Male"] * 100 + ["male"] * 47 + ["M"] * 3 + ["Female"] * 50, "age": [50] * N}
        ),
    }


def fold_frame() -> pl.DataFrame:
    """Arms of 200, 150 and 5 patients: the 5 is primary-suppressed, the 150 its complement."""
    arms = ["A"] * 200 + ["B"] * 150 + ["C"] * 5
    return pl.DataFrame({"patient_id": [f"Q{i:03d}" for i in range(len(arms))], "arm": arms})


def predictor_frame(name: str) -> pl.DataFrame:
    """60 patients at sites East, North and West. Died for every third patient, so no level
    separates the outcome; age varies.

    "predictor": West has 3 patients. "predictor_complete_cases": West has 12, but 3 of them
    have no age, so the model (complete cases) rests on 9 West patients.
    """
    size = 60
    west = 3 if name == "predictor" else 12
    frame = pl.DataFrame(
        {
            "patient_id": [f"Q{i:03d}" for i in range(size)],
            "site": ["West" if i < west else ("East" if i % 2 else "North") for i in range(size)],
            "died": [1 if i % 3 == 0 else 0 for i in range(size)],
            "age": [30 + (i * 13) % 40 for i in range(size)],
        }
    )
    if name == "predictor_complete_cases":
        frame = frame.with_columns(pl.when(pl.int_range(pl.len()) < 3).then(None).otherwise(pl.col("age")).alias("age"))
    return frame


# Predictor answers: (frame, outcome, predictors). The NL parser does not pass named predictors
# through to the plan, so the probe fixes them; the grader reads the claim and the term labels.
PREDICTOR_PROBES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("predictor", "died", ("age", "site")),
    ("predictor_complete_cases", "died", ("age", "site")),
)


def visits_died_frame() -> pl.DataFrame:
    """200 visit rows of 170 patients; died is the same at every visit of a patient (40 of 170)."""
    ids = IDS[:170] + IDS[:30]
    died = [1 if int(pid[1:]) < 40 else 0 for pid in ids]
    return pl.DataFrame({"patient_id": ids, "died": died, "age": [50] * N})


# (frame, question or fixed plan, confirm, expected presented answer): graded on what the app
# shows. The 150-patient arm keeps its own row, hidden, never the "groups under 11" label. After
# rows are confirmed as visits, the share of a patient-level event counts patients (40 of 170;
# by visits it would read 35.0%). The share is asked as a fixed plan: the NL parser does not
# read "what percentage of patients died?" as a share on this small frame, and parsing is not
# what the probe tests.
DIED_SHARE = QueryPlan(intent="COUNT", proportion_column="died", proportion_value=1, confidence=0.99)
SHOWN_PROBES: tuple[tuple[str, str | QueryPlan, bool, object], ...] = (
    ("fold", "how many patients by arm?", False, {"A": 200, "B": "hidden", "Other (groups under 11)": "hidden"}),
    ("visits_died", DIED_SHARE, True, "23.5% of patients had died=1 (n=170 patients)."),
)


# (probe, question): every one must be held or refused, never answered.
DQ_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("dupids", "how many patients are there?"),
    ("dupids", "what percentage of patients died?"),
    ("dupids", "average age"),
    ("range", "average age"),
    ("range", "what is the mean age?"),
    ("range", "average glucose"),
    ("crp_arm", "compare crp by arm"),
    ("crp_arm", "average crp"),
    ("textnum", "average crp"),
    ("allnull", "average hba1c"),
    ("sex", "how many patients by sex?"),
    ("sex", "how many male patients are there?"),
    ("textglucose", "average glucose"),
    ("sex_rare", "how many patients by sex?"),
)
