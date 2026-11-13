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
    }


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
)
