"""Adversarial probe set: 103 questions from the second independent review of PR #60, plus
geography probes from the aggregate-data triage.

Three datasets: the committed synthetic cohort, a generated frame with other column
conventions ("Age (years)", "Gender", coded "Smoker 1: Yes 2: No", "LDL mg/dL", "hba1c"), and
a synthetic geography frame whose values contain question words ("Las Animas County" and
"Count Lake" contain "count"; "North Dakota" and "South Dakota" share "dakota").
Truth is computed here with Polars from each question's plain meaning; None means no single
correct answer exists, so the question must be held or refused.
"""

from __future__ import annotations

import random
from typing import Any

import polars as pl

c = pl.col
S = c("Statin use 1: Yes 2: No")


def own_frame(size: int = 500) -> pl.DataFrame:
    """The reviewer's generated dataset, reproduced exactly from its seed."""
    rng = random.Random(7)
    n = size

    def R(xs: list[Any]) -> list[Any]:  # noqa: N802 - kept from the review script
        return [rng.choice(xs) for _ in range(n)]

    return pl.DataFrame(
        {
            "patient_id": [f"Z{i:04d}" for i in range(n)],
            "Age (years)": [rng.randint(18, 95) for _ in range(n)],
            "Gender": R(["Male", "Female"]),
            "Smoker 1: Yes 2: No": R([1, 2]),
            "diabetes_type": R(["Type 1", "Type 2", "None"]),
            "Hypertension": R(["Yes", "No"]),
            "Region": R(["North", "North East", "South", "West"]),
            "arm": R(["drug A", "drug B", "placebo"]),
            "statin": R(["atorvastatin", "simvastatin", "none"]),
            "Weight (kg)": [round(rng.uniform(45, 140), 1) for _ in range(n)],
            "LDL mg/dL": [round(rng.uniform(60, 200), 1) for _ in range(n)],
            "hba1c": [round(rng.uniform(4.5, 11), 1) for _ in range(n)],
            "SBP mmHg": [rng.randint(95, 200) for _ in range(n)],
            "readmitted": R(["yes", "no"]),
        }
    )


# Synthetic only. "Las Animas County" is the longest county name, so a tie broken by value
# length would pick it; "Count Lake" has "count" as a whole word.
GEO_LOCATIONS = ("Colorado", "Las Animas County", "Adams County", "Boulder County", "El Paso County", "Count Lake")
GEO_STATES = ("New Mexico", "North Dakota", "South Dakota")


def geo_frame(size: int = 240) -> pl.DataFrame:
    """A long-format aggregate-shaped frame: location, state, sex and a rate per row."""
    return pl.DataFrame(
        {
            "patient_id": [f"G{i:04d}" for i in range(size)],
            "location_name": [GEO_LOCATIONS[i % len(GEO_LOCATIONS)] for i in range(size)],
            "state": [GEO_STATES[(i // 7) % len(GEO_STATES)] for i in range(size)],
            "sex": [("Male", "Female")[(i // 18) % 2] for i in range(size)],
            "mx": [100.0 + (i * 37) % 250 for i in range(size)],
        }
    )


def geo_probes(geo: pl.DataFrame) -> list[tuple[str, str, Any]]:
    """Grammar words ("count", "number", "by", "per", "total") never select a value, and a
    word shared by several values ("county", "dakota") never selects one of them."""
    loc = c("location_name")
    return [
        ("geo", "count by sex", by(geo, True, "sex")),
        ("geo", "count", cnt(geo)),
        ("geo", "number of patients by sex", by(geo, True, "sex")),
        ("geo", "how many patients per sex?", by(geo, True, "sex")),
        ("geo", "count by state", by(geo, True, "state")),
        ("geo", "mean mx by sex", by(geo, True, "sex", "mx")),
        ("geo", "average mx by county", H),
        ("geo", "count by county", H),
        ("geo", "how many patients in a county?", H),
        ("geo", "how many patients in the counties?", H),
        ("geo", "how many patients in dakota?", H),
        ("geo", "how many patients in Las Animas County?", cnt(geo, loc == "Las Animas County")),
        ("geo", "count in las animas county", cnt(geo, loc == "Las Animas County")),
        ("geo", "average mx by location", by(geo, True, "location_name", "mx")),
        ("geo", "how many patients in El Paso County?", cnt(geo, loc == "El Paso County")),
        ("geo", "how many patients in Boulder County by sex?", by(geo, loc == "Boulder County", "sex")),
        ("geo", "how many patients in count lake?", cnt(geo, loc == "Count Lake")),
        ("geo", "how many patients in north dakota?", cnt(geo, c("state") == "North Dakota")),
        ("geo", "average mx in New Mexico", mean(geo, "mx", c("state") == "New Mexico")),
    ]


def cnt(df: pl.DataFrame, e: Any = True) -> int:
    return df.filter(e).height


def mean(df: pl.DataFrame, col: str, e: Any = True) -> Any:
    return df.filter(e)[col].mean()


def prop(df: pl.DataFrame, e: Any, cond: pl.Expr) -> Any:
    return df.filter(e).select(cond.mean()).item()


def by(df: pl.DataFrame, e: Any, g: str, col: str | None = None) -> dict[str, Any]:
    x = df.filter(e).group_by(g)
    r = x.agg(c(col).mean().alias("v")) if col else x.agg(pl.len().alias("v"))
    return {str(k): v for k, v in r.iter_rows()}


def rate(df: pl.DataFrame, g: str, cond: pl.Expr) -> dict[str, Any]:
    """Share of rows meeting ``cond`` within each value of ``g`` ("are men more likely to ...")."""
    r = df.group_by(g).agg(cond.mean().alias("v"))
    return {str(k): v for k, v in r.iter_rows()}


H = None  # must hold or refuse


def probes(coh: pl.DataFrame, own: pl.DataFrame, geo: pl.DataFrame) -> list[tuple[str, str, Any]]:
    """(dataset, question, truth) for every probe."""
    return [*_review_probes(coh, own), *geo_probes(geo)]


def _review_probes(coh: pl.DataFrame, own: pl.DataFrame) -> list[tuple[str, str, Any]]:
    """The second review's 103 probes."""
    return [
        ("coh", "how many patients given placebo?", cnt(coh, c("treatment") == "placebo")),
        (
            "coh",
            "how many people from the north treated with drug?",
            cnt(coh, (c("site") == "North") & (c("treatment") == "drug")),
        ),
        ("coh", "how many users of statin?", cnt(coh, S == 1)),
        ("coh", "how many patients taking statins?", cnt(coh, S == 1)),
        (
            "coh",
            "how many patients receiving placebo in the west?",
            cnt(coh, (c("treatment") == "placebo") & (c("site") == "West")),
        ),
        ("coh", "average ldl among patients given the drug", mean(coh, "ldl", c("treatment") == "drug")),
        ("coh", "average ldl within the south site", mean(coh, "ldl", c("site") == "South")),
        ("coh", "mean hdl of women from the west", mean(coh, "hdl", (c("sex") == "F") & (c("site") == "West"))),
        ("coh", "what is the average ldl of patients over 65?", mean(coh, "ldl", c("age") > 65)),
        ("coh", "how many patients over 65 with ldl above 130?", cnt(coh, (c("age") > 65) & (c("ldl") > 130))),
        ("coh", "how many patients with bmi 30+?", cnt(coh, c("bmi") >= 30)),
        ("coh", "how many patients with a bmi of 30 or more?", cnt(coh, c("bmi") >= 30)),
        ("coh", "how many patients with bmi under 25?", cnt(coh, c("bmi") < 25)),
        ("coh", "average hdl in patients with bmi over 30", mean(coh, "hdl", c("bmi") > 30)),
        ("coh", "how many patients with ldl over 130 and hdl under 50?", cnt(coh, (c("ldl") > 130) & (c("hdl") < 50))),
        ("coh", "how many diabetic patients?", cnt(coh, c("diabetes") == 1)),
        ("coh", "how many diabetics are women?", cnt(coh, (c("diabetes") == 1) & (c("sex") == "F"))),
        ("coh", "what is the average ldl of diabetics?", mean(coh, "ldl", c("diabetes") == 1)),
        ("coh", "how many patients on statin not in the north?", cnt(coh, (S == 1) & (c("site") != "North"))),
        ("coh", "how many women not on placebo?", cnt(coh, (c("sex") == "F") & (c("treatment") != "placebo"))),
        (
            "coh",
            "how many men on drug not using statin?",
            cnt(coh, (c("sex") == "M") & (c("treatment") == "drug") & (S == 2)),
        ),
        (
            "coh",
            "how many patients are not in the west and on drug?",
            cnt(coh, (c("site") != "West") & (c("treatment") == "drug")),
        ),
        ("coh", "how many patients in the north or west?", cnt(coh, c("site").is_in(["North", "West"]))),
        (
            "coh",
            "how many women in the north or west on drug?",
            cnt(coh, (c("sex") == "F") & c("site").is_in(["North", "West"]) & (c("treatment") == "drug")),
        ),
        ("coh", "average ldl by site", by(coh, True, "site", "ldl")),
        ("coh", "average ldl by site for women", by(coh, c("sex") == "F", "site", "ldl")),
        ("coh", "average ldl by sex in the north", by(coh, c("site") == "North", "sex", "ldl")),
        ("coh", "how many patients by treatment in the south?", by(coh, c("site") == "South", "treatment")),
        ("coh", "how many women by treatment?", by(coh, c("sex") == "F", "treatment")),
        ("coh", "number of patients by sex on drug", by(coh, c("treatment") == "drug", "sex")),
        ("coh", "compare ldl between women and men", by(coh, True, "sex", "ldl")),
        (
            "coh",
            "compare ldl between the north and the south",
            by(coh, c("site").is_in(["North", "South"]), "site", "ldl"),
        ),
        (
            "coh",
            "compare hdl by sex for patients between 50 and 65",
            by(coh, c("age").is_between(50, 65), "sex", "hdl"),
        ),
        ("coh", "compare bmi by treatment among statin users", by(coh, S == 1, "treatment", "bmi")),
        ("coh", "what proportion of women are diabetic?", prop(coh, c("sex") == "F", c("diabetes") == 1)),
        ("coh", "what proportion of patients in the north use statins?", prop(coh, c("site") == "North", S == 1)),
        ("coh", "what percentage of drug patients are on statin?", prop(coh, c("treatment") == "drug", S == 1)),
        ("coh", "what proportion of statin users are women?", prop(coh, S == 1, c("sex") == "F")),
        ("coh", "are statin users more likely to be women?", H),
        # The review script gave counts by sex here; the question asks for the statin rate of each sex.
        ("coh", "are men more likely to use statins than women?", rate(coh, "sex", S == 1)),
        ("coh", "what share of women are not on statin?", prop(coh, c("sex") == "F", S == 2)),
        (
            "coh",
            "how many patients with osteopenia?",
            cnt(coh, c("DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis") == 2),
        ),
        (
            "coh",
            "how many patients with an osteoporosis DEXA result?",
            cnt(coh, c("DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis") == 3),
        ),
        (
            "coh",
            "how many women with normal dexa?",
            cnt(coh, (c("sex") == "F") & (c("DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis") == 1)),
        ),
        (
            "coh",
            "how many patients without a normal dexa result?",
            cnt(coh, c("DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis") != 1),
        ),
        ("coh", "average age of statin users", mean(coh, "age", S == 1)),
        ("coh", "average age of statin non-users", mean(coh, "age", S == 2)),
        ("coh", "what is the statistic for ldl?", H),
        ("coh", "average followup in days for women", mean(coh, "followup_days", c("sex") == "F")),
        ("coh", "how many patients followed for more than 365 days?", cnt(coh, c("followup_days") > 365)),
        ("coh", "how many patients with followup under 100 days?", cnt(coh, c("followup_days") < 100)),
        ("coh", "how many events in women?", cnt(coh, (c("event") == 1) & (c("sex") == "F"))),
        ("coh", "how many patients had no outcome?", cnt(coh, c("outcome") == 0)),
        (
            "coh",
            "average age of patients older than 50 and younger than 70",
            mean(coh, "age", (c("age") > 50) & (c("age") < 70)),
        ),
        ("coh", "how many patients aged 40 to 60?", cnt(coh, c("age").is_between(40, 60))),
        ("coh", "how many patients under 40 or over 80?", cnt(coh, (c("age") < 40) | (c("age") > 80))),
        ("coh", "maximum ldl for men over 70", coh.filter((c("sex") == "M") & (c("age") > 70))["ldl"].max()),
        (
            "coh",
            "minimum bmi among women on placebo",
            coh.filter((c("sex") == "F") & (c("treatment") == "placebo"))["bmi"].min(),
        ),
        ("coh", "median hdl of patients in the north", coh.filter(c("site") == "North")["hdl"].median()),
        ("coh", "how many patients?", coh.height),
        ("coh", "total number of patients on drug", cnt(coh, c("treatment") == "drug")),
        ("coh", "count women", cnt(coh, c("sex") == "F")),
        ("coh", "how many females on placebo?", cnt(coh, (c("sex") == "F") & (c("treatment") == "placebo"))),
        ("coh", "average bmi", mean(coh, "bmi")),
        ("coh", "what is the mean hdl?", mean(coh, "hdl")),
        ("coh", "average ldl for men", mean(coh, "ldl", c("sex") == "M")),
        ("coh", "how many patients in each site?", by(coh, True, "site")),
        # own
        ("own", "how many smokers?", cnt(own, c("Smoker 1: Yes 2: No") == 1)),
        ("own", "how many non-smokers?", cnt(own, c("Smoker 1: Yes 2: No") == 2)),
        ("own", "how many female smokers?", cnt(own, (c("Smoker 1: Yes 2: No") == 1) & (c("Gender") == "Female"))),
        ("own", "how many patients with type 2 diabetes?", cnt(own, c("diabetes_type") == "Type 2")),
        ("own", "how many diabetic patients?", cnt(own, c("diabetes_type") != "None")),
        ("own", "how many patients without diabetes?", cnt(own, c("diabetes_type") == "None")),
        ("own", "how many patients in the north east?", cnt(own, c("Region") == "North East")),
        ("own", "how many patients in the north?", cnt(own, c("Region") == "North")),
        ("own", "how many patients on drug A?", cnt(own, c("arm") == "drug A")),
        ("own", "how many patients on a drug?", cnt(own, c("arm").is_in(["drug A", "drug B"]))),
        ("own", "how many patients not on placebo?", cnt(own, c("arm") != "placebo")),
        ("own", "how many patients taking atorvastatin?", cnt(own, c("statin") == "atorvastatin")),
        ("own", "how many patients on statins?", cnt(own, c("statin") != "none")),
        ("own", "how many patients not on a statin?", cnt(own, c("statin") == "none")),
        ("own", "average weight of women", mean(own, "Weight (kg)", c("Gender") == "Female")),
        ("own", "how many patients over 100 kg?", cnt(own, c("Weight (kg)") > 100)),
        ("own", "how many patients weighing over 100?", cnt(own, c("Weight (kg)") > 100)),
        ("own", "how many patients with weight above 100 kg?", cnt(own, c("Weight (kg)") > 100)),
        ("own", "how many patients over 65?", cnt(own, c("Age (years)") > 65)),
        ("own", "average ldl of patients over 65", mean(own, "LDL mg/dL", c("Age (years)") > 65)),
        ("own", "how many patients with ldl above 160 mg/dl?", cnt(own, c("LDL mg/dL") > 160)),
        ("own", "average sbp in hypertensive patients", mean(own, "SBP mmHg", c("Hypertension") == "Yes")),
        ("own", "how many patients with hypertension?", cnt(own, c("Hypertension") == "Yes")),
        ("own", "how many patients without hypertension?", cnt(own, c("Hypertension") == "No")),
        ("own", "how many patients were readmitted?", cnt(own, c("readmitted") == "yes")),
        (
            "own",
            "what proportion of smokers were readmitted?",
            prop(own, c("Smoker 1: Yes 2: No") == 1, c("readmitted") == "yes"),
        ),
        # Readmission rate by smoking status / by gender (the review script held the first and
        # gave counts for the second; both have one correct answer, the rate within each group).
        (
            "own",
            "are smokers more likely to be readmitted?",
            rate(own, "Smoker 1: Yes 2: No", c("readmitted") == "yes"),
        ),
        ("own", "are men more likely to be readmitted?", rate(own, "Gender", c("readmitted") == "yes")),
        ("own", "average hba1c in type 1 diabetes", mean(own, "hba1c", c("diabetes_type") == "Type 1")),
        ("own", "average hba1c above 7", H),
        ("own", "how many patients with hba1c over 7?", cnt(own, c("hba1c") > 7)),
        ("own", "how many patients with sbp 140+?", cnt(own, c("SBP mmHg") >= 140)),
        ("own", "average age by region", by(own, True, "Region", "Age (years)")),
        ("own", "how many men by arm?", by(own, c("Gender") == "Male", "arm")),
        (
            "own",
            "how many patients older than 80 in the south?",
            cnt(own, (c("Age (years)") > 80) & (c("Region") == "South")),
        ),
        ("own", "how many patients aged 65 or older?", cnt(own, c("Age (years)") >= 65)),
    ]
