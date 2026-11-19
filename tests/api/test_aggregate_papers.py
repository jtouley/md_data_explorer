# SPDX-License-Identifier: MIT
"""A fellow's six paper ideas, end to end on synthetic public tables (ADR014 §10).

Each idea from "Michigan & US Infectious Disease Paper Ideas" is run the way a fellow would:
upload the public tables it needs through the real upload route (declared public population
estimates), then ask plain-English questions through POST /api/aggregate/questions, confirming
holds with the defaults offered. Every expected number comes from an independent Polars read of
the synthetic frames (the oracle), never from the engine. All values are made up; no GBD or CDC
data is used.

Ideas 3 and 5 cannot be carried all the way: NOAA daily station rainfall and PLACES county
estimates do not join at state-year grain; they are refused, not guessed (PLACES here, NOAA in
tests/core/test_estimate_join.py).
"""

from __future__ import annotations

import io
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

from clinical_analytics.contracts.config import MIN_CELL_SIZE, POPULATION_ESTIMATES_FORM_VALUE
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED

pytestmark = [pytest.mark.slow, pytest.mark.integration]

GREAT_LAKES = ("Illinois", "Indiana", "Michigan", "Minnesota", "New York", "Ohio", "Pennsylvania", "Wisconsin")


def _upload(client: TestClient, frame: pl.DataFrame, name: str) -> str:
    buffer = io.BytesIO()
    frame.write_csv(buffer)
    response = client.post(
        "/api/datasets/upload",
        files={"file": (f"{name}.csv", buffer.getvalue(), "text/csv")},
        data={"dataset_name": name, "population_estimates": POPULATION_ESTIMATES_FORM_VALUE},
    )
    body = response.json()
    assert response.status_code == 200, body
    assert body["population_estimates"] is True
    return str(body["upload_id"])


def _ask(client: TestClient, ids: list[str], question: str, confirm: bool = True) -> dict[str, Any]:
    """Ask; when held and ``confirm``, ask again with the offered defaults (a fellow clicking OK)."""
    response = client.post("/api/aggregate/questions", json={"dataset_ids": ids, "question": question})
    assert response.status_code == 200, response.text
    body = response.json()
    if body["status"] == "held" and confirm:
        choices: dict[str, dict[str, Any]] = {}
        for option in body["options"]:
            choices.setdefault(option["dataset_id"], {})[option["column"]] = option["default"]
        response = client.post(
            "/api/aggregate/questions", json={"dataset_ids": ids, "question": question, "choices": choices}
        )
        assert response.status_code == 200, response.text
        body = response.json()
    return body


def _gbd(frame: pl.DataFrame, **where: Any) -> pl.DataFrame:
    return frame.filter(pl.all_horizontal([pl.col(k) == v for k, v in where.items()]))


def test_idea1_hcvWomenMidwest_changeBySexAndWonderDeaths(test_client, isolated_data_home, make_public_table):
    # Arrange
    gbd = make_public_table("gbd", locations=(*GREAT_LAKES, "United States"), years=tuple(range(2010, 2024)))
    wonder = make_public_table("wonder", states=GREAT_LAKES)
    ids = [_upload(test_client, gbd, "GBD"), _upload(test_client, wonder, "WONDER")]

    # Act
    change = _ask(
        test_client,
        ids[:1],
        "How did the age-standardized hepatitis C death rate change from "
        "2010 to 2023 in Michigan women compared with men?",
    )
    neighbours = _ask(
        test_client,
        ids[:1],
        "How did the age-standardized HCV death rate change from 2010 to 2023 in Great Lakes women?",
    )
    side = _ask(test_client, ids, "GBD crude HCV death rate and WONDER deaths in Michigan by sex, 2023")

    # Assert: change per sex, each year's own interval, percent change from the oracle
    assert change["status"] == "answered", change["message"]
    rows = {r["group"]: r for r in change["table"]["rows"]}
    for sex in ("Female", "Male"):
        frame = _gbd(gbd, location_name="Michigan", sex_name=sex, age_name="Age-standardized", metric_name="Rate")
        first = frame.filter(pl.col("year") == 2010).row(0, named=True)
        last = frame.filter(pl.col("year") == 2023).row(0, named=True)
        assert rows[f"Michigan · {sex}"]["percent change"] == round(
            (last["val"] - first["val"]) / first["val"] * 100, 1
        )
    assert "No interval is computed for the change" in " ".join(change["notes"])
    assert neighbours["status"] == "answered", neighbours["message"]
    assert {r["group"].split(" · ")[0] for r in neighbours["table"]["rows"]} == set(GREAT_LAKES)
    # Assert: WONDER deaths beside GBD, small and suppressed counts never shown
    assert side["status"] == "answered", side["message"]
    for row in side["table"]["rows"]:
        source = wonder.filter(
            (pl.col("State") == "Michigan") & (pl.col("Year") == "2023") & (pl.col("Gender") == row["sex"])
        ).row(0, named=True)
        deaths = source["Deaths"]
        if deaths == "Suppressed" or int(deaths) < MIN_CELL_SIZE:
            assert row["WONDER: Deaths"] == SECONDARY_SUPPRESSED
        else:
            assert row["WONDER: Deaths"] == float(deaths)
    assert any("WONDER" in m and "count" in m for m in side["method"])


def test_idea2_syphilisModelVsSurveillance_screenThenSavedChoice(test_client, isolated_data_home, make_public_table):
    # Arrange
    gbd = make_public_table("gbd", causes=("Syphilis",), measures=("Incidence",), years=(2022, 2023))
    atlas = make_public_table("atlasplus")
    ids = [_upload(test_client, gbd, "GBD syphilis"), _upload(test_client, atlas, "AtlasPlus")]
    question = (
        "Which states disagree most between GBD syphilis incidence and AtlasPlus rates in 2023, both sexes, all ages?"
    )

    # Act
    first = _ask(test_client, ids, question, confirm=False)
    confirmed = _ask(test_client, ids, question)
    again = _ask(test_client, ids, question, confirm=False)

    # Assert: held once on the measure, then answered; the choice is saved for the dataset
    assert first["status"] == "held"
    assert first["options"][0]["column"] == "__standardization__"
    assert confirmed["status"] == "answered", confirmed["message"]
    assert again["status"] == "answered", again["message"]
    model = _gbd(gbd, sex_name="Both", age_name="All ages", metric_name="Rate", year=2023).select(
        pl.col("location_name").alias("Geography"), "lower", "upper"
    )
    joined = (
        atlas.filter(pl.col("Year") == "2023")
        .with_columns(pl.col("Rate per 100000").cast(pl.Float64))
        .join(model, on="Geography")
    )
    above = joined.filter(pl.col("Rate per 100000") > pl.col("upper")).height
    below = joined.filter(pl.col("Rate per 100000") < pl.col("lower")).height
    assert confirmed["message"].startswith(
        f"Of {joined.height} rows, AtlasPlus's value is above GBD "
        f"syphilis's interval in {above} and below it in {below}"
    )


def test_idea3_legionella_gbdBesideNndss(test_client, isolated_data_home, make_public_table):
    # Arrange (NOAA daily station rainfall is refused by the engine: tests/core/test_estimate_join.py)
    gbd = make_public_table(
        "gbd", causes=("Lower respiratory infections",), metrics=("Percent",), years=(2023,), rei="Legionella spp.",
        ages=("All ages",),
    )  # fmt: skip
    nndss = make_public_table("nndss")
    ids = [_upload(test_client, gbd, "GBD etiology"), _upload(test_client, nndss, "NNDSS")]

    # Act
    side = _ask(
        test_client,
        ids,
        "Share of pneumonia deaths due to Legionella and NNDSS legionellosis "
        "cases in the Great Lakes states, 2023, both sexes",
    )

    # Assert
    assert side["status"] == "answered", side["message"]
    assert len(side["table"]["rows"]) == len(GREAT_LAKES)
    for row in side["table"]["rows"]:
        cases = int(
            nndss.filter((pl.col("Reporting Area") == row["state"].upper()) & (pl.col("Year") == "2023"))[
                "Legionellosis Cases"
            ].item()
        )
        expected = SECONDARY_SUPPRESSED if cases < MIN_CELL_SIZE else float(cases)
        assert row["NNDSS: Legionellosis Cases"] == expected


def test_idea4_hivGreatLakes_changeAndPrepSideBySide(test_client, isolated_data_home, make_public_table):
    # Arrange
    gbd = make_public_table("gbd", causes=("HIV/AIDS",), years=(2010, 2023))
    aidsvu = make_public_table("aidsvu", years=(2023,))
    ids = [_upload(test_client, gbd, "GBD HIV"), _upload(test_client, aidsvu, "AIDSVu")]

    # Act
    change = _ask(
        test_client,
        ids[:1],
        "How did age-standardized HIV death rates change from 2010 to "
        "2023 in Michigan, Ohio and Wisconsin, both sexes?",
    )
    prep = _ask(
        test_client,
        ids,
        "Age-standardized HIV death rate and AIDSVu PrEP rate in the Great Lakes states, 2023, both sexes",
    )

    # Assert
    assert change["status"] == "answered", change["message"]
    assert {r["group"] for r in change["table"]["rows"]} == {"Michigan · Both", "Ohio · Both", "Wisconsin · Both"}
    assert prep["status"] == "answered", prep["message"]
    assert len(prep["table"]["rows"]) == len(GREAT_LAKES)
    mi = next(r for r in prep["table"]["rows"] if r["state"] == "Michigan")
    assert mi["AIDSVu: PrEP Rate"] == float(aidsvu.filter(pl.col("State Abbreviation") == "MI")["PrEP Rate"].item())


def test_idea5_preventablePneumonia_gbdBesideFluVaxViewButPlacesRefused(
    test_client, isolated_data_home, make_public_table
):
    # Arrange
    gbd = make_public_table(
        "gbd", causes=("Lower respiratory infections",), metrics=("Percent",), years=(2023,), rei="Smoking",
        ages=("All ages",),
    )  # fmt: skip
    flu = make_public_table("fluvaxview")
    places = make_public_table("places")
    ids = [
        _upload(test_client, gbd, "GBD smoking"),
        _upload(test_client, flu, "FluVaxView"),
        _upload(test_client, places, "PLACES"),
    ]

    # Act
    side = _ask(
        test_client,
        ids[:2],
        "Pneumonia deaths attributable to smoking and flu vaccination for "
        "adults 18 years and older in Michigan, 2023, both sexes",
    )
    county = _ask(
        test_client,
        [ids[0], ids[2]],
        "Smoking-attributable pneumonia deaths and PLACES smoking in Michigan, 2023, both sexes",
    )

    # Assert
    assert side["status"] == "answered", side["message"]
    row = side["table"]["rows"][0]
    oracle = flu.filter(
        (pl.col("Geography") == "Michigan")
        & (pl.col("Season/Survey Year") == "2022-23")
        & (pl.col("Dimension") == "≥18 Years")
    )
    assert row["FluVaxView: Estimate (%)"] == float(oracle["Estimate (%)"].item())
    assert any("percent" in m for m in side["method"])
    assert county["status"] == "refused"
    assert county["message"].startswith("PLACES is by county or city")


def test_idea6_precision_widestIntervalsAcrossStatesAndCauses(test_client, isolated_data_home, make_public_table):
    # Arrange
    causes = ("Syphilis", "Hepatitis C", "HIV/AIDS", "Tuberculosis", "Lower respiratory infections")
    gbd = make_public_table("gbd", causes=causes, years=(2023,))
    ids = [_upload(test_client, gbd, "GBD ID causes")]

    # Act
    result = _ask(
        test_client,
        ids,
        "Which states and causes have the widest uncertainty intervals for age-standardized death rates in 2023?",
    )

    # Assert
    assert result["status"] == "answered", result["message"]
    oracle = (
        _gbd(gbd, sex_name="Both", age_name="Age-standardized", metric_name="Rate")
        .filter(pl.col("location_name") != "United States")
        .with_columns(((pl.col("upper") - pl.col("lower")) / pl.col("val")).round(3).alias("width"))
        .sort("width", descending=True)
    )
    shown = result["table"]["rows"]
    assert len(shown) == oracle.height
    top = oracle.row(0, named=True)
    assert shown[0]["relative width"] == top["width"]
    assert shown[0]["group"] == f"{top['location_name']} · {top['cause_name']}"


def test_aggregateQuestion_undeclaredUpload_refused(test_client, isolated_data_home, make_public_table):
    # Arrange: uploaded without the mark
    buffer = io.BytesIO()
    make_public_table("atlasplus").write_csv(buffer)
    response = test_client.post(
        "/api/datasets/upload",
        files={"file": ("a.csv", buffer.getvalue(), "text/csv")},
        data={"dataset_name": "AtlasPlus"},
    )
    dataset_id = response.json()["upload_id"]

    # Act
    result = _ask(test_client, [dataset_id], "AtlasPlus syphilis rate in Michigan, 2023")

    # Assert
    assert result["status"] == "refused"
    assert result["message"].startswith("AtlasPlus wasn't marked")


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"dataset_ids": [], "question": "x"}, 422),
        ({"dataset_ids": ["../etc"], "question": "x"}, 400),
        ({"dataset_ids": ["nope_123"], "question": "x"}, 404),
        ({"dataset_ids": ["a", "a"], "question": "x"}, 400),
        ({"dataset_ids": ["a"], "question": "x", "extra": 1}, 422),
    ],
    ids=["no_tables", "bad_id", "unknown", "twice", "extra_field"],
)
def test_aggregateQuestion_badRequest_rejected(test_client, isolated_data_home, payload, status):
    # Act
    response = test_client.post("/api/aggregate/questions", json=payload)

    # Assert
    assert response.status_code == status


def test_aggregateQuestion_choiceNotALevel_rejected(test_client, isolated_data_home, make_public_table):
    # Arrange
    dataset_id = _upload(test_client, make_public_table("gbd", years=(2023,)), "GBD")

    # Act
    response = test_client.post(
        "/api/aggregate/questions",
        json={"dataset_ids": [dataset_id], "question": "x", "choices": {dataset_id: {"sex_name": "Nobody"}}},
    )

    # Assert
    assert response.status_code == 400
    assert "'Nobody' is not a value of sex_name" in response.json()["detail"]
