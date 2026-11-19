"""Ground truth for tests/eval/corpus/questions/sepsis_2019.yaml (PhysioNet/CinC Challenge 2019,
training sets A and B: one PSV file per ICU patient, one row per hour).

Written from the PhysioNet project page, not from the app's code. The patient ID is only in the
file name (pNNNNNN.psv); "NaN" is missing; Gender is 0 female, 1 male; SepsisLabel is 1 from 6
hours before sepsis onset on, so a patient with sepsis is one with any hour labelled 1; Age and
Gender are repeated on every hour. A question about hours counts rows (the app's "visits"); one
about patients counts files.

The oracles read the corpus snapshot's upload ZIP: every .psv member, its rows prefixed with the
member's file-name stem, parsed in one pass (only the columns the questions use).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path, PurePosixPath

import polars as pl

from clinical_analytics.eval.corpus import patient_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer, OracleError

ID = "patient"
COLUMNS = ("Age", "Gender", "HR", "Lactate", "SepsisLabel")
SEPSIS = kit.text("SepsisLabel") == "1"


def scan(path: Path) -> pl.LazyFrame:
    """Every hour of every patient in the ZIP, with the patient's ID, as text columns.

    Raises:
        OracleError: the ZIP holds no .psv file, or two files disagree on the header.
    """
    header: bytes | None = None
    body = io.BytesIO()
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            name = PurePosixPath(info.filename)
            if name.suffix != ".psv":
                continue
            first, _, rows = archive.read(info).partition(b"\n")
            if header is None:
                header = first.rstrip(b"\r")
                body.write(f"{ID}|".encode() + header + b"\n")
            elif first.rstrip(b"\r") != header:
                raise OracleError(f"{info.filename}: header differs from the first file's")
            prefix = f"{name.stem}|".encode()
            for line in rows.splitlines():
                if line.strip():
                    body.write(prefix + line + b"\n")
    if header is None:
        raise OracleError(f"{path.name} holds no .psv file")
    body.seek(0)
    return pl.read_csv(body, separator="|", infer_schema=False, columns=[ID, *COLUMNS]).lazy()


ORACLES: dict[str, kit.Oracle] = {
    "sepsis_2019-patients-001": kit.count(patient=ID),
    "sepsis_2019-hours-002": kit.count(unit="visits", patient=ID),
    "sepsis_2019-gender-003": kit.count_by("Gender", patient=ID),
    "sepsis_2019-sepsis-004": kit.count(SEPSIS, patient=ID),
    "sepsis_2019-sexrate-005": kit.rates(SEPSIS, "Gender", patient=ID),
    "sepsis_2019-age-006": kit.stat_by("median", "Age", "Gender", patient=ID),
    "sepsis_2019-hr-007": kit.stat_by("mean", "HR", "SepsisLabel"),
    "sepsis_2019-lactate-008": kit.count(kit.number("Lactate").is_not_null(), unit="visits", patient=ID),
    "sepsis_2019-listing-009": kit.held("listing", SEPSIS, patient=ID),
    "sepsis_2019-after2020-010": kit.held(),
}


def answers(path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(scan(path), oracles or ORACLES)
