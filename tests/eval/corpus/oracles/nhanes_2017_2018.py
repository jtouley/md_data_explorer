"""Ground truth for tests/eval/corpus/questions/nhanes_2017_2018.yaml (NHANES 2017-2018).

Written from the NCHS codebooks (DEMO_J, BIOPRO_J, HEPC_J, HEPBD_J, BMX_J), not from the app's
code. One row per participant (SEQN) in DEMO_J; the lab and exam tables join 1:1 on SEQN, and a
participant missing from one has no value there. The tables are SAS XPORT files, read here with
pyreadstat. Codes: RIAGENDR 1 male, 2 female; RIDRETH3 race/Hispanic origin; LBXHCR (HCV RNA)
1 positive, 2 negative, 3 negative screening antibody; LBXSCR creatinine in mg/dL; RIDAGEYR is
top-coded at 80. Estimates here are unweighted: a question asking for the survey-weighted
estimate must be held, since the app does not use the survey design.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyreadstat

from clinical_analytics.eval.corpus import patient_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

ID = "SEQN"
JOINED = ("BIOPRO_J.xpt", "HEPC_J.xpt", "HEPBD_J.xpt", "BMX_J.xpt")
HCV_RNA_POSITIVE = pl.when(kit.text("LBXHCR").is_null()).then(None).otherwise(kit.text("LBXHCR") == "1")


def read_xpt(path: Path) -> pl.LazyFrame:
    frame, _meta = pyreadstat.read_xport(str(path), output_format="polars")
    return frame.lazy()


def scan(demo_path: Path) -> pl.LazyFrame:
    """DEMO_J with every other table left-joined on SEQN, all columns as text."""
    frame = read_xpt(demo_path)
    for name in JOINED:
        frame = frame.join(read_xpt(demo_path.parent / name), on=ID, how="left", validate="1:1")
    return kit.as_text(frame)


ORACLES: dict[str, kit.Oracle] = {
    "nhanes_2017_2018-total-001": kit.count(),
    "nhanes_2017_2018-sex-002": kit.count_by("RIAGENDR"),
    "nhanes_2017_2018-race-003": kit.count_by("RIDRETH3"),
    "nhanes_2017_2018-age-004": kit.stat_by("mean", "RIDAGEYR", "RIAGENDR"),
    "nhanes_2017_2018-crea-005": kit.count((kit.number("RIDAGEYR") > 65) & (kit.number("LBXSCR") > 1.5)),
    "nhanes_2017_2018-hcvsex-006": kit.rates(HCV_RNA_POSITIVE, "RIAGENDR"),
    "nhanes_2017_2018-bmi-007": kit.stat_by("mean", "BMXBMI", "RIAGENDR"),
    "nhanes_2017_2018-hcv-008": kit.count(kit.text("LBXHCR") == "1"),
    "nhanes_2017_2018-weighted-009": kit.held(),
    "nhanes_2017_2018-cycles-010": kit.held(),
}


def answers(path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(scan(path), oracles or ORACLES)
