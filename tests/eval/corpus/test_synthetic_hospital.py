"""Synthetic hospital: seeded and reproducible, pinned in corpus.yaml, and carrying its traps."""

from __future__ import annotations

import hashlib
import io

import polars as pl
import pytest

from clinical_analytics.eval.corpus.fetch import load_corpus
from clinical_analytics.eval.corpus.score import is_small
from clinical_analytics.eval.corpus.synthetic_hospital import (
    DEFAULT_SEED,
    EXTRACTS,
    SyntheticDataError,
    generate,
    main,
)

FELLOW = [f"fellow_{extract}" for extract in EXTRACTS]
_AGE = pl.col("age_years").cast(pl.Int64)


def _frame(extract: str) -> pl.DataFrame:
    return pl.read_csv(io.BytesIO(generate(extract, "table")), infer_schema=False)


class TestGenerate:
    @pytest.mark.parametrize("extract", EXTRACTS)
    def test_generate_sameSeedTwice_sameBytes(self, extract):
        # Arrange / Act
        first, second = generate(extract, "table"), generate(extract, "table")

        # Assert
        assert first == second

    @pytest.mark.parametrize("extract", EXTRACTS)
    def test_generate_otherSeed_otherBytes(self, extract):
        # Arrange / Act
        default, other = generate(extract, "table"), generate(extract, "table", DEFAULT_SEED + 1)

        # Assert
        assert default != other

    @pytest.mark.parametrize("name", FELLOW)
    def test_corpusPins_matchGeneratedBytes(self, name):
        # Arrange
        dataset = load_corpus()[name]

        # Act
        actual = {
            item.path: hashlib.sha256(
                generate(item.synthetic.extract, item.synthetic.part, item.synthetic.seed)
            ).hexdigest()
            for item in dataset.files
            if item.synthetic is not None
        }

        # Assert: a generator change must update the pins (and be checked against the oracles).
        assert actual == {item.path: item.sha256 for item in dataset.files}

    @pytest.mark.parametrize("name", FELLOW)
    def test_corpusShape_matchesGeneratedTable(self, name):
        # Arrange
        dataset = load_corpus()[name]

        # Act
        frame = _frame(name.removeprefix("fellow_"))

        # Assert
        assert (frame.height, frame.width) == (dataset.rows, dataset.columns)

    @pytest.mark.parametrize(
        ("extract", "part"),
        [pytest.param("tb", "table", id="unknown_extract"), pytest.param("cdiff", "rows", id="unknown_part")],
    )
    def test_generate_unknownName_raises(self, extract, part):
        # Arrange / Act / Assert
        with pytest.raises(SyntheticDataError, match="unknown"):
            generate(extract, part)

    def test_main_badArguments_exitTwoNoTraceback(self, capsys):
        # Arrange / Act
        code = main(["cdiff"])

        # Assert
        assert code == 2
        assert "usage" in capsys.readouterr().err


class TestTraps:
    @pytest.mark.parametrize("extract", ["cdiff", "hepb", "measles"])
    def test_extract_hasExactDuplicateRows(self, extract):
        # Arrange
        frame = _frame(extract)

        # Act
        duplicates = frame.height - frame.unique().height

        # Assert
        assert duplicates > 0

    @pytest.mark.parametrize("extract", EXTRACTS)
    def test_extract_hasPatientsWithSeveralRows(self, extract):
        # Arrange
        frame = _frame(extract)

        # Act / Assert
        assert frame["patient_id"].n_unique() < frame.unique().height

    def test_pneumonia_secondPathogen_sameAdmissionTwoRows(self):
        # Arrange
        frame = _frame("pneumonia")

        # Act
        rows_per_admission = frame.group_by("encounter_id").agg(pl.col("pathogen").n_unique().alias("n"))

        # Assert
        assert rows_per_admission["n"].max() == 2

    def test_measles_iggInTwoUnits(self):
        # Arrange / Act
        units = set(_frame("measles")["measles_igg_unit"].drop_nulls().unique().to_list())

        # Assert
        assert units == {"IU/mL", "mIU/mL"}

    @pytest.mark.parametrize(
        ("extract", "cell"),
        [
            pytest.param(
                "cdiff",
                (pl.col("test_year") == "2017") & (pl.col("cdiff_result") == "positive") & (_AGE >= 65),
                id="cdiff_partial_first_year",
            ),
            pytest.param(
                "hepb",
                (pl.col("birth_region") == "Middle East") & (pl.col("new_hbv_diagnosis") == "yes"),
                id="hepb_new_diagnoses_one_region",
            ),
            pytest.param("pneumonia", (pl.col("pathogen") == "Legionella") & (_AGE >= 70), id="pneumonia_legionella"),
            pytest.param("measles", pl.col("risk_group") == "Childcare worker", id="measles_small_group"),
        ],
    )
    def test_extract_hasSmallCell(self, extract, cell):
        # Arrange
        frame = _frame(extract)

        # Act
        patients = frame.filter(cell)["patient_id"].n_unique()

        # Assert
        assert is_small(patients)
