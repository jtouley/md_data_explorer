# SPDX-License-Identifier: MIT
"""Staging order for a declared upload (ADR014 §7): stage without the public markers, run the
guard, and re-stage as public only once accepted. A declared file that is refused is staged
exactly as an undeclared one ("<5" stays text). Same for a declared CSV inside a ZIP.
Synthetic files only; every value is made up."""

from __future__ import annotations

import functools
import io
import zipfile

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.contracts.geography import STATES
from clinical_analytics.storage.staging import read_csv_typed
from clinical_analytics.ui.storage.user_datasets import extract_zip_tables, stage_declared


def _csv(frame: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.write_csv(buffer)
    return buffer.getvalue()


def _public_table(marker: str) -> bytes:
    """25 states x 2 years of counts and rates; Alaska's cells carry the source's marker."""
    rows = [
        {
            "State": s.name,
            "Year": y,
            "Cases": marker if s.name == "Alaska" else str(200 + i),
            "Rate per 100000": marker if s.name == "Alaska" else f"{10 + i / 10:.1f}",
        }
        for i, s in enumerate(STATES[:25])
        for y in (2022, 2023)
    ]
    return _csv(pl.DataFrame(rows))


def _labs() -> bytes:
    """A patient lab file: one row per patient visit, with "<5" limits in the lab values."""
    rows = [
        {
            "patient_id": f"P{p:03d}",
            "visit_month": f"2024-0{v}",
            "crp_mg_l": "<5" if (p + v) % 4 == 0 else f"{10 + p % 60}.5",
            "viral_load": "<20" if (p + v) % 5 == 0 else str(100 + p),
        }
        for p in range(30)
        for v in (1, 2, 3)
    ]
    return _csv(pl.DataFrame(rows))


def _stage(content: bytes, declared: bool) -> pl.DataFrame:
    return stage_declared(functools.partial(_read, content), declared).data


def _read(content: bytes, public: bool):
    return read_csv_typed(content, public=public)


class TestDeclaredStagingOrder:
    def test_stageDeclared_declaredPatientFile_sameAsUndeclared(self):
        # Arrange
        content = _labs()

        # Act
        declared = _stage(content, declared=True)
        undeclared = _stage(content, declared=False)

        # Assert
        assert_frame_equal(declared, undeclared)
        assert declared.schema["crp_mg_l"] == pl.String
        assert "crp_mg_l suppressed" not in declared.columns

    @pytest.mark.parametrize("marker", ["Suppressed", "<10", "(D)", "*", "1-9", "<5", "Data suppressed"])
    def test_stageDeclared_publicTableWithMarkers_flaggedSuppressed(self, marker):
        # Act
        frame = _stage(_public_table(marker), declared=True)

        # Assert
        alaska = frame.filter(pl.col("State") == "Alaska")
        assert frame.schema["Cases"].is_numeric()
        assert alaska.get_column("Cases").null_count() == alaska.height
        assert alaska.get_column("Cases suppressed").to_list() == [True] * alaska.height

    @pytest.mark.parametrize("marker", ["<5", "(D)", "*"])
    def test_stageDeclared_undeclaredPublicTable_markersStayText(self, marker):
        # Act
        frame = _stage(_public_table(marker), declared=False)

        # Assert
        assert frame.schema["Cases"] == pl.String
        assert "Cases suppressed" not in frame.columns

    @pytest.mark.parametrize("marker", ["^", "NR", "<0.1"])
    def test_stageDeclared_unrecognisedMarker_staysTextFailClosed(self, marker):
        # Act
        frame = _stage(_public_table(marker), declared=True)

        # Assert
        assert frame.schema["Cases"] == pl.String

    def test_extractZipTables_declaredPatientCsv_sameAsUndeclared(self):
        # Arrange
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("labs.csv", _labs())
        content = buffer.getvalue()

        # Act
        declared = extract_zip_tables(content, declared=True)
        undeclared = extract_zip_tables(content, declared=False)

        # Assert
        assert_frame_equal(declared[0]["data"], undeclared[0]["data"])
        assert declared[0]["data"].schema["crp_mg_l"] == pl.String

    def test_extractZipTables_declaredPublicCsv_flaggedSuppressed(self):
        # Arrange
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("cases.csv", _public_table("<10"))

        # Act
        [table] = extract_zip_tables(buffer.getvalue(), declared=True)

        # Assert
        assert "Cases suppressed" in table["data"].columns
