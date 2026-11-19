# SPDX-License-Identifier: MIT
"""The log of shown counts (core.shown_counts): the pairwise rule, persistence, versions and
failing closed. Sets are built directly as row positions (synthetic, no data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clinical_analytics.core import shown_counts
from clinical_analytics.core.shown_counts import (
    FULL_REASON,
    ROWS_UNIT,
    UNAVAILABLE_REASON,
    WITHHELD_REASON,
    FileShownCountLog,
    MemoryShownCountLog,
    ShownSet,
    Universe,
)

ROWS = 200
UNIVERSE = Universe(rows=ROWS, fingerprint="200:1")


def _rows(start: int, stop: int) -> ShownSet:
    return ShownSet(unit=ROWS_UNIT, positions=np.arange(start, stop))


class TestDerivableRule:
    @pytest.mark.parametrize(
        ("overlap", "shown"),
        [(0, True), (1, False), (10, False), (11, True)],
        ids=["disjoint", "one", "ten", "eleven"],
    )
    def test_admit_unionOfTwoShownSets_withheldWhenOverlap1To10(self, overlap: int, shown: bool) -> None:
        # Arrange: v1c's attack. "A 2019 row" and "a 2020 row" (50 each) are shown; "a row before
        # 2021" is their union, and |A| + |B| - |A or B| is the overlap
        log = MemoryShownCountLog()
        log.admit("d", "v1", UNIVERSE, [_rows(0, 50)])
        log.admit("d", "v1", UNIVERSE, [_rows(50 - overlap, 100 - overlap)])

        # Act
        verdict = log.admit("d", "v1", UNIVERSE, [_rows(0, 100 - overlap)])

        # Assert
        assert verdict.shown is shown
        assert verdict.reason == (None if shown else WITHHELD_REASON)

    def test_admit_overlapNotDerivable_shown(self) -> None:
        # Arrange: two sets overlapping in 5, nothing else shown about them: |A & B| can't be
        # worked out from |A|, |B| and the total (over-blocking matters as much as leaks)
        log = MemoryShownCountLog()

        # Act
        verdict = log.admit("d", "v1", UNIVERSE, [_rows(0, 60), _rows(55, 100)])

        # Assert
        assert verdict.shown is True

    def test_admit_threeWayRegionDerivable_withheld(self) -> None:
        # Arrange: A, B, C and A or B, B or C, A or C shown; with A & B & C small, inclusion-exclusion
        # gives it, though no pair of shown sets isolates it
        log = MemoryShownCountLog()
        a, b = np.arange(0, 100), np.arange(50, 150)
        c = np.concatenate([np.arange(30, 45), np.arange(50, 55), np.arange(100, 120), np.arange(150, 180)])
        sets = [a, b, c, np.union1d(a, b), np.union1d(b, c), np.union1d(a, c)]
        earlier = [log.admit("d", "v1", UNIVERSE, [ShownSet(unit=ROWS_UNIT, positions=p)]) for p in sets]

        # Act: the union of all three makes A & B & C a combination of shown counts
        verdict = log.admit("d", "v1", UNIVERSE, [ShownSet(unit=ROWS_UNIT, positions=np.union1d(np.union1d(a, b), c))])

        # Assert: A & B & C holds rows 50-54 (5); every pair overlaps in 20 or more
        assert np.intersect1d(np.intersect1d(a, b), c).size == 5
        assert [e.shown for e in earlier] == [True] * len(sets)
        assert verdict.shown is False

    @pytest.mark.parametrize(("dropped", "shown"), [(1, False), (10, False), (11, True)], ids=["one", "ten", "eleven"])
    def test_admit_nestedSetsDifferBy_withheldOnlyFrom1To10(self, dropped: int, shown: bool) -> None:
        # Arrange: "age over 65" then "age 65 and over": the second holds `dropped` more
        log = MemoryShownCountLog()
        log.admit("d", "v1", UNIVERSE, [_rows(0, 60)])

        # Act
        verdict = log.admit("d", "v1", UNIVERSE, [_rows(0, 60 + dropped)])

        # Assert
        assert verdict.shown is shown

    @pytest.mark.parametrize(("size", "shown"), [(ROWS - 10, False), (ROWS - 11, True), (ROWS, True)])
    def test_admit_setNearTheDatasetTotal_comparedWithTheTotal(self, size: int, shown: bool) -> None:
        # Arrange: the app shows the row count, so it is always a logged set
        log = MemoryShownCountLog()

        # Act
        verdict = log.admit("d", "v1", UNIVERSE, [_rows(0, size)])

        # Assert
        assert verdict.shown is shown

    def test_admit_emptyAndRepeatedSets_shownAndNotAddedTwice(self) -> None:
        # Arrange
        log = MemoryShownCountLog()
        log.admit("d", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        repeated = log.admit("d", "v1", UNIVERSE, [_rows(0, 50)])
        empty = log.admit("d", "v1", UNIVERSE, [])

        # Assert
        assert (repeated.shown, repeated.added) == (True, 0)
        assert (empty.shown, empty.added) == (True, 0)

    def test_admit_withheldAnswer_logsNothing(self) -> None:
        # Arrange: B is withheld beside A; C conflicts with B only
        log = MemoryShownCountLog()
        log.admit("d", "v1", UNIVERSE, [_rows(0, 50)])
        withheld = log.admit("d", "v1", UNIVERSE, [_rows(0, 55), _rows(100, 150)])

        # Act: C differs from B's second set by 5, from A by 50 or more
        verdict = log.admit("d", "v1", UNIVERSE, [_rows(100, 145)])

        # Assert
        assert withheld.shown is False
        assert verdict.shown is True

    def test_admit_otherVersionOrDataset_startsEmpty(self) -> None:
        # Arrange
        log = MemoryShownCountLog()
        log.admit("d", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        other_version = log.admit("d", "v2", UNIVERSE, [_rows(0, 55)])
        other_dataset = log.admit("e", "v1", UNIVERSE, [_rows(0, 55)])

        # Assert
        assert other_version.shown is True
        assert other_dataset.shown is True


class TestVisitRows:
    """Patients as their first row: rows 0-149 are patients P0-P149, rows 150-154 are second
    visits of P0-P4, rows 155-199 have no ID."""

    @staticmethod
    def _universe() -> Universe:
        patient_of_row = np.concatenate([np.arange(150), np.arange(5), np.full(45, -1)])
        return Universe(rows=ROWS, fingerprint="200:1", patient_key="pid", patient_of_row=patient_of_row)

    @staticmethod
    def _visits() -> ShownSet:
        """Visit rows of P0-P99: 105 rows, 5 readmitted."""
        return ShownSet(unit=ROWS_UNIT, positions=np.concatenate([np.arange(100), np.arange(150, 155)]))

    def test_admit_visitRowsBesideTheirPatients_withheld(self) -> None:
        # Arrange: 100 patients shown, then 105 visit rows with a value: 5 readmitted
        universe = self._universe()
        log = MemoryShownCountLog()
        patients = ShownSet(unit="patients:pid", positions=np.arange(100))
        log.admit("d", "v1", universe, [patients])

        # Act
        verdict = log.admit("d", "v1", universe, [self._visits()])

        # Assert
        assert verdict.shown is False

    def test_admit_patientsAfterTheirVisitRows_withheld(self) -> None:
        # Arrange: the same two answers in the other order
        universe = self._universe()
        log = MemoryShownCountLog()
        visits = log.admit("d", "v1", universe, [self._visits()])

        # Act
        verdict = log.admit("d", "v1", universe, [ShownSet(unit="patients:pid", positions=np.arange(100))])

        # Assert
        assert visits.shown is True
        assert verdict.shown is False

    def test_admit_visitRowsWithoutRepeats_shown(self) -> None:
        # Arrange: rows of 60 patients with one visit each: no extra visits
        universe = self._universe()
        log = MemoryShownCountLog()
        log.admit("d", "v1", universe, [ShownSet(unit="patients:pid", positions=np.arange(20, 80))])

        # Act
        verdict = log.admit("d", "v1", universe, [_rows(20, 80)])

        # Assert
        assert verdict.shown is True


class TestFileLog:
    def test_admit_afterRestart_checksAgainstWhatWasShown(self, tmp_path: Path) -> None:
        # Arrange: one process shows A; a new log object (an app restart) reads the file
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        verdict = FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 55)])

        # Assert
        assert verdict.shown is False

    def test_admit_newVersion_hasItsOwnEmptyLog(self, tmp_path: Path) -> None:
        # Arrange
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        verdict = FileShownCountLog(tmp_path).admit("d1", "v2", UNIVERSE, [_rows(0, 55)])

        # Assert
        assert verdict.shown is True
        assert len(list((tmp_path / shown_counts.SHOWN_COUNTS_DIRNAME / "d1").glob("*.json"))) == 2

    def test_file_holdsNoIdsOrValues_onlyBitmaps(self, tmp_path: Path) -> None:
        # Arrange / Act
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Assert
        (path,) = (tmp_path / shown_counts.SHOWN_COUNTS_DIRNAME / "d1").glob("*.json")
        stored = json.loads(path.read_text())
        assert set(stored) == {"format", "dataset", "version", "rows", "fingerprint", "sets"}
        assert set(stored["sets"][0]) == {"unit", "size", "excess", "bits"}

    @pytest.mark.parametrize(
        "damage",
        ["not json", "truncated", "wrong_size", "other_version", "unknown_key"],
    )
    def test_admit_unreadableLog_withholdsEveryAnswer(self, tmp_path: Path, damage: str) -> None:
        # Arrange
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])
        (path,) = (tmp_path / shown_counts.SHOWN_COUNTS_DIRNAME / "d1").glob("*.json")
        stored = json.loads(path.read_text())
        if damage == "not json":
            path.write_text("{")
        elif damage == "truncated":
            path.write_text(path.read_text()[:40])
        elif damage == "wrong_size":
            stored["sets"][0]["size"] = 49
            path.write_text(json.dumps(stored))
        elif damage == "other_version":
            stored["version"] = "v0"
            path.write_text(json.dumps(stored))
        else:
            stored["extra"] = 1
            path.write_text(json.dumps(stored))

        # Act: a set that conflicts with nothing
        verdict = FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(100, 150)])

        # Assert
        assert (verdict.shown, verdict.reason) == (False, UNAVAILABLE_REASON)

    def test_admit_logOfOtherRows_withheld(self, tmp_path: Path) -> None:
        # Arrange: the same version's rows hashed differently (reordered or changed)
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        verdict = FileShownCountLog(tmp_path).admit(
            "d1", "v1", Universe(rows=ROWS, fingerprint="200:2"), [_rows(100, 150)]
        )

        # Assert
        assert (verdict.shown, verdict.reason) == (False, UNAVAILABLE_REASON)

    def test_admit_logCannotBeSaved_withheldAndNothingShown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        def fail(*_args: object, **_kwargs: object) -> bool:
            raise OSError("disk full")

        monkeypatch.setattr(shown_counts, "atomic_write_bytes", fail)

        # Act
        verdict = FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Assert
        assert (verdict.shown, verdict.reason) == (False, UNAVAILABLE_REASON)

    def test_admit_invalidDatasetId_withheld(self, tmp_path: Path) -> None:
        # Arrange / Act
        verdict = FileShownCountLog(tmp_path).admit("../x", "v1", UNIVERSE, [_rows(0, 50)])

        # Assert
        assert verdict.shown is False


class TestBounds:
    def test_admit_logFull_withholdsNewSetsButNotRepeats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: room for two sets
        monkeypatch.setattr(shown_counts, "MAX_LOGGED_SETS", 2)
        log = MemoryShownCountLog()
        log.admit("d", "v1", UNIVERSE, [_rows(0, 50), _rows(100, 150)])

        # Act
        new = log.admit("d", "v1", UNIVERSE, [_rows(50, 100)])
        repeat = log.admit("d", "v1", UNIVERSE, [_rows(0, 50)])

        # Assert
        assert (new.shown, new.reason) == (False, FULL_REASON)
        assert repeat.shown is True

    def test_admit_positionOutsideTheRows_withheld(self) -> None:
        # Arrange / Act
        verdict = MemoryShownCountLog().admit(
            "d", "v1", UNIVERSE, [ShownSet(unit=ROWS_UNIT, positions=np.array([ROWS]))]
        )

        # Assert
        assert verdict.shown is False


class TestDatasetDeletion:
    def test_purgeDatasetFiles_removesTheDatasetsLog(self, tmp_path: Path) -> None:
        # Arrange
        from clinical_analytics.storage.dataset_deletion import purge_dataset_files

        uploads = tmp_path / "uploads"
        uploads.mkdir()
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])
        FileShownCountLog(tmp_path).admit("d2", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        purge_dataset_files("d1", uploads)

        # Assert
        assert not (tmp_path / shown_counts.SHOWN_COUNTS_DIRNAME / "d1").exists()
        assert (tmp_path / shown_counts.SHOWN_COUNTS_DIRNAME / "d2").is_dir()
