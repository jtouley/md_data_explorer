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
    to_bitmap,
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

    def test_admit_otherRowsOrSameRowsUnderAnotherName_logOfTheRows(self) -> None:
        # Arrange: the log belongs to the rows: another dataset id or version on the same rows
        # (the same file uploaded twice) shares it; other rows have their own
        log = MemoryShownCountLog()
        log.admit("d", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        same_rows = log.admit("e", "v2", UNIVERSE, [_rows(0, 55)])
        other_rows = log.admit("d", "v1", Universe(rows=ROWS, fingerprint="200:2"), [_rows(0, 55)])

        # Assert
        assert same_rows.shown is False
        assert other_rows.shown is True

    @pytest.mark.parametrize(("men_young", "shown"), [(5, False), (10, False), (11, True), (0, True)])
    def test_admit_unionOfAtomsDerivable_withheld(self, men_young: int, shown: bool) -> None:
        # Arrange: the review's case. Total, "over 65", "women", then "women over 65" give the men
        # of 65 or under (total - over 65 - women + women over 65). "Diabetes" splits that group
        # into atoms of 3 and 2 first, so no single atom or pair region is small.
        women_old, women_young, men_old = np.arange(0, 60), np.arange(60, 120), np.arange(120, 120 + 75)
        men_young_rows = np.arange(195, 195 + men_young)
        rows = 195 + men_young
        universe = Universe(rows=rows, fingerprint=f"{rows}:1")
        diabetes = np.concatenate([women_old[:25], women_young[:30], men_old[:40], men_young_rows[:3]])
        log = MemoryShownCountLog()
        for positions in (diabetes, np.union1d(women_old, men_old), np.union1d(women_old, women_young)):
            log.admit("d", "v1", universe, [ShownSet(unit=ROWS_UNIT, positions=positions)])

        # Act
        verdict = log.admit("d", "v1", universe, [ShownSet(unit=ROWS_UNIT, positions=women_old)])

        # Assert
        assert verdict.shown is shown

    def test_admit_twoRegionsAndAnAgeBand_withheld(self) -> None:
        # Arrange: the review's second form. North 53 and South 53 of 106; "aged 65+" 100; the
        # region sets and the age band leave 6 patients under 65 across both regions
        universe = Universe(rows=106, fingerprint="106:1")
        north, south = np.arange(0, 53), np.arange(53, 106)
        young = np.concatenate([np.arange(0, 3), np.arange(53, 56)])
        old = np.setdiff1d(np.arange(106), young)
        log = MemoryShownCountLog()
        log.admit(
            "d", "v1", universe, [ShownSet(unit=ROWS_UNIT, positions=north), ShownSet(unit=ROWS_UNIT, positions=south)]
        )

        # Act
        verdict = log.admit("d", "v1", universe, [ShownSet(unit=ROWS_UNIT, positions=old)])

        # Assert
        assert verdict.shown is False


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
        patients = ShownSet(unit=str(universe.patients_unit), positions=np.arange(100))
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
        verdict = log.admit("d", "v1", universe, [ShownSet(unit=str(universe.patients_unit), positions=np.arange(100))])

        # Assert
        assert visits.shown is True
        assert verdict.shown is False

    def test_admit_visitRowsWithoutRepeats_shown(self) -> None:
        # Arrange: rows of 60 patients with one visit each: no extra visits
        universe = self._universe()
        log = MemoryShownCountLog()
        log.admit("d", "v1", universe, [ShownSet(unit=str(universe.patients_unit), positions=np.arange(20, 80))])

        # Act
        verdict = log.admit("d", "v1", universe, [_rows(20, 80)])

        # Assert
        assert verdict.shown is True


def _log_file(data_dir: Path) -> Path:
    """The one log file under ``data_dir`` (its name is a keyed digest of the rows)."""
    (path,) = (data_dir / shown_counts.SHOWN_COUNTS_DIRNAME).glob("*.json")
    return path


class TestFileLog:
    def test_admit_afterRestart_checksAgainstWhatWasShown(self, tmp_path: Path) -> None:
        # Arrange: one process shows A; a new log object (an app restart) reads the file
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        verdict = FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 55)])

        # Assert
        assert verdict.shown is False

    def test_admit_sameRowsUploadedTwice_shareOneFile(self, tmp_path: Path) -> None:
        # Arrange: two uploads (ids, versions) of the same rows
        FileShownCountLog(tmp_path).admit("upload-a", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        verdict = FileShownCountLog(tmp_path).admit("upload-b", "v9", UNIVERSE, [_rows(0, 55)])

        # Assert
        assert verdict.shown is False
        assert len(list((tmp_path / shown_counts.SHOWN_COUNTS_DIRNAME).glob("*.json"))) == 1

    def test_admit_newRows_haveTheirOwnEmptyLog(self, tmp_path: Path) -> None:
        # Arrange: a corrected re-upload is other rows
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        verdict = FileShownCountLog(tmp_path).admit(
            "d1", "v2", Universe(rows=ROWS, fingerprint="200:2"), [_rows(0, 55)]
        )

        # Assert
        assert verdict.shown is True

    def test_file_holdsNoIdsOrValues_onlyBitmaps(self, tmp_path: Path) -> None:
        # Arrange / Act
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Assert: the rows' fingerprint appears only as a keyed digest (name and file)
        path = _log_file(tmp_path)
        stored = json.loads(path.read_text())
        assert set(stored) == {"format", "rows", "fingerprint", "key_id", "sets"}
        assert set(stored["sets"][0]) == {"unit", "size", "bits"}
        assert UNIVERSE.fingerprint not in path.read_text() and UNIVERSE.fingerprint not in path.name

    @pytest.mark.parametrize("damage", ["not json", "truncated", "wrong_size", "other_rows", "unknown_key"])
    def test_admit_unreadableLog_withholdsEveryAnswer(self, tmp_path: Path, damage: str) -> None:
        # Arrange
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])
        path = _log_file(tmp_path)
        stored = json.loads(path.read_text())
        if damage == "not json":
            path.write_text("{")
        elif damage == "truncated":
            path.write_text(path.read_text()[:40])
        elif damage == "wrong_size":
            stored["sets"][0]["size"] = 49
            path.write_text(json.dumps(stored))
        elif damage == "other_rows":
            stored["fingerprint"] = "200:9"
            path.write_text(json.dumps(stored))
        else:
            stored["extra"] = 1
            path.write_text(json.dumps(stored))

        # Act: a set that conflicts with nothing
        verdict = FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(100, 150)])

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
    def test_purgeDatasetFiles_keepsTheLogOfTheRows(self, tmp_path: Path) -> None:
        # Arrange: deleting and re-uploading the same rows must not reset what was shown on them
        from clinical_analytics.storage.dataset_deletion import purge_dataset_files

        uploads = tmp_path / "uploads"
        uploads.mkdir()
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Act
        purge_dataset_files("d1", uploads)
        verdict = FileShownCountLog(tmp_path).admit("d2", "v1", UNIVERSE, [_rows(0, 55)])

        # Assert
        assert verdict.shown is False


class TestExactCheckBounds:
    @pytest.mark.parametrize("bound", ["MAX_SPAN_WORK", "MAX_MATRIX_FLOATS"])
    def test_admit_aboveABound_withheldNeverShownUnchecked(self, monkeypatch: pytest.MonkeyPatch, bound: str) -> None:
        # Arrange: the review's fallback leak (total, over 65, women, women over 65) with each
        # bound at 0: no stricter-but-leaky rule runs instead, the answer is withheld
        log = MemoryShownCountLog()
        log.admit("d", "v1", UNIVERSE, [_rows(0, 60), _rows(55, 100)])
        monkeypatch.setattr(shown_counts, bound, 0)

        # Act: their union makes the overlap of 5 derivable
        verdict = log.admit("d", "v1", UNIVERSE, [_rows(0, 100)])

        # Assert
        assert (verdict.shown, verdict.reason) == (False, shown_counts.NOT_CHECKED_REASON)

    def test_admit_integerProgramUnfinished_withheldNeverShownUnchecked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: the union case, which only the integer program decides; it does not finish
        from scipy import optimize

        women_old, women_young, men_old, men_young = (
            np.arange(0, 60),
            np.arange(60, 120),
            np.arange(120, 195),
            np.arange(195, 200),
        )
        diabetes = np.concatenate([women_old[:25], women_young[:30], men_old[:40], men_young[:3]])
        log = MemoryShownCountLog()
        for positions in (diabetes, np.union1d(women_old, men_old), np.union1d(women_old, women_young)):
            log.admit("d", "v1", UNIVERSE, [ShownSet(unit=ROWS_UNIT, positions=positions)])
        unfinished = optimize.OptimizeResult(status=1, x=None)
        monkeypatch.setattr(optimize, "milp", lambda *args, **kwargs: unfinished)

        # Act
        verdict = log.admit("d", "v1", UNIVERSE, [ShownSet(unit=ROWS_UNIT, positions=women_old)])

        # Assert
        assert (verdict.shown, verdict.reason) == (False, shown_counts.NOT_CHECKED_REASON)

    def test_admit_checkFails_withheldAsNotChecked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        def fail(*_args: object) -> None:
            raise np.linalg.LinAlgError("SVD did not converge")

        monkeypatch.setattr(shown_counts, "find_conflict", fail)

        # Act
        verdict = MemoryShownCountLog().admit("d", "v1", UNIVERSE, [_rows(0, 50)])

        # Assert
        assert (verdict.shown, verdict.reason) == (False, shown_counts.NOT_CHECKED_REASON)

    def test_admit_patientSetsWithoutPatientMap_withheldAsUnavailable(self) -> None:
        # Arrange: patient sets logged while rows were declared visits; the map is gone now
        patient_of_row = np.concatenate([np.arange(150), np.full(50, -1)])
        visits = Universe(rows=ROWS, fingerprint="200:1", patient_key="pid", patient_of_row=patient_of_row)
        log = MemoryShownCountLog()
        log.admit("d", "v1", visits, [ShownSet(unit=str(visits.patients_unit), positions=np.arange(60))])

        # Act
        verdict = log.admit("d", "v1", UNIVERSE, [ShownSet(unit=str(visits.patients_unit), positions=np.arange(70))])

        # Assert
        assert (verdict.shown, verdict.reason) == (False, UNAVAILABLE_REASON)

    def test_admit_tooManyCountsInOneAnswer_ownReason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setattr(shown_counts, "MAX_NEW_SETS", 1)

        # Act
        verdict = MemoryShownCountLog().admit("d", "v1", UNIVERSE, [_rows(0, 50), _rows(100, 150)])

        # Assert
        assert (verdict.shown, verdict.reason) == (False, shown_counts.TOO_MANY_COUNTS_REASON)


class TestPresenterCounts:
    @pytest.mark.parametrize(("event_text", "listed"), [(None, False), ("1", True)])
    def test_presentWithCounts_survivalEventsWithoutEventText_notListed(
        self, event_text: str | None, listed: bool
    ) -> None:
        # Arrange: an ungrouped survival result that shows its events (40 of 200)
        from clinical_analytics.core.result_presentation import present_with_counts

        analysis = {
            "type": "survival",
            "time_col": "months",
            "event_col": "died",
            "n_used": 200,
            "n_excluded": 0,
            "n_events": 40,
            "median_survival": 12.0,
            "event_level": "1",
            **({"event_text": event_text} if event_text is not None else {}),
        }

        # Act
        presentation, counts = present_with_counts(analysis)

        # Assert: without the event's text the events can't be rebuilt, so nothing is listed
        # and the log withholds the answer
        assert presentation["table"] is not None
        assert (counts is not None) is listed
        if counts is not None:
            assert 40 in [c.value for c in counts]


class TestInstallKey:
    def test_admit_keyLostWhileLogsExist_withheld(self, tmp_path: Path) -> None:
        # Arrange: losing the key would make every log unfindable (a reset): fail closed
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])
        (tmp_path / shown_counts.SHOWN_COUNTS_DIRNAME / shown_counts.INSTALL_KEY_NAME).unlink()

        # Act
        verdict = FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(100, 150)])

        # Assert
        assert (verdict.shown, verdict.reason) == (False, UNAVAILABLE_REASON)

    def test_admit_otherInstall_otherFileNames(self, tmp_path: Path) -> None:
        # Arrange / Act: two data directories (installs) log the same rows
        FileShownCountLog(tmp_path / "a").admit("d1", "v1", UNIVERSE, [_rows(0, 50)])
        FileShownCountLog(tmp_path / "b").admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Assert: the names are keyed per install, so they differ
        assert _log_file(tmp_path / "a").name != _log_file(tmp_path / "b").name


class TestReviewTwo:
    def test_admit_rowsLoggedBeforeVisitsDeclared_extraVisitsCheckedAfter(self) -> None:
        # Arrange: 105 rows of 100 patients logged while rows were not yet declared visits (no
        # patient map); once declared, a patient count of those 100 gives the 5 extra visits
        patient_of_row = np.concatenate([np.arange(150), np.arange(5), np.full(45, -1)])
        visits = Universe(rows=ROWS, fingerprint="200:1", patient_key="pid", patient_of_row=patient_of_row)
        log = MemoryShownCountLog()
        rows = ShownSet(unit=ROWS_UNIT, positions=np.concatenate([np.arange(100), np.arange(150, 155)]))
        before = log.admit("d", "v1", UNIVERSE, [rows])

        # Act
        after = log.admit("d", "v1", visits, [ShownSet(unit=str(visits.patients_unit), positions=np.arange(100))])

        # Assert
        assert before.shown is True
        assert after.shown is False

    def test_patientsUnit_namedByWhichRowsArePatients_notTheColumn(self) -> None:
        # Arrange / Act
        mapping = np.concatenate([np.arange(150), np.arange(5), np.full(45, -1)])
        a = Universe(rows=ROWS, fingerprint="f", patient_key="patient_id", patient_of_row=mapping)
        b = Universe(rows=ROWS, fingerprint="f", patient_key="subject_id", patient_of_row=mapping.copy())

        # Assert
        assert a.patients_unit == b.patients_unit

    def test_smallGroupDerivable_secondSvdAboveBound_notChecked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: atoms built within the bounds, then the bound lowered for the second SVD
        sets = [
            to_bitmap(np.arange(0, 60), ROWS),
            to_bitmap(np.arange(55, 100), ROWS),
            to_bitmap(np.arange(ROWS), ROWS),
        ]
        atoms = shown_counts._Atoms.of(sets, np.arange(ROWS), ROWS)
        monkeypatch.setattr(shown_counts, "MAX_SPAN_WORK", 1)

        # Act / Assert
        with pytest.raises(shown_counts._TooLargeError):
            atoms.small_group_derivable()

    @pytest.mark.slow
    @pytest.mark.integration
    def test_admit_manyRandomHalves_shownOrNotCheckedNeverUnchecked(self) -> None:
        # Arrange: the review's b4. Random halves of 3,000 rows (tiny atoms); each answer is
        # either shown or withheld, within seconds, never shown on a check that did not finish
        rng = np.random.default_rng(3)
        universe = Universe(rows=3000, fingerprint="3000:1")
        log = MemoryShownCountLog()

        # Act
        verdicts = [
            log.admit("d", "v1", universe, [ShownSet(unit=ROWS_UNIT, positions=np.flatnonzero(rng.random(3000) < 0.5))])
            for _ in range(15)
        ]

        # Assert
        assert all(v.shown or v.reason in (WITHHELD_REASON, shown_counts.NOT_CHECKED_REASON) for v in verdicts)


class TestInstallKeyHandling:
    def test_key_permissions_ownerOnly(self, tmp_path: Path) -> None:
        # Arrange / Act
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])

        # Assert
        root = tmp_path / shown_counts.SHOWN_COUNTS_DIRNAME
        assert root.stat().st_mode & 0o777 == 0o700
        assert (root / shown_counts.INSTALL_KEY_NAME).stat().st_mode & 0o777 == 0o600

    def test_admit_keyReplacedWhileLogsExist_withholdsEverything(self, tmp_path: Path) -> None:
        # Arrange: a log written under one key; the key file then replaced by another
        FileShownCountLog(tmp_path).admit("d1", "v1", UNIVERSE, [_rows(0, 50)])
        key = tmp_path / shown_counts.SHOWN_COUNTS_DIRNAME / shown_counts.INSTALL_KEY_NAME
        key.write_bytes(bytes(32))

        # Act: other rows, whose log would be new under the new key
        verdict = FileShownCountLog(tmp_path).admit(
            "d2", "v1", Universe(rows=ROWS, fingerprint="other"), [_rows(0, 50)]
        )

        # Assert
        assert (verdict.shown, verdict.reason) == (False, UNAVAILABLE_REASON)

    def test_key_madeByManyAtOnce_oneKey(self, tmp_path: Path) -> None:
        # Arrange: eight logs (threads) make the key at the same time
        from concurrent.futures import ThreadPoolExecutor

        def make(_: int) -> bytes:
            return FileShownCountLog(tmp_path)._key()

        # Act
        with ThreadPoolExecutor(max_workers=8) as pool:
            keys = set(pool.map(make, range(8)))

        # Assert
        assert len(keys) == 1
