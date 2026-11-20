# SPDX-License-Identifier: MIT
""" "Is each row one patient?": asked once at upload, answered for that dataset version.

When no column is named as a patient ID, the upload keys rows (``patient_id_is_row_key``) and
every answer counts rows. If the file has a column unique on every row, or nothing that
identifies a row, and no ID-like column repeats (``identifier_rules.rows_may_be_patients``), the
upload asks the user whether each row is one patient.

* "Yes" stores the dataset version's one-row-per-patient grain: the same ``GrainDeclaration``
  (grain ``patient``, no ID column, the version's row count) that governed SQL stores when its
  refusal is confirmed, bound by the same ``confirmation_token``. Questions then count patients
  (``answer_checks.declared_patient_rows``) and governed SQL counts patients.
* "No" keeps rows, and removes an earlier "yes" on this version.
* Never answered: nothing is stored and every answer counts rows, exactly as before.

The token names the version, no ID column and the row count asked about, so an answer sent for
an earlier version (a re-upload in between) or a made-up token is refused and nothing is stored.
A new version asks again: its token differs, and declarations are kept per version.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from typing import Literal

import structlog

from clinical_analytics.contracts.declarations import (
    Declarations,
    DeclarationStore,
    GrainDeclaration,
    confirmation_token,
)
from clinical_analytics.core.identifier_rules import rows_may_be_patients

logger = structlog.get_logger()

RowsCount = Literal["patients", "rows"]

# Said after "Your answer wasn't saved, so answers still count rows." (the app's prefix).
NOT_ASKED = "This cohort is not asked whether each row is one patient."
STALE = "The cohort changed after you were asked. Upload the file again to answer for the new data."


class PatientRowsError(Exception):
    """An answer to "Is each row one patient?" that is not stored. ``str(e)`` is for the user."""


class PatientRowsNotAskedError(PatientRowsError):
    """The dataset's current version is not asked (an ID column repeats, IDs name patients, ...)."""


class PatientRowsStaleError(PatientRowsError):
    """The token is not this version's: asked about other data, or not a token the app gave."""


def _asked_about(metadata: Mapping[str, object]) -> tuple[str, int] | None:
    """(dataset version, row count) the question is asked about, or None when it is not asked."""
    if not rows_may_be_patients(metadata):
        return None
    version, rows = metadata.get("dataset_version"), metadata.get("row_count")
    if not isinstance(version, str) or not version or not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        return None
    return version, rows


def question_token(metadata: Mapping[str, object]) -> str | None:
    """The confirmation token of this version's question, or None when it is not asked."""
    asked = _asked_about(metadata)
    return None if asked is None else confirmation_token(asked[0], None, asked[1])


def confirmed_patient_rows(metadata: Mapping[str, object], declarations: Declarations) -> bool:
    """Whether this version's rows were confirmed one per patient (here or by governed SQL)."""
    asked = _asked_about(metadata)
    if asked is None:
        return False
    version, rows = asked
    grain = declarations.grain_for(version)
    return grain is not None and grain.grain == "patient" and grain.key is None and grain.rows == rows


def pending_token(metadata: Mapping[str, object], declarations: Declarations) -> str | None:
    """The token to show with the question, or None: not asked, or already answered "yes"."""
    if confirmed_patient_rows(metadata, declarations):
        return None
    return question_token(metadata)


def record_answer(
    dataset_id: str,
    metadata: Mapping[str, object],
    store: DeclarationStore,
    *,
    confirmation: str,
    one_row_per_patient: bool,
) -> RowsCount:
    """Store the user's answer for the dataset's current version and say what answers count.

    Safe to run twice: the same answer stores the same declaration.

    Raises:
        PatientRowsNotAskedError: the current version is not asked.
        PatientRowsStaleError: ``confirmation`` is not the current version's token.
        ContractStoreError: the declarations could not be read or saved (nothing changed).
        InvalidDeclarationKeyError: ``dataset_id`` is not a valid key.
    """
    log = logger.bind(dataset_id=dataset_id)
    asked = _asked_about(metadata)
    if asked is None:
        log.info("patient_rows_not_asked")
        raise PatientRowsNotAskedError(NOT_ASKED)
    version, rows = asked
    if not hmac.compare_digest(confirmation.encode(), confirmation_token(version, None, rows).encode()):
        log.info("patient_rows_answer_stale", dataset_version=version)
        raise PatientRowsStaleError(STALE)
    if one_row_per_patient:
        grain = GrainDeclaration(grain="patient", key=None, rows=rows, patients=rows, dataset_version=version)
        store.update(dataset_id, lambda current: current.with_grain(grain))
        log.info("patient_rows_confirmed", dataset_version=version)
        return "patients"
    if confirmed_patient_rows(metadata, store.load(dataset_id)):
        store.update(dataset_id, lambda current: current.without_patient_grain(version))
    log.info("patient_rows_declined", dataset_version=version)
    return "rows"
