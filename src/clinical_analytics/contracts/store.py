"""Contracts and quality reports in the data home.

Layout under the data directory (``<data home>/data``, the parent of ``uploads/``)::

    contracts/<upload_id>/<dataset_version>.odcs.yaml     written once per version, never replaced
    dq/<upload_id>/<dataset_version>/<run_ts>.json        one file per quality run

A contract is immutable: it is written to a temporary file and hard-linked into place, which
fails if the target exists, so two concurrent uploads of the same content cannot interleave and
a finished contract is never rewritten. File systems without hard links (exFAT, many SMB shares)
fall back to an exclusive create (``open(path, "x")``) plus fsync: still never overwritten.
Reports are written to a temporary file and renamed.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pydantic
import structlog
import yaml
from open_data_contract_standard.model import OpenDataContractStandard

from clinical_analytics.contracts import config
from clinical_analytics.contracts.errors import ContractStoreError, ContractValidationError
from clinical_analytics.contracts.quality import QualityReport
from clinical_analytics.storage.dataset_deletion import validate_dataset_id

logger = structlog.get_logger()

_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_RUN_FILE = re.compile(r"^\d{8}T\d{12}Z\.json$")


def _version(dataset_version: str) -> str:
    if not _VERSION_PATTERN.fullmatch(dataset_version):
        raise ContractStoreError(f"Invalid dataset version: {dataset_version!r}")
    return dataset_version


def contracts_root(data_dir: Path) -> Path:
    return data_dir / config.CONTRACTS_DIRNAME


def dq_root(data_dir: Path) -> Path:
    return data_dir / config.DQ_DIRNAME


def contract_path(data_dir: Path, upload_id: str, dataset_version: str) -> Path:
    validate_dataset_id(upload_id)
    return contracts_root(data_dir) / upload_id / f"{_version(dataset_version)}{config.CONTRACT_SUFFIX}"


def dq_dir(data_dir: Path, upload_id: str, dataset_version: str) -> Path:
    validate_dataset_id(upload_id)
    return dq_root(data_dir) / upload_id / _version(dataset_version)


def _write_temp(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise
    return Path(tmp)


def _create_exclusive(path: Path, text: str) -> None:
    """Create ``path`` with ``text``; FileExistsError if it exists. A partial file is removed."""
    with open(path, "x", encoding="utf-8") as f:
        try:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        except OSError:
            f.close()
            path.unlink(missing_ok=True)
            raise


def write_contract(data_dir: Path, upload_id: str, dataset_version: str, contract: OpenDataContractStandard) -> bool:
    """Write the contract for a version unless one exists. Returns True when this call wrote it.

    Raises:
        ContractStoreError: the file could not be written.
    """
    path = contract_path(data_dir, upload_id, dataset_version)
    log = logger.bind(upload_id=upload_id, dataset_version=dataset_version)
    if path.exists():
        log.info("contract_exists_kept")
        return False
    text = contract.to_yaml()
    try:
        tmp = _write_temp(path.parent, text)
        try:
            os.link(tmp, path)
        except FileExistsError:
            log.info("contract_exists_kept")
            return False
        except OSError as e:
            log.info("contract_link_unsupported", errno=e.errno)
            try:
                _create_exclusive(path, text)
            except FileExistsError:
                log.info("contract_exists_kept")
                return False
        finally:
            tmp.unlink(missing_ok=True)
    except OSError as e:
        raise ContractStoreError(f"Could not write contract {path.name}: {e.strerror or type(e).__name__}") from e
    log.info("contract_written")
    return True


def read_contract(path: Path) -> OpenDataContractStandard:
    """Load and validate a stored contract.

    Raises:
        ContractStoreError: missing or unreadable file.
        ContractValidationError: not valid YAML or not ODCS.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ContractStoreError(f"Could not read contract {path.name}: {e.strerror or type(e).__name__}") from e
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ContractValidationError(f"Contract {path.name} is not a mapping")
        return OpenDataContractStandard.model_validate(data)
    except yaml.YAMLError as e:
        raise ContractValidationError(f"Contract {path.name} is not valid YAML") from e
    except pydantic.ValidationError as e:
        raise ContractValidationError(f"Contract {path.name} is not valid ODCS ({e.error_count()} errors)") from e


def run_filename(run_ts: datetime) -> str:
    """Sortable, filesystem-safe name: 20261103T192600123456Z.json."""
    return run_ts.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"


def write_report(data_dir: Path, report: QualityReport) -> Path:
    """Persist a quality report; returns its path.

    Raises:
        ContractStoreError: the file could not be written.
    """
    directory = dq_dir(data_dir, report.upload_id, report.dataset_version)
    run_ts = datetime.fromisoformat(report.run_ts.replace("Z", "+00:00"))
    path = directory / run_filename(run_ts)
    try:
        tmp = _write_temp(directory, report.model_dump_json(indent=2))
        os.replace(tmp, path)
    except OSError as e:
        raise ContractStoreError(f"Could not write quality report: {e.strerror or type(e).__name__}") from e
    logger.info("quality_report_written", upload_id=report.upload_id, dataset_version=report.dataset_version)
    return path


def latest_report(data_dir: Path, upload_id: str, dataset_version: str) -> QualityReport | None:
    """Most recent report for a version, or None if none was written.

    Raises:
        ContractStoreError: the newest report cannot be read or parsed.
    """
    directory = dq_dir(data_dir, upload_id, dataset_version)
    if not directory.is_dir():
        return None
    runs = sorted(p for p in directory.iterdir() if _RUN_FILE.fullmatch(p.name))
    if not runs:
        return None
    try:
        return QualityReport.model_validate_json(runs[-1].read_text(encoding="utf-8"))
    except OSError as e:
        raise ContractStoreError(f"Could not read quality report: {e.strerror or type(e).__name__}") from e
    except pydantic.ValidationError as e:
        raise ContractStoreError(f"Quality report {runs[-1].name} is invalid ({e.error_count()} errors)") from e
