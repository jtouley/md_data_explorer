"""The contract as the model's context: the typed read view of an upload version's contract."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from clinical_analytics.contracts.context import ContractColumn, context_from_contract, load_contract_context
from clinical_analytics.contracts.dictionary import ColumnDoc
from clinical_analytics.contracts.errors import ContractDamagedError
from clinical_analytics.contracts.store import contract_path, write_contract
from clinical_analytics.core.metadata_patch import ResolvedColumnMetadata, ResolvedDatasetMetadata


@pytest.fixture
def documented_frame() -> pl.DataFrame:
    """Synthetic: 20 patients (above the tiny-table cut-off), a coded age band, a weight and a free-text note."""
    return pl.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(20)],
            "age_band": [0, 1] * 10,
            "weight": [60.0 + i for i in range(20)],
            "note": ["x"] * 20,
        }
    )


DOCS = {
    "age_band": ColumnDoc(description="Age in categories.", codebook={"0": "under 40", "1": "40 or more"}),
    "weight": ColumnDoc(description="Weight at baseline", unit="kg"),
}


def test_contextFromContract_documentedUpload_columnsKeyAndGrain(make_contract, documented_frame):
    # Arrange
    contract = make_contract(documented_frame, dictionary=DOCS)

    # Act
    context = context_from_contract(contract)

    # Assert
    by_name = {c.name: c for c in context.columns}
    assert context.upload_id == "u1"
    assert context.dataset_version == "v1"
    assert context.key == "patient_id"
    assert context.grain == "One row per patient_id"
    assert by_name["age_band"] == ContractColumn(
        name="age_band",
        logical_type="integer",
        label="age_band",
        description="Age in categories.",
        codebook={"0": "under 40", "1": "40 or more"},
    )
    assert by_name["weight"].unit == "kg"
    assert by_name["patient_id"].is_key and by_name["patient_id"].is_phi
    assert by_name["note"].description is None


def test_contextFromContract_acceptedMetadata_synonymsAndSemanticType(make_contract, documented_frame):
    # Arrange
    from clinical_analytics.core.metadata_patch import SemanticType

    resolved = ResolvedDatasetMetadata(
        columns={
            "weight": ResolvedColumnMetadata(
                name="weight", aliases=["body weight"], semantic_type=SemanticType.MEASUREMENT
            )
        }
    )
    contract = make_contract(documented_frame, resolved=resolved)

    # Act
    weight = next(c for c in context_from_contract(contract).columns if c.name == "weight")

    # Assert
    assert weight.synonyms == ("body weight",)
    assert weight.semantic_type == SemanticType.MEASUREMENT.value


def test_contextFromContract_noUnifiedObject_raisesDamaged(make_contract, documented_frame):
    # Arrange
    contract = make_contract(documented_frame)
    contract.schema_[0].name = "other"

    # Act / Assert
    with pytest.raises(ContractDamagedError):
        context_from_contract(contract)


def test_loadContractContext_noContractForVersion_returnsNone(tmp_path: Path):
    # Arrange
    path = contract_path(tmp_path, "u1", "v1")

    # Act
    context = load_contract_context(path)

    # Assert
    assert context is None


def test_loadContractContext_savedContract_readsIt(tmp_path: Path, make_contract, documented_frame):
    # Arrange
    write_contract(tmp_path, "u1", "v1", make_contract(documented_frame, dictionary=DOCS))

    # Act
    context = load_contract_context(contract_path(tmp_path, "u1", "v1"))

    # Assert
    assert context is not None
    assert [c.name for c in context.columns] == documented_frame.columns


@pytest.mark.parametrize("text", ["", "not: [a contract", "apiVersion: v3.2.0\nkind: DataContract\n"])
def test_loadContractContext_damagedFile_raisesDamaged(tmp_path: Path, text: str):
    # Arrange
    path = contract_path(tmp_path, "u1", "v1")
    path.parent.mkdir(parents=True)
    path.write_text(text)

    # Act / Assert
    with pytest.raises(ContractDamagedError):
        load_contract_context(path)
