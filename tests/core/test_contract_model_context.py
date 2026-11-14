"""Tier 3 context from the data contract: what the model is told about an uploaded dataset."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from clinical_analytics.contracts.context import ContractColumn, ContractContext
from clinical_analytics.contracts.errors import ContractDamagedError
from clinical_analytics.core.autocontext import build_autocontext
from clinical_analytics.core.nl_query_config import SCHEMA_CONTEXT_MAX_CODES
from clinical_analytics.core.nl_query_engine import NLQueryEngine, _rag_context_from_contract

CONTRACT = ContractContext(
    upload_id="u1",
    dataset_version="v1",
    grain="One row per patient_id",
    key="patient_id",
    columns=(
        ContractColumn(
            name="smoker", logical_type="string", description="Ever smoked", codebook={"y": "yes", "n": "no"}
        ),
        ContractColumn(name="weight_kg", logical_type="number", description="Weight at baseline", unit="kg"),
        ContractColumn(name="patient_id", logical_type="string", is_key=True, is_phi=True),
        ContractColumn(name="onset", logical_type="date", synonyms=("first symptoms",)),
    ),
)


def _engine(mock_semantic_layer) -> NLQueryEngine:
    return NLQueryEngine(mock_semantic_layer(columns={c.name: c.name for c in CONTRACT.columns}))


def test_buildAutocontext_contract_onlySourceKeyFirst(mock_semantic_layer):
    # Arrange: documentation that would add a glossary on the schema path
    layer = mock_semantic_layer(columns={"other": "other"})

    # Act
    autocontext = build_autocontext(
        semantic_layer=layer, doc_context="BMI: Body Mass Index", query_terms=["weight"], contract=CONTRACT
    )

    # Assert
    names = [c.name for c in autocontext.columns]
    assert names[:2] == ["patient_id", "weight_kg"]  # the key, then the column the question names
    assert set(names) == {c.name for c in CONTRACT.columns}
    assert autocontext.entity_keys == ["patient_id"]
    assert autocontext.glossary == {}
    assert autocontext.grain == "One row per patient_id"
    dtypes = {c.name: c.dtype for c in autocontext.columns}
    assert dtypes == {"patient_id": "id", "smoker": "coded", "weight_kg": "numeric", "onset": "datetime"}


def test_buildAutocontext_questionWordInDescriptionOnly_ranksThatColumnFirst(mock_semantic_layer):
    # Arrange
    layer = mock_semantic_layer()

    # Act
    autocontext = build_autocontext(semantic_layer=layer, query_terms=["smoked", "the"], contract=CONTRACT)

    # Assert
    assert [c.name for c in autocontext.columns][:2] == ["patient_id", "smoker"]


def test_buildLlmPrompt_contractContext_rendersDescriptionsUnitsCodesAndGrain(mock_semantic_layer):
    # Arrange
    engine = _engine(mock_semantic_layer)
    context = _rag_context_from_contract({"columns": [], "aliases": {"x": "y"}, "examples": []}, CONTRACT)
    autocontext = build_autocontext(semantic_layer=engine.semantic_layer, contract=CONTRACT)

    # Act
    system_prompt, _ = engine._build_llm_prompt("weight by smoking", context, autocontext=autocontext)

    # Assert
    assert "One row is: One row per patient_id" in system_prompt
    assert "- weight_kg [numeric, kg]: Weight at baseline" in system_prompt
    assert "- smoker [coded]: Ever smoked (codes: y=yes; n=no)" in system_prompt
    assert "- onset [datetime] (aliases: first symptoms)" in system_prompt
    assert "Aliases: {'first symptoms': 'onset'}" in system_prompt  # nothing but the contract's synonyms


def test_buildLlmPrompt_longCodebookAndDescription_clippedToLimits(mock_semantic_layer):
    # Arrange
    many = {str(i): f"level {i}" for i in range(SCHEMA_CONTEXT_MAX_CODES + 3)}
    contract = ContractContext(
        upload_id="u1",
        dataset_version="v1",
        grain=None,
        key=None,
        columns=(ContractColumn(name="drug", logical_type="string", description="d" * 400, codebook=many),),
    )
    engine = _engine(mock_semantic_layer)
    autocontext = build_autocontext(semantic_layer=engine.semantic_layer, contract=contract)

    # Act
    system_prompt, _ = engine._build_llm_prompt(
        "q", {"columns": [], "aliases": {}, "examples": []}, autocontext=autocontext
    )

    # Assert
    line = next(line for line in system_prompt.splitlines() if line.startswith("- drug"))
    assert "d" * 159 + "…" in line and "d" * 161 not in line
    assert f"{SCHEMA_CONTEXT_MAX_CODES - 1}=level" in line
    assert f"{SCHEMA_CONTEXT_MAX_CODES}=level" not in line
    assert "and 3 more" in line
    assert "One row is" not in system_prompt


def test_buildLlmPrompt_overBudget_namesTheColumnsLeftOut(mock_semantic_layer, monkeypatch):
    # Arrange
    monkeypatch.setattr("clinical_analytics.core.nl_query_engine.SCHEMA_CONTEXT_MAX_TOKENS", 30)
    engine = _engine(mock_semantic_layer)
    autocontext = build_autocontext(semantic_layer=engine.semantic_layer, contract=CONTRACT)

    # Act
    system_prompt, _ = engine._build_llm_prompt(
        "q", {"columns": [], "aliases": {}, "examples": []}, autocontext=autocontext
    )

    # Assert
    assert "less relevant columns)" in system_prompt
    assert "- patient_id [id]" in system_prompt


def test_ragContextFromContract_types_numericAndDates():
    # Arrange / Act
    context = _rag_context_from_contract({"columns": ["stale"], "examples": ["e"]}, CONTRACT)

    # Assert
    assert context["columns"] == ["smoker", "weight_kg", "patient_id", "onset"]
    assert context["column_types"]["weight_kg"] == {"type": "number", "numeric": True}
    assert context["column_types"]["onset"] == {"type": "datetime", "numeric": False}
    assert context["examples"] == ["e"]


@pytest.mark.parametrize("path", [None, "not a path"])
def test_contractContext_layerWithoutContractPath_none(mock_semantic_layer, path):
    # Arrange
    engine = _engine(mock_semantic_layer)
    engine.semantic_layer.contract_path = path

    # Act / Assert
    assert engine._contract_context() is None


def test_contractContext_versionWithoutContract_noneAndLogged(mock_semantic_layer, tmp_path: Path):
    # Arrange
    engine = _engine(mock_semantic_layer)
    engine.semantic_layer.contract_path = tmp_path / "missing.odcs.yaml"

    # Act / Assert
    assert engine._contract_context() is None


def test_contractContext_damagedContract_raises(mock_semantic_layer, tmp_path: Path):
    # Arrange
    path = tmp_path / "v1.odcs.yaml"
    path.write_text("")
    engine = _engine(mock_semantic_layer)
    engine.semantic_layer.contract_path = path

    # Act / Assert
    with pytest.raises(ContractDamagedError):
        engine._contract_context()


def test_llmParse_uploadWithContract_promptReadsOnlyTheContract(mock_semantic_layer, tmp_path):
    # Arrange: the layer's config documents a column the contract does not
    import polars as pl

    from clinical_analytics.contracts.builder import ContractInputs, build_contract
    from clinical_analytics.contracts.dictionary import ColumnDoc
    from clinical_analytics.contracts.store import contract_path, write_contract

    df = pl.DataFrame({"patient_id": [f"p{i}" for i in range(20)], "weight": [70.0 + i for i in range(20)]})
    inputs = ContractInputs(
        upload_id="u1",
        dataset_version="v1",
        dataset_name="synthetic",
        id_column="patient_id",
        dictionary={"weight": ColumnDoc(unit="kg")},
    )
    write_contract(tmp_path, "u1", "v1", build_contract(df, inputs))
    engine = _engine(mock_semantic_layer)
    engine.semantic_layer.contract_path = contract_path(tmp_path, "u1", "v1")
    engine.semantic_layer.config = {"metadata": {"doc_context": "LEGACY: only on the schema path"}}
    client = MagicMock()
    client.is_available.return_value = True
    client.generate.return_value = json.dumps(
        {"intent": "DESCRIBE", "metric": "weight", "group_by": None, "filters": [], "confidence": 0.8}
    )
    engine._ollama_client = client

    # Act
    engine._llm_parse("average weight")

    # Assert
    system_prompt = client.generate.call_args.kwargs["system_prompt"]
    assert "- weight [numeric, kg]" in system_prompt
    assert "Available columns: patient_id, weight" in system_prompt
    assert "LEGACY" not in system_prompt
