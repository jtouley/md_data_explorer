"""Tier 3 context from the data contract: what the model is told about an uploaded dataset."""

from __future__ import annotations

import dataclasses
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
    context = _rag_context_from_contract(CONTRACT, {"x": "y"}, [], "weight by smoking")
    autocontext = build_autocontext(semantic_layer=engine.semantic_layer, contract=CONTRACT)

    # Act
    system_prompt, _ = engine._build_llm_prompt("weight by smoking", context, autocontext=autocontext)

    # Assert
    assert "One row is: One row per patient_id" in system_prompt
    assert "- weight_kg [numeric, kg]: Weight at baseline" in system_prompt
    assert "- smoker [coded]: Ever smoked (codes: y=yes; n=no)" in system_prompt
    assert "- onset [datetime] (aliases: first symptoms)" in system_prompt
    assert "Aliases: {'first symptoms': 'onset'}" in system_prompt  # "x" names no contract column
    assert 'Filter on a coded column with its code (the value before "=")' in system_prompt


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


def test_ragContextFromContract_types_numericDatesAndCodes():
    # Arrange / Act
    context = _rag_context_from_contract(CONTRACT, {}, ["e"], "q")

    # Assert
    assert context["columns"] == ["smoker", "weight_kg", "patient_id", "onset"]
    assert context["column_types"]["weight_kg"] == {"type": "numeric", "numeric": True, "coded": False, "labels": {}}
    assert context["column_types"]["onset"]["type"] == "datetime"
    assert context["column_types"]["smoker"] == {
        "type": "string",
        "numeric": False,
        "coded": True,
        "labels": {"y": "yes", "n": "no"},
    }
    assert context["examples"] == ["e"]
    assert context["query"] == "q"


@pytest.mark.parametrize(
    ("logical_type", "catalog", "rule"),
    [
        ("time", "datetime", "DATE/TIME"),
        ("timestamp", "datetime", "DATE/TIME"),
        ("date", "datetime", "DATE/TIME"),
        ("integer", "numeric", "NUMERIC"),
        ("number", "numeric", "NUMERIC"),
        ("string", "categorical", "STRING"),
    ],
)
def test_contractTypes_eachLogicalType_catalogAndTypeRulesAgree(
    mock_semantic_layer, logical_type: str, catalog: str, rule: str
):
    # Arrange: a pl.Time column is "time" in the contract
    contract = ContractContext(
        upload_id="u1",
        dataset_version="v1",
        grain=None,
        key=None,
        columns=(ContractColumn(name="visit", logical_type=logical_type),),
    )
    engine = _engine(mock_semantic_layer)

    # Act
    autocontext = build_autocontext(semantic_layer=engine.semantic_layer, contract=contract)
    rules = engine._build_type_rules_section(_rag_context_from_contract(contract, {}, [], "q")["column_types"])

    # Assert
    assert autocontext.columns[0].dtype == catalog
    assert rules.startswith(f"- visit: {rule}")


def test_ragContextFromContract_layerAliases_keptForContractColumnsOnly():
    # Arrange: aliases users added after upload, a respelling of a name, a column not in the contract
    alias_index = {"body mass": "weight_kg", "smoking": "smoker", "weight kg": "weight_kg", "x": "not_in_contract"}

    # Act
    aliases = _rag_context_from_contract(CONTRACT, alias_index, [], "q")["aliases"]

    # Assert
    assert aliases == {"first symptoms": "onset", "body mass": "weight_kg", "smoking": "smoker"}


def test_ragContextFromContract_manyLayerAliases_questionWordsFirstWithinBudget(monkeypatch):
    # Arrange
    monkeypatch.setattr("clinical_analytics.core.nl_query_engine.SCHEMA_CONTEXT_ALIAS_MAX_TOKENS", 40)
    alias_index = {f"filler alias {i}": "weight_kg" for i in range(50)} | {"tobacco use": "smoker"}

    # Act
    aliases = _rag_context_from_contract(CONTRACT, alias_index, [], "how many report tobacco use")["aliases"]

    # Assert
    assert aliases["tobacco use"] == "smoker"
    assert aliases["first symptoms"] == "onset"
    assert 2 < len(aliases) < 20


def test_buildLlmPrompt_codesNotChecked_noRuleToFilterOnCodes(mock_semantic_layer):
    # Arrange: the schema path's codebook was read from documentation, never checked against data
    engine = _engine(mock_semantic_layer)
    autocontext = build_autocontext(semantic_layer=engine.semantic_layer, contract=CONTRACT)
    unchecked = dataclasses.replace(autocontext, codes_checked=False)

    # Act
    system_prompt, _ = engine._build_llm_prompt(
        "q", {"columns": [], "aliases": {}, "examples": []}, autocontext=unchecked
    )

    # Assert
    assert "Filter on a coded column" not in system_prompt


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


def _write_contract(tmp_path: Path, *, dictionary_read: bool = True, **overrides) -> Path:
    """A real contract for a synthetic 20-patient frame (weight, sex coded 1/2), written to disk;
    ``dictionary_read=False`` writes it as a builder from before dictionaries did."""
    import polars as pl

    from clinical_analytics.contracts import config
    from clinical_analytics.contracts.builder import ContractInputs, build_contract
    from clinical_analytics.contracts.store import contract_path, write_contract

    df = pl.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(20)],
            "weight": [70.0 + i for i in range(20)],
            "sex": [1, 2] * 10,
        }
    )
    fields = {"upload_id": "u1", "dataset_version": "v1", "dataset_name": "synthetic", "id_column": "patient_id"}
    contract = build_contract(df, ContractInputs(**{**fields, **overrides}))
    if not dictionary_read:
        contract.customProperties = [
            c for c in contract.customProperties or [] if c.property != config.DICTIONARY_READ_PROPERTY
        ]
    write_contract(tmp_path, "u1", "v1", contract)
    return contract_path(tmp_path, "u1", "v1")


def _contract_engine(mock_semantic_layer, path: Path) -> NLQueryEngine:
    engine = NLQueryEngine(mock_semantic_layer(columns={"patient_id": "patient_id", "weight": "weight", "sex": "sex"}))
    engine.semantic_layer.contract_path = path
    engine.semantic_layer.loaded_version = "v1"
    engine.semantic_layer.config = {"metadata": {}}
    return engine


@pytest.mark.parametrize("method", ["_dba_validate_llm", "_retry_with_dba_feedback_llm"])
def test_validationAndRetryPrompts_uploadWithContract_columnTypesFromContractOnly(
    mock_semantic_layer, tmp_path: Path, method: str
):
    # Arrange: the layer's own column metadata says something the contract does not
    from clinical_analytics.contracts.dictionary import ColumnDoc
    from clinical_analytics.core.nl_query_engine import QueryIntent

    path = _write_contract(tmp_path, dictionary={"sex": ColumnDoc(codebook={"1": "male", "2": "female"})})
    engine = _contract_engine(mock_semantic_layer, path)
    engine.semantic_layer.get_column_metadata.return_value = {"type": "string", "labels": {"9": "LAYER"}}
    client = MagicMock()
    client.is_available.return_value = True
    client.generate.return_value = json.dumps({"is_valid": True, "errors": [], "warnings": []})
    engine._ollama_client = client

    # Act
    if method == "_dba_validate_llm":
        engine._dba_validate_llm(QueryIntent(intent_type="COUNT", confidence=0.9), "how many male")
    else:
        engine._retry_with_dba_feedback_llm("how many male", ["bad column"])

    # Assert
    user_prompt = client.generate.call_args.args[1]
    assert "LAYER" not in user_prompt
    assert '"female"' in user_prompt
    engine.semantic_layer.get_column_metadata.assert_not_called()


def test_contractContext_askedTwice_readFromDiskOnce(mock_semantic_layer, tmp_path: Path, monkeypatch):
    # Arrange
    from clinical_analytics.contracts import context as contract_context

    engine = _contract_engine(mock_semantic_layer, _write_contract(tmp_path))
    reads = MagicMock(wraps=contract_context.load_contract_context)
    monkeypatch.setattr(contract_context, "load_contract_context", reads)

    # Act
    first, second = engine._contract_context(), engine._contract_context()

    # Assert
    assert first is second
    assert reads.call_count == 1


def test_contractContext_versionWithoutContract_loggedOnce(mock_semantic_layer, tmp_path: Path, monkeypatch):
    # Arrange
    warnings = MagicMock()
    monkeypatch.setattr("clinical_analytics.core.nl_query_engine.logger.warning", warnings)
    engine = _contract_engine(mock_semantic_layer, tmp_path / "missing.odcs.yaml")

    # Act
    results = [engine._contract_context() for _ in range(3)]

    # Assert
    assert results == [None, None, None]
    assert [c.args[0] for c in warnings.call_args_list] == ["model_context_without_contract"]


def test_contractContext_glossaryInContract_reachesThePrompt(mock_semantic_layer, tmp_path: Path):
    # Arrange
    engine = _contract_engine(mock_semantic_layer, _write_contract(tmp_path, glossary={"BMI": "Body mass index"}))

    # Act
    autocontext = build_autocontext(semantic_layer=engine.semantic_layer, contract=engine._contract_context())

    # Assert
    assert autocontext.glossary == {"BMI": "Body mass index"}


@pytest.mark.parametrize(
    ("stored", "codes"),
    [
        ([1, 2, None], {"1": "male", "2": "female"}),  # the data uses only documented codes
        ([1, 2, 3], {}),  # 3 is not documented: the dictionary's codes are not the data's
        ([None], {}),  # nothing stored to check them against
    ],
)
def test_contractContext_contractFromBeforeDictionaries_readsUploadDictionaryCheckedAgainstData(
    mock_semantic_layer, tmp_path: Path, stored: list[int | None], codes: dict[str, str]
):
    # Arrange: an upload whose contract predates dictionaries; its metadata keeps the dictionary
    import polars as pl

    engine = _contract_engine(mock_semantic_layer, _write_contract(tmp_path, dictionary_read=False))
    engine.semantic_layer.config = {
        "metadata": {
            "doc_context": "weight: Body weight (kg) at baseline\nsex: Sex at birth\n1: male\n2: female\n"
            "BMI: Body mass index"
        }
    }
    engine.semantic_layer.base_value_counts.side_effect = lambda column: (
        pl.Series(column, stored, dtype=pl.Int64),
        pl.Series([1] * len(stored)),
    )

    # Act
    context = engine._contract_context()

    # Assert
    columns = {c.name: c for c in context.columns}
    assert columns["weight"].description == "Body weight (kg) at baseline"
    assert columns["weight"].unit == "kg"
    assert columns["sex"].codebook == codes
    assert context.glossary["BMI"] == "Body mass index"
