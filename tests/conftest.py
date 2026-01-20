"""
Pytest configuration and fixtures for clinical analytics tests.
"""

import sys
import tempfile
from pathlib import Path

import polars as pl
import pytest
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Register performance tracking plugin
# Plugin checks --track-performance flag internally and only tracks when enabled
pytest_plugins = ["performance.plugin"]


# ============================================================================
# LLM Mocking Fixture (Phase 2.1: Performance Optimization)
# ============================================================================


@pytest.fixture
def mock_llm_calls(request):
    """
    Explicit fixture to mock LLM calls for unit tests.

    Usage: Add 'mock_llm_calls' to test function parameters.
    Integration tests should NOT use this fixture to get real LLM calls.

    This fixture mocks all LLMFeature types with realistic responses,
    providing 30-50x speedup (10-30s → <1s per test).

    Automatically skips mocking for tests marked with @pytest.mark.integration
    to allow real Ollama validation.
    """
    # Skip mocking for integration tests that need real Ollama
    integration_marker = request.node.get_closest_marker("integration")
    if integration_marker:
        # Don't mock - let real Ollama run for integration tests
        yield None
        return

    from unittest.mock import patch

    from clinical_analytics.core.llm_feature import LLMCallResult, LLMFeature

    # Patch call_llm in all modules that import it
    patches = [
        patch("clinical_analytics.core.llm_feature.call_llm"),
        patch("clinical_analytics.core.filter_extraction.call_llm"),
        patch("clinical_analytics.core.result_interpretation.call_llm"),
        patch("clinical_analytics.core.error_translation.call_llm"),
        patch("clinical_analytics.core.golden_question_generator.call_llm"),
        # Also patch OllamaClient.generate() for code that uses it directly (e.g., NLQueryEngine)
        patch("clinical_analytics.core.llm_client.OllamaClient.generate"),
        # CRITICAL: Patch is_available() to avoid real HTTP requests (30s timeout when Ollama isn't running)
        patch("clinical_analytics.core.llm_client.OllamaClient.is_available", return_value=True),
        # CRITICAL: Patch OllamaManager methods to ensure client is returned (not None)
        patch("clinical_analytics.core.ollama_manager.OllamaManager.is_service_running", return_value=True),
        # Return default and fallback models so get_client() won't return None
        patch(
            "clinical_analytics.core.ollama_manager.OllamaManager.get_available_models",
            return_value=["llama3.1:8b", "llama3.2:3b"],
        ),
    ]

    # Start all patches
    mock_objects = [p.start() for p in patches]

    # Use the first mock as the main one
    mock_call_llm = mock_objects[0]

    # Mock for OllamaClient.generate() - returns raw JSON string
    # Note: When used as side_effect, receives all arguments including self
    def _mock_ollama_generate(*args, **kwargs):
        """Mock OllamaClient.generate() to return JSON string matching QueryPlan schema."""
        import json

        # Extract arguments (self is first arg for instance methods, but we ignore it)
        # OllamaClient.generate() signature: generate(self, prompt, system_prompt=None, json_mode=False, model=None)
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt", "")
        system_prompt = kwargs.get("system_prompt", None)

        prompt_lower = prompt.lower() if prompt else ""
        system_prompt_lower = system_prompt.lower() if system_prompt else ""
        combined_text = f"{prompt_lower} {system_prompt_lower}"

        import re

        # Detect refinement queries - check for refinement keywords in prompt or conversation history in system prompt
        is_refinement = (
            "refinement" in combined_text
            or "remove" in prompt_lower
            or "exclude" in prompt_lower
            or "without" in prompt_lower
            or "get rid of" in prompt_lower
            or "also exclude" in prompt_lower
            or "only active" in prompt_lower
            or (
                "conversation_history" in system_prompt_lower
                and ("remove" in prompt_lower or "exclude" in prompt_lower)
            )
        )

        # Return appropriate JSON based on prompt content (heuristic)
        if is_refinement:
            # Refinement query - return query plan with filters
            # Extract group_by from conversation history if present in system prompt
            group_by = None
            if system_prompt:
                # Look for previous group_by in conversation history JSON
                # Pattern: "group_by": "column_name" or "group_by": null
                # Also handle multi-line strings (the statin column name is very long)
                group_by_match = re.search(r'"group_by":\s*"([^"]+)"', system_prompt, re.DOTALL)
                if group_by_match:
                    group_by = group_by_match.group(1)
                # Also check for previous_group_by in logs (format: previous_group_by='...')
                prev_group_by_match = re.search(r"previous_group_by[=:]\s*['\"]([^'\"]+)['\"]", system_prompt)
                if prev_group_by_match:
                    group_by = prev_group_by_match.group(1)

            # Extract existing filters from conversation history (for merging)
            existing_filters = []
            if system_prompt:
                # Look for Previous Filters in conversation context
                # Format is Python dict repr: Previous Filters: [{'column': 'age', ...}]
                import ast

                filters_match = re.search(
                    r"Previous Filters:\s*(\[[^\]]*\])",
                    system_prompt,
                    re.DOTALL,
                )
                if filters_match:
                    try:
                        # Parse Python dict repr to actual dict
                        filter_str = filters_match.group(1)
                        existing_filters = ast.literal_eval(filter_str)
                    except (ValueError, SyntaxError):
                        pass

            # Determine filter column and value based on context
            filter_column = "status"  # default
            filter_operator = "!="
            filter_value = 0

            # Check for age filter updates ("actually make it over 65")
            # Look for age-related keywords in prompt or system prompt
            has_age_context = (
                "age" in prompt_lower
                or "age" in system_prompt_lower
                or "over" in prompt_lower
                or "65" in prompt_lower
                or "50" in prompt_lower
            )

            # Case 1: Age filter update (e.g., "actually make it over 65")
            if has_age_context and ("over" in prompt_lower or ">" in prompt_lower or "make it" in prompt_lower):
                filter_column = "age"
                filter_operator = ">"
                # Extract age value from prompt (e.g., "over 65", "> 50")
                age_match = re.search(r"(?:over|>)\s*(\d+)", prompt_lower)
                if age_match:
                    filter_value = int(age_match.group(1))
                elif "65" in prompt_lower:
                    filter_value = 65
                elif "50" in prompt_lower:
                    filter_value = 50
                else:
                    filter_value = 65  # default
                # For age updates, replace existing age filter (not merge)
                response = {
                    "intent": "COUNT",
                    "metric": None,
                    "group_by": group_by,
                    "filters": [
                        {
                            "column": filter_column,
                            "operator": filter_operator,
                            "value": filter_value,
                            "exclude_nulls": True,
                        }
                    ],
                    "confidence": 0.8,
                    "explanation": "Refining previous query with updated age filter",
                }
                return json.dumps(response)

            # Case 2: Adding new filter (merge with existing)
            elif "statin" in combined_text:
                filter_column = "statin_used"
            elif "treatment" in combined_text:
                filter_column = "treatment_group"
            elif "status" in combined_text:
                filter_column = "status"

            # Build new filter
            new_filter = {
                "column": filter_column,
                "operator": filter_operator,
                "value": filter_value,
                "exclude_nulls": True,
            }

            # Merge: keep existing filters that aren't on the same column
            merged_filters = [f for f in existing_filters if f.get("column") != filter_column]
            merged_filters.append(new_filter)

            response = {
                "intent": "COUNT",
                "metric": None,
                "group_by": group_by,
                "filters": merged_filters,
                "confidence": 0.8,
                "explanation": "Refining previous query by excluding n/a values",
            }
            return json.dumps(response)
        elif "filter" in prompt_lower:
            # Filter extraction
            return '{"filters": []}'
        elif "follow" in prompt_lower:
            # Follow-ups
            return '{"follow_ups": []}'
        elif "describe" in prompt_lower:
            # DESCRIBE query
            response = {
                "intent": "DESCRIBE",
                "metric": None,
                "group_by": None,
                "filters": [],
                "confidence": 0.8,
                "explanation": "Describe the data",
            }
            return json.dumps(response)
        else:
            # Default parse response
            response = {
                "intent": "DESCRIBE",
                "metric": None,
                "group_by": None,
                "filters": [],
                "confidence": 0.8,
                "explanation": "Default query plan",
            }
            return json.dumps(response)

    # Mock responses for all LLMFeature types
    def _mock_call_llm(feature, system, user, timeout_s, model=None):
        if feature == LLMFeature.PARSE:
            return LLMCallResult(
                raw_text='{"intent": "DESCRIBE", "confidence": 0.8}',
                payload={"intent": "DESCRIBE", "confidence": 0.8},
                latency_ms=10.0,
                timed_out=False,
                error=None,
            )
        elif feature == LLMFeature.FILTER_EXTRACTION:
            return LLMCallResult(
                raw_text='{"filters": []}',
                payload={"filters": []},
                latency_ms=10.0,
                timed_out=False,
                error=None,
            )
        elif feature == LLMFeature.FOLLOWUPS:
            return LLMCallResult(
                raw_text='{"follow_ups": []}',
                payload={"follow_ups": []},
                latency_ms=10.0,
                timed_out=False,
                error=None,
            )
        elif feature == LLMFeature.RESULT_INTERPRETATION:
            return LLMCallResult(
                raw_text='{"interpretation": "Test interpretation"}',
                payload={"interpretation": "Test interpretation"},
                latency_ms=10.0,
                timed_out=False,
                error=None,
            )
        elif feature == LLMFeature.ERROR_TRANSLATION:
            return LLMCallResult(
                raw_text='{"translation": "Test error translation"}',
                payload={"translation": "Test error translation"},
                latency_ms=10.0,
                timed_out=False,
                error=None,
            )
        elif feature == LLMFeature.INTERPRETATION:
            return LLMCallResult(
                raw_text='{"interpretation": "Test interpretation", "confidence": 0.8}',
                payload={"interpretation": "Test interpretation", "confidence": 0.8},
                latency_ms=10.0,
                timed_out=False,
                error=None,
            )
        # Default fallback
        return LLMCallResult(
            raw_text="{}",
            payload={},
            latency_ms=10.0,
            timed_out=False,
            error=None,
        )

    # Set side_effect on all mocks EXCEPT OllamaClient.generate (which uses _mock_ollama_generate)
    # Note: is_available() patch (index 6) uses return_value=True, not side_effect
    ollama_generate_mock_index = 5  # Index of OllamaClient.generate mock
    for i, mock in enumerate(mock_objects):
        if i == ollama_generate_mock_index:
            mock.side_effect = _mock_ollama_generate
        elif i < 6:  # Only set side_effect for call_llm mocks (indices 0-4)
            mock.side_effect = _mock_call_llm
        # Index 6 (is_available) already has return_value=True from patch() call

    yield mock_call_llm

    # Stop all patches
    for p in patches:
        p.stop()


@pytest.fixture(scope="session")
def cached_sentence_transformer():
    """
    Session-scoped fixture to cache SentenceTransformer model.

    Loads the model once per test session and reuses it across all tests.
    This provides 2-5 second speedup per test that uses semantic matching.

    The model is loaded lazily on first use in NLQueryEngine._semantic_match(),
    but this fixture pre-loads it once for the entire test session.
    """
    from sentence_transformers import SentenceTransformer

    # Use same model name as NLQueryEngine default
    model_name = "all-MiniLM-L6-v2"
    encoder = SentenceTransformer(model_name)

    yield encoder

    # Cleanup (if needed)
    del encoder


@pytest.fixture
def mock_encoder():
    """Mock SentenceTransformer encoder for fast unit tests (10-50x speedup)."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    mock = MagicMock()
    embedding_dim = 384  # all-MiniLM-L6-v2 dimension

    def mock_encode(texts, *args, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            seed = hash(text) % (2**32)
            rng = np.random.RandomState(seed)
            embeddings.append(rng.randn(embedding_dim).astype(np.float32))
        return np.array(embeddings)

    mock.encode = MagicMock(side_effect=mock_encode)

    with patch("clinical_analytics.core.nl_query_engine.SentenceTransformer", mock, create=True):
        with patch("sentence_transformers.SentenceTransformer", return_value=mock):
            yield mock


@pytest.fixture
def nl_query_engine_with_cached_model(make_semantic_layer, cached_sentence_transformer):
    """
    Factory fixture that creates NLQueryEngine with pre-loaded SentenceTransformer.

    This avoids reloading the model for each test, providing 2-5s speedup per test.

    Usage:
        def test_example(nl_query_engine_with_cached_model, make_semantic_layer):
            semantic = make_semantic_layer(...)
            engine = nl_query_engine_with_cached_model(semantic_layer=semantic)
            result = engine.parse_query("describe outcome")
    """
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    def _create_engine(semantic_layer=None, **kwargs):
        if semantic_layer is None:
            semantic_layer = make_semantic_layer()

        engine = NLQueryEngine(semantic_layer=semantic_layer, **kwargs)
        # Inject pre-loaded encoder
        engine.encoder = cached_sentence_transformer
        # Pre-compute template embeddings if not already done
        if engine.template_embeddings is None:
            template_texts = [t["template"] for t in engine.query_templates]
            engine.template_embeddings = cached_sentence_transformer.encode(template_texts)

        return engine

    return _create_engine


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return test data directory."""
    return project_root / "data" / "raw"


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_data = {
        "test_dataset": {
            "name": "test",
            "display_name": "Test Dataset",
            "status": "available",
            "init_params": {"source_path": "data/raw/test/test.csv"},
            "column_mapping": {"id": "patient_id", "date": "time_zero", "result": "outcome"},
            "analysis": {
                "default_outcome": "outcome",
                "default_predictors": ["age", "sex"],
                "categorical_variables": ["sex"],
            },
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture(scope="module")
def ask_questions_page():
    """
    Import the Ask Questions page module.

    Uses importlib because the filename contains an emoji.
    """
    import importlib.util
    import sys
    from pathlib import Path

    # Add src to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    # Import from the page file (has emoji in name, so use importlib)
    page_path = project_root / "src" / "clinical_analytics" / "ui" / "pages" / "03_💬_Ask_Questions.py"
    spec = importlib.util.spec_from_file_location("ask_questions_page", page_path)
    ask_questions_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask_questions_page)

    return ask_questions_page


# ============================================================================
# Test Data Generation Fixtures (DRY - Single Source of Truth)
# ============================================================================


@pytest.fixture(scope="module")
def large_test_data_csv(num_records: int = 1000000) -> str:
    """Generate large CSV data string (1M records) with patient_id and age columns."""
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import make_large_csv

    return make_large_csv(
        columns={
            "patient_id": lambda i: f"P{i:06d}",
            "age": lambda i: str(20 + i % 100),
        },
        num_records=num_records,
    )


@pytest.fixture(scope="module")
def large_patients_csv(num_records: int = 1000000) -> str:
    """Generate large patients CSV (1M records) with patient_id, age, sex columns."""
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import make_large_csv

    return make_large_csv(
        columns={
            "patient_id": lambda i: f"P{i:06d}",
            "age": lambda i: str(20 + i % 100),
            "sex": lambda i: ["M", "F"][i % 2],
        },
        num_records=num_records,
    )


@pytest.fixture(scope="module")
def large_admissions_csv(num_records: int = 1000000) -> str:
    """Generate large admissions CSV (1M records) with patient_id and date columns."""
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import make_large_csv

    return make_large_csv(
        columns={
            "patient_id": lambda i: f"P{i:06d}",
            "date": lambda i: f"2020-01-{1 + i % 30:02d}",
        },
        num_records=num_records,
    )


@pytest.fixture(scope="module")
def large_admissions_with_admission_date_csv(num_records: int = 1000000) -> str:
    """Generate large admissions CSV (1M records) with patient_id and admission_date."""
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import make_large_csv

    return make_large_csv(
        columns={
            "patient_id": lambda i: f"P{i:06d}",
            "admission_date": lambda i: f"2020-01-{1 + i % 30:02d}",
        },
        num_records=num_records,
    )


@pytest.fixture(scope="module")
def large_admissions_with_discharge_csv(num_records: int = 1000000) -> str:
    """Generate large admissions CSV (1M records) with admission/discharge dates."""
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import make_large_csv

    return make_large_csv(
        columns={
            "patient_id": lambda i: f"P{i:06d}",
            "admission_date": lambda i: f"2020-01-{1 + i % 30:02d}",
            "discharge_date": lambda i: f"2020-01-{5 + i % 30:02d}",
        },
        num_records=num_records,
    )


@pytest.fixture(scope="module")
def large_diagnoses_csv(num_records: int = 1000000) -> str:
    """Generate large diagnoses CSV (1M records) with patient_id, icd_code, diagnosis."""
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import make_large_csv

    return make_large_csv(
        columns={
            "patient_id": lambda i: f"P{i:06d}",
            "icd_code": lambda i: "E11.9",
            "diagnosis": lambda i: "Diabetes",
        },
        num_records=num_records,
    )


@pytest.fixture(scope="module")
def large_zip_with_csvs(large_patients_csv, large_admissions_csv) -> bytes:
    """Create ZIP with patients.csv and admissions.csv (large test data)."""
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import make_large_zip

    return make_large_zip(
        csv_files={
            "patients.csv": large_patients_csv,
            "admissions.csv": large_admissions_csv,
        }
    )


@pytest.fixture(scope="module")
def large_zip_with_three_tables(large_patients_csv, large_admissions_csv, large_diagnoses_csv) -> bytes:
    """Create ZIP with patients, admissions, diagnoses CSVs (large test data)."""
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import make_large_zip

    return make_large_zip(
        csv_files={
            "patients.csv": large_patients_csv,
            "admissions.csv": large_admissions_csv,
            "diagnoses.csv": large_diagnoses_csv,
        }
    )


@pytest.fixture
def sample_patients_df() -> pl.DataFrame:
    """Create sample patients Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "age": [45, 62, 38],
            "sex": ["M", "F", "M"],
        }
    )


@pytest.fixture
def sample_admissions_df() -> pl.DataFrame:
    """Create sample admissions Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "patient_id": ["P001", "P002"],
            "admission_date": ["2020-01-01", "2020-02-01"],
            "discharge_date": ["2020-01-05", "2020-02-10"],
        }
    )


@pytest.fixture
def sample_upload_df() -> pl.DataFrame:
    """Create sample upload Polars DataFrame for testing lazy frames."""
    return pl.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "outcome": [0, 1, 0],
            "age": [50, 60, 70],
        }
    )


@pytest.fixture
def sample_upload_metadata() -> dict:
    """Create sample upload metadata for testing."""
    return {
        "upload_timestamp": "2024-01-01T00:00:00",
        "original_filename": "test.csv",
    }


@pytest.fixture
def sample_variable_mapping() -> dict:
    """Create sample variable mapping for uploaded datasets."""
    return {
        "patient_id": "patient_id",
        "outcome": "outcome",
        "predictors": ["age"],
    }


# ============================================================================
# Excel Test Data Fixtures (DRY - Reusable across all Excel tests)
# ============================================================================


@pytest.fixture(scope="module")
def synthetic_dexa_excel_file(tmp_path_factory):
    """
    Create synthetic DEXA-like Excel file with headers in row 0 (standard format).

    Mimics: data/raw/LWTest/de-identified DEXA.xlsx
    - Headers in first row
    - 25-50 rows of clinical data
    - Mixed data types (strings, numbers, dates)

    Uses caching to avoid regeneration across test runs.

    Returns:
        Path to Excel file
    """
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import _create_synthetic_excel_file

    # Create DataFrame (deterministic data)
    data = {
        "Race": ["Black or African-American"] * 30 + ["White"] * 20,
        "Gender": ["Male", "Female"] * 25,
        "Age": list(range(40, 90)),
        "Had DEXA Scan? Yes: 1 No: 2": [1] * 35 + [2] * 15,
        "Results of DEXA? 1: Normal 2: Osteopenia 3: Osteoporosis": [1, 2, 3] * 16 + [1, 2],
        "DEXA Score (T score)": [-2.5 + i * 0.1 for i in range(50)],
        "DEXA Score (Z score)": [-1.8 + i * 0.08 for i in range(50)],
        "CD4 Count": list(range(200, 1200, 20)),
        "Viral Load": ["<20"] * 30 + ["40", "120", "240"] * 6 + ["<20"] * 2,
        "Prior Tenofovir (TDF) use? 1: Yes 2: No 3: Unknown": [1, 2, 3] * 16 + [1, 2],
    }

    return _create_synthetic_excel_file(
        tmp_path_factory,
        data,
        "synthetic_dexa.xlsx",
        excel_config={"header_row": 0, "use_dataframe_hash": True},
    )


@pytest.fixture(scope="module")
def synthetic_statin_excel_file(tmp_path_factory):
    """
    Create synthetic Statin-like Excel file with empty row 1, headers in row 2.

    Mimics: data/raw/LWTest/Statin use - deidentified.xlsx
    - Empty first row
    - Headers in row 2 (index 1)
    - 25-50 rows of clinical data
    - Complex multi-column structure (27 columns)

    Returns:
        Path to Excel file
    """
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import _create_synthetic_excel_file

    # All columns must have exactly 50 rows
    n_rows = 50
    data = {
        "Race": (["Black or African-American"] * 30 + ["White", "Asian"] * 10)[:n_rows],
        "Gender": (["Male", "Female", "Transgender MtF"] * 17)[:n_rows],
        "Age": list(range(40, 90))[:n_rows],
        "Most Recent VL copies/mL": (["<20"] * 25 + ["40", "120", "240", "1740"] * 6 + ["<20"])[:n_rows],
        "Most Recent CD4 /uL": list(range(185, 185 + n_rows * 26, 26))[:n_rows],
        "Current Regimen 1: Biktarvy 2: Symtuza 3: Triumeq": ([1, 2, 3, 4, 5] * 10)[:n_rows],
        "Regimen (if 9 or other)": ([None] * 45 + ["Biktarvy, Darunavir/Ritonavir"] * 5)[:n_rows],
        "Total Cholesterol mg/dL": list(range(85, 85 + n_rows * 3, 3))[:n_rows],
        "LDL mg/dL": list(range(25, 25 + n_rows * 2, 2))[:n_rows],
        "HDL mg/dL": list(range(33, 33 + n_rows))[:n_rows],
        "Triglycerides mg/dL": list(range(5, 5 + n_rows * 5, 5))[:n_rows],
        "Systolic blood pressure": list(range(100, 100 + n_rows))[:n_rows],
        "Diastolic blood pressure": list(range(60, 60 + n_rows))[:n_rows],
        "Diabetes 1: Yes 2: No": ([1, 2] * 25)[:n_rows],
        "HTN 1: Yes 2: No": ([1, 2] * 25)[:n_rows],
    }

    return _create_synthetic_excel_file(
        tmp_path_factory,
        data,
        "synthetic_statin.xlsx",
        excel_config={
            "header_row": 1,
            "metadata_rows": [{"row_index": 0, "cells": [""] * len(data)}],
            "use_dataframe_hash": False,  # Must hash file due to metadata row
        },
    )


@pytest.fixture(scope="module")
def synthetic_complex_excel_file(tmp_path_factory):
    """
    Create complex Excel file with metadata rows before headers.

    - Row 1: Metadata/notes (mostly empty, one cell with "Units")
    - Row 2: Actual headers
    - Row 3+: Data

    Returns:
        Path to Excel file
    """
    import sys
    from pathlib import Path

    # Add tests to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.factories import _create_synthetic_excel_file

    # All columns must have exactly 45 rows
    n_rows = 45
    data = {
        "Race": (["Black or African-American"] * 25 + ["White"] * 20)[:n_rows],
        "Gender": (["Male", "Female"] * 23)[:n_rows],
        "Age": list(range(35, 80))[:n_rows],
        "Viral Load copies/mL": (["<20"] * 30 + ["40", "120", "240"] * 5)[:n_rows],
        "CD4 Count /uL": list(range(200, 200 + n_rows * 26, 26))[:n_rows],
        "Total Cholesterol mg/dL": list(range(100, 100 + n_rows * 3, 3))[:n_rows],
        "LDL mg/dL": list(range(30, 30 + n_rows * 2, 2))[:n_rows],
        "HDL mg/dL": list(range(35, 35 + n_rows))[:n_rows],
        "Triglycerides mg/dL": list(range(50, 50 + n_rows * 5, 5))[:n_rows],
        "Systolic BP mmHg": list(range(100, 100 + n_rows))[:n_rows],
        "Diastolic BP mmHg": list(range(60, 60 + n_rows))[:n_rows],
        "Diabetes Yes:1 No:2": ([1, 2] * 23)[:n_rows],
    }

    # Create metadata row with "Units" in column 8 (index 7)
    metadata_cells = [None] * len(data)
    metadata_cells[7] = "Units"

    return _create_synthetic_excel_file(
        tmp_path_factory,
        data,
        "synthetic_complex.xlsx",
        excel_config={
            "header_row": 1,
            "metadata_rows": [{"row_index": 0, "cells": metadata_cells}],
            "use_dataframe_hash": False,  # Must hash file due to metadata row
        },
    )


# ============================================================================
# Consolidated Test Fixtures (DRY - Single Source of Truth)
# ============================================================================


@pytest.fixture
def sample_cohort():
    """
    Standard Polars cohort fixture used across all tests.

    Returns Polars DataFrame with:
    - patient_id: Integer IDs [1, 2, 3, 4, 5]
    - outcome: Binary outcome [0, 1, 0, 1, 0]
    - age: Age values [45, 62, 38, 71, 55]
    - treatment: Treatment arm ["A", "A", "B", "B", "A"]
    """
    return pl.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5],
            "outcome": [0, 1, 0, 1, 0],
            "age": [45, 62, 38, 71, 55],
            "treatment": ["A", "A", "B", "B", "A"],
        }
    )


@pytest.fixture
def mock_cohort():
    """
    Pandas cohort fixture for Streamlit UI tests.

    Returns Pandas DataFrame with UnifiedCohort schema:
    - patient_id: String IDs ["P0" - "P19"]
    - time_zero: Date range
    - outcome: Binary [0, 1] alternating
    - outcome_label: ["alive", "dead"] alternating
    - Predictors: age, score, group
    """
    # PANDAS EXCEPTION: Required for legacy cohort format compatibility
    import pandas as pd
    from clinical_analytics.core.schema import UnifiedCohort

    return pd.DataFrame(
        {
            UnifiedCohort.PATIENT_ID: [f"P{i}" for i in range(20)],
            UnifiedCohort.TIME_ZERO: pd.date_range("2023-01-01", periods=20),
            UnifiedCohort.OUTCOME: [0, 1] * 10,
            UnifiedCohort.OUTCOME_LABEL: ["alive", "dead"] * 10,
            # Predictors
            "age": [25, 30, 35, 40] * 5,
            "score": [1.5, 2.5, 3.5, 4.5] * 5,
            "group": ["A", "B"] * 10,
        }
    )


@pytest.fixture
def sample_context():
    """
    Direct AnalysisContext fixture for backward compatibility.

    Returns AnalysisContext with DESCRIBE intent and confidence=0.9.
    This is a simple, ready-to-use fixture.
    """
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext(
        inferred_intent=AnalysisIntent.DESCRIBE,
        primary_variable="all",
    )
    context.confidence = 0.9
    return context


@pytest.fixture
def low_confidence_context():
    """
    AnalysisContext fixture with low confidence (0.4).

    Returns AnalysisContext configured for low-confidence feedback testing:
    - inferred_intent: COMPARE_GROUPS
    - primary_variable: "mortality"
    - grouping_variable: "treatment_arm"
    - confidence: 0.4 (below auto-execute threshold)
    - match_suggestions: Dictionary with collision suggestions
    """
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="mortality",
        grouping_variable="treatment_arm",
        research_question="compare mortality by treatment",
        match_suggestions={"mortality": ["mortality", "death", "outcome"]},
    )
    context.confidence = 0.4  # Low confidence
    return context


@pytest.fixture
def high_confidence_context():
    """
    AnalysisContext fixture with high confidence (0.9).

    Returns AnalysisContext configured for high-confidence auto-execute testing:
    - inferred_intent: COMPARE_GROUPS
    - primary_variable: "mortality"
    - grouping_variable: "treatment_arm"
    - confidence: 0.9 (above auto-execute threshold)
    """
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="mortality",
        grouping_variable="treatment_arm",
        research_question="compare mortality by treatment",
    )
    context.confidence = 0.9  # High confidence
    return context


@pytest.fixture
def mock_semantic_layer():
    """
    Factory fixture for creating mock SemanticLayer instances.

    Returns a function that creates a MagicMock with configurable column mappings.

    Usage:
        def test_example(mock_semantic_layer):
            mock = mock_semantic_layer(columns={
                "mortality": "mortality",
                "treatment": "treatment_arm"
            })
    """
    from unittest.mock import MagicMock

    def _make(columns=None, collision_suggestions=None):
        mock = MagicMock()
        default_columns = {
            "mortality": "mortality",
            "treatment": "treatment_arm",
            "age": "age",
        }
        column_map = columns or default_columns
        mock.get_column_alias_index.return_value = column_map
        mock.get_collision_suggestions.return_value = collision_suggestions
        mock.get_collision_warnings.return_value = set()
        mock._normalize_alias = lambda x: x.lower().replace(" ", "_")

        # Mock get_base_view() for filter validation
        # Filter validation checks if column exists in view.columns
        base_view_mock = MagicMock()
        base_view_mock.columns = list(column_map.values())  # Use actual column names
        mock.get_base_view.return_value = base_view_mock

        return mock

    return _make


# ============================================================================
# Semantic Layer Factory Fixtures (Test Suite DRY Refactoring - Phase 1.1)
# ============================================================================


@pytest.fixture
def make_semantic_layer(tmp_path):
    """
    Factory fixture for creating SemanticLayer instances.

    Eliminates duplicate mock_semantic_layer fixtures across 21 test files.

    Usage:
        def test_example(make_semantic_layer):
            layer = make_semantic_layer(
                dataset_name="custom",
                data={"patient_id": [1, 2, 3], "age": [45, 62, 38]},
                config_overrides={"time_zero": {"value": "2024-01-01"}}
            )
    """
    from clinical_analytics.core.semantic import SemanticLayer

    def _make(
        dataset_name: str = "test_dataset",
        data: dict | pl.DataFrame | None = None,
        config_overrides: dict | None = None,
        workspace_name: str | None = None,
    ) -> SemanticLayer:
        workspace = tmp_path / (workspace_name or "workspace")
        workspace.mkdir()
        (workspace / "pyproject.toml").write_text("[project]\nname = 'test'")

        data_dir = workspace / "data" / "raw" / dataset_name
        data_dir.mkdir(parents=True)

        # Default data if not provided
        if data is None:
            data = {
                "patient_id": [1, 2, 3],
                "age": [45, 62, 38],
                "status": ["active", "inactive", "active"],
            }

        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pl.DataFrame(data)
        else:
            df = data

        # Write CSV
        df.write_csv(data_dir / "test.csv")

        # Build config
        config = {
            "init_params": {"source_path": f"data/raw/{dataset_name}/test.csv"},
            "column_mapping": {"patient_id": "patient_id"},
            "time_zero": {"value": "2024-01-01"},
            "outcomes": {},
            "analysis": {"default_outcome": "outcome"},
        }
        if config_overrides:
            config.update(config_overrides)

        semantic = SemanticLayer(dataset_name, config=config, workspace_root=workspace)
        semantic.dataset_version = "test_v1"
        return semantic

    return _make


# ============================================================================
# DataFrame Factory Fixtures (Test Suite DRY Refactoring - Phase 2.1)
# ============================================================================


@pytest.fixture
def make_cohort_with_categorical():
    """
    Factory for cohort DataFrames with categorical encoding.

    Eliminates duplicate DataFrame creation across test files with
    common patterns like "1: Yes", "2: No" categorical variables.

    Usage:
        def test_example(make_cohort_with_categorical):
            cohort = make_cohort_with_categorical(
                patient_ids=["P001", "P002"],
                treatment=["1: Yes", "2: No"],
                ages=[45, 52]
            )
    """

    def _make(
        patient_ids: list[str] | None = None,
        treatment: list[str] | None = None,
        status: list[str] | None = None,
        ages: list[int] | None = None,
        n_patients: int | None = None,
    ) -> pl.DataFrame:
        # If n_patients provided, generate defaults
        if n_patients is not None:
            if patient_ids is None:
                patient_ids = [f"P{i:03d}" for i in range(1, n_patients + 1)]
            elif len(patient_ids) != n_patients:
                raise ValueError(f"patient_ids length ({len(patient_ids)}) != n_patients ({n_patients})")

            if treatment is None:
                treatment = ["control"] * n_patients
            if status is None:
                status = ["active"] * n_patients
            if ages is None:
                ages = [30 + i for i in range(n_patients)]

        # Fallback for when n_patients not provided (backward compatibility)
        if patient_ids is None:
            patient_ids = [f"P{i:03d}" for i in range(1, 6)]

        # Ensure all arrays have same length as patient_ids
        n = len(patient_ids)

        if treatment is None:
            treatment = ["1: Yes", "2: No"] * (n // 2) + ["1: Yes"] * (n % 2)
        if status is None:
            status = ["1: Active", "2: Inactive"] * (n // 2) + ["1: Active"] * (n % 2)
        if ages is None:
            ages = [45 + i * 5 for i in range(n)]

        return pl.DataFrame(
            {
                "patient_id": patient_ids,
                "treatment": treatment,
                "status": status,
                "age": ages,
            }
        )

    return _make


def test_make_cohort_with_categorical_n_patients_generates_defaults(make_cohort_with_categorical):
    """Test that n_patients parameter generates default patient IDs."""
    # Arrange
    n_patients = 5

    # Act
    cohort = make_cohort_with_categorical(n_patients=n_patients)

    # Assert
    assert cohort.height == 5
    assert cohort["patient_id"].to_list() == ["P001", "P002", "P003", "P004", "P005"]


@pytest.fixture
def make_multi_table_setup():
    """
    Factory for multi-table test setups (patients, medications, bridge).

    Eliminates duplicate multi-table DataFrame creation across test files.
    Returns a dictionary with 3 DataFrames: patients, medications, patient_medications.

    Usage:
        def test_example(make_multi_table_setup):
            tables = make_multi_table_setup(num_patients=5, num_medications=4)
            patients = tables["patients"]
            medications = tables["medications"]
            bridge = tables["patient_medications"]
    """

    def _make(
        num_patients: int = 3,
        num_medications: int = 3,
    ) -> dict[str, pl.DataFrame]:
        patients = pl.DataFrame(
            {
                "patient_id": [f"P{i}" for i in range(1, num_patients + 1)],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"][:num_patients],
                "age": [30, 45, 28, 52, 39][:num_patients],
            }
        )

        medications = pl.DataFrame(
            {
                "medication_id": [f"M{i}" for i in range(1, num_medications + 1)],
                "drug_name": ["Aspirin", "Metformin", "Lisinopril", "Atorvastatin"][:num_medications],
                "dosage": ["100mg", "500mg", "10mg", "20mg"][:num_medications],
            }
        )

        # Default bridge: P1->M1,M2; P2->M1; P3->M3
        patient_medications = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P3"][: min(4, num_patients * num_medications)],
                "medication_id": ["M1", "M2", "M1", "M3"][: min(4, num_patients * num_medications)],
                "start_date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-03-01"][
                    : min(4, num_patients * num_medications)
                ],
            }
        )

        return {
            "patients": patients,
            "medications": medications,
            "patient_medications": patient_medications,
        }

    return _make


# ============================================================================
# Analysis Context Fixtures (for compute tests)
# ============================================================================


@pytest.fixture
def sample_numeric_df():
    """Create sample Polars DataFrame with numeric columns."""
    return pl.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "score": [10, 20, 30, 40, 50, 60, 70, 80],
            "value": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        }
    )


@pytest.fixture
def sample_categorical_df():
    """Create sample Polars DataFrame with categorical columns."""
    return pl.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "status": ["active", "inactive", "active", "inactive", "active", "inactive", "active", "inactive"],
        }
    )


@pytest.fixture
def sample_mixed_df():
    """Create sample Polars DataFrame with mixed column types."""
    return pl.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "category": ["A", "B", "A", "B", "A"],
            "score": [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture
def sample_context_describe():
    """Create AnalysisContext for descriptive analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.DESCRIBE
    context.primary_variable = "all"
    return context


@pytest.fixture
def sample_context_compare():
    """Create AnalysisContext for comparison analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
    context.primary_variable = "score"
    context.grouping_variable = "category"
    return context


@pytest.fixture
def sample_context_predictor():
    """Create AnalysisContext for predictor analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.FIND_PREDICTORS
    context.primary_variable = "outcome"
    context.predictor_variables = ["age", "score"]
    return context


@pytest.fixture
def sample_context_survival():
    """Create AnalysisContext for survival analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.EXAMINE_SURVIVAL
    context.time_variable = "time"
    context.event_variable = "event"
    return context


@pytest.fixture
def sample_context_relationship():
    """Create AnalysisContext for relationship analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.EXPLORE_RELATIONSHIPS
    context.predictor_variables = ["age", "score", "value"]
    return context


@pytest.fixture
def sample_context_count():
    """Create AnalysisContext for count analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.COUNT
    return context


# ============================================================================
# Mock Session State Fixture
# ============================================================================


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session_state for UI tests."""
    return {}


# ============================================================================
# Real-World Query Test Cases Fixture (ADR003 - Query Parsing Validation)
# ============================================================================


@pytest.fixture(scope="module")
def real_world_query_test_cases():
    """
    Fixture providing real-world query test cases with expected outputs.

    This fixture centralizes all real-world queries and their expected parsing results,
    making it easy to:
    - Track expected outputs
    - Update expectations as parsing improves
    - Add new queries without hardcoding
    - Reuse across multiple test files

    Structure:
        Each test case is a dict with:
        - query: str - The natural language query
        - expected_intent: str - Expected intent type (COUNT, DESCRIBE, COMPARE_GROUPS, etc.)
        - expected_primary_variable: str | None - Expected primary variable (canonical column name)
        - expected_grouping_variable: str | None - Expected grouping variable (for breakdowns)
        - expected_filters: list[dict] | None - Expected filter specifications
        - min_confidence: float - Minimum acceptable confidence (0.0-1.0)
        - parsing_tier: str | None - Expected parsing tier (pattern_match, semantic_match, llm_fallback)
        - notes: str | None - Notes about the query or expected behavior

    Returns:
        dict: Test cases organized by category
    """
    return {
        "count_queries": [
            {
                "query": "how many patients were on statins",
                "expected_intent": "COUNT",
                "expected_primary_variable": None,
                "expected_grouping_variable": None,
                "expected_filters": [{"column": "statins", "operator": "in", "values": ["yes", "1", "true"]}],
                "min_confidence": 0.75,
                "parsing_tier": "pattern_match",
                "notes": "Simple count query with filter on statins",
            },
            {
                "query": "which statin was most prescribed?",
                "expected_intent": "COUNT",
                "expected_primary_variable": None,
                "expected_grouping_variable": "statins",  # Canonical column name (normalized from "statin")
                "expected_filters": None,
                "min_confidence": 0.75,
                "parsing_tier": "pattern_match",
                "notes": "Count with grouping to find most common statin",
            },
            {
                "query": "what was the most common HIV regiment?",
                "expected_intent": "COUNT",
                "expected_primary_variable": None,
                "expected_grouping_variable": "hiv_regiment",
                "expected_filters": None,
                "min_confidence": 0.75,
                "parsing_tier": "pattern_match",
                "notes": "Count with grouping to find most common regimen",
            },
            {
                "query": "what was the most common Current Regimen",
                "expected_intent": "COUNT",
                "expected_primary_variable": None,
                "expected_grouping_variable": "Current Regimen",
                "expected_filters": None,
                "min_confidence": 0.75,
                "parsing_tier": "pattern_match",
                "notes": "Count with grouping on Current Regimen column",
            },
            {
                "query": "excluding those not on statins, which was the most prescribed statin?",
                "expected_intent": "COUNT",
                "expected_primary_variable": None,
                "expected_grouping_variable": "statins",
                "expected_filters": [{"column": "statins", "operator": "in", "values": ["yes", "1", "true"]}],
                "min_confidence": 0.7,
                "parsing_tier": "semantic_match",  # More complex, may need semantic matching
                "notes": "Count with filter and grouping - complex query",
            },
            {
                "query": "what statins were those patients on, broken down by count of patients per statin?",
                "expected_intent": "COUNT",
                "expected_primary_variable": None,
                "expected_grouping_variable": "statins",
                "expected_filters": None,
                "min_confidence": 0.7,
                "parsing_tier": "semantic_match",  # Complex phrasing
                "notes": "Count breakdown by statin type",
            },
            {
                "query": (
                    "what statins were those patients on, broken down by count of patients by their Current Regimen"
                ),
                "expected_intent": "COUNT",
                "expected_primary_variable": None,
                "expected_grouping_variable": "Current Regimen",
                "expected_filters": None,
                "min_confidence": 0.7,
                "parsing_tier": "semantic_match",  # Complex phrasing with multiple grouping hints
                "notes": "Count breakdown by Current Regimen (complex query)",
            },
        ],
        "describe_queries": [
            {
                "query": "average BMI of patients",
                "expected_intent": "DESCRIBE",
                "expected_primary_variable": "BMI",
                "expected_grouping_variable": None,
                "expected_filters": None,
                "min_confidence": 0.85,
                "parsing_tier": "pattern_match",
                "notes": "Average/mean query - should extract BMI variable",
            },
            {
                "query": "average ldl of all patients",
                "expected_intent": "DESCRIBE",
                "expected_primary_variable": "LDL mg/dL",
                "expected_grouping_variable": None,
                "expected_filters": None,
                "min_confidence": 0.85,
                "parsing_tier": "pattern_match",
                "notes": "Average query with 'of all patients' phrasing",
            },
        ],
    }


@pytest.fixture
def semantic_layer_with_clinical_columns():
    """
    Create a mock semantic layer with clinical columns for real-world query testing.

    Includes columns commonly found in clinical datasets:
    - Statins-related columns
    - HIV regimen columns
    - Clinical measurements (BMI, LDL)
    - Current Regimen
    """
    from unittest.mock import MagicMock

    mock = MagicMock()

    # Alias index with clinical column mappings
    alias_index = {
        # Statins
        "statins": "statins",
        "statin": "statins",
        "on statins": "statins",
        "statin medication": "statins",
        # HIV Regimen
        "hiv regiment": "hiv_regiment",
        "hiv regimen": "hiv_regiment",
        "hiv_regiment": "hiv_regiment",
        # Current Regimen
        "current regimen": "Current Regimen",
        "current_regimen": "Current Regimen",
        # Note: "regimen" intentionally omitted to avoid collision - tests handle this explicitly
        # Clinical measurements
        "bmi": "BMI",
        "body mass index": "BMI",
        "ldl": "LDL mg/dL",
        "ldl cholesterol": "LDL mg/dL",
        "ldl mg/dl": "LDL mg/dL",
    }

    mock.get_column_alias_index.return_value = alias_index
    mock.get_collision_suggestions.return_value = None
    mock.get_collision_warnings.return_value = set()
    mock._normalize_alias = lambda x: x.lower().replace(" ", "_")

    def mock_fuzzy_match(term: str):
        """Mock fuzzy matching for clinical variables."""
        term_lower = term.lower().strip()
        var_map = {
            "bmi": ("BMI", 0.9, None),
            "ldl": ("LDL mg/dL", 0.9, None),
            "statins": ("statins", 0.9, None),
            "statin": ("statins", 0.9, None),
            "hiv regiment": ("hiv_regiment", 0.85, None),
            "hiv_regiment": ("hiv_regiment", 0.9, None),
            "regimen": ("hiv_regiment", 0.8, None),  # Lower confidence due to potential collision
            "current regimen": ("Current Regimen", 0.9, None),
            "current_regimen": ("Current Regimen", 0.9, None),
        }
        return var_map.get(term_lower, (None, 0.0, None))

    mock._fuzzy_match_variable = mock_fuzzy_match

    # Mock base view with clinical columns
    base_view = pl.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "statins": ["yes", "no", "yes"],
            "statin": ["atorvastatin", "none", "simvastatin"],
            "hiv_regiment": ["regimen_a", "regimen_b", "regimen_a"],
            "Current Regimen": ["regimen_1", "regimen_2", "regimen_1"],
            "BMI": [25.5, 28.3, 22.1],
            "LDL mg/dL": [120, 150, 100],
        }
    )
    mock.get_base_view.return_value = base_view

    return mock


# ============================================================================
# Dataset Discovery Fixtures (Performance Optimization - Session Scoped)
# ============================================================================


@pytest.fixture(scope="session")
def discovered_datasets():
    """
    Session-scoped fixture to discover datasets once per test run.

    Caches dataset discovery to avoid expensive module imports on every test.
    This dramatically improves test performance by:
    - Discovering datasets once per session instead of per test
    - Pre-loading configs to avoid repeated YAML parsing
    - Eliminating redundant registry resets

    Returns:
        dict with keys:
        - available: List of available dataset names (excluding built-ins)
        - configs: Dict mapping dataset names to their configs
        - all_datasets: List of all discovered dataset names
    """
    import logging

    from clinical_analytics.core.mapper import load_dataset_config
    from clinical_analytics.core.registry import DatasetRegistry

    logger = logging.getLogger(__name__)

    # Discover once per session
    DatasetRegistry.reset()
    DatasetRegistry.discover_datasets()
    DatasetRegistry.load_config()

    all_datasets = DatasetRegistry.list_datasets()
    excluded = ["uploaded"]  # Exclude the class itself, not instances
    available = [d for d in all_datasets if d not in excluded]

    logger.info(
        "session_dataset_discovery",
        all_datasets=all_datasets,
        total_count=len(all_datasets),
        available_after_filter=available,
        available_count=len(available),
    )

    # Pre-load configs for available datasets
    configs = {}
    for dataset_name in available:
        try:
            configs[dataset_name] = load_dataset_config(dataset_name)
        except Exception as e:
            logger.warning(
                "session_config_load_failed",
                dataset_name=dataset_name,
                error=str(e),
            )

    return {
        "available": available,
        "configs": configs,
        "all_datasets": all_datasets,
    }


@pytest.fixture(scope="session")
def dataset_registry():
    """
    Session-scoped dataset registry for lazy loading.

    Discovers datasets but does NOT pre-load all configs.
    Configs are loaded lazily when get_dataset_by_name() is called.

    Returns:
        DatasetRegistry class (not instance) - use DatasetRegistry.get_dataset() directly
    """
    from clinical_analytics.core.registry import DatasetRegistry

    # Discover datasets once per session (but don't load configs)
    DatasetRegistry.reset()
    DatasetRegistry.discover_datasets()
    DatasetRegistry.load_config()  # Load config registry, but not individual dataset configs

    return DatasetRegistry


@pytest.fixture
def get_dataset_by_name(dataset_registry):
    """
    Helper fixture to load specific dataset by name (lazy loading).

    This fixture allows tests to load only the dataset they need,
    avoiding the cost of loading all dataset configs upfront.

    Args:
        dataset_registry: Session-scoped DatasetRegistry class

    Returns:
        Function that takes dataset name and returns ClinicalDataset instance

    Example:
        def test_example(get_dataset_by_name):
            dataset = get_dataset_by_name("my_dataset")
            cohort = dataset.get_cohort()
    """
    import logging

    logger = logging.getLogger(__name__)

    def _get(name: str):
        """
        Load dataset by name, skipping if not available.

        Args:
            name: Dataset name to load

        Returns:
            ClinicalDataset instance

        Raises:
            pytest.skip: If dataset doesn't exist or doesn't validate
        """
        try:
            dataset = dataset_registry.get_dataset(name)
            if dataset is None:
                pytest.skip(f"Dataset '{name}' not found in registry")

            # Validate dataset (check if data is available)
            if not dataset.validate():
                pytest.skip(f"Dataset '{name}' data not available")

            logger.info(f"selective_dataset_loaded: dataset={name}")

            return dataset
        except Exception as e:
            logger.warning(
                f"selective_dataset_load_failed: dataset={name}, error={e}",
            )
            pytest.skip(f"Failed to load dataset '{name}': {e}")

    return _get


# ============================================================================
# Performance Tracking Plugin Registration
# ============================================================================
# Test Fixture Enforcement (Phase 8: DRY Test Patterns)
# ============================================================================


@pytest.fixture
def upload_storage(tmp_path):
    """
    Create UserDatasetStorage with temp directory.

    DRY principle: Extract common UserDatasetStorage setup used across
    10+ test files to avoid duplicate inline creation.
    """
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    return UserDatasetStorage(upload_dir=tmp_path)


@pytest.fixture
def large_test_df(num_records: int = 150) -> pl.DataFrame:
    """
    Create large test Polars DataFrame with patient_id and other columns.

    DRY principle: Extract common DataFrame creation pattern used across
    multiple test files to meet 1KB minimum requirement.
    """
    return pl.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(num_records)],
            "age": [25 + (i % 50) for i in range(num_records)],
            "sex": ["M" if i % 2 == 0 else "F" for i in range(num_records)],
            "outcome": [1 if i % 3 == 0 else 0 for i in range(num_records)],
            "medication": [f"Med{i % 5}" for i in range(num_records)],
            "dosage": [100 + (i % 200) for i in range(num_records)],
        }
    )


@pytest.fixture
def large_test_df_pd(num_records: int = 150):
    """
    Create large test pandas DataFrame with patient_id and other columns.

    DRY principle: Extract common pandas DataFrame creation pattern used across
    multiple test files to meet 1KB minimum requirement.
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(num_records)],
            "age": [25 + (i % 50) for i in range(num_records)],
            "sex": ["M" if i % 2 == 0 else "F" for i in range(num_records)],
            "outcome": [1 if i % 3 == 0 else 0 for i in range(num_records)],
            "medication": [f"Med{i % 5}" for i in range(num_records)],
            "dosage": [100 + (i % 200) for i in range(num_records)],
        }
    )


@pytest.fixture
def sample_patient_medication_df() -> pl.DataFrame:
    """
    Create sample patient-medication DataFrame.

    DRY principle: Extract common multi-table test pattern.
    """
    return pl.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003", "P004", "P005"],
            "medication": ["Aspirin", "Lisinopril", "Metformin", "Atorvastatin", "Warfarin"],
            "dosage_mg": [81, 10, 500, 20, 5],
            "start_date": ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15", "2023-03-01"],
        }
    )


@pytest.fixture
def simple_test_df() -> pl.DataFrame:
    """
    Create simple single-row test DataFrame.

    DRY principle: Extract common single-row DataFrame pattern used in tests.
    """
    return pl.DataFrame({"patient_id": ["P001"], "age": [25]})


@pytest.fixture
def dummy_table():
    """
    Create dummy table dict for MultiTableHandler tests.

    DRY principle: Extract common dummy table pattern used in multi-table tests.
    """
    return {"dummy": pl.DataFrame({"id": [1]})}


@pytest.fixture
def make_patient_value_df():
    """
    Factory fixture for creating patient-value DataFrames.

    DRY principle: Single factory for all patient-value DataFrame patterns.
    Eliminates duplicate DataFrame creation code.

    Usage:
        df = make_patient_value_df()  # Default: with some nulls
        df = make_patient_value_df(pattern="empty")  # Empty
        df = make_patient_value_df(pattern="all_nulls")  # All nulls
    """

    def _make(pattern: str = "default") -> pl.DataFrame:
        if pattern == "default":
            return pl.DataFrame(
                {
                    "patient_id": ["P1", "P2", None, "P3", None],
                    "value": [100, 200, 300, 400, 500],
                }
            )
        elif pattern == "empty":
            return pl.DataFrame(
                {
                    "patient_id": pl.Series([], dtype=pl.Utf8),
                    "value": pl.Series([], dtype=pl.Int64),
                }
            )
        elif pattern == "all_nulls":
            return pl.DataFrame({"patient_id": [None, None, None], "value": [100, 200, 300]})
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    return _make


@pytest.fixture
def patient_value_df(make_patient_value_df):
    """
    Create patient-value DataFrame with nulls for testing.

    DRY principle: Uses factory fixture to avoid duplication.
    """
    return make_patient_value_df("default")


@pytest.fixture
def empty_patient_df(make_patient_value_df):
    """
    Create empty patient-value DataFrame for testing.

    DRY principle: Uses factory fixture to avoid duplication.
    """
    return make_patient_value_df("empty")


@pytest.fixture
def all_nulls_patient_df(make_patient_value_df):
    """
    Create patient-value DataFrame with all null patient_ids.

    DRY principle: Uses factory fixture to avoid duplication.
    """
    return make_patient_value_df("all_nulls")


# ============================================================================


# Import performance plugin to register pytest_addoption hook
# The plugin checks --track-performance flag internally
try:
    import performance.plugin  # noqa: F401
except ImportError:
    # Plugin not available, skip
    pass
