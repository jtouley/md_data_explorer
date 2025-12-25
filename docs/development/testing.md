# Testing

## Testing Philosophy

- **Write tests first** (TDD when possible)
- **Test behavior, not implementation**
- **Keep tests fast** (<1s per unit test)
- **Use fixtures** for common test data
- **Mock external dependencies** (APIs, file I/O)

## Test Structure

```
tests/
├── fixtures/                  # Test data
│   ├── test_data.csv
│   └── mimic_demo/
├── test_core/                 # Core module tests
│   ├── test_semantic.py
│   ├── test_registry.py
│   └── test_nl_query_engine.py
├── test_datasets/             # Dataset tests
├── test_ui/                   # UI component tests
└── conftest.py                # Shared fixtures
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core/test_nl_query_engine.py

# Run specific test function
pytest tests/test_core/test_nl_query_engine.py::test_pattern_matching

# Run with coverage
pytest tests/ --cov=src/clinical_analytics --cov-report=html

# Run with verbose output
pytest tests/ -v -s

# Run fast tests only (skip slow integration tests)
pytest tests/ -m "not slow"
```

## Writing Tests

### Unit Test Example

```python
import pytest
from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent


def test_pattern_matching_compare_groups():
    """Test pattern matching for group comparison queries."""
    # Arrange
    engine = NLQueryEngine(semantic_layer=None)  # Mock semantic layer

    # Act
    intent = engine._pattern_match("compare mortality by treatment")

    # Assert
    assert intent is not None
    assert intent.intent_type == "COMPARE_GROUPS"
    assert intent.confidence > 0.9


def test_fuzzy_variable_matching():
    """Test fuzzy matching of variable names."""
    engine = NLQueryEngine(semantic_layer=None)
    engine.semantic_layer = MockSemanticLayer(columns=["age_years", "mortality", "treatment_arm"])

    result = engine._fuzzy_match_variable("age")

    assert result == "age_years"


def test_invalid_query_raises_error():
    """Test that invalid queries raise appropriate errors."""
    engine = NLQueryEngine(semantic_layer=None)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        engine.parse_query("")
```

### Integration Test Example

```python
import pandas as pd
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.core.nl_query_engine import NLQueryEngine


def test_end_to_end_query_execution(sample_dataset):
    """Test full workflow from query to results."""
    # Register dataset
    DatasetRegistry.register_from_dataframe("test_dataset", sample_dataset)

    # Get semantic layer
    semantic_layer = DatasetRegistry.get_semantic_layer("test_dataset")

    # Parse query
    engine = NLQueryEngine(semantic_layer)
    intent = engine.parse_query("compare mortality by treatment")

    # Execute analysis
    results = run_group_comparison(semantic_layer, intent)

    # Verify results
    assert "p_value" in results
    assert "effect_size" in results
    assert isinstance(results["p_value"], float)
```

## Test Fixtures

### conftest.py

```python
import pytest
import pandas as pd


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    return pd.DataFrame({
        'patient_id': range(1, 101),
        'age_years': np.random.randint(18, 90, 100),
        'sex': np.random.choice(['M', 'F'], 100),
        'treatment_arm': np.random.choice(['A', 'B'], 100),
        'mortality': np.random.binomial(1, 0.2, 100)
    })


@pytest.fixture
def semantic_layer(sample_dataset):
    """Create semantic layer for testing."""
    config = {
        'outcomes': {
            'mortality': {'source_column': 'mortality', 'type': 'binary'}
        },
        'column_mapping': {
            'patient_id': 'patient_id'
        }
    }
    return SemanticLayer(sample_dataset, config)


@pytest.fixture
def nl_query_engine(semantic_layer):
    """Create NL query engine for testing."""
    return NLQueryEngine(semantic_layer)
```

### Using Fixtures

```python
def test_with_fixtures(nl_query_engine):
    """Test using shared fixture."""
    intent = nl_query_engine.parse_query("what predicts mortality")

    assert intent.intent_type == "FIND_PREDICTORS"
    assert intent.primary_variable == "mortality"
```

## Mocking

### Mock External APIs

```python
from unittest.mock import Mock, patch


@patch('clinical_analytics.core.nl_query_engine.anthropic.Client')
def test_llm_fallback(mock_client):
    """Test LLM fallback with mocked API."""
    # Configure mock
    mock_response = Mock()
    mock_response.content = [Mock(text='{"intent_type": "DESCRIBE", "confidence": 0.8}')]
    mock_client.return_value.messages.create.return_value = mock_response

    # Test
    engine = NLQueryEngine(semantic_layer=None)
    intent = engine._llm_parse("show me everything")

    assert intent.intent_type == "DESCRIBE"
    assert intent.confidence == 0.8
```

### Mock File I/O

```python
from unittest.mock import mock_open, patch


@patch('builtins.open', mock_open(read_data='patient_id,age,mortality\n1,45,0\n2,67,1'))
@patch('pandas.read_csv')
def test_csv_upload(mock_read_csv, mock_file):
    """Test CSV file upload with mocked file."""
    mock_read_csv.return_value = pd.DataFrame({
        'patient_id': [1, 2],
        'age': [45, 67],
        'mortality': [0, 1]
    })

    df = load_csv_file("test.csv")

    assert len(df) == 2
    assert 'mortality' in df.columns
```

## Parameterized Tests

Test multiple inputs with one test function:

```python
@pytest.mark.parametrize("query,expected_intent", [
    ("compare mortality by treatment", "COMPARE_GROUPS"),
    ("what predicts mortality", "FIND_PREDICTORS"),
    ("survival analysis", "SURVIVAL"),
    ("correlation between age and outcome", "CORRELATIONS"),
    ("descriptive statistics", "DESCRIBE"),
])
def test_intent_classification(query, expected_intent, nl_query_engine):
    """Test intent classification for various queries."""
    intent = nl_query_engine.parse_query(query)
    assert intent.intent_type == expected_intent
```

## Testing Statistical Functions

```python
import numpy as np
from scipy import stats


def test_t_test():
    """Test t-test implementation."""
    group_a = np.array([1, 2, 3, 4, 5])
    group_b = np.array([3, 4, 5, 6, 7])

    result = run_t_test(group_a, group_b)

    # Verify with scipy
    expected = stats.ttest_ind(group_a, group_b)

    assert np.isclose(result['t_statistic'], expected.statistic)
    assert np.isclose(result['p_value'], expected.pvalue)


def test_logistic_regression():
    """Test logistic regression."""
    X = np.random.randn(100, 2)
    y = np.random.binomial(1, 0.5, 100)

    result = run_logistic_regression(X, y)

    assert 'coefficients' in result
    assert 'p_values' in result
    assert 'auc' in result
    assert 0 <= result['auc'] <= 1
```

## Testing UI Components

For Streamlit components, use `streamlit.testing.v1`:

```python
from streamlit.testing.v1 import AppTest


def test_upload_component():
    """Test file upload UI component."""
    at = AppTest.from_file("src/clinical_analytics/ui/components/upload.py")
    at.run()

    # Simulate file upload
    at.file_uploader[0].upload("tests/fixtures/test_data.csv")

    assert at.success[0].value == "File uploaded successfully"
    assert len(at.dataframe) == 1  # Dataset displayed
```

## Testing Semantic Layer

```python
def test_semantic_layer_sql_generation(semantic_layer):
    """Test SQL generation from semantic layer."""
    # Get base view
    view = semantic_layer.get_base_view()

    # Apply filter
    filtered = view.filter(view.age_years > 50)

    # Generate SQL
    sql = semantic_layer.compile_to_sql(filtered)

    assert "WHERE" in sql
    assert "age_years > 50" in sql


def test_outcome_computation(semantic_layer):
    """Test outcome column computation."""
    df = semantic_layer.get_outcome_df('mortality')

    assert 'mortality' in df.columns
    assert df['mortality'].dtype == np.int64
    assert set(df['mortality'].unique()).issubset({0, 1})
```

## Coverage

### Check Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src/clinical_analytics --cov-report=html

# Open report
open htmlcov/index.html
```

### Coverage Goals

- **Unit tests**: >80% coverage
- **Integration tests**: Cover critical paths
- **UI components**: Test core functionality

### Coverage Configuration

In `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src/clinical_analytics"]
omit = [
    "tests/*",
    "src/clinical_analytics/ui/app.py",  # Streamlit app entry point
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

## Continuous Integration

Tests run automatically on GitHub Actions:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ --cov
```

## Best Practices

1. **Arrange-Act-Assert**: Structure tests clearly
2. **One assertion per test**: Tests should be focused
3. **Descriptive names**: `test_pattern_matching_compare_groups` not `test1`
4. **Use fixtures**: Share setup code
5. **Mock external dependencies**: Keep tests fast and reliable
6. **Test edge cases**: Empty input, missing data, invalid types
7. **Test error handling**: Verify exceptions are raised correctly

## Next Steps

- Review [Contributing Guidelines](contributing.md)
- Set up [Development Environment](setup.md)
- Explore the [Architecture](../architecture/overview.md)
