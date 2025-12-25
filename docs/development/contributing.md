# Contributing

## Getting Started

Thank you for contributing to the Clinical Analytics Platform! This guide will help you get set up.

### Prerequisites

- Python 3.10+
- Git
- uv package manager (recommended) or pip

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/md_data_explorer.git
cd md_data_explorer

# Create a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Install pre-commit hooks (if available)
pre-commit install
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run src/clinical_analytics/ui/app.py

# Run tests
pytest tests/

# Run type checking
mypy src/

# Format code
black src/ tests/
isort src/ tests/
```

## Project Structure

```
src/clinical_analytics/
├── core/              # Core business logic
│   ├── semantic.py    # Semantic layer (Ibis)
│   ├── registry.py    # Dataset registry
│   └── nl_query_engine.py  # NL query parsing
├── datasets/          # Built-in datasets
├── ui/                # Streamlit UI components
│   ├── app.py         # Main application
│   └── components/    # Reusable UI components
└── utils/             # Utility functions

tests/                 # Test suite
docs/                  # Documentation (MkDocs)
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the coding standards (see below).

### 3. Write Tests

Add tests for new functionality in `tests/`:

```python
def test_nl_query_parsing():
    engine = NLQueryEngine(semantic_layer)
    intent = engine.parse_query("compare mortality by treatment")

    assert intent.intent_type == "COMPARE_GROUPS"
    assert intent.primary_variable == "mortality"
    assert intent.grouping_variable == "treatment"
```

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Commit Changes

Follow conventional commit format:

```bash
git commit -m "feat(nl-query): add pattern matching for survival queries"
```

**Commit Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Coding Standards

### Python Style

Follow PEP 8 with these conventions:

- **Line length**: 100 characters (not 79)
- **Imports**: Absolute imports, organized by standard lib → third-party → local
- **Type hints**: Use for all function signatures
- **Docstrings**: Google-style for all public classes/methods

### Example

```python
from typing import Optional, List
import pandas as pd
from clinical_analytics.core.semantic import SemanticLayer


def parse_query(query: str, semantic_layer: SemanticLayer) -> Optional[QueryIntent]:
    """
    Parse natural language query into structured intent.

    Args:
        query: User's question in plain English
        semantic_layer: Semantic layer for metadata access

    Returns:
        QueryIntent if parsing successful, None otherwise

    Raises:
        ValueError: If query is empty or invalid

    Example:
        >>> intent = parse_query("compare survival by treatment", sl)
        >>> print(intent.intent_type)
        COMPARE_GROUPS
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    # Implementation...
    return intent
```

### UI Components

For Streamlit components:

- Use `st.cache_data` for expensive computations
- Add `help` text for all inputs
- Provide clear error messages
- Show progress indicators for long operations

## Testing Guidelines

### Unit Tests

Test individual functions and classes:

```python
def test_fuzzy_variable_matching():
    columns = ["age_years", "mortality", "treatment_arm"]
    result = fuzzy_match_variable("age", columns)
    assert result == "age_years"
```

### Integration Tests

Test component interactions:

```python
def test_end_to_end_analysis():
    # Upload dataset
    dataset = upload_csv("tests/fixtures/test_data.csv")

    # Parse query
    intent = parse_query("compare mortality by treatment")

    # Execute analysis
    results = run_analysis(dataset, intent)

    # Verify results
    assert "p_value" in results
    assert results["p_value"] < 0.05
```

### Test Fixtures

Place test data in `tests/fixtures/`:

```
tests/fixtures/
├── test_data.csv
├── mimic_demo/
│   ├── patients.csv
│   └── admissions.csv
└── expected_results.json
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def compute_odds_ratio(a: int, b: int, c: int, d: int) -> float:
    """
    Compute odds ratio from 2x2 contingency table.

    Args:
        a: Exposed cases
        b: Exposed non-cases
        c: Unexposed cases
        d: Unexposed non-cases

    Returns:
        Odds ratio (OR)

    Example:
        >>> compute_odds_ratio(10, 20, 5, 30)
        3.0
    """
    return (a * d) / (b * c)
```

### Updating Docs

Edit markdown files in `docs/`:

```bash
# Edit documentation
vim docs/user-guide/new-feature.md

# Preview changes
mkdocs serve

# Build documentation
mkdocs build
```

## Pull Request Process

### PR Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Code formatted (`black` and `isort`)
- [ ] Type hints added (`mypy src/`)
- [ ] Docstrings updated
- [ ] Documentation updated (if user-facing change)
- [ ] CHANGELOG.md updated (for releases)

### PR Template

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

Describe testing performed.

## Screenshots (if applicable)

Add screenshots for UI changes.
```

### Review Process

1. Automated checks run (tests, linting)
2. Maintainer reviews code
3. Feedback addressed
4. PR approved and merged

## Reporting Issues

### Bug Reports

Include:

1. Python version and OS
2. Steps to reproduce
3. Expected vs actual behavior
4. Error messages and stack trace
5. Sample data (if possible)

### Feature Requests

Include:

1. Use case / problem to solve
2. Proposed solution
3. Alternative approaches considered
4. Example usage

## Code of Conduct

Be respectful and constructive. We're all here to improve clinical data analysis.

## Questions?

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas

## Next Steps

- Review [Development Setup](setup.md)
- Learn about [Testing](testing.md)
- Explore the [Architecture](../architecture/overview.md)
