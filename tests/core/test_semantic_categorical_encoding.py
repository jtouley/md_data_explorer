"""SemanticLayer._detect_categorical_encoding samples low-cardinality columns."""

import pytest


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1: Yes", "2: No", "1: Yes", "2: No"], True),
        ([0, 1, 2, 1], True),
        ([1, 2, None, 1], True),
        ([100, 200, 300, 100], False),
        (["north", "south", "north", "east"], False),
        (list(range(30)), False),
    ],
    ids=["encoded-labels", "small-int-codes", "codes-with-null", "large-ints", "plain-strings", "high-cardinality"],
)
def test_detectCategoricalEncoding_sampledValues_classifiesColumn(make_semantic_layer, values, expected) -> None:
    # Arrange
    n = len(values)
    layer = make_semantic_layer(data={"patient_id": list(range(n)), "measure": values})

    # Act
    result = layer._detect_categorical_encoding("measure")

    # Assert
    assert result is expected
