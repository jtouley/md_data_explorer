"""The population-estimates declaration and saved dimension choices (ADR014)."""

from __future__ import annotations

import pytest

from clinical_analytics.contracts import config
from clinical_analytics.contracts.declarations import (
    Declarations,
    DimensionChoice,
    FileDeclarationStore,
    PopulationEstimatesDeclaration,
)
from clinical_analytics.contracts.service import population_estimates_properties

DECLARED = PopulationEstimatesDeclaration(statement=config.POPULATION_ESTIMATES_STATEMENT)


class TestDeclarations:
    def test_forVersion_keepsDatasetWideDeclarationAndOwnChoices(self):
        # Arrange
        declarations = (
            Declarations()
            .with_population_estimates(DECLARED)
            .with_choice(DimensionChoice(column="sex", value="Both", dataset_version="v1"))
            .with_choice(DimensionChoice(column="sex", value="Male", dataset_version="v2"))
        )

        # Act
        v1 = declarations.for_version("v1")

        # Assert
        assert v1.population_estimates == DECLARED
        assert v1.choices_for("v1") == {"sex": "Both"}

    def test_withChoice_sameColumnTwice_replacedNotDuplicated(self):
        # Arrange
        first = DimensionChoice(column="sex", value="Both", dataset_version="v1")
        second = DimensionChoice(column="sex", value="Female", dataset_version="v1")

        # Act
        declarations = Declarations().with_choice(first).with_choice(second)

        # Assert
        assert declarations.choices == (second,)

    def test_fileStore_roundTrip_declarationAndYearChoice(self, tmp_path):
        # Arrange
        store = FileDeclarationStore(tmp_path)
        choice = DimensionChoice(column="year_id", value=2014, dataset_version="v1")

        # Act
        store.update("ds1", lambda d: d.with_population_estimates(DECLARED).with_choice(choice))
        loaded = store.load("ds1")

        # Assert
        assert loaded.population_estimates == DECLARED
        assert loaded.choices_for("v1") == {"year_id": 2014}


class TestContractProperties:
    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            pytest.param({"population_estimates": True}, config.POPULATION_ESTIMATES_PRIVACY_CLASS, id="declared"),
            pytest.param({"population_estimates": "yes"}, None, id="not_a_bool"),
            pytest.param({}, None, id="absent"),
        ],
    )
    def test_populationEstimatesProperties_onlyWhenDeclared(self, metadata, expected):
        # Act
        properties = population_estimates_properties(metadata)

        # Assert
        assert properties.get("privacyClass") == expected
