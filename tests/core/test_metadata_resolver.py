"""
Tests for MetadataResolver - merge precedence golden tests.

Phase 0: ADR011 Metadata Enrichment
Golden tests for deterministic merge resolution with precedence rules.
"""

from datetime import UTC, datetime


class TestMergePrecedence:
    """Golden tests for merge precedence rules."""

    def test_base_only_returns_column_names(self):
        """
        Test that base schema (InferredSchema) provides column names only.

        Merge precedence rule 1: Base provides column names from schema.
        When no patches or dictionary metadata exist, resolved metadata
        should contain only the column names.
        """
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import InferredSchema

        # Arrange: InferredSchema with no dictionary metadata
        schema = InferredSchema(
            patient_id_column="patient_id",
            outcome_columns=["mortality"],
            categorical_columns=["sex", "treatment_arm"],
            continuous_columns=["age", "bmi"],
        )
        patches = []  # No patches

        # Act: Resolve metadata
        resolved = resolve_metadata(schema, patches)

        # Assert: All columns present with names only
        assert "patient_id" in resolved.columns
        assert "mortality" in resolved.columns
        assert "sex" in resolved.columns
        assert "age" in resolved.columns
        assert "bmi" in resolved.columns

        # No descriptions or labels (only base column names)
        assert resolved.columns["age"].label is None
        assert resolved.columns["age"].description is None

    def test_inferred_overrides_base(self):
        """
        Test that DictionaryMetadata overrides base column names.

        Merge precedence rule 2: Inferred (from DictionaryMetadata) overrides base.
        """
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import (
            DictionaryMetadata,
            InferredSchema,
        )

        # Arrange: Schema with dictionary metadata
        dict_metadata = DictionaryMetadata(
            column_descriptions={
                "age": "Patient age at enrollment in years",
                "bmi": "Body Mass Index (kg/m²)",
            },
        )
        schema = InferredSchema(
            patient_id_column="patient_id",
            continuous_columns=["age", "bmi"],
            dictionary_metadata=dict_metadata,
        )
        patches = []  # No patches

        # Act: Resolve metadata
        resolved = resolve_metadata(schema, patches)

        # Assert: Dictionary descriptions are used
        assert resolved.columns["age"].description == "Patient age at enrollment in years"
        assert resolved.columns["bmi"].description == "Body Mass Index (kg/m²)"

    def test_accepted_patches_override_inferred(self):
        """
        Test that accepted patches override inferred metadata.

        Merge precedence rule 3: Accepted patches override inferred.
        """
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import (
            DictionaryMetadata,
            InferredSchema,
        )

        # Arrange: Schema with dictionary + accepted patch
        dict_metadata = DictionaryMetadata(
            column_descriptions={"age": "Patient age"},
        )
        schema = InferredSchema(
            continuous_columns=["age"],
            dictionary_metadata=dict_metadata,
        )

        patch = MetadataPatch(
            patch_id="patch-1",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age at enrollment (years), validated against medical records",
            status=PatchStatus.ACCEPTED,
            created_at=datetime.now(UTC),
            provenance="user",
        )
        patches = [patch]

        # Act: Resolve metadata
        resolved = resolve_metadata(schema, patches)

        # Assert: Patch description overrides dictionary
        assert "validated against medical records" in resolved.columns["age"].description

    def test_later_patches_override_earlier(self):
        """
        Test that later patches in chronological order override earlier patches.

        Merge precedence rule 4: Later patches override earlier patches.
        """
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import InferredSchema

        # Arrange: Two patches for same column, different times
        early_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        late_time = datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC)

        schema = InferredSchema(continuous_columns=["age"])

        patch_early = MetadataPatch(
            patch_id="patch-early",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Early description",
            status=PatchStatus.ACCEPTED,
            created_at=early_time,
            provenance="user",
        )

        patch_late = MetadataPatch(
            patch_id="patch-late",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Late description (final)",
            status=PatchStatus.ACCEPTED,
            created_at=late_time,
            provenance="user",
        )

        # Pass in wrong order to verify sorting
        patches = [patch_late, patch_early]

        # Act: Resolve metadata
        resolved = resolve_metadata(schema, patches)

        # Assert: Later patch wins
        assert resolved.columns["age"].description == "Late description (final)"

    def test_rejected_patches_excluded(self):
        """
        Test that rejected patches are not applied.

        Merge precedence rule 5: Rejected patches are excluded.
        """
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import InferredSchema

        # Arrange: One rejected patch
        schema = InferredSchema(continuous_columns=["bmi"])

        patch = MetadataPatch(
            patch_id="patch-rejected",
            operation=PatchOperation.SET_DESCRIPTION,
            column="bmi",
            value="This description should NOT appear",
            status=PatchStatus.REJECTED,
            created_at=datetime.now(UTC),
            provenance="llm",
        )
        patches = [patch]

        # Act: Resolve metadata
        resolved = resolve_metadata(schema, patches)

        # Assert: Rejected patch not applied
        assert resolved.columns["bmi"].description is None

    def test_pending_patches_excluded(self):
        """
        Test that pending patches are not applied.

        Merge precedence rule 6: Pending patches are excluded (require explicit acceptance).
        """
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import InferredSchema

        # Arrange: One pending patch
        schema = InferredSchema(categorical_columns=["sex"])

        patch = MetadataPatch(
            patch_id="patch-pending",
            operation=PatchOperation.SET_DESCRIPTION,
            column="sex",
            value="Pending description - should NOT appear",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )
        patches = [patch]

        # Act: Resolve metadata
        resolved = resolve_metadata(schema, patches)

        # Assert: Pending patch not applied
        assert resolved.columns["sex"].description is None

    def test_deterministic_output_same_inputs_same_output(self):
        """
        Test that same inputs always produce identical output.

        Merge precedence rule 7: Resolution is deterministic.
        """
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import (
            DictionaryMetadata,
            InferredSchema,
        )

        # Arrange: Same schema and patches
        dict_metadata = DictionaryMetadata(
            column_descriptions={"age": "Base description"},
        )
        schema = InferredSchema(
            continuous_columns=["age", "bmi"],
            dictionary_metadata=dict_metadata,
        )

        patches = [
            MetadataPatch(
                patch_id="p1",
                operation=PatchOperation.SET_DESCRIPTION,
                column="age",
                value="Updated description",
                status=PatchStatus.ACCEPTED,
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
                provenance="user",
            ),
            MetadataPatch(
                patch_id="p2",
                operation=PatchOperation.SET_LABEL,
                column="bmi",
                value="BMI",
                status=PatchStatus.ACCEPTED,
                created_at=datetime(2024, 1, 2, tzinfo=UTC),
                provenance="user",
            ),
        ]

        # Act: Resolve twice
        resolved1 = resolve_metadata(schema, patches)
        resolved2 = resolve_metadata(schema, patches)

        # Assert: Identical output
        assert resolved1.columns["age"].description == resolved2.columns["age"].description
        assert resolved1.columns["bmi"].label == resolved2.columns["bmi"].label


class TestResolvedMetadataStructure:
    """Tests for ResolvedDatasetMetadata structure."""

    def test_resolved_column_metadata_structure(self):
        """Test ResolvedColumnMetadata has expected fields."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
            SemanticType,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import InferredSchema

        # Arrange: Schema with patches for various operations
        schema = InferredSchema(continuous_columns=["hba1c"])

        patches = [
            MetadataPatch(
                patch_id="p1",
                operation=PatchOperation.SET_LABEL,
                column="hba1c",
                value="HbA1c",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="user",
            ),
            MetadataPatch(
                patch_id="p2",
                operation=PatchOperation.SET_DESCRIPTION,
                column="hba1c",
                value="Hemoglobin A1c percentage",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="user",
            ),
            MetadataPatch(
                patch_id="p3",
                operation=PatchOperation.SET_SEMANTIC_TYPE,
                column="hba1c",
                value="measurement",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="llm",
            ),
            MetadataPatch(
                patch_id="p4",
                operation=PatchOperation.SET_UNIT,
                column="hba1c",
                value="%",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="user",
            ),
        ]

        # Act
        resolved = resolve_metadata(schema, patches)

        # Assert: All fields populated
        col_meta = resolved.columns["hba1c"]
        assert col_meta.label == "HbA1c"
        assert col_meta.description == "Hemoglobin A1c percentage"
        assert col_meta.semantic_type == SemanticType.MEASUREMENT
        assert col_meta.unit == "%"

    def test_resolved_metadata_includes_codebook(self):
        """Test that codebook entries are resolved into column metadata."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import InferredSchema

        # Arrange
        schema = InferredSchema(categorical_columns=["statin_used"])

        patches = [
            MetadataPatch(
                patch_id="cb1",
                operation=PatchOperation.SET_CODEBOOK_ENTRY,
                column="statin_used",
                value={"code": "0", "label": "n/a"},
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="llm",
            ),
            MetadataPatch(
                patch_id="cb2",
                operation=PatchOperation.SET_CODEBOOK_ENTRY,
                column="statin_used",
                value={"code": "1", "label": "Atorvastatin"},
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="llm",
            ),
            MetadataPatch(
                patch_id="cb3",
                operation=PatchOperation.SET_CODEBOOK_ENTRY,
                column="statin_used",
                value={"code": "2", "label": "Simvastatin"},
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="llm",
            ),
        ]

        # Act
        resolved = resolve_metadata(schema, patches)

        # Assert: Codebook entries aggregated
        codebook = resolved.columns["statin_used"].codebook
        assert codebook is not None
        assert codebook["0"] == "n/a"
        assert codebook["1"] == "Atorvastatin"
        assert codebook["2"] == "Simvastatin"

    def test_resolved_metadata_includes_aliases(self):
        """Test that multiple aliases are accumulated."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import InferredSchema

        # Arrange
        schema = InferredSchema(continuous_columns=["patient_id"])

        patches = [
            MetadataPatch(
                patch_id="a1",
                operation=PatchOperation.ADD_ALIAS,
                column="patient_id",
                value="subject_id",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="user",
            ),
            MetadataPatch(
                patch_id="a2",
                operation=PatchOperation.ADD_ALIAS,
                column="patient_id",
                value="participant_id",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="user",
            ),
        ]

        # Act
        resolved = resolve_metadata(schema, patches)

        # Assert: Aliases accumulated
        aliases = resolved.columns["patient_id"].aliases
        assert "subject_id" in aliases
        assert "participant_id" in aliases

    def test_resolved_metadata_includes_exclusion_patterns(self):
        """Test that exclusion patterns are resolved into column metadata."""
        from clinical_analytics.core.metadata_patch import (
            ExclusionPatternPatch,
            PatchStatus,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import InferredSchema

        # Arrange
        schema = InferredSchema(categorical_columns=["statin_used"])

        patch = ExclusionPatternPatch(
            patch_id="excl-1",
            column="statin_used",
            pattern="n/a",
            coded_value=0,
            context="Use != 0 to exclude patients not on statins",
            auto_apply=False,
            status=PatchStatus.ACCEPTED,
            created_at=datetime.now(UTC),
            provenance="llm",
        )
        patches = [patch]

        # Act
        resolved = resolve_metadata(schema, patches)

        # Assert: Exclusion pattern present
        exclusions = resolved.columns["statin_used"].exclusion_patterns
        assert exclusions is not None
        assert len(exclusions) >= 1
        assert exclusions[0].pattern == "n/a"
        assert exclusions[0].coded_value == 0

    def test_resolved_metadata_includes_relationships(self):
        """Test that cross-column relationships are resolved."""
        from clinical_analytics.core.metadata_patch import (
            PatchStatus,
            RelationshipPatch,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import InferredSchema

        # Arrange
        schema = InferredSchema(categorical_columns=["statin_used", "statin_prescribed"])

        patch = RelationshipPatch(
            patch_id="rel-1",
            columns=["statin_used", "statin_prescribed"],
            relationship_type="coded_exclusion",
            rule="Statin Used = 0 implies not prescribed",
            inference="Cross-column filter hint",
            confidence=0.9,
            status=PatchStatus.ACCEPTED,
            created_at=datetime.now(UTC),
            provenance="llm",
        )
        patches = [patch]

        # Act
        resolved = resolve_metadata(schema, patches)

        # Assert: Relationship present
        assert resolved.relationships is not None
        assert len(resolved.relationships) >= 1
        assert resolved.relationships[0].columns == ["statin_used", "statin_prescribed"]


class TestResolverPerformance:
    """Performance tests for resolver (< 100ms per spec)."""

    def test_resolver_performance_under_100ms(self):
        """Test that resolver completes in under 100ms for reasonable input sizes."""
        import time

        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )
        from clinical_analytics.core.metadata_resolver import resolve_metadata
        from clinical_analytics.core.schema_inference import InferredSchema

        # Arrange: Schema with 50 columns and 100 patches
        columns = [f"col_{i}" for i in range(50)]
        schema = InferredSchema(continuous_columns=columns)

        patches = []
        for i, col in enumerate(columns):
            patches.append(
                MetadataPatch(
                    patch_id=f"p{i}",
                    operation=PatchOperation.SET_DESCRIPTION,
                    column=col,
                    value=f"Description for {col}",
                    status=PatchStatus.ACCEPTED,
                    created_at=datetime.now(UTC),
                    provenance="user",
                )
            )
            patches.append(
                MetadataPatch(
                    patch_id=f"p{i}b",
                    operation=PatchOperation.SET_LABEL,
                    column=col,
                    value=f"Label {i}",
                    status=PatchStatus.ACCEPTED,
                    created_at=datetime.now(UTC),
                    provenance="user",
                )
            )

        # Act: Time the resolution
        start = time.perf_counter()
        resolved = resolve_metadata(schema, patches)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Assert: Under 100ms
        assert elapsed_ms < 100, f"Resolver took {elapsed_ms:.2f}ms, expected < 100ms"
        assert len(resolved.columns) == 50
