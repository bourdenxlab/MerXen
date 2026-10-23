"""Species, anatomical scope, and custom-reference configuration contracts."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from merxen.config import PciSeqReferenceConfig


@pytest.mark.parametrize("species,source", [("mouse", "wmb"), ("human", "whb")])
def test_auto_reference_uses_the_configured_species(species: str, source: str) -> None:
    config = PciSeqReferenceConfig.model_validate(
        {"species": species, "region_mode": "whole_brain"}
    )
    assert config.source == source


def test_reference_requires_explicit_anatomical_scope() -> None:
    with pytest.raises(ValidationError, match="region_labels"):
        PciSeqReferenceConfig()


def test_mouse_default_uses_whole_brain_until_ccfv3_registration_is_available() -> None:
    config = PciSeqReferenceConfig(species="mouse")
    assert config.region_mode == "whole_brain"
    assert config.region_labels == []


def test_mouse_manual_regions_override_the_automatic_whole_brain_default() -> None:
    config = PciSeqReferenceConfig(species="mouse", region_labels=["MOp"])
    assert config.region_mode == "required"


def test_reference_regions_are_canonicalized_without_losing_case() -> None:
    config = PciSeqReferenceConfig(
        species="mouse", region_labels='["MOp", "MOs", "MOp", " "]'
    )
    assert config.region_labels == ["MOp", "MOs"]


def test_custom_prebuilt_matrix_requires_membership_for_regional_filtering() -> None:
    with pytest.raises(ValidationError, match="custom_metadata_path"):
        PciSeqReferenceConfig(reference_path="custom.csv", region_labels=["frontal"])
    config = PciSeqReferenceConfig(
        reference_path="custom.csv",
        region_labels=["frontal"],
        custom_metadata_path="class_regions.csv",
    )
    assert config.source == "custom"


def test_custom_prebuilt_matrix_can_explicitly_use_all_supplied_types() -> None:
    config = PciSeqReferenceConfig(
        reference_path="custom.csv", region_mode="whole_brain"
    )
    assert config.source == "custom"


@pytest.mark.parametrize("source", ["SEA-AD", "sea_ad", "SEAAD"])
def test_seaad_source_aliases_use_one_cache_identity(source: str) -> None:
    config = PciSeqReferenceConfig.model_validate(
        {"source": source, "region_labels": ["DFC"]}
    )
    assert config.source == "seaad"


@pytest.mark.parametrize(
    "fields",
    [
        {"species": "human", "source": "wmb"},
        {"species": "mouse", "source": "whb"},
        {"species": "mouse", "source": "seaad"},
        {"source": "custom"},
        {"source": "whb", "reference_path": "custom.csv"},
        {"source": "whb", "seaad_cohort": "ad"},
        {"normalization_target_sum": float("nan")},
        {"normalization_target_sum": 0},
        {"min_cells_per_type": 0},
        {"chunk_rows": 0},
        {"region_mode": "whole_brain", "region_labels": ["DFC"]},
    ],
)
def test_reference_rejects_incompatible_or_ambiguous_settings(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        PciSeqReferenceConfig.model_validate({"region_mode": "whole_brain"} | fields)
