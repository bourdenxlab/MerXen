"""Tests for the hook-H8 fields of merxen.config (plan §3.7, §4.9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from merxen.annotation.config import (
    DEFAULT_CLUSTERING_MODE,
    LEGACY_UNASSIGNED_STATE_POLICY,
    MAP_FIRST_TABLE_KEY_SUFFIX,
    AdaptiveSplitConfig,
    AnnotationConfig,
    AnnotationThresholds,
    resolve_clustering_mode,
    resolve_table_key_suffix,
)
from merxen.config import (
    ClusteringSquidpyConfig,
    ClusteringSquidpySampleConfig,
    MenderConfig,
)

H8_CLUSTERING_FIELDS = [
    "mode",
    "labels_dir",
    "leaf_source",
    "adaptive_split",
    "qc_leiden_resolution",
    "table_key_suffix",
]


def _clustering(**overrides: Any) -> ClusteringSquidpyConfig:
    data: dict[str, Any] = {"pair_id": "P1", "output_dir": "out", "samples": []}
    data.update(overrides)
    return ClusteringSquidpyConfig.model_validate(data)


def _legacy_stage_json(tmp_path: Path) -> dict[str, Any]:
    """A config in the shape the legacy PREPARE heredoc writes (no H8 keys)."""
    return {
        "pair_id": "P1",
        "output_dir": str(tmp_path),
        "samples": [
            {
                "sample_id": "P1_XENIUM",
                "platform": "XENIUM",
                "zarr_path": str(tmp_path / "x.zarr"),
                "segmentation": "reseg",
                "table_key": "table_MOSAIK_proseg",
                "shape_key": "MOSAIK_proseg",
            }
        ],
        "min_counts": 10,
        "use_gpu": True,
        "hierarchical_enabled": True,
        "broad_round": {"leiden_resolution": 0.2},
    }


def _mender(**overrides: Any) -> MenderConfig:
    data: dict[str, Any] = {
        "pair_id": "P1",
        "sample_id": "P1_XENIUM",
        "platform": "XENIUM",
        "segmentation": "reseg",
        "source_h5ad": "clustered.h5ad",
        "spatialdata_path": "x.zarr",
        "source_spatialdata_table": "table_MOSAIK_proseg_clustering_squidpy",
        "native_shape_key": "MOSAIK_proseg",
        "output_dir": "mender_out",
    }
    data.update(overrides)
    return MenderConfig.model_validate(data)


def test_clustering_config_h8_defaults_are_legacy() -> None:
    config = _clustering()

    assert list(ClusteringSquidpyConfig.model_fields)[-6:] == H8_CLUSTERING_FIELDS
    assert config.mode == "legacy"
    assert config.labels_dir is None
    assert config.leaf_source == "mapped"
    assert config.adaptive_split == AdaptiveSplitConfig()
    assert config.adaptive_split.rule == "none"
    assert config.qc_leiden_resolution == 0.5
    assert config.table_key_suffix == ""
    assert {DEFAULT_CLUSTERING_MODE[s] for s in ("human", "mouse")} == {config.mode}


def test_legacy_stage_json_loads_unchanged(tmp_path: Path) -> None:
    """The JSON the legacy heredoc writes parses to the legacy defaults."""
    path = tmp_path / "clustering_squidpy_config.json"
    path.write_text(json.dumps(_legacy_stage_json(tmp_path)))

    config = ClusteringSquidpyConfig.model_validate_json(path.read_text())

    assert config.mode == "legacy"
    assert config.table_key_suffix == ""
    sample = config.samples[0]
    assert sample.anatomical_region is None
    assert sample.mouse_section_regions is None
    assert config.min_counts == 10


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_resolved_defaults_give_a_valid_legacy_config(species: str) -> None:
    """Nextflow's default mode and suffix for either species validate."""
    mode = resolve_clustering_mode(species)  # type: ignore[arg-type]
    suffix = resolve_table_key_suffix(species, mode)  # type: ignore[arg-type]

    config = _clustering(mode=mode, table_key_suffix=suffix)

    assert (config.mode, config.table_key_suffix) == ("legacy", "")


def test_map_first_fields_round_trip_through_json() -> None:
    config = _clustering(
        mode="map_first",
        labels_dir="annotation_out",
        qc_leiden_resolution=0.8,
        table_key_suffix=MAP_FIRST_TABLE_KEY_SUFFIX,
    )

    restored = ClusteringSquidpyConfig.model_validate_json(config.model_dump_json())

    assert restored == config
    assert restored.labels_dir == Path("annotation_out")
    assert restored.table_key_suffix == "mapfirst"


def test_table_key_suffix_is_stripped_and_checked() -> None:
    config = _clustering(mode="map_first", table_key_suffix=" trial_2 ")
    assert config.table_key_suffix == "trial_2"
    with pytest.raises(ValidationError, match="lower-case token"):
        _clustering(mode="map_first", table_key_suffix="Map/First")


def test_legacy_mode_rejects_a_table_key_suffix() -> None:
    """Legacy runs always write the unsuffixed clustered table (plan §4.8)."""
    with pytest.raises(ValidationError, match="map_first runs only"):
        _clustering(table_key_suffix="mapfirst")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mode": "denovo"}, "mode"),
        ({"leaf_source": "clusters"}, "leaf_source"),
        ({"mode": "map_first", "leaf_source": "denovo"}, "adaptive_split rule"),
        ({"adaptive_split": {"rule": "count_split_merge"}}, "adaptive_split"),
        ({"adaptive_split": {"rule": "none", "tau": 0.9}}, "Extra inputs"),
        ({"qc_leiden_resolution": 0.0}, "qc_leiden_resolution"),
    ],
)
def test_clustering_mode_fields_reject_invalid_values(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _clustering(**overrides)


def test_sample_config_annotation_columns() -> None:
    base = {"sample_id": "S", "platform": "MERSCOPE", "zarr_path": "s.zarr"}
    fields = list(ClusteringSquidpySampleConfig.model_fields)

    assert fields[-2:] == ["anatomical_region", "mouse_section_regions"]
    blank = ClusteringSquidpySampleConfig.model_validate(
        {**base, "anatomical_region": " ", "mouse_section_regions": ""}
    )
    assert (blank.anatomical_region, blank.mouse_section_regions) == (None, None)
    parsed = ClusteringSquidpySampleConfig.model_validate(
        {
            **base,
            "anatomical_region": "Frontal cortex",
            "mouse_section_regions": "isocortex; hpf",
        }
    )
    assert parsed.anatomical_region == "frontal_cortex"
    assert parsed.mouse_section_regions == "Isocortex;HPF"
    assert (
        ClusteringSquidpySampleConfig.model_validate(
            {**base, "mouse_section_regions": "AUTO"}
        ).mouse_section_regions
        == "auto"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("anatomical_region", "frontal/cortex", "invalid anatomical_region"),
        ("mouse_section_regions", "Isocortex;Striatum", "unknown mouse section"),
    ],
)
def test_sample_config_rejects_invalid_annotation_columns(
    field: str, value: str, message: str
) -> None:
    base = {"sample_id": "S", "platform": "MERSCOPE", "zarr_path": "s.zarr"}
    with pytest.raises(ValidationError, match=message):
        ClusteringSquidpySampleConfig.model_validate({**base, field: value})


def test_mender_unassigned_state_policy() -> None:
    assert list(MenderConfig.model_fields)[-1] == "unassigned_state_policy"
    assert _mender().unassigned_state_policy == LEGACY_UNASSIGNED_STATE_POLICY
    assert _mender().unassigned_state_policy == "state"
    policy = "exclude_from_features"
    assert _mender(unassigned_state_policy=policy).unassigned_state_policy == policy
    with pytest.raises(ValidationError, match="unassigned_state_policy"):
        _mender(unassigned_state_policy="drop")


def test_hard_min_counts_follows_the_clustering_min_counts() -> None:
    clustering = _clustering(min_counts=15)

    uncoupled = AnnotationConfig()
    coupled = clustering.coupled_annotation_config(uncoupled)

    assert coupled.min_counts == coupled.thresholds.hard_min_counts == 15
    assert coupled.require_min_counts() == 15
    with pytest.raises(ValueError, match="not coupled to the clustering run"):
        uncoupled.require_min_counts()
    default_run = _clustering().coupled_annotation_config(AnnotationConfig())
    assert default_run.min_counts == default_run.thresholds.hard_min_counts == 10
    explicit_equal = AnnotationConfig(
        min_counts=15, thresholds=AnnotationThresholds(hard_min_counts=15)
    )
    assert clustering.coupled_annotation_config(explicit_equal).min_counts == 15


def test_hard_min_counts_coupling_rejects_a_different_threshold() -> None:
    """An annotation config naming another min_counts cannot be coupled."""
    with pytest.raises(ValueError, match="must equal the clustering min_counts"):
        _clustering(min_counts=15).coupled_annotation_config(
            AnnotationConfig(min_counts=10)
        )
