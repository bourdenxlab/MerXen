"""Tests for the pipeline configuration models.

Covers the hook-H8 annotation fields of merxen.config (plan §3.7, §4.9) and
the MERSCOPE bead-removal build config.
"""

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
    MerscopeBeadRemovalConfig,
    MerscopeBuildConfig,
    SpatialDataBuildConfig,
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
    # The stage config defaults to legacy (the legacy heredoc writes no mode);
    # map_first rows set it. Both species default to map_first since their
    # flips (human M8, mouse M9), so a row's mode always comes from the row
    # settings, never from this field's default.
    assert set(DEFAULT_CLUSTERING_MODE.values()) == {"map_first"}


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
def test_resolved_defaults_give_a_valid_config(species: str) -> None:
    """Nextflow's default mode and suffix for either species validate.

    Both species run map_first, unsuffixed, since their flips (human §20,
    mouse §25).
    """
    mode = resolve_clustering_mode(species)  # type: ignore[arg-type]
    suffix = resolve_table_key_suffix(species, mode)  # type: ignore[arg-type]

    config = _clustering(mode=mode, table_key_suffix=suffix)

    assert (config.mode, config.table_key_suffix) == ("map_first", "")


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


# MERSCOPE bead-removal build config


def test_merscope_build_config_has_no_bead_removal_by_default() -> None:
    """Existing build configs must validate unchanged, without the new key."""
    config = MerscopeBuildConfig()

    assert config.bead_removal is None


def test_bead_removal_defaults_apply_when_only_enabled_is_given() -> None:
    config = MerscopeBuildConfig.model_validate({"bead_removal": {"enabled": True}})

    bead = config.bead_removal
    assert bead is not None
    assert bead.enabled is True
    assert bead.method == "psf_subtract"
    assert bead.stains == ["PolyT", "DAPI"]
    assert bead.reference_stain == "PolyT"
    assert bead.detection_plane == 0
    assert bead.halo_px == 256
    assert bead.max_workers == 5
    assert bead.calibration_tiles == 6
    assert bead.calibration_tile_px == 2048
    assert bead.min_offtissue_distance_px == 500
    assert bead.min_ring_prominence == pytest.approx(0.010)
    # The fill is on by default, in hybrid mode.
    assert bead.fill is True
    assert bead.fill_mode == "hybrid"
    assert bead.device == "cpu"


def test_bead_removal_is_disabled_by_default_when_constructed_directly() -> None:
    assert MerscopeBeadRemovalConfig().enabled is False


def test_bead_removal_rejects_unknown_nested_key() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MerscopeBuildConfig.model_validate(
            {"bead_removal": {"enabled": True, "halo": 300}}
        )


def test_bead_removal_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        MerscopeBeadRemovalConfig.model_validate({"method": "median_filter"})


def test_bead_removal_rejects_reference_stain_outside_stains() -> None:
    with pytest.raises(ValidationError, match="reference_stain"):
        MerscopeBeadRemovalConfig(stains=["DAPI"], reference_stain="PolyT")


def test_bead_removal_accepts_custom_stains_with_matching_reference() -> None:
    config = MerscopeBeadRemovalConfig(stains=["DAPI"], reference_stain="DAPI")

    assert config.stains == ["DAPI"]


@pytest.mark.parametrize("detection_plane", [1, 3, -1])
def test_bead_removal_rejects_detection_plane_other_than_zero(
    detection_plane: int,
) -> None:
    with pytest.raises(ValidationError, match="detection_plane"):
        MerscopeBeadRemovalConfig(detection_plane=detection_plane)


@pytest.mark.parametrize("halo_px", [0, 240, 255, 257, 265])
def test_bead_removal_rejects_halo_below_256_or_off_grid(halo_px: int) -> None:
    with pytest.raises(ValidationError, match="halo_px"):
        MerscopeBeadRemovalConfig(halo_px=halo_px)


@pytest.mark.parametrize("halo_px", [256, 272, 512])
def test_bead_removal_accepts_halo_on_the_16_px_grid(halo_px: int) -> None:
    assert MerscopeBeadRemovalConfig(halo_px=halo_px).halo_px == halo_px


@pytest.mark.parametrize("max_workers", [0, -2])
def test_bead_removal_rejects_non_positive_max_workers(max_workers: int) -> None:
    with pytest.raises(ValidationError, match="max_workers"):
        MerscopeBeadRemovalConfig(max_workers=max_workers)


def test_bead_removal_fill_requires_enabled_when_given_explicitly() -> None:
    with pytest.raises(ValidationError, match="fill requires enabled"):
        MerscopeBeadRemovalConfig(enabled=False, fill=True)


def test_bead_removal_fill_default_does_not_need_enabled() -> None:
    """The default config is valid with removal off although the fill defaults on."""
    config = MerscopeBeadRemovalConfig()

    assert config.enabled is False
    assert config.fill is True
    assert MerscopeBeadRemovalConfig(enabled=False, fill=False).fill is False


def test_bead_removal_fill_is_accepted_when_enabled() -> None:
    config = MerscopeBeadRemovalConfig(enabled=True, fill=True)

    assert config.fill is True


def test_bead_removal_fill_and_its_mode_default_to_hybrid_on() -> None:
    assert MerscopeBeadRemovalConfig().fill_mode == "hybrid"
    config = MerscopeBeadRemovalConfig(enabled=True)
    assert config.fill is True
    assert config.fill_mode == "hybrid"
    assert MerscopeBeadRemovalConfig(enabled=True, fill=True).fill_mode == "hybrid"


def test_bead_removal_fill_false_switches_the_fill_off() -> None:
    config = MerscopeBeadRemovalConfig(enabled=True, fill=False)

    assert config.fill is False
    # The default mode is ignored with the fill off, not refused.
    assert config.fill_mode == "hybrid"


@pytest.mark.parametrize("mode", ["flat", "hybrid"])
def test_bead_removal_fill_mode_requires_the_fill_when_given_explicitly(
    mode: str,
) -> None:
    with pytest.raises(ValidationError, match=f"fill_mode '{mode}' requires fill"):
        MerscopeBeadRemovalConfig(
            enabled=True,
            fill=False,
            fill_mode=mode,  # type: ignore[arg-type]
        )


def test_bead_removal_accepts_the_flat_fill_mode() -> None:
    config = MerscopeBeadRemovalConfig(enabled=True, fill=True, fill_mode="flat")

    assert config.fill_mode == "flat"
    assert MerscopeBeadRemovalConfig(enabled=True, fill_mode="flat").fill is True


def test_bead_removal_rejects_an_unknown_fill_mode() -> None:
    with pytest.raises(ValidationError, match="fill_mode"):
        MerscopeBeadRemovalConfig(
            enabled=True,
            fill=True,
            fill_mode="aggressive",  # type: ignore[arg-type]
        )


def test_spatialdata_build_config_loads_bead_removal_from_json(
    tmp_path: Path,
) -> None:
    """The JSON shape Nextflow writes must round-trip into the nested model."""
    raw = {
        "dataset_name": "synthetic_MERSCOPE",
        "platform": "MERSCOPE",
        "input_path": str(tmp_path / "raw"),
        "output_path": "spatialdata_out/source_spatialdata.zarr",
        "merscope": {
            "z_layers": [1, 2, 3],
            "bead_removal": {"enabled": True, "fill": True},
        },
    }

    config = SpatialDataBuildConfig.model_validate(raw)

    assert config.merscope.z_layers == [1, 2, 3]
    assert config.merscope.bead_removal is not None
    assert config.merscope.bead_removal.fill is True


def test_spatialdata_build_config_loads_the_fill_settings_the_workflow_writes(
    tmp_path: Path,
) -> None:
    """The workflow writes ``fill`` and, with the fill on, ``fill_mode``."""
    raw = {
        "dataset_name": "synthetic_MERSCOPE",
        "platform": "MERSCOPE",
        "input_path": str(tmp_path / "raw"),
        "output_path": "spatialdata_out/source_spatialdata.zarr",
        "merscope": {
            "z_layers": [1, 2, 3],
            "bead_removal": {"enabled": True, "fill": True, "fill_mode": "hybrid"},
        },
    }

    config = SpatialDataBuildConfig.model_validate(raw)

    assert config.merscope.bead_removal is not None
    assert config.merscope.bead_removal.fill_mode == "hybrid"


def test_spatialdata_build_config_loads_a_bead_build_without_the_fill(
    tmp_path: Path,
) -> None:
    """``--merscope_bead_removal_fill false`` writes ``fill: false`` and no mode."""
    raw = {
        "dataset_name": "synthetic_MERSCOPE",
        "platform": "MERSCOPE",
        "input_path": str(tmp_path / "raw"),
        "output_path": "spatialdata_out/source_spatialdata.zarr",
        "merscope": {
            "z_layers": [1, 2, 3],
            "bead_removal": {"enabled": True, "fill": False},
        },
    }

    config = SpatialDataBuildConfig.model_validate(raw)

    assert config.merscope.bead_removal is not None
    assert config.merscope.bead_removal.fill is False


def test_spatialdata_build_config_refuses_a_fill_mode_without_the_fill(
    tmp_path: Path,
) -> None:
    raw = {
        "dataset_name": "synthetic_MERSCOPE",
        "platform": "MERSCOPE",
        "input_path": str(tmp_path / "raw"),
        "output_path": "spatialdata_out/source_spatialdata.zarr",
        "merscope": {
            "bead_removal": {"enabled": True, "fill": False, "fill_mode": "hybrid"},
        },
    }

    with pytest.raises(ValidationError, match="requires fill=True"):
        SpatialDataBuildConfig.model_validate(raw)
