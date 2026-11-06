"""Config models of the ATERA ProSeg route (``native_proseg``, plan AT-9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from merxen.config import (
    AteraProsegModelParams,
    AteraProsegPlanConfig,
    AteraProsegTileConfig,
    EnrichmentConfig,
    MaskImageQuantificationConfig,
    MiqLayer,
)


def _plan(**overrides: Any) -> AteraProsegPlanConfig:
    values: dict[str, Any] = {
        "dataset_name": "PAIR_A_ATERA",
        "platform": "ATERA",
        "source_zarr_path": Path("source.zarr"),
        "output_dir": Path("plan_out"),
    }
    values.update(overrides)
    return AteraProsegPlanConfig(**values)


def test_model_defaults_are_the_benchmark_schedule() -> None:
    model = AteraProsegModelParams()
    assert (model.burnin_samples, model.samples, model.hillclimb) == (100, 200, 20)
    assert model.recorded_samples == 100
    assert model.prior_weight_effective == 140.0
    assert model.prior_seg_reassignment_prob is None
    assert model.use_cell_initialization is False
    assert model.nuclear_compartment is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("burnin_samples", -1, "burnin_samples"),
        ("hillclimb", -1, "hillclimb"),
        ("samples", 0, "samples must be at least 1"),
        ("recorded_samples", 300, "must not exceed"),
        ("prior_weight_effective", 0.0, "prior_weight_effective"),
        ("prior_weight_tolerance", 1.5, "prior_weight_tolerance"),
        ("voxel_layers", 0, "voxel_layers"),
    ],
)
def test_model_schedule_refusals(field: str, value: Any, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        AteraProsegModelParams(**{field: value})


def test_plan_defaults_and_grid_refusals() -> None:
    plan = _plan()
    assert (plan.tile_um, plan.halo_um, plan.owned_margin_um) == (2000.0, 100.0, 20.0)
    with pytest.raises(ValidationError, match="below the minimum"):
        _plan(halo_um=50)
    with pytest.raises(ValidationError, match="twice the halo"):
        _plan(tile_um=200, halo_um=100)
    with pytest.raises(ValidationError, match="at least one tile"):
        _plan(tile_subset=[])
    # Tests lower the minimum for the synthetic bundle.
    assert _plan(tile_um=8, halo_um=2, min_halo_um=1).halo_um == 2


def test_tile_config_needs_a_window_and_owned_labels() -> None:
    values: dict[str, Any] = {
        "dataset_name": "PAIR_A_ATERA",
        "platform": "ATERA",
        "tile_id": "c0_r0",
        "source_zarr_path": Path("source.zarr"),
        "source_build_fingerprint": "f" * 64,
        "pixel_size_um": 0.2125,
        "core_um": [0, 0, 10, 10],
        "window_px": [0, 50, 0, 50],
        "mask_shape_px": [97, 67],
        "owned_labels": [1, 2],
        "output_dir": Path("tile_c0_r0"),
    }
    assert AteraProsegTileConfig(**values).num_threads == 16
    with pytest.raises(ValidationError, match="owns no label"):
        AteraProsegTileConfig(**{**values, "owned_labels": []})
    with pytest.raises(ValidationError, match="empty or negative window"):
        AteraProsegTileConfig(**{**values, "window_px": [10, 10, 0, 50]})


def test_enrichment_native_proseg_needs_no_masks_and_is_atera_only() -> None:
    values: dict[str, Any] = {
        "dataset_name": "PAIR_A_ATERA",
        "platform": "ATERA",
        "latest_zarr_path": Path("latest.zarr"),
        "mask_path": None,
        "nuclei_mask_path": None,
        "original_data_path": Path("source.zarr"),
        "output_dir": Path("enrich_out"),
        "segmentation_source": "native_proseg",
    }
    assert EnrichmentConfig(**values).segmentation_source == "native_proseg"
    with pytest.raises(ValidationError, match="ATERA rows only"):
        EnrichmentConfig(**{**values, "platform": "XENIUM"})


def test_miq_extra_layers_default_empty_and_need_distinct_tables() -> None:
    values: dict[str, Any] = {
        "dataset_name": "PAIR_A_ATERA",
        "platform": "ATERA",
        "latest_zarr_path": Path("latest.zarr"),
        "mask_path": None,
        "mask_label_key": "cell_labels",
        "output_dir": Path("miq_out"),
        "shape_key": "atera_cell_boundaries",
        "table_key": "table_atera_cell_boundaries_image_quantification",
    }
    assert MaskImageQuantificationConfig(**values).extra_layers == []
    layer = MiqLayer(
        mask_label_key="proseg_cell_labels",
        shape_key="MOSAIK_proseg",
        table_key="table_MOSAIK_proseg_image_quantification",
    )
    assert MaskImageQuantificationConfig(**values, extra_layers=[layer]).extra_layers
    clash = layer.model_copy(update={"table_key": values["table_key"]})
    with pytest.raises(ValidationError, match="its own table"):
        MaskImageQuantificationConfig(**values, extra_layers=[clash])
