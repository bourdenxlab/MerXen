"""Synthetic multi-tile ATERA store and the fake ProSeg of the AT-9 tests.

:func:`build_multitile_latest` writes the synthetic Atera bundle with cells
spread over two grid columns and transcripts in four 10 micron vendor tiles,
builds its source store and its NATIVE_LATEST latest store. With
:data:`TILE_UM` 8 and :data:`HALO_UM` 2 the ProSeg grid has several tiles,
cells near tile edges and halo rows that two tiles see.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

from merxen.config import (
    AteraProsegPlanConfig,
    AteraProsegStitchConfig,
    AteraProsegTileConfig,
    NativeLatestConfig,
)
from merxen.segmentation.native import build_native_latest

TILE_UM: Final = 8.0
HALO_UM: Final = 2.0
MARGIN_UM: Final = 0.3
DATASET: Final = "ATERA_SYNTH"
FAKE_PROSEG: Final = Path(__file__).with_name("fake_proseg.py")
BUNDLE_OPTIONS: Final[dict[str, Any]] = {
    "n_cells": 12,
    "tiles": {"0,0": 400, "1,0": 250, "0,1": 300, "1,1": 200},
    "grid_keys_order": ("0,0", "1,0", "0,1", "1,1"),
    "unassigned_fraction": 0.35,
    "low_qv_fraction": 0.1,
    "seed": 3,
}


def fake_proseg_binary(directory: Path, **env: str) -> Path:
    """Write an executable wrapper that runs the fake ProSeg with this Python.

    Args:
        directory: Folder for the wrapper.
        **env: ``FAKE_PROSEG_*`` variables baked into the wrapper.

    Returns:
        The wrapper path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = directory / "proseg"
    exports = "".join(f"export {key}='{value}'\n" for key, value in env.items())
    wrapper.write_text(
        f"#!/bin/sh\n{exports}exec '{sys.executable}' -I '{FAKE_PROSEG}' \"$@\"\n"
    )
    wrapper.chmod(0o755)
    return wrapper


def build_multitile_latest(tmp_path: Path) -> tuple[Path, Path]:
    """Build the synthetic source store and its native latest store.

    Args:
        tmp_path: Test directory.

    Returns:
        ``(source_store, latest_store)``.
    """
    from test_io.atera_native_fixture import build_atera_source

    source = build_atera_source(tmp_path, **BUNDLE_OPTIONS)
    latest = tmp_path / "results" / "latest" / "latest_spatialdata.zarr"
    build_native_latest(
        NativeLatestConfig(
            dataset_name=DATASET,
            platform="ATERA",
            source_zarr_path=source,
            output_dir=tmp_path / "native_out",
            persistent_latest_zarr_path=latest,
        )
    )
    return source, latest


def plan_config(
    source: Path, output_dir: Path, **overrides: Any
) -> AteraProsegPlanConfig:
    """Return the plan config of the synthetic grid."""
    values: dict[str, Any] = {
        "dataset_name": DATASET,
        "platform": "ATERA",
        "source_zarr_path": source,
        "output_dir": output_dir,
        "tile_um": TILE_UM,
        "halo_um": HALO_UM,
        "min_halo_um": 1.0,
        "owned_margin_um": MARGIN_UM,
    }
    values.update(overrides)
    return AteraProsegPlanConfig(**values)


def tile_configs(
    plan: dict[str, Any], source: Path, tiles_root: Path, **overrides: Any
) -> list[AteraProsegTileConfig]:
    """Return the tile configs of every tile a plan runs."""
    configs = []
    for tile in plan["tiles"]:
        if not tile["run"]:
            continue
        values: dict[str, Any] = {
            "dataset_name": DATASET,
            "platform": "ATERA",
            "tile_id": tile["tile_id"],
            "source_zarr_path": source,
            "source_build_fingerprint": plan["source_build_fingerprint"],
            "pixel_size_um": plan["pixel_size_um"],
            "core_um": tile["core_um"],
            "window_px": tile["window_px"],
            "mask_shape_px": plan["mask_shape_px"],
            "owned_labels": tile["owned_labels"],
            "output_dir": tiles_root / f"tile_{tile['tile_id']}",
            "num_threads": 1,
            "model": {
                "burnin_samples": 2,
                "samples": 2,
                "hillclimb": 0,
                "recorded_samples": 1,
                "prior_weight_effective": 140.0,
            },
        }
        values.update(overrides)
        configs.append(AteraProsegTileConfig(**values))
    return configs


def stitch_config(
    latest: Path,
    plan_path: Path,
    tile_dirs: list[Path],
    output_dir: Path,
    **overrides: Any,
) -> AteraProsegStitchConfig:
    """Return the stitch config of the synthetic run."""
    values: dict[str, Any] = {
        "dataset_name": DATASET,
        "platform": "ATERA",
        "latest_zarr_path": latest,
        "plan_path": plan_path,
        "tile_dirs": tile_dirs,
        "output_dir": output_dir,
        "n_workers": 2,
    }
    values.update(overrides)
    return AteraProsegStitchConfig(**values)
