"""The ATERA ProSeg route with the real ProSeg 3.2.0 (plan AT-9).

- ``integration``: a synthetic bundle of about 40 cells and 50k transcripts in
  two tiles, a 5 + 5 + 0 schedule. ProSeg must accept the Xenium-layout
  parquet, its outputs must parse, the recount must equal its count matrix and
  the stitched store must pass the layer check. Needs ``PROSEG_BINARY``
  (``.env`` or the environment) pointing at ProSeg 3.2.0.
- ``slow``: a 300 x 300 micron core of a built ATERA source store
  (``MERXEN_ATERA_SOURCE_STORE``, ``.env``), schedule 20 + 20 + 5, checked
  against the AT-5(b) benchmark ranges.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from merxen import local_config
from merxen.config import AteraProsegTileConfig, NativeLatestConfig
from merxen.segmentation.atera_proseg.plan import read_vendor_cells, write_plan
from merxen.segmentation.atera_proseg.stitch import check_proseg_layer, stitch
from merxen.segmentation.atera_proseg.tile import run_tile
from merxen.segmentation.native import build_native_latest
from test_segmentation.atera_proseg_fixture import (
    DATASET,
    plan_config,
    stitch_config,
    tile_configs,
)

_SCHEDULE = {"burnin_samples": 5, "samples": 5, "hillclimb": 0, "recorded_samples": 5}


def _proseg_binary() -> Path:
    raw = local_config.get("PROSEG_BINARY")
    binary = Path(raw) if raw else None
    if binary is None or not binary.is_file():
        pytest.skip("PROSEG_BINARY is not set to an existing ProSeg binary")
    result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    if result.stdout.strip().split()[-1:] != ["3.2.0"]:
        pytest.skip(f"PROSEG_BINARY is not ProSeg 3.2.0 ({result.stdout.strip()})")
    return binary


@pytest.mark.integration
def test_real_proseg_on_a_synthetic_two_tile_bundle(tmp_path: Path) -> None:
    from test_io.atera_native_fixture import build_atera_source

    binary = _proseg_binary()
    source = build_atera_source(
        tmp_path,
        n_cells=40,
        tiles={"0,0": 16000, "1,0": 12000, "0,1": 14000, "1,1": 10000},
        grid_keys_order=("0,0", "1,0", "0,1", "1,1"),
        unassigned_fraction=0.3,
        low_qv_fraction=0.05,
        seed=5,
    )
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
    plan_path = write_plan(
        plan_config(source, tmp_path / "plan_out", tile_um=16.0, halo_um=2.0)
    )
    plan = json.loads(plan_path.read_text())
    assert plan["n_tiles_run"] >= 2
    dirs = []
    for config in tile_configs(
        plan, source, tmp_path / "tiles", num_threads=2, model=_SCHEDULE
    ):
        manifest = run_tile(config, binary)
        assert manifest["proseg"]["version"] == "3.2.0"
        reported = manifest["prior_weight"]["reported"]
        # A tile whose mean is below 100 transcripts per cell keeps the
        # multiplier; ProSeg then prints the weight itself.
        assert reported == pytest.approx(140.0, rel=0.02)
        dirs.append(config.output_dir)
    report = stitch(stitch_config(latest, plan_path, dirs, tmp_path / "stitch_out"))
    assert report["n_cells"] > 0
    assert check_proseg_layer(latest)["passed"]


@pytest.mark.slow
def test_real_proseg_on_a_window_of_a_real_store(tmp_path: Path) -> None:
    binary = _proseg_binary()
    store = local_config.path("MERXEN_ATERA_SOURCE_STORE")
    if store is None or not store.is_dir():
        pytest.skip("MERXEN_ATERA_SOURCE_STORE is not set to a built ATERA store")
    plan = json.loads(
        write_plan(plan_config(store, tmp_path / "plan", tile_um=2000.0, halo_um=100.0,
                               min_halo_um=100.0, owned_margin_um=20.0)).read_text()
    )  # fmt: skip
    ps = float(plan["pixel_size_um"])
    cells = read_vendor_cells(store)
    # A 300 um core in the densest 2 mm tile, with a 30 um halo.
    densest = max(
        (t for t in plan["tiles"] if t["run"]), key=lambda t: t["est_rows"] or 0
    )
    cx = (densest["core_um"][0] + densest["core_um"][2]) / 2
    cy = (densest["core_um"][1] + densest["core_um"][3]) / 2
    core = (cx - 150.0, cy - 150.0, cx + 150.0, cy + 150.0)
    centroid = cells["centroid"]
    owned = (
        (centroid[:, 0] >= core[0])
        & (centroid[:, 0] < core[2])
        & (centroid[:, 1] >= core[1])
        & (centroid[:, 1] < core[3])
    )
    window = (core[0] - 30.0, core[1] - 30.0, core[2] + 30.0, core[3] + 30.0)
    config = AteraProsegTileConfig(
        dataset_name=DATASET,
        platform="ATERA",
        tile_id="window",
        source_zarr_path=store,
        source_build_fingerprint=plan["source_build_fingerprint"],
        pixel_size_um=ps,
        core_um=list(core),
        window_px=[
            int(np.floor(window[0] / ps)),
            int(np.ceil(window[2] / ps)),
            int(np.floor(window[1] / ps)),
            int(np.ceil(window[3] / ps)),
        ],
        mask_shape_px=plan["mask_shape_px"],
        owned_labels=[int(v) for v in cells["instance_id"][owned]],
        output_dir=tmp_path / "tile_window",
        num_threads=8,
        keep_inputs=True,
        model={"burnin_samples": 20, "samples": 20, "hillclimb": 5,
               "recorded_samples": 20},
    )  # fmt: skip
    manifest = run_tile(config, binary)
    assert manifest["prior_weight"]["reported"] == pytest.approx(140.0, rel=0.02)
    work = tmp_path / "tile_window" / "work"
    exported = pq.read_table(
        work / "inputs" / "tile.parquet", columns=["transcript_id", "cell_id"]
    ).to_pandas()
    meta = pq.read_table(
        work / "proseg" / "transcript-metadata.parquet",
        columns=["transcript_id", "assignment"],
    ).to_pandas()
    merged = exported.merge(meta, on="transcript_id")
    unassigned = merged["cell_id"] == "0"
    absorption = float(merged.loc[unassigned, "assignment"].notna().mean())
    assert 0.05 <= absorption <= 0.30, absorption
    import shapely

    owned_cells = pq.read_table(tmp_path / "tile_window" / "cells.parquet").to_pandas()
    areas = shapely.area(shapely.from_wkb(owned_cells["geometry_wkb"].to_numpy()))
    vendor_area = _vendor_cell_area(store)
    ratio = areas / owned_cells["instance_id"].map(vendor_area).to_numpy()
    assert 0.6 <= float(np.median(ratio)) <= 1.1, float(np.median(ratio))
    shutil.rmtree(work, ignore_errors=True)


def _vendor_cell_area(store: Path) -> dict[int, float]:
    import anndata as ad
    import zarr

    group = zarr.open_group(str(store / "tables" / "table"), mode="r")
    obs = ad.io.read_elem(group["obs"])
    return dict(
        zip(
            obs["instance_id"].astype(int),
            obs["cell_area"].astype(float),
            strict=True,
        )
    )
