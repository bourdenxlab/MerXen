"""Atera builder on the public 10x dataset (slow; skipped without the data).

A small build (one small transcript tile, points and metadata only) checks on
the real bundle what the synthetic one cannot: the vendor files are copied byte
for byte, the full header reaches the attrs, the points reconcile with the
vendor tile counts, and the label pyramid has the stored image levels. The
data under ``$MERXEN_ATERA_RAW_ROOT`` is only read; no manifest value is
printed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import spatialdata as sd
import zarr

from merxen.config import AteraBuildConfig
from merxen.io import atera_reader as ar
from merxen.io.builders import atera as atera_builder
from merxen.io.spatialdata_schema import validate_merxen_schema_streaming
from merxen.platforms import find_um_per_px

pytestmark = pytest.mark.slow

PUBLIC_DATASET = "Atera_v1_WTA_S4A_Brain_Hippocampus_ND_FF"
SMALL_TILE = "5,4"


@pytest.fixture
def outs(real_path: Callable[[str], Path]) -> Path:
    """The extracted public dataset folder."""
    folder = real_path("MERXEN_ATERA_RAW_ROOT") / PUBLIC_DATASET / "outs"
    if not (folder / ".extract_complete").exists():
        pytest.skip("the public Atera bundle is not extracted on this host")
    return folder


def test_small_real_build_keeps_the_metadata_whole(outs: Path, tmp_path: Path) -> None:
    config = AteraBuildConfig(
        cells_labels=False,
        nucleus_labels=False,
        cells_boundaries=False,
        nucleus_boundaries=False,
        cells_table=False,
        morphology_2d=False,
        tile_allowlist=[SMALL_TILE],
        species="human",
        max_rss_gb=40,
    )
    store = atera_builder.write_atera_spatialdata(
        input_path=outs,
        output_path=tmp_path / "small.zarr",
        build_config=config,
        n_workers=8,
        dataset_name="ATERA_PUBLIC",
    )
    bundle = atera_builder.resolve_atera_input(outs, config)
    marker = atera_builder.read_build_marker(store)
    assert marker is not None
    assert marker["build_fingerprint"] == atera_builder.atera_build_fingerprint(
        bundle, config
    )

    header = dict(zarr.open_group(store, mode="r").attrs)["merxen_atera"]
    experiment = json.loads((outs / "experiment.spatial").read_text())
    assert header["experiment"] == experiment
    assert header["metrics_summary"] and len(header["metrics_summary"][0]) > 50
    (payload,) = header["panel_config"]["payloads"]
    assert {"customer", "designer"} <= set(payload)
    for sidecar in bundle.sidecars:
        (record,) = [
            item
            for item in header["sidecars"]
            if item["original_name"] == sidecar.original_name
        ]
        data = sidecar.source.read_bytes()
        assert (store / record["path"]).read_bytes() == data
        assert record["sha256"] == hashlib.sha256(data).hexdigest()

    tiles = {tile.key: tile for tile in ar.read_atera_tile_index(bundle)}
    transcripts = header["transcripts"]
    assert transcripts["rows_read_per_tile"][SMALL_TILE] == tiles[SMALL_TILE].n_rows
    sdata = sd.read_zarr(store)
    summaries = validate_merxen_schema_streaming(
        sdata,
        group_column="source_tile",
        expected_group_rows={"transcripts": transcripts["rows_kept_per_tile"]},
    )
    assert summaries["transcripts"].n_rows == transcripts["n_points_written"]
    assert find_um_per_px(store, "ATERA") == pytest.approx(0.21249222)


def test_real_label_pyramid_matches_the_stored_image_levels(outs: Path) -> None:
    """Lazy only: spatialdata's label coarsening rounds like the TIFF pyramid."""
    bundle = ar.resolve_atera_bundle(outs)
    tree = ar.read_atera_morphology_2d(bundle, n_levels=5)
    image_levels = atera_builder._level_shapes(tree)
    assert image_levels[0] == bundle.morphology_shape_yx
    labels = atera_builder._label_elements(bundle, AteraBuildConfig(), image_levels)
    for element in labels.values():
        assert atera_builder._level_shapes(element) == image_levels
