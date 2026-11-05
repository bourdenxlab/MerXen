"""Atera readers on the synthetic bundle (plan §7.2: readers, shapes, metadata)."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
import threading
import time
import weakref
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import psutil
import pytest
import shapely
import spatialdata as sd
import tifffile
import zarr
from spatialdata.models import Image2DModel, PointsModel
from spatialdata.transformations import Identity, Scale, get_transformation

from merxen.io import atera_reader as ar
from merxen.io.atera_reader import (
    AteraFormatError,
    AteraTile,
    CellKeyLookup,
    PartitionSpec,
)
from merxen.segmentation import mask_geometry
from merxen.segmentation.cellpose import assign_labels_from_masks
from merxen.segmentation.mask_geometry import masks_to_labeled_polygons

from .atera_synth import (
    COMPANION_PREFIX,
    EMPTY_POLYGON_CELL,
    FAKE_IDENTIFIERS,
    MULTI_COMPONENT_CELL,
    NUCLEUS_LESS_CELL,
    RUN_SUFFIX,
    TWO_NUCLEUS_CELL,
    AteraSynth,
    cell_id_strings,
    rewrite_zip_store,
    write_atera_bundle,
)

BundleFactory = Callable[..., tuple[AteraSynth, ar.AteraBundle]]

#: Stored sidecar path (under ``atera/``) -> original bundle path, default bundle.
EXPECTED_SIDECARS = {
    "experiment.spatial": "experiment.spatial",
    "metrics_summary.csv": f"metrics_summary_{RUN_SUFFIX}.csv",
    "analysis_summary.html": f"analysis_summary_{RUN_SUFFIX}.html",
    "qc_images.html": "qc_images.html",
    "panel_config.json": "panel_config.json",
    "morphology_fov_locations.json": "morphology_fov_locations.json",
    "overview_scan_fov_locations.json": "overview_scan_fov_locations.json",
    "overview_scan.png": "overview_scan.png",
    "diffexp_genes/gene_expression_graphclust.csv": (
        "diffexp_genes/gene_expression_graphclust.csv"
    ),
    "he_imagealignment.csv": f"{COMPANION_PREFIX}_he_imagealignment.csv",
    "keypoints.csv": f"{COMPANION_PREFIX}_keypoints.csv",
}


@pytest.fixture(scope="module")
def atera(tmp_path_factory: pytest.TempPathFactory) -> BundleFactory:
    """Write each synthetic bundle variant once per module and resolve it."""
    cache: dict[tuple[tuple[str, Any], ...], tuple[AteraSynth, ar.AteraBundle]] = {}

    def factory(**kwargs: Any) -> tuple[AteraSynth, ar.AteraBundle]:
        key = tuple(sorted(kwargs.items()))
        if key not in cache:
            root = tmp_path_factory.mktemp("atera") / "roi"
            synth = write_atera_bundle(root, **kwargs)
            cache[key] = (synth, ar.resolve_atera_bundle(root))
        return cache[key]

    return factory


@pytest.fixture
def default_bundle(atera: BundleFactory) -> tuple[AteraSynth, ar.AteraBundle]:
    """The default bundle: two channels, five special cells and one extra."""
    return atera()


def _copy_bundle(synth: AteraSynth, tmp_path: Path) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(synth.root, target)
    return target


def _all_rows(bundle: ar.AteraBundle, **kwargs: Any) -> pd.DataFrame:
    points, _, _ = ar.read_atera_transcripts(bundle, **kwargs)
    with dask.config.set(scheduler="sync"):
        return points.compute().reset_index(drop=True)


def _truth(synth: AteraSynth, min_qv: float | None) -> pd.DataFrame:
    frame = synth.transcripts
    if min_qv is not None:
        frame = frame[frame["qv"].astype(np.float64) >= min_qv]
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Bundle resolution
# --------------------------------------------------------------------------- #


def test_resolve_from_experiment_spatial(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    assert bundle.transcripts_zip == synth.root / "transcripts.zarr.zip"
    assert bundle.pixel_size_um == pytest.approx(synth.pixel_size_um)
    assert bundle.analysis_sw_version.startswith("atera-")
    assert bundle.morphology_channel_names == ("DAPI", "18S")
    assert [p.name for p in bundle.morphology_2d_files] == list(synth.channel_files)
    assert bundle.morphology_shape_yx == synth.cell_mask.shape
    assert bundle.morphology_3d_file is None
    assert bundle.panel_config_file == synth.root / "panel_config.json"
    assert {sidecar.stored_path for sidecar in bundle.sidecars} == set(
        EXPECTED_SIDECARS
    )


def test_resolve_globs_prefixed_companions(
    atera: BundleFactory, tmp_path: Path
) -> None:
    synth, bundle = atera(prefixed_companions=True)
    assert bundle.transcripts_zip == synth.zip_path("transcripts")
    assert bundle.cells_zip == synth.zip_path("cells")
    assert bundle.cell_feature_matrix_zip == synth.zip_path("cell_feature_matrix")
    # The prefixed viz stores never stand in for the primary ones.
    root = _copy_bundle(synth, tmp_path)
    prefix = synth.zip_path("transcripts").name.removesuffix("transcripts.zarr.zip")
    source = root / f"{prefix}transcripts.zarr.zip"
    for name in ("binned_transcripts", "csc_cell_feature_matrix"):
        shutil.copy(source, root / f"{prefix}{name}.zarr.zip")
    copy = ar.resolve_atera_bundle(root)
    assert copy.transcripts_zip.name == f"{prefix}transcripts.zarr.zip"
    assert copy.cell_feature_matrix_zip.name == f"{prefix}cell_feature_matrix.zarr.zip"


def test_explicit_paths_win(default_bundle: tuple, tmp_path: Path) -> None:
    synth, _ = default_bundle
    moved = tmp_path / "elsewhere.zarr.zip"
    shutil.copy(synth.root / "cells.zarr.zip", moved)
    bundle = ar.resolve_atera_bundle(synth.root, cells_path=moved)
    assert bundle.cells_zip == moved


def test_spec_override_is_metadata_only(default_bundle: tuple, tmp_path: Path) -> None:
    synth, _ = default_bundle
    experiment = json.loads((synth.root / "experiment.spatial").read_text())
    experiment["panel_wta"]["panel_organism"] = "Mouse"
    experiment["explorer_files"]["cells_zarr_filepath"] = "not_here.zarr.zip"
    override = tmp_path / "override.spatial"
    override.write_text(json.dumps(experiment))
    bundle = ar.resolve_atera_bundle(
        synth.root, spec_override=override, species="mouse"
    )
    assert bundle.experiment["panel_wta"]["panel_organism"] == "Mouse"
    assert bundle.cells_zip == synth.root / "cells.zarr.zip"
    assert bundle.root == synth.root
    # Both manifests are kept: the bundle's own and the override.
    stored = {sidecar.stored_path: sidecar for sidecar in bundle.sidecars}
    assert stored["experiment.spatial"].source == synth.root / "experiment.spatial"
    assert stored[ar.EXPERIMENT_OVERRIDE_SIDECAR].source == override
    assert stored[ar.EXPERIMENT_OVERRIDE_SIDECAR].original_name == "override.spatial"


def _edit_experiment(root: Path, **fields: Any) -> None:
    path = root / "experiment.spatial"
    experiment = json.loads(path.read_text())
    experiment.update(fields)
    path.write_text(json.dumps(experiment))


def _set_attrs(**attrs: Any) -> Callable[[zarr.Group], None]:
    def edit(group: zarr.Group) -> None:
        group.attrs.update(attrs)

    return edit


@pytest.mark.parametrize(
    ("zip_name", "attrs", "message"),
    [
        ("transcripts", {"major_version": 7}, "RnaDataset major_version 7"),
        ("cells", {"major_version": 5}, "CellSegmentationDataset major_version 5"),
        ("transcripts", {"coordinate_space": "pixels"}, "coordinate_space"),
        ("transcripts", {"spatial_units": "pixels"}, "transcripts spatial_units"),
        ("cells", {"spatial_units": "mm"}, "cells spatial_units"),
    ],
)
def test_unknown_store_headers_fail_loudly(
    default_bundle: tuple,
    tmp_path: Path,
    zip_name: str,
    attrs: dict[str, Any],
    message: str,
) -> None:
    synth, _ = default_bundle
    root = _copy_bundle(synth, tmp_path)
    rewrite_zip_store(root / f"{zip_name}.zarr.zip", _set_attrs(**attrs))
    with pytest.raises(AteraFormatError, match=message):
        ar.resolve_atera_bundle(root)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"analysis_sw_version": "xenium-3.0.0"}, "not an Atera version"),
        ({"major_version": 2}, "experiment.spatial major_version 2"),
    ],
)
def test_unknown_experiment_versions_fail_loudly(
    default_bundle: tuple, tmp_path: Path, fields: dict[str, Any], message: str
) -> None:
    synth, _ = default_bundle
    root = _copy_bundle(synth, tmp_path)
    _edit_experiment(root, **fields)
    with pytest.raises(AteraFormatError, match=message):
        ar.resolve_atera_bundle(root)


def test_species_mismatch_fails(default_bundle: tuple) -> None:
    synth, _ = default_bundle
    ar.resolve_atera_bundle(synth.root, species="human")
    with pytest.raises(AteraFormatError, match="does not match species"):
        ar.resolve_atera_bundle(synth.root, species="mouse")


def test_missing_morphology_sibling_fails(
    default_bundle: tuple, tmp_path: Path
) -> None:
    synth, _ = default_bundle
    root = _copy_bundle(synth, tmp_path)
    (root / "morphology_2d" / synth.channel_files[1]).unlink()
    with pytest.raises(AteraFormatError, match="is missing"):
        ar.resolve_atera_bundle(root)


def test_bundle_paths_may_not_leave_the_roi(
    default_bundle: tuple, tmp_path: Path
) -> None:
    synth, _ = default_bundle
    root = _copy_bundle(synth, tmp_path)
    experiment = json.loads((root / "experiment.spatial").read_text())
    experiment["explorer_files"]["cells_zarr_filepath"] = "../cells.zarr.zip"
    (root / "experiment.spatial").write_text(json.dumps(experiment))
    with pytest.raises(AteraFormatError, match="leaves the ROI directory"):
        ar.resolve_atera_bundle(root)


def test_missing_directory_asks_for_extraction(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="extract"):
        ar.resolve_atera_bundle(tmp_path / "outs.zip")


def test_pixel_size_must_match_ome(default_bundle: tuple, tmp_path: Path) -> None:
    synth, _ = default_bundle
    root = _copy_bundle(synth, tmp_path)
    experiment = json.loads((root / "experiment.spatial").read_text())
    experiment["pixel_size"] = 0.2125
    (root / "experiment.spatial").write_text(json.dumps(experiment))
    with pytest.raises(AteraFormatError, match="PhysicalSizeX"):
        ar.resolve_atera_bundle(root)


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


def test_tile_index_follows_attribute_order(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    tiles = ar.read_atera_tile_index(bundle)
    assert [tile.key for tile in tiles] == list(synth.grid_keys_order)
    assert list(synth.grid_keys_order) != sorted(synth.grid_keys_order)
    offsets = np.cumsum([0] + [synth.tile_rows[key] for key in synth.grid_keys_order])
    assert [tile.global_offset for tile in tiles] == offsets[:-1].tolist()
    negative = next(tile for tile in tiles if tile.key == "-1,1")
    assert (negative.x_index, negative.y_index) == (-1, 1)
    assert negative.bounds_um == (-10.0, 0.0, 10.0, 20.0)


def test_rows_match_vendor_truth_without_filter(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    rows = _all_rows(bundle, min_qv=None, partition_rows=16, emit_codeword_index=True)
    truth = _truth(synth, None)
    assert len(rows) == len(truth)
    np.testing.assert_array_equal(rows["transcript_id"], truth["transcript_id"])
    np.testing.assert_array_equal(rows["x"], truth["x"])
    np.testing.assert_array_equal(rows["y"], truth["y"])
    np.testing.assert_array_equal(rows["feature_name"].cat.codes, truth["gene"])
    # The gene from gene_offset equals the gene from the codeword mapping.
    np.testing.assert_array_equal(
        rows["feature_name"].cat.codes,
        synth.codeword_gene_mapping[rows["codeword_index"].to_numpy()],
    )
    np.testing.assert_array_equal(rows["is_gene"], synth.is_gene[truth["gene"]])
    np.testing.assert_array_equal(
        rows["overlaps_nucleus"], truth["overlaps_nucleus"] > 0
    )
    expected = pd.array(
        np.where(truth["cell_row"] >= 0, truth["cell_row"] + 1, 0).astype(np.uint64),
        dtype="UInt64",
    )
    expected[truth["cell_row"].to_numpy() < 0] = pd.NA
    pd.testing.assert_extension_array_equal(
        rows["original_assignment"].array, expected, check_exact=True
    )
    assert rows["original_assignment"].isna().sum() > 0
    assert (rows["x"] < 0).any() and (rows["y"] < 0).any()


def test_dtypes_are_known_and_lean(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    points, specs, _ = ar.read_atera_transcripts(bundle, partition_rows=16)
    meta = points._meta
    assert list(meta.columns) == list(ar.TRANSCRIPT_COLUMNS)
    assert list(meta["feature_name"].cat.categories) == list(synth.gene_names)
    assert list(meta["source_tile"].cat.categories) == list(synth.grid_keys_order)
    expected = {
        "x": "float32",
        "y": "float32",
        "z": "float32",
        "is_gene": "bool",
        "qv": "float32",
        "overlaps_nucleus": "bool",
        "original_assignment": "UInt64",
        "transcript_id": "uint64",
    }
    assert {name: str(meta[name].dtype) for name in expected} == expected
    with dask.config.set(scheduler="sync"):
        for index in range(points.npartitions):
            part = points.get_partition(index).compute()
            assert part["feature_name"].dtype == meta["feature_name"].dtype
            assert part["source_tile"].dtype == meta["source_tile"].dtype
            assert part["source_tile"].nunique() == 1
            assert part["source_tile"].iloc[0] == specs[index].tile_key


def test_is_gene_is_selected_by_column_name(atera: BundleFactory) -> None:
    synth, bundle = atera(shuffle_category_columns=True)
    rows = _all_rows(bundle, min_qv=None)
    np.testing.assert_array_equal(
        rows["is_gene"], synth.is_gene[rows["feature_name"].cat.codes]
    )
    with zarr.storage.ZipStore(bundle.transcripts_zip, mode="r") as store:
        names = zarr.open_group(store, mode="r")["gene_category"].attrs["column_names"]
    assert names[0] != "is_gene"


def test_qv_filter_keeps_ids_stable(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    everything = _all_rows(bundle, min_qv=None)
    filtered = _all_rows(bundle, min_qv=20.0)
    truth = _truth(synth, 20.0)
    assert len(filtered) == len(truth) < len(everything)
    assert filtered["qv"].min() >= 20.0
    np.testing.assert_array_equal(filtered["transcript_id"], truth["transcript_id"])
    assert filtered["transcript_id"].is_unique
    assert filtered["transcript_id"].min() >= 1
    kept = everything.set_index("transcript_id").loc[filtered["transcript_id"]]
    np.testing.assert_array_equal(kept["x"].to_numpy(), filtered["x"].to_numpy())
    # QV < 20 rows are never assigned by the vendor.
    low = everything[everything["qv"] < 20]
    assert low["original_assignment"].isna().all()


def test_empty_tile_gets_no_partition(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    _, specs, _ = ar.read_atera_transcripts(bundle, partition_rows=16)
    assert synth.tile_rows["1,0"] == 0
    assert "1,0" not in {spec.tile_key for spec in specs}
    by_tile = Counter(spec.tile_key for spec in specs)
    assert by_tile == {"0,1": 2, "-1,1": 1, "0,0": 3}


def test_row_range_and_tile_blocks_give_the_same_rows(default_bundle: tuple) -> None:
    _, bundle = default_bundle
    row_range = _all_rows(bundle, partition_rows=16)
    blocks = _all_rows(
        bundle, spatial_order="tile_blocks", block_um=2.5, min_block_rows=1
    )
    assert len(blocks) == len(row_range)
    pd.testing.assert_frame_equal(
        blocks.sort_values("transcript_id").reset_index(drop=True),
        row_range.sort_values("transcript_id").reset_index(drop=True),
    )


def test_plan_partitions_row_ranges_and_blocks() -> None:
    tiles = [
        AteraTile("0,0", 0, 0, 100, 0, 2000.0),
        AteraTile("1,0", 1, 0, 0, 100, 2000.0),
        AteraTile("2,0", 2, 0, 10, 100, 2000.0),
    ]
    specs = ar.plan_transcript_partitions(tiles, partition_rows=40)
    assert [(s.tile_key, s.row_start, s.row_stop) for s in specs] == [
        ("0,0", 0, 40),
        ("0,0", 40, 80),
        ("0,0", 80, 100),
        ("2,0", 0, 10),
    ]
    blocks = ar.plan_transcript_partitions(
        tiles,
        partition_rows=40,
        spatial_order="tile_blocks",
        block_um=500.0,
        min_block_rows=5,
    )
    # 100 rows over 16 blocks is above the 5-row minimum; 10 rows is not.
    assert Counter(s.tile_key for s in blocks) == {"0,0": 16, "2,0": 1}
    tile_blocks = [s for s in blocks if s.tile_key == "0,0"]
    assert all(s.block_bounds_um is not None for s in tile_blocks)
    x_edges = sorted({s.block_bounds_um[0] for s in tile_blocks})  # type: ignore[index]
    assert x_edges == [-np.inf, 500.0, 1000.0, 1500.0]
    assert blocks[-1].block_bounds_um is None
    only = ar.plan_transcript_partitions(
        tiles, partition_rows=40, tile_allowlist=["2,0"]
    )
    assert [s.tile_key for s in only] == ["2,0"]
    with pytest.raises(ValueError, match="unknown tiles"):
        ar.plan_transcript_partitions(tiles, partition_rows=40, tile_allowlist=["9,9"])
    with pytest.raises(ValueError, match="partition_rows"):
        ar.plan_transcript_partitions(tiles, partition_rows=0)


def test_aggregates_match_a_direct_count(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    points, _, aggregates = ar.read_atera_transcripts(bundle, partition_rows=16)
    with pytest.raises(RuntimeError, match="0 of 6 transcript partitions"):
        aggregates.summary()
    with dask.config.set(scheduler="sync"):
        points.compute()
    summary = aggregates.summary()
    truth = synth.transcripts
    kept = truth[truth["qv"].astype(np.float64) >= 20.0]
    n_genes = len(synth.gene_names)
    np.testing.assert_array_equal(
        summary.gene_counts_all_qv, np.bincount(truth["gene"], minlength=n_genes)
    )
    np.testing.assert_array_equal(
        summary.gene_counts_kept, np.bincount(kept["gene"], minlength=n_genes)
    )
    np.testing.assert_array_equal(summary.gene_counts_q20, summary.gene_counts_kept)
    with zarr.storage.ZipStore(bundle.transcripts_zip, mode="r") as store:
        vendor = zarr.open_group(store, mode="r")["grid"].attrs["gene_counts"]
    np.testing.assert_array_equal(summary.gene_counts_all_qv, vendor)
    assert summary.n_rows_read == len(truth)
    assert summary.n_rows_kept == len(kept)
    assert summary.rows_read_per_tile == synth.tile_rows
    for key, histogram in summary.qv_histogram_per_tile.items():
        qv = truth.loc[truth["tile"] == key, "qv"].astype(np.float64)
        expected = np.bincount((qv / 0.25).round().astype(int), minlength=161)
        np.testing.assert_array_equal(histogram, expected)
        mean = summary.mean_qv_per_tile[key]
        assert mean == (pytest.approx(qv.mean()) if len(qv) else None)
    assert summary.bounds_um["x_min"] == pytest.approx(float(truth["x"].min()))
    assert summary.bounds_um["y_min"] < 0
    assert int(summary.density.sum()) == len(kept)
    attrs = summary.to_attrs()
    json.dumps(attrs)
    assert attrs["n_points_written"] == len(kept)
    assert attrs["gene_counts_q20"] == summary.gene_counts_q20.tolist()


def test_q20_totals_are_recorded_without_a_filter(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    points, _, aggregates = ar.read_atera_transcripts(
        bundle, min_qv=None, partition_rows=16
    )
    with dask.config.set(scheduler="sync"):
        points.compute()
        # A re-executed partition is counted once.
        points.get_partition(0).compute()
    summary = aggregates.summary()
    truth = synth.transcripts
    q20 = truth[truth["qv"].astype(np.float64) >= 20.0]
    n_genes = len(synth.gene_names)
    np.testing.assert_array_equal(
        summary.gene_counts_q20, np.bincount(q20["gene"], minlength=n_genes)
    )
    np.testing.assert_array_equal(
        summary.gene_counts_kept, np.bincount(truth["gene"], minlength=n_genes)
    )
    assert summary.n_rows_read == summary.n_rows_kept == len(truth)


def test_reads_for_genes_use_gene_offset_slices(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    genes = ["SNAP25", "Intergenic_Region_12"]
    frame = ar.read_atera_transcripts_for_genes(bundle, genes, min_qv=20.0)
    everything = _all_rows(bundle, min_qv=20.0)
    expected = everything[everything["feature_name"].isin(genes)]
    assert sorted(frame["transcript_id"]) == sorted(expected["transcript_id"])
    assert frame["feature_name"].dtype == everything["feature_name"].dtype
    with pytest.raises(KeyError, match="NOTAGENE"):
        ar.read_atera_transcripts_for_genes(bundle, ["NOTAGENE"])


def test_cell_lookup_handles_shuffled_keys_and_rejects_unknown_ids() -> None:
    prefix = np.array([500, 100, 300], dtype=np.uint32)
    suffix = np.ones(3, dtype=np.uint32)
    lookup = CellKeyLookup.from_cell_ids(prefix, suffix)
    sentinel = np.uint32(0xFFFFFFFF)
    ids = lookup.instance_ids(
        np.array([300, sentinel, 500, 100], dtype=np.uint32),
        np.array([1, sentinel, 1, 1], dtype=np.uint32),
    )
    assert ids.tolist() == [3, pd.NA, 1, 2]
    with pytest.raises(AteraFormatError, match="not in cells.zarr.zip"):
        lookup.instance_ids(np.array([999], np.uint32), np.array([1], np.uint32))
    with pytest.raises(AteraFormatError, match="half-unassigned"):
        lookup.instance_ids(np.array([sentinel], np.uint32), np.array([1], np.uint32))


def test_gene_offset_must_cover_the_tile() -> None:
    with pytest.raises(AteraFormatError, match="contiguous"):
        ar._gene_codes(np.array([[0, 3], [4, 6]], dtype=np.uint32), 0, 6, 6, 2)
    offsets = np.array([[0, 3], [3, 3], [3, 6]], dtype=np.uint32)
    with pytest.raises(AteraFormatError, match=r"not \(4, 2\)"):
        ar._gene_codes(offsets, 0, 6, 6, 4)
    assert ar._gene_codes(offsets, 2, 5, 6, 3).tolist() == [0, 2, 2]
    # Selected rows (tile_blocks) get the same codes as the full range.
    full = ar._gene_codes(offsets, 0, 6, 6, 3)
    selected = np.array([0, 2, 3, 5])
    assert ar._gene_codes(offsets, 0, 6, 6, 3, selected).tolist() == (
        full[selected].tolist()
    )
    assert ar._gene_codes(offsets, 2, 5, 6, 3, np.array([0, 1])).tolist() == [0, 2]


# --------------------------------------------------------------------------- #
# Pass and concurrency budget
# --------------------------------------------------------------------------- #


def test_one_read_per_partition_for_write_and_aggregates(
    default_bundle: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = default_bundle
    calls: Counter[tuple[str, int]] = Counter()
    original = ar._read_partition_with_stats

    def counting(context: Any, spec: PartitionSpec) -> Any:
        calls[(spec.tile_key, spec.row_start)] += 1
        return original(context, spec)

    monkeypatch.setattr(ar, "_read_partition_with_stats", counting)
    points, specs, aggregates = ar.read_atera_transcripts(bundle, partition_rows=16)
    parsed = PointsModel.parse(
        points,
        coordinates={"x": "x", "y": "y", "z": "z"},
        feature_key="feature_name",
        instance_key="original_assignment",
        transformations={"global": Scale([1.0, 1.0], axes=("x", "y"))},
        sort=True,
    )
    assert not calls, "PointsModel.parse(sort=True) must not read partitions"
    with dask.config.set(scheduler="threads", num_workers=4):
        parsed.to_parquet(tmp_path / "points.parquet")
    assert set(calls.values()) == {1}
    assert len(calls) == len(specs)
    summary = aggregates.summary()
    assert summary.n_rows_kept == len(dd.read_parquet(tmp_path / "points.parquet"))


def test_row_range_never_reads_a_whole_tile(
    default_bundle: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    synth, bundle = default_bundle
    reads: list[tuple[str, int]] = []
    original = ar._read_rows

    def recording(tile: Any, name: str, rows: slice) -> np.ndarray:
        reads.append((name, rows.stop - rows.start))
        return original(tile, name, rows)

    monkeypatch.setattr(ar, "_read_rows", recording)
    _all_rows(bundle, partition_rows=16)
    location_reads = [n for name, n in reads if name == "location"]
    assert max(location_reads) == 16
    assert max(synth.tile_rows.values()) > 16


def test_tile_blocks_bound_live_whole_tile_columns(
    default_bundle: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = default_bundle
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}
    original = ar._read_rows

    def release() -> None:
        with lock:
            state["now"] -= 1

    def tracked(tile: Any, name: str, rows: slice) -> np.ndarray:
        values = original(tile, name, rows)
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        # Counts every whole-tile column until its memory is freed.
        weakref.finalize(values, release)
        time.sleep(0.01)
        return values

    points, specs, _ = ar.read_atera_transcripts(
        bundle,
        spatial_order="tile_blocks",
        block_um=2.5,
        min_block_rows=1,
        max_tiles_in_flight=2,
        tile_allowlist=["0,0", "0,1"],
    )
    assert all(spec.block_bounds_um is not None for spec in specs)
    assert len(specs) == 32
    monkeypatch.setattr(ar, "_read_rows", tracked)
    with dask.config.set(scheduler="threads", num_workers=8):
        points.compute()
    # One whole-tile column per tile in flight: each is cut to its block (or
    # dropped) before the next column of that tile is read.
    assert 1 <= state["peak"] <= 2
    assert state["now"] == 0


def test_transcript_reads_refuse_other_processes(default_bundle: tuple) -> None:
    _, bundle = default_bundle
    points, _, _ = ar.read_atera_transcripts(bundle, partition_rows=16)
    with (
        pytest.raises(TypeError, match="threaded or synchronous scheduler"),
        dask.config.set(scheduler="processes", num_workers=2),
    ):
        points.compute()
    reader = ar._PartitionReader(*_reader_parts(bundle))
    with pytest.raises(TypeError, match="threaded or synchronous scheduler"):
        pickle.dumps(reader)


def _reader_parts(
    bundle: ar.AteraBundle,
) -> tuple[Any, ar.TranscriptAggregates]:
    tiles = ar.read_atera_tile_index(bundle)
    context = ar._transcript_context(
        bundle, tiles, min_qv=20.0, emit_codeword_index=False
    )
    aggregates = ar.TranscriptAggregates(
        1,
        tile_keys=[tile.key for tile in tiles],
        n_genes=len(context.gene_dtype.categories),
        min_qv=20.0,
        density_grid=context.density_grid,
    )
    return context, aggregates


# --------------------------------------------------------------------------- #
# Cells, labels and shapes
# --------------------------------------------------------------------------- #


def test_read_cells(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)
    assert cells.n_cells == synth.n_cells
    assert list(cells.source_cell_id) == cell_id_strings(
        synth.cell_prefix, synth.cell_suffix
    )
    np.testing.assert_array_equal(
        cells.vendor_cell_id,
        (synth.cell_suffix.astype(np.int64) << 32) | synth.cell_prefix,
    )
    np.testing.assert_array_equal(cells.nucleus_cell_index, synth.nucleus_cell_index)
    np.testing.assert_array_equal(
        cells.nucleus_cell_instance_id, synth.nucleus_cell_index + 1
    )
    assert cells.summary.loc[NUCLEUS_LESS_CELL, ["nucleus_area"]].isna().all()
    assert cells.summary.loc[TWO_NUCLEUS_CELL, "nucleus_count"] == 2
    assert cells.cell_segmentation_method[1] == "Segmented by interior stain (18S)"
    assert set(cells.nucleus_segmentation_method) <= {
        "Segmented by nuclear stain (DAPI) multi-focus",
        "Segmented by nuclear stain (DAPI) z stack",
    }


def test_labels_are_rows_plus_one(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    labels = ar.read_atera_labels(bundle, chunks=(32, 32))
    np.testing.assert_array_equal(labels["cell_labels"].compute(), synth.cell_mask)
    np.testing.assert_array_equal(
        labels["nucleus_labels"].compute(), synth.nucleus_mask
    )
    rows, cols = np.nonzero(synth.cell_mask == 2)
    r0, r1, c0, c1 = synth.cell_rects[1][0]
    assert (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1) == (r0, r1, c0, c1)


def _bump_nucleus_count(group: zarr.Group) -> None:
    array = group["cell_summary"]
    values = np.asarray(array[:])
    values[0, list(array.attrs["column_names"]).index("nucleus_count")] += 1
    array[...] = values


def _duplicate_first_cell_id(group: zarr.Group) -> None:
    values = np.asarray(group["cell_id"][:])
    values[1] = values[0]
    group["cell_id"][...] = values


def _swap_cell_polygon_rows(group: zarr.Group) -> None:
    array = group["polygon_sets/1/cell_index"]
    values = np.asarray(array[:])
    values[[0, 1]] = values[[1, 0]]
    array[...] = values


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_bump_nucleus_count, "nucleus_count disagrees"),
        (_duplicate_first_cell_id, "duplicate cell IDs"),
        (_swap_cell_polygon_rows, "not the cell row order"),
    ],
)
def test_inconsistent_cell_records_fail(
    default_bundle: tuple,
    tmp_path: Path,
    edit: Callable[[zarr.Group], None],
    message: str,
) -> None:
    synth, _ = default_bundle
    root = _copy_bundle(synth, tmp_path)
    rewrite_zip_store(root / "cells.zarr.zip", edit)
    with pytest.raises(AteraFormatError, match=message):
        ar.read_atera_cells(ar.resolve_atera_bundle(root))


def test_grid_counts_must_match_location_rows(
    default_bundle: tuple, tmp_path: Path
) -> None:
    synth, _ = default_bundle
    root = _copy_bundle(synth, tmp_path)

    def bump(group: zarr.Group) -> None:
        counts = list(group["grid"].attrs["grid_number_objects"])
        counts[0] += 1
        group["grid"].attrs["grid_number_objects"] = counts

    rewrite_zip_store(root / "transcripts.zarr.zip", bump)
    bundle = ar.resolve_atera_bundle(root)
    with pytest.raises(AteraFormatError, match="location rows"):
        ar.read_atera_tile_index(bundle)


def test_morphology_uuid_must_match_tiffdata(
    default_bundle: tuple, tmp_path: Path
) -> None:
    synth, _ = default_bundle
    root = _copy_bundle(synth, tmp_path)
    path = root / "morphology_2d" / synth.channel_files[1]
    xml = tifffile.tiffcomment(path)
    assert isinstance(xml, str)
    other = 'UUID="urn:uuid:00000000-0000-4000-8000-0000000000ff"'
    edited = re.sub(r'UUID="urn:uuid:[^"]*"', other, xml, count=1)
    tifffile.tiffcomment(path, edited.encode())
    with pytest.raises(AteraFormatError, match="UUID does not match"):
        ar.resolve_atera_bundle(root)


def test_labels_must_match_the_morphology_shape(default_bundle: tuple) -> None:
    _, bundle = default_bundle
    with pytest.raises(AteraFormatError, match="morphology shape"):
        ar.read_atera_labels(replace(bundle, morphology_shape_yx=(1, 1)))


def test_homogeneous_transform_check() -> None:
    ps = 0.21249222
    ar.check_homogeneous_transform(
        np.diag([np.float32(1 / ps), np.float32(1 / ps), 1, 1]).astype(np.float32), ps
    )
    shifted = np.diag([1 / ps, 1 / ps, 1, 1])
    shifted[0, 3] = 1.0
    sheared = np.diag([1 / ps, 1 / ps, 1, 1])
    sheared[0, 1] = 0.01
    for matrix in (shifted, sheared, np.diag([1 / 0.2125, 1 / 0.2125, 1, 1])):
        with pytest.raises(AteraFormatError, match="pure 1/pixel_size scale"):
            ar.check_homogeneous_transform(matrix, ps)


def _corner_centroids(mask: np.ndarray, n_labels: int, ps: float) -> np.ndarray:
    out = np.zeros((n_labels, 2))
    for label in range(1, n_labels + 1):
        rows, cols = np.nonzero(mask == label)
        out[label - 1] = ((cols.mean() + 0.5) * ps, (rows.mean() + 0.5) * ps)
    return out


@pytest.fixture(scope="module")
def mask_shapes(
    atera: BundleFactory, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    synth, bundle = atera()
    cells = ar.read_atera_cells(bundle)
    workdir = tmp_path_factory.mktemp("polygons")
    shapes = ar.atera_shapes_from_masks(bundle, cells, n_jobs=1, workdir=workdir)
    return {"shapes": shapes, "cells": cells, "workdir": workdir}


def _centroids(frame: Any) -> np.ndarray:
    return np.column_stack([frame.geometry.centroid.x, frame.geometry.centroid.y])


def test_mask_shape_centroids_match_cell_summary(
    default_bundle: tuple, tmp_path: Path
) -> None:
    synth, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)
    shapes = ar.atera_shapes_from_masks(
        bundle, cells, n_jobs=1, workdir=tmp_path, simplify_px=0.0
    )
    expected = cells.summary[["cell_centroid_x", "cell_centroid_y"]].to_numpy()
    assert np.abs(_centroids(shapes["cell_boundaries"]) - expected).max() < 0.01
    truth = _corner_centroids(synth.nucleus_mask, cells.n_nuclei, bundle.pixel_size_um)
    assert np.abs(_centroids(shapes["nucleus_boundaries"]) - truth).max() < 0.01
    # Plain c * ps (no half-pixel shift) would put every centroid 0.5 px off.
    assert 0.5 * bundle.pixel_size_um > 0.1


def test_simplified_mask_shapes_stay_close(
    default_bundle: tuple, mask_shapes: dict[str, Any]
) -> None:
    _, bundle = default_bundle
    cells = mask_shapes["cells"]
    gdf = mask_shapes["shapes"]["cell_boundaries"]
    expected = cells.summary[["cell_centroid_x", "cell_centroid_y"]].to_numpy()
    # Simplifying the tiny synthetic cells by 0.5 px moves centroids < 0.1 px.
    assert np.abs(_centroids(gdf) - expected).max() < 0.1 * bundle.pixel_size_um
    area = gdf.geometry.area.to_numpy() / cells.summary["cell_area"].to_numpy()
    assert np.all(area > 0.9)


def test_mask_shapes_columns_and_ids(mask_shapes: dict[str, Any]) -> None:
    cells = mask_shapes["cells"]
    cell_shapes = mask_shapes["shapes"]["cell_boundaries"]
    nuclei = mask_shapes["shapes"]["nucleus_boundaries"]
    for frame in (cell_shapes, nuclei):
        assert frame.index.name == "instance_id"
        assert str(frame.index.dtype) == "uint64"
        assert frame["instance_id"].dtype == np.uint64
        np.testing.assert_array_equal(frame.index.to_numpy(), frame["instance_id"])
    np.testing.assert_array_equal(cell_shapes["instance_id"], cells.instance_id)
    np.testing.assert_array_equal(nuclei["instance_id"], cells.nucleus_instance_id)
    np.testing.assert_array_equal(
        nuclei["cell_instance_id"], cells.nucleus_cell_instance_id
    )
    np.testing.assert_array_equal(
        nuclei["source_cell_id"], cells.source_cell_id[cells.nucleus_cell_index]
    )
    assert cell_shapes.geometry.iloc[MULTI_COMPONENT_CELL].geom_type == "MultiPolygon"
    stats = ar.polygon_vertex_stats(cell_shapes)
    assert stats["n_multipolygons"] == 1
    assert stats["n_shapes"] == cells.n_cells
    assert not list(mask_shapes["workdir"].glob("*.npy"))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda labeled: labeled[:-1], "without a polygon"),
        (
            lambda labeled: [*labeled, (len(labeled) + 1, labeled[0][1])],
            "exceeds",
        ),
    ],
)
def test_mask_labels_must_match_the_records(
    default_bundle: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[list[Any]], list[Any]],
    message: str,
) -> None:
    _, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)

    def polygonise(*args: Any, **kwargs: Any) -> list[Any]:
        return change(masks_to_labeled_polygons(*args, **kwargs))

    monkeypatch.setattr(ar, "masks_to_labeled_polygons", polygonise)
    with pytest.raises(AteraFormatError, match=message):
        ar.atera_shapes_from_masks(bundle, cells, n_jobs=1, workdir=tmp_path)
    assert not list(tmp_path.glob("*.npy"))


def test_mask_shapes_refuse_fork(default_bundle: tuple, tmp_path: Path) -> None:
    _, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)
    with pytest.raises(ValueError, match="fork is unsafe"):
        ar.atera_shapes_from_masks(
            bundle,
            cells,
            n_jobs=1,
            workdir=tmp_path,
            mp_context="fork",  # type: ignore[arg-type]
        )


def test_failed_mask_export_leaves_no_npy(
    default_bundle: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)

    def partial_export(bundle: Any, member: str, path: Path, **kwargs: Any) -> Path:
        path.write_bytes(b"partial")
        raise OSError("no space left on device")

    monkeypatch.setattr(ar, "export_mask_npy", partial_export)
    with pytest.raises(OSError, match="no space"):
        ar.atera_shapes_from_masks(bundle, cells, n_jobs=1, workdir=tmp_path)
    assert not list(tmp_path.glob("*.npy"))


def test_default_simplification_keeps_curved_contours() -> None:
    # An irregular blob: a disc joined to an offset ellipse, unlike the
    # fixture's rectangles, so simplifying really removes vertices.
    rows, cols = np.mgrid[0:80, 0:90]
    blob = ((rows - 35) ** 2 + (cols - 30) ** 2 < 22**2) | (
        ((rows - 45) / 14.0) ** 2 + ((cols - 60) / 25.0) ** 2 < 1
    )
    mask = blob.astype(np.int32)
    ps = 0.21249222
    [(_, contour)] = masks_to_labeled_polygons(mask, n_jobs=1)
    full = ar._pixel_polygons_to_microns(np.array([contour]), ps, 0.0)[0]
    simple = ar._pixel_polygons_to_microns(
        np.array([contour]), ps, ar.DEFAULT_SIMPLIFY_PX, n_threads=2
    )[0]
    assert ar.DEFAULT_SIMPLIFY_PX == 0.25
    assert shapely.get_num_coordinates(simple) < shapely.get_num_coordinates(full)
    # Sub-0.5 px tolerances only drop collinear staircase vertices.
    assert shapely.hausdorff_distance(simple, full) <= 0.25 * ps + 1e-9
    r, c = np.nonzero(mask)
    centroid = ((c.mean() + 0.5) * ps, (r.mean() + 0.5) * ps)
    assert abs(simple.centroid.x - centroid[0]) < 0.01
    assert abs(simple.centroid.y - centroid[1]) < 0.01
    assert simple.area == pytest.approx(mask.sum() * ps**2, rel=0.01)


def test_vendor_polygons_with_too_few_vertices_fail(default_bundle: tuple) -> None:
    _, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)
    counts = cells.cell_num_vertices.copy()
    counts[0] = 3
    with pytest.raises(AteraFormatError, match="1-3 vertices"):
        ar.atera_shapes_from_vendor_polygons(
            replace(cells, cell_num_vertices=counts), bundle.pixel_size_um
        )


def test_vendor_polygon_fallback(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)
    shapes = ar.atera_shapes_from_vendor_polygons(cells, bundle.pixel_size_um)
    cell_shapes = shapes["cell_boundaries"]
    centroid = np.column_stack(
        [cell_shapes.geometry.centroid.x, cell_shapes.geometry.centroid.y]
    )
    expected = cells.summary[["cell_centroid_x", "cell_centroid_y"]].to_numpy()
    single = np.arange(cells.n_cells) != MULTI_COMPONENT_CELL
    assert np.abs(centroid[single] - expected[single]).max() < 0.01
    # Rectangles: 4 distinct vertices, closed again after dropping the stored copy.
    assert all(len(geom.exterior.coords) == 5 for geom in cell_shapes.geometry)
    nuclei = shapes["nucleus_boundaries"]
    empty_row = int(np.flatnonzero(synth.nucleus_cell_index == EMPTY_POLYGON_CELL)[0])
    assert empty_row + 1 not in set(nuclei["instance_id"])
    assert len(nuclei) == cells.n_nuclei - 1
    assert nuclei.index.name == "instance_id"


def test_polygonisation_pool_failure_raises(
    atera: BundleFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = atera(n_cells=70)
    cells = ar.read_atera_cells(bundle)

    def broken_pool(*args: Any, **kwargs: Any) -> Any:
        raise OSError("worker died")

    monkeypatch.setattr(mask_geometry, "_polygonise_in_pool", broken_pool)
    with pytest.raises(RuntimeError, match="Polygonisation pool"):
        ar.atera_shapes_from_masks(bundle, cells, n_jobs=2, workdir=tmp_path)
    assert not list(tmp_path.glob("*.npy"))


def test_polygonisation_pool_maps_the_exported_mask(
    atera: BundleFactory, tmp_path: Path
) -> None:
    _, bundle = atera(n_cells=70)
    cells = ar.read_atera_cells(bundle)
    shapes = ar.atera_shapes_from_masks(
        bundle,
        cells,
        n_jobs=2,
        workdir=tmp_path,
        simplify_px=0.0,
        mp_context="forkserver",
    )
    assert len(shapes["cell_boundaries"]) == 70
    centroid = shapes["cell_boundaries"].geometry.centroid
    expected = cells.summary[["cell_centroid_x", "cell_centroid_y"]].to_numpy()
    assert np.abs(np.column_stack([centroid.x, centroid.y]) - expected).max() < 0.01


def test_nucleus_area_total_sums_every_nucleus(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)
    total = ar.atera_nucleus_area_total(bundle, cells, slab_bytes=1)
    pixels = np.bincount(synth.nucleus_mask.ravel(), minlength=cells.n_nuclei + 1)[1:]
    expected = np.bincount(
        synth.nucleus_cell_index, weights=pixels, minlength=cells.n_cells
    ) * (bundle.pixel_size_um**2)
    assert np.isnan(total[NUCLEUS_LESS_CELL])
    has = np.arange(cells.n_cells) != NUCLEUS_LESS_CELL
    np.testing.assert_allclose(total[has], expected[has])
    largest = cells.summary["nucleus_area"].to_numpy()
    assert total[TWO_NUCLEUS_CELL] > largest[TWO_NUCLEUS_CELL]


# --------------------------------------------------------------------------- #
# Pixel convention
# --------------------------------------------------------------------------- #


def test_corner_convention_reproduces_vendor_assignment(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    rows = _all_rows(bundle, min_qv=20.0)
    ps = bundle.pixel_size_um
    a_inv = np.diag([1.0 / ps, 1.0 / ps])
    vendor = rows["original_assignment"].fillna(0).to_numpy(dtype=np.int64)
    corner = assign_labels_from_masks(
        rows["x"].to_numpy(np.float64),
        rows["y"].to_numpy(np.float64),
        synth.cell_mask,
        a_inv,
        np.zeros(2),
        pixel_convention="corner",
    )
    np.testing.assert_array_equal(corner, vendor)
    center = assign_labels_from_masks(
        rows["x"].to_numpy(np.float64),
        rows["y"].to_numpy(np.float64),
        synth.cell_mask,
        a_inv,
        np.zeros(2),
    )
    assert (center != vendor).any()


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #


def test_table_gene_only_with_derived_controls(
    default_bundle: tuple, mask_shapes: dict[str, Any]
) -> None:
    synth, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)
    table = ar.read_atera_table(bundle, cells)
    genes = np.flatnonzero(np.asarray(synth.feature_types) == "Gene Expression")
    assert list(table.var_names) == [synth.gene_names[g] for g in genes]
    np.testing.assert_array_equal(table.X.toarray(), synth.cfm_counts[:, genes])
    assert table.X.dtype == np.float32
    assert list(table.var["gene_ids"]) == [synth.feature_ids[g] for g in genes]
    assert table.var.loc["ENSG00000289701", "gene_ids"] == "ENSG00000289701"
    types = np.asarray(synth.feature_types)
    for column, feature_type in ar.DERIVED_CONTROL_COUNT_COLUMNS.items():
        np.testing.assert_array_equal(
            table.obs[column], synth.cfm_counts[:, types == feature_type].sum(axis=1)
        )
    assert table.obs["vendor_deprecated_codeword_counts"].sum() > 0
    np.testing.assert_array_equal(
        table.obs["vendor_total_counts"], synth.cfm_counts.sum(axis=1)
    )
    np.testing.assert_array_equal(table.obs["instance_id"], cells.instance_id)
    assert list(table.obs_names) == [str(i) for i in cells.instance_id]
    np.testing.assert_array_equal(
        table.obs["source_cell_id"],
        mask_shapes["shapes"]["cell_boundaries"]["source_cell_id"],
    )
    assert "cell_id" not in table.obs
    assert table.obs.loc["2", "nucleus_count"] == 2
    assert table.obs.loc["2", "nucleus_area_total"] > table.obs.loc["2", "nucleus_area"]
    assert np.isnan(table.obs.loc[str(NUCLEUS_LESS_CELL + 1), "nucleus_area"])
    np.testing.assert_array_equal(table.obs["vendor_filtered"], synth.cfm_filtered)
    np.testing.assert_allclose(
        table.obsm["spatial"],
        cells.summary[["cell_centroid_x", "cell_centroid_y"]].to_numpy(),
    )
    assert set(table.obsm) == {"spatial", "X_pca_vendor", "X_umap_vendor"}
    assert "highly_variable" not in table.var
    assert table.var["vendor_highly_variable"].dtype == bool
    assert not [c for c in table.var.columns if c.startswith("de_")]
    coverage = table.var["vendor_gene_coverage"]
    assert coverage.isna().sum() == 1 and pd.isna(coverage["BOOSTGENE1"])
    assert "vendor_is_sketch" not in table.obs


def test_table_options_and_optional_vendor_columns(atera: BundleFactory) -> None:
    synth, bundle = atera(with_is_sketch=True)
    cells = ar.read_atera_cells(bundle)
    area = np.zeros(cells.n_cells)
    table = ar.read_atera_table(bundle, cells, nucleus_area_total=area)
    assert "vendor_is_sketch" in table.obs
    np.testing.assert_array_equal(table.obs["nucleus_area_total"], area)
    full = ar.read_atera_table(
        bundle,
        cells,
        gex_only=False,
        keep_vendor_annotations=False,
        nucleus_area_total=area,
    )
    assert full.n_vars == len(synth.gene_names)
    assert not [c for c in full.obs.columns if c.startswith("vendor_leiden")]
    assert set(full.obsm) == {"spatial"}


def test_table_rejects_a_different_cell_order(default_bundle: tuple) -> None:
    _, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)
    reordered = replace(cells, source_cell_id=cells.source_cell_id[::-1].copy())
    with pytest.raises(AteraFormatError, match="barcodes differ"):
        ar.read_atera_table(bundle, reordered, nucleus_area_total=np.zeros(6))
    packed = cells.vendor_cell_id.copy()
    packed[0] += 1
    with pytest.raises(AteraFormatError, match=r"obs\.cell_id differs"):
        ar.read_atera_table(
            bundle, replace(cells, vendor_cell_id=packed), nucleus_area_total=None
        )


def test_table_features_must_match_transcript_genes(
    default_bundle: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = default_bundle
    cells = ar.read_atera_cells(bundle)
    genes = ar.read_atera_gene_names(bundle)
    monkeypatch.setattr(
        ar, "read_atera_gene_names", lambda bundle: (*genes[1:], genes[0])
    )
    with pytest.raises(AteraFormatError, match="differ from transcripts gene_names"):
        ar.read_atera_table(bundle, cells)


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def test_channels_follow_ome_tiffdata(atera: BundleFactory) -> None:
    synth, bundle = atera(n_channels=4, file_order_differs=True)
    assert bundle.morphology_channel_names == (
        "DAPI",
        "ATP1A1/CD45/E-Cadherin",
        "18S",
        "alphaSMA/Vimentin",
    )
    files = [path.name for path in bundle.morphology_2d_files]
    assert files == list(synth.channel_files)
    assert files != sorted(files)
    tree = ar.read_atera_morphology_2d(bundle, chunks=(32, 32))
    image = tree["scale0"]["image"]
    assert list(image.coords["c"].values) == list(synth.channel_names)
    np.testing.assert_array_equal(image.data.compute(), np.stack(synth.channel_level0))


def test_stored_pyramid_levels_are_used(default_bundle: tuple, tmp_path: Path) -> None:
    synth, bundle = default_bundle
    tree = ar.read_atera_morphology_2d(bundle, n_levels=3)
    shapes = [tree[f"scale{k}"]["image"].shape for k in range(3)]
    assert shapes == [(2, 97, 67), (2, 48, 33), (2, 24, 16)]
    Image2DModel().validate(tree)
    assert isinstance(get_transformation(tree, "global"), Identity)
    # Level scales on disk come from shape ratios, not exact powers of two.
    store = tmp_path / "image.zarr"
    sd.SpatialData(images={"morphology_focus": tree}).write(store)
    attrs = zarr.open_group(store / "images" / "morphology_focus", mode="r").attrs
    multiscales = attrs.get("multiscales") or attrs["ome"]["multiscales"]
    scales = [
        dataset["coordinateTransformations"][0]["scale"]
        for dataset in multiscales[0]["datasets"]
    ]
    assert scales[1][1:] == pytest.approx([97 / 48, 67 / 33])
    assert scales[2][1:] == pytest.approx([97 / 24, 67 / 16])


def test_morphology_reads_leave_no_open_tiff(default_bundle: tuple) -> None:
    _, bundle = default_bundle
    tree = ar.read_atera_morphology_2d(bundle, chunks=(16, 16))
    tree["scale0"]["image"].data.compute()
    open_paths = {Path(f.path) for f in psutil.Process().open_files()}
    assert not open_paths & set(bundle.morphology_2d_files)


# --------------------------------------------------------------------------- #
# Vendor metadata: verbatim sidecars and the full header
# --------------------------------------------------------------------------- #


def test_sidecars_are_byte_identical_copies(
    default_bundle: tuple, tmp_path: Path
) -> None:
    synth, bundle = default_bundle
    store = tmp_path / "store.zarr"
    records = ar.copy_atera_sidecars(bundle, store)
    by_path = {record["path"]: record for record in records}
    assert set(by_path) == {f"atera/{name}" for name in EXPECTED_SIDECARS}
    for stored, original in EXPECTED_SIDECARS.items():
        record = by_path[f"atera/{stored}"]
        assert record["original_name"] == original
        source = (synth.root / original).read_bytes()
        assert (store / "atera" / stored).read_bytes() == source
        assert record["size_bytes"] == len(source)
        assert record["sha256"] == hashlib.sha256(source).hexdigest()
    # Nothing is redacted: customer, designer and run fields are all there.
    panel = (store / "atera" / "panel_config.json").read_text()
    metrics = (store / "atera" / "metrics_summary.csv").read_text()
    report = (store / "atera" / "analysis_summary.html").read_text()
    for value in (
        FAKE_IDENTIFIERS["customer_email"],
        FAKE_IDENTIFIERS["designer_name"],
    ):
        assert value in panel
    for value in (FAKE_IDENTIFIERS["run_name"], FAKE_IDENTIFIERS["reagent_lot"]):
        assert value in metrics
    assert FAKE_IDENTIFIERS["run_name"] in report


def test_metadata_attrs_carry_the_full_header(default_bundle: tuple) -> None:
    synth, bundle = default_bundle
    records = [{"path": "atera/experiment.spatial", "original_name": "x"}]
    attrs = ar.atera_metadata_attrs(bundle, records)
    assert json.loads(json.dumps(attrs, allow_nan=False)) == attrs
    experiment = json.loads((synth.root / "experiment.spatial").read_text())
    assert attrs["experiment"] == experiment
    for key in (
        "run_name",
        "slide_name",
        "slide_id",
        "region_name",
        "instrument_sn",
        "analysis_uuid",
        "roi_uuid",
    ):
        assert attrs["experiment"][key] == experiment[key]
    assert attrs["experiment"]["instrument_sn"] == FAKE_IDENTIFIERS["instrument_sn"]
    (metrics,) = attrs["metrics_summary"]
    assert metrics["run_name"] == FAKE_IDENTIFIERS["run_name"]
    assert metrics["region_name"] == FAKE_IDENTIFIERS["region_name"]
    assert metrics["run_start_time"] == FAKE_IDENTIFIERS["run_start_time"]
    assert metrics["reagent_plate_1"] == FAKE_IDENTIFIERS["reagent_lot"]
    assert metrics["reagent_pouch_1"] == 424242
    assert metrics["num_cells_detected"] == synth.n_cells
    assert metrics["fraction_transcripts_assigned"] == 0.5
    assert metrics["median_genes_per_cell_panel_a"] is None
    (payload,) = attrs["panel_config"]["payloads"]
    assert payload["customer"]["email"] == FAKE_IDENTIFIERS["customer_email"]
    assert (
        payload["designer"]["designed_by"]["name"]
        == (FAKE_IDENTIFIERS["designer_name"])
    )
    assert payload["panel"]["identity"]["design_id"] == "hWTA_v1"
    assert "targets" not in payload
    panel = json.loads((synth.root / "panel_config.json").read_text())
    assert payload["num_targets"] == len(panel["payloads"][0]["targets"])
    assert attrs["panel_config"]["metadata"] == {
        "ui_props": {"visible": True, "display_name": ""}
    }
    assert attrs["bundle_files"]["transcripts"] == "transcripts.zarr.zip"
    assert attrs["bundle_files"]["morphology_2d"] == list(synth.channel_files)
    assert attrs["sidecars"] == records


def test_metrics_summary_values_are_structured(tmp_path: Path) -> None:
    path = tmp_path / "metrics_summary.csv"
    path.write_text("a,b,c,d,e\n7,0.25,,NaN,text\n,,,,\n")
    assert ar.read_metrics_summary(path) == [
        {"a": 7, "b": 0.25, "c": None, "d": "NaN", "e": "text"}
    ]


def test_duplicate_sidecar_names_fail_loudly(
    default_bundle: tuple, tmp_path: Path
) -> None:
    synth, _ = default_bundle
    root = _copy_bundle(synth, tmp_path)
    (root / "metrics_summary_second_run.csv").write_text("num_cells\n1\n")
    with pytest.raises(AteraFormatError, match="metrics_summary.csv"):
        ar.resolve_atera_bundle(root)


def test_stable_sidecar_names() -> None:
    for name, stable in (
        ("metrics_summary_SN_SLIDE_ROI.csv", "metrics_summary.csv"),
        ("analysis_summary_SN_SLIDE_ROI.html", "analysis_summary.html"),
        ("Some_dataset_he_imagealignment.csv", "he_imagealignment.csv"),
        ("Some_dataset_keypoints.csv", "keypoints.csv"),
        ("Some_dataset_transcripts.zarr.zip", "transcripts.zarr.zip"),
        ("binned_transcripts.zarr.zip", "binned_transcripts.zarr.zip"),
        ("Some_dataset_binned_transcripts.zarr.zip", "binned_transcripts.zarr.zip"),
        ("csc_cell_feature_matrix.zarr.zip", "csc_cell_feature_matrix.zarr.zip"),
        ("Some_csc_cell_feature_matrix.zarr.zip", "csc_cell_feature_matrix.zarr.zip"),
        ("Some_dataset_cell_feature_matrix.zarr.zip", "cell_feature_matrix.zarr.zip"),
        ("panel_config.json", "panel_config.json"),
    ):
        assert ar.stable_sidecar_name(name) == stable
