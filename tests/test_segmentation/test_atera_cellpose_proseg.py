"""Cellpose-prior tiled ProSeg and the Cellpose layer of ATERA (plan AT-7).

The synthetic multi-tile ATERA store of the AT-9 tests gets a fake Cellpose
mask on its mask grid: the vendor cell labels moved one pixel right, renumbered
(``1000 + 7 i``) and with one cell dropped, so every check that would pass on
the vendor labels by accident (ids, pixels, counts) fails if the vendor
segmentation leaks into the Cellpose route. The plan, the tiles (fake ProSeg),
the stitch and ENRICH then run on it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import spatialdata as sd
from test_io.atera_synth import PIXEL_SIZE_UM

from merxen.analysis_layers import validate_analysis_layer
from merxen.config import AteraProsegPlanConfig, AteraProsegPrior, EnrichmentConfig
from merxen.enrichment.assignment import run_per_shape_assignment_for_dataset
from merxen.enrichment.enrich import enrich_single_latest
from merxen.io.points_aggregates import points_parquet_dir, read_points_aggregates
from merxen.io.spatialdata_schema import MERXEN_SCHEMA_ATTR, PROSEG_ID_NAMESPACE
from merxen.segmentation.atera_proseg import (
    CELLPOSE_ASSIGNMENT_COLUMN,
    PROSEG_LAYERS,
    AteraProsegError,
)
from merxen.segmentation.atera_proseg.plan import (
    _level0,
    build_plan,
    read_plan,
    write_plan,
)
from merxen.segmentation.atera_proseg.prior import label_stats, lookup_labels
from merxen.segmentation.atera_proseg.runner import build_command
from merxen.segmentation.atera_proseg.stitch import check_proseg_layer, stitch
from merxen.segmentation.atera_proseg.tile import run_tile
from test_segmentation.atera_proseg_fixture import (
    DATASET,
    build_multitile_latest,
    fake_proseg_binary,
    plan_config,
    stitch_config,
    tile_configs,
)

PS = PIXEL_SIZE_UM


def _fake_cellpose(source: Path, out: Path) -> tuple[Path, Path, dict[int, int]]:
    """Write a Cellpose-like mask and logits derived from the vendor labels."""
    vendor = np.asarray(_level0(source / "labels" / "cell_labels")[:], dtype=np.int64)
    shifted = np.zeros_like(vendor)
    shifted[:, 1:] = vendor[:, :-1]
    labels = np.unique(shifted[shifted > 0])
    mapping = {int(v): 1000 + 7 * i for i, v in enumerate(labels)}
    mapping[int(labels[-1])] = 0
    lut = np.zeros(int(shifted.max()) + 1, dtype=np.uint32)
    for old, new in mapping.items():
        lut[old] = new
    mask = lut[shifted]
    out.mkdir(parents=True, exist_ok=True)
    mask_path = out / "cellpose_masks_tiled.npy"
    cellprob_path = out / "cellpose_cellprobs_tiled.npy"
    np.save(mask_path, mask.astype(np.uint32))
    np.save(cellprob_path, np.where(mask > 0, 3.0, -3.0).astype(np.float32))
    return mask_path, cellprob_path, mapping


@dataclass(frozen=True)
class Route:
    source: Path
    latest: Path
    mask_path: Path
    cellprob_path: Path
    plan_path: Path
    tile_dirs: list[Path]
    root: Path


def _cellpose_plan_config(
    root: Path, source: Path, mask: Path, cellprob: Path | None
) -> AteraProsegPlanConfig:
    return plan_config(
        source,
        root / "plan_out",
        prior=AteraProsegPrior(kind="cellpose", mask_path=mask, cellprob_path=cellprob),
    )


@pytest.fixture(scope="module")
def route(tmp_path_factory: pytest.TempPathFactory) -> Route:
    root = tmp_path_factory.mktemp("cellpose_route")
    source, latest = build_multitile_latest(root)
    mask, cellprob, _ = _fake_cellpose(source, root / "segmentation")
    plan_path = write_plan(_cellpose_plan_config(root, source, mask, cellprob))
    plan = read_plan(plan_path)
    # greedy_halo: ProSeg also gives rows off the Cellpose cells to nearby
    # cells, so its assignment differs from the Cellpose mask's.
    binary = fake_proseg_binary(root / "bin", FAKE_PROSEG_MODE="greedy_halo")
    dirs = []
    for config in tile_configs(
        plan,
        source,
        root / "tiles",
        prior=_tile_prior(plan),
        model=_model(plan),
    ):
        run_tile(config, binary)
        dirs.append(config.output_dir)
    stitch(
        stitch_config(
            latest, plan_path, dirs, root / "stitch_out", prior_kind="cellpose"
        )
    )
    return Route(source, latest, mask, cellprob, plan_path, dirs, root)


def _tile_prior(plan: dict) -> dict:
    prior = plan["prior"]
    return {
        key: prior[key]
        for key in (
            "kind",
            "mask_path",
            "cellprob_path",
            "mask_stat",
            "cellprob_stat",
            "fingerprint",
        )
    }


def _model(plan: dict) -> dict:
    return {
        "burnin_samples": 2,
        "samples": 2,
        "hillclimb": 0,
        "recorded_samples": 1,
        "prior_weight_effective": plan["prior_weight"]["effective"],
        "nuclear_compartment": False,
    }


@pytest.fixture
def store(route: Route, tmp_path: Path) -> Path:
    copy = tmp_path / "latest.zarr"
    shutil.copytree(route.latest, copy)
    return copy


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


def test_label_stats_equal_a_brute_force_pass() -> None:
    rng = np.random.default_rng(0)
    mask = np.zeros((40, 30), dtype=np.uint32)
    for label in (3, 9, 27):
        r, c = rng.integers(0, 35), rng.integers(0, 25)
        mask[r : r + rng.integers(1, 5), c : c + rng.integers(1, 5)] = label
    stats = label_stats(mask, 0.5, strip_rows=7)
    assert stats["instance_id"].tolist() == [3, 9, 27]
    for row, label in enumerate(stats["instance_id"]):
        rr, cc = np.nonzero(mask == label)
        assert stats["area_px"][row] == rr.size
        np.testing.assert_allclose(
            stats["centroid"][row], [(cc.mean() + 0.5) * 0.5, (rr.mean() + 0.5) * 0.5]
        )
        np.testing.assert_allclose(
            stats["bounds"][row],
            [
                cc.min() * 0.5,
                rr.min() * 0.5,
                (cc.max() + 1) * 0.5,
                (rr.max() + 1) * 0.5,
            ],
        )


def test_cellpose_plan_owns_every_label_once_and_records_the_prior(
    route: Route,
) -> None:
    plan = read_plan(route.plan_path)
    mask = np.load(route.mask_path)
    labels = np.unique(mask[mask > 0])
    owned = [label for tile in plan["tiles"] for label in tile["owned_labels"]]
    assert sorted(owned) == labels.tolist()
    assert len(owned) == len(set(owned))
    assert plan["n_labels"] == labels.size
    assert plan["prior"]["kind"] == "cellpose"
    assert len(plan["prior"]["mask_sha256"]) == 64
    assert plan["prior"]["cellprob_sha256"] is not None
    # The whole-section automatic weight: max(gene rows / labels / 100, 1).
    aggregates = read_points_aggregates(route.source)
    assert aggregates is not None
    weight = plan["prior_weight"]
    assert weight["rule"] == "whole_section_auto"
    assert weight["n_labels"] == labels.size
    assert weight["n_gene_rows"] <= int(aggregates.gene_counts_kept.sum())
    assert weight["effective"] == pytest.approx(
        max(weight["n_gene_rows"] / labels.size / 100.0, 1.0)
    )


def test_a_configured_weight_and_a_vendor_plan(route: Route, tmp_path: Path) -> None:
    config = _cellpose_plan_config(
        tmp_path, route.source, route.mask_path, route.cellprob_path
    ).model_copy(update={"prior_weight_effective": 140.0})
    assert build_plan(config)["prior_weight"] == {
        "rule": "configured",
        "effective": 140.0,
    }
    vendor = build_plan(plan_config(route.source, tmp_path / "vendor"))
    assert "prior" not in vendor and "prior_weight" not in vendor


def test_a_mask_off_the_store_grid_is_refused(route: Route, tmp_path: Path) -> None:
    small = tmp_path / "small.npy"
    np.save(small, np.ones((3, 3), dtype=np.uint32))
    config = _cellpose_plan_config(tmp_path, route.source, small, None)
    with pytest.raises(AteraProsegError, match="store's mask grid"):
        build_plan(config)


# --------------------------------------------------------------------------- #
# Tiles
# --------------------------------------------------------------------------- #


def test_tiles_seed_cell_id_from_the_cellpose_mask(
    route: Route, tmp_path: Path
) -> None:
    plan = read_plan(route.plan_path)
    # The tile farthest from the origin: its window starts off row and column
    # 0, so a window cut from the wrong place shows.
    config = max(
        tile_configs(
            plan,
            route.source,
            tmp_path / "tiles",
            prior=_tile_prior(plan),
            model=_model(plan),
            keep_inputs=True,
        ),
        key=lambda item: item.window_px[0] + item.window_px[2],
    )
    assert config.window_px[0] > 0 and config.window_px[2] > 0
    manifest = run_tile(config, fake_proseg_binary(tmp_path / "bin"))
    assert manifest["prior"]["kind"] == "cellpose"
    assert "--cellpose-cellprobs" in manifest["proseg"]["arguments"]
    inputs = Path(config.output_dir) / "work" / "inputs"
    table = pq.read_table(inputs / "tile.parquet")
    mask = np.load(route.mask_path)
    expected = lookup_labels(
        mask,
        table.column("x_location").to_numpy(),
        table.column("y_location").to_numpy(),
        plan["pixel_size_um"],
    )
    assert table.column("cell_id").to_pylist() == [str(int(v)) for v in expected]
    assert set(table.column("overlaps_nucleus").to_pylist()) == {0}
    window = np.load(inputs / "prior.npy")
    col0, col1, row0, row1 = config.window_px
    np.testing.assert_array_equal(window, mask[row0:row1, col0:col1])
    cellprob = np.load(inputs / "prior_cellprob.npy")
    assert cellprob.dtype == np.float32 and cellprob.shape == window.shape
    np.testing.assert_array_equal(
        cellprob, np.load(route.cellprob_path)[row0:row1, col0:col1]
    )


def test_vendor_command_is_unchanged_and_cellprobs_follow_the_masks() -> None:
    from merxen.config import AteraProsegModelParams

    common = {
        "parquet_path": "in/tile.parquet",
        "prior_path": "in/prior.npy",
        "output_dir": "out",
        "model": AteraProsegModelParams(),
        "prior_weight": 2.0,
        "pixel_size_um": 0.5,
        "col0": 10,
        "row0": 20,
        "num_threads": 4,
    }
    vendor = build_command("proseg", **common)
    cellpose = build_command("proseg", cellprob_path="in/prior_cellprob.npy", **common)
    assert "--cellpose-cellprobs" not in vendor
    at = cellpose.index("--cellpose-masks")
    assert cellpose[at : at + 4] == [
        "--cellpose-masks",
        "in/prior.npy",
        "--cellpose-cellprobs",
        "in/prior_cellprob.npy",
    ]
    assert [arg for arg in cellpose if arg not in {"--cellpose-cellprobs"}] == [
        *vendor[: at + 2],
        "in/prior_cellprob.npy",
        *vendor[at + 2 :],
    ]
    # The ProSeg transforms are the vendor ones (corner convention).
    x_at = cellpose.index("--cellpose-x-transform")
    assert cellpose[x_at + 1 : x_at + 4] == ["0.5", "0", "5.0"]


def test_a_changed_cellpose_mask_is_refused_by_a_tile(
    route: Route, tmp_path: Path
) -> None:
    plan = read_plan(route.plan_path)
    prior = _tile_prior(plan)
    prior["mask_stat"] = [prior["mask_stat"][0] + 1, prior["mask_stat"][1]]
    (config, *_) = tile_configs(
        plan, route.source, tmp_path / "tiles", prior=prior, model=_model(plan)
    )
    with pytest.raises(AteraProsegError, match="Cellpose re-ran"):
        run_tile(config, fake_proseg_binary(tmp_path / "bin"))


# --------------------------------------------------------------------------- #
# Stitch
# --------------------------------------------------------------------------- #


def test_stitch_writes_the_xenium_named_layer_in_the_cellpose_namespace(
    route: Route,
) -> None:
    layer = PROSEG_LAYERS["cellpose"]
    sdata = sd.read_zarr(route.latest)
    for name in (layer.shapes_key,):
        assert name in sdata.shapes
    assert layer.table_key in sdata.tables
    assert layer.label_key in sdata.labels
    assert "MOSAIK_proseg_vendor" not in sdata.shapes
    entry = sdata.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]["proseg"]
    assert entry["id_namespace"] == PROSEG_ID_NAMESPACE
    assert entry["prior"] == "cellpose_cell_labels"
    assert entry["legacy_aliases"] == ["reseg"]
    table = sdata.tables[layer.table_key]
    mask = np.load(route.mask_path)
    assert set(table.obs["instance_id"].astype(int)) <= set(np.unique(mask[mask > 0]))
    assert set(table.obs["source_cell_id"]) == {
        f"cellpose_{int(v)}" for v in table.obs["instance_id"]
    }
    provenance = sdata.attrs["merxen_atera_proseg"]
    assert provenance["prior_kind"] == "cellpose"
    assert provenance["prior"]["mask_file"] == "cellpose_masks_tiled.npy"
    assert "/" not in json.dumps(provenance["prior"]).replace("\\/", "")
    assert check_proseg_layer(route.latest)["passed"]
    result = validate_analysis_layer(
        route.latest,
        platform="ATERA",
        segmentation="reseg",
        table_key=layer.table_key,
        shape_key=layer.shapes_key,
    )
    assert result["n_cells"] == table.n_obs


def test_cellpose_assignment_is_the_mask_label_of_every_row(route: Route) -> None:
    mask = np.load(route.mask_path)
    n_rows = 0
    for path in sorted(
        points_parquet_dir(route.latest, "transcripts").glob("*.parquet")
    ):
        table = pq.read_table(path, columns=["x", "y", CELLPOSE_ASSIGNMENT_COLUMN])
        expected = lookup_labels(
            mask, table.column("x").to_numpy(), table.column("y").to_numpy(), PS
        )
        found = table.column(CELLPOSE_ASSIGNMENT_COLUMN).combine_chunks()
        assert found.type == pa.uint64()
        assert found.fill_null(0).to_numpy().tolist() == expected.tolist()
        assert (found.is_null().to_numpy(zero_copy_only=False) == (expected == 0)).all()
        n_rows += table.num_rows
    assert n_rows > 0


def test_a_vendor_stitch_replaces_the_cellpose_layer(
    route: Route, store: Path, tmp_path: Path
) -> None:
    vendor_plan = write_plan(plan_config(route.source, tmp_path / "vendor_plan"))
    plan = read_plan(vendor_plan)
    binary = fake_proseg_binary(tmp_path / "bin")
    dirs = []
    for config in tile_configs(plan, route.source, tmp_path / "vendor_tiles"):
        run_tile(config, binary)
        dirs.append(config.output_dir)
    with pytest.raises(AteraProsegError, match="vendor prior"):
        stitch(
            stitch_config(
                store, vendor_plan, dirs, tmp_path / "wrong", prior_kind="cellpose"
            )
        )
    # The row ran ENRICH as cellpose_proseg first: its proseg_mask layer reads
    # cellpose_assignment, which the vendor stitch drops (review 2, F5).
    enrich_single_latest(
        _enrich_config(route, store, tmp_path / "enrich_out"), polygon_n_jobs=2
    )
    assert "MOSAIK_cellpose" in sd.read_zarr(store).shapes
    # And VIEWER_CACHE: base masks of both layers and a derived level.
    viewer_labels = (
        "MOSAIK_proseg_labels",
        "_napari_compare_outline__MOSAIK_proseg_labels__w1",
        "MOSAIK_cellpose_labels",
        "_napari_compare_labelpyr__MOSAIK_cellpose_labels__ds4",
    )
    _write_viewer_labels(store, viewer_labels)
    report = stitch(stitch_config(store, vendor_plan, dirs, tmp_path / "vendor_out"))
    removed = report["removed_foreign_layer"]
    assert "shapes/MOSAIK_proseg" in removed
    for element in (
        "shapes/MOSAIK_cellpose",
        "tables/table_MOSAIK_cellpose",
        "labels/cellpose_cell_labels",
        "registry/cellpose",
        "attrs/merxen_atera_cellpose_layer",
        *(f"labels/{name}" for name in viewer_labels),
    ):
        assert element in removed, element
    sdata = sd.read_zarr(store)
    assert "MOSAIK_proseg" not in sdata.shapes
    assert "MOSAIK_cellpose" not in sdata.shapes
    assert "table_MOSAIK_cellpose" not in sdata.tables
    assert "cellpose_cell_labels" not in sdata.labels
    assert "merxen_atera_cellpose_layer" not in sdata.attrs
    assert "MOSAIK_proseg_vendor" in sdata.shapes
    registry = sdata.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]
    assert "proseg" not in registry and "proseg_vendor" in registry
    assert "cellpose" not in registry
    assert not set(viewer_labels) & set(sdata.labels)
    # Leftovers in a store whose layer is current (same tiles) are removed
    # without a re-stitch, so the tables derived from the layer stay.
    _write_viewer_labels(store, viewer_labels[:2])
    derived = sdata.tables["table_MOSAIK_proseg_vendor"].copy()
    sdata.tables["table_MOSAIK_proseg_vendor_image_quantification"] = derived
    sdata.write_element("table_MOSAIK_proseg_vendor_image_quantification")
    again = stitch(stitch_config(store, vendor_plan, dirs, tmp_path / "again_out"))
    assert again["skipped"] is True
    assert sorted(again["removed_foreign_layer_without_restitch"]) == sorted(
        f"labels/{name}" for name in viewer_labels[:2]
    )
    after = sd.read_zarr(store)
    assert not set(viewer_labels) & set(after.labels)
    assert "table_MOSAIK_proseg_vendor_image_quantification" in after.tables
    schema = pq.read_schema(
        next(points_parquet_dir(store, "transcripts").glob("*.parquet"))
    )
    assert CELLPOSE_ASSIGNMENT_COLUMN not in schema.names
    assert check_proseg_layer(store)["passed"]


def _write_viewer_labels(store: Path, names: tuple[str, ...]) -> None:
    from spatialdata.models import Labels2DModel

    sdata = sd.read_zarr(store, selection=("labels",))
    for name in names:
        sdata[name] = Labels2DModel.parse(
            np.zeros((4, 4), dtype=np.uint32), dims=("y", "x")
        )
        sdata.write_element(name)


# --------------------------------------------------------------------------- #
# ENRICH
# --------------------------------------------------------------------------- #


def _enrich_config(route: Route, store: Path, out: Path) -> EnrichmentConfig:
    return EnrichmentConfig(
        dataset_name=DATASET,
        platform="ATERA",
        latest_zarr_path=store,
        mask_path=route.mask_path,
        nuclei_mask_path=None,
        original_data_path=route.source,
        output_dir=out,
        segmentation_source="cellpose_proseg",
    )


def test_enrich_config_of_an_atera_cellpose_row() -> None:
    values = {
        "dataset_name": "X",
        "platform": "ATERA",
        "latest_zarr_path": Path("l.zarr"),
        "nuclei_mask_path": None,
        "original_data_path": Path("s.zarr"),
        "output_dir": Path("o"),
        "segmentation_source": "cellpose_proseg",
    }
    EnrichmentConfig(**values, mask_path=Path("m.npy"))
    with pytest.raises(ValueError, match="ATERA Cellpose mask"):
        EnrichmentConfig(**values, mask_path=None)
    with pytest.raises(ValueError, match="nuclei_mask_path"):
        EnrichmentConfig(**{**values, "platform": "XENIUM"}, mask_path=Path("m.npy"))


def test_enrich_builds_the_cellpose_layer(
    route: Route, store: Path, tmp_path: Path
) -> None:
    enrich_single_latest(
        _enrich_config(route, store, tmp_path / "enrich_out"), polygon_n_jobs=2
    )
    check = json.loads(
        (tmp_path / "enrich_out" / "atera_cellpose_layer_check.json").read_text()
    )
    assert check["check"]["passed"]
    assert json.loads(
        (tmp_path / "enrich_out" / "atera_proseg_layer_check.json").read_text()
    )["passed"]
    sdata = sd.read_zarr(store)
    mask = np.load(route.mask_path)
    labels = np.unique(mask[mask > 0])
    table = sdata.tables["table_MOSAIK_cellpose"]
    shapes = sdata.shapes["MOSAIK_cellpose"]
    assert sorted(table.obs["instance_id"].astype(int)) == labels.tolist()
    assert sorted(shapes["instance_id"].astype(int)) == labels.tolist()
    np.testing.assert_array_equal(
        np.asarray(sdata.labels["cellpose_cell_labels"]["scale0"]["image"]), mask
    )
    # Counts: every gene row on a Cellpose cell, by the corner lookup.
    gene_position = {name: i for i, name in enumerate(table.var_names)}
    expected = np.zeros(table.shape, dtype=np.int64)
    row_of = {int(v): i for i, v in enumerate(table.obs["instance_id"])}
    for path in sorted(points_parquet_dir(store, "transcripts").glob("*.parquet")):
        points = pq.read_table(path, columns=["x", "y", "is_gene", "feature_name"])
        hit = lookup_labels(
            mask, points.column("x").to_numpy(), points.column("y").to_numpy(), PS
        )
        genes = points.column("is_gene").to_numpy(zero_copy_only=False).astype(bool)
        names = points.column("feature_name").to_pylist()
        for label, gene, name in zip(hit, genes, names, strict=True):
            if label and gene:
                expected[row_of[int(label)], gene_position[name]] += 1
    np.testing.assert_array_equal(np.asarray(table.X.todense()), expected)
    # ProSeg moved rows (greedy_halo), so a table counted from its assignment
    # instead of the Cellpose mask's would differ.
    proseg = sdata.tables["table_MOSAIK_proseg"]
    proseg_rows = {int(v): i for i, v in enumerate(proseg.obs["instance_id"])}
    order = [proseg_rows[int(v)] for v in table.obs["instance_id"]]
    assert not np.array_equal(
        np.asarray(proseg.X[order].todense()), np.asarray(table.X.todense())
    )
    entry = sdata.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]["cellpose"]
    assert entry["assignment_column"] == CELLPOSE_ASSIGNMENT_COLUMN
    assert entry["id_namespace"] == PROSEG_ID_NAMESPACE
    assert entry["label"] == "cellpose_cell_labels"
    assert entry["legacy_aliases"] == ["proseg_mask", "cellpose_mask"]
    # The vendor layer was enriched natively.
    assert "merxen_native_enrichment" in sdata.attrs
    for segmentation, table_key, shape_key in (
        ("reseg", "table_MOSAIK_proseg", "MOSAIK_proseg"),
        ("original_seg", "table_original", "atera_cell_boundaries"),
        ("proseg_mask", "table_MOSAIK_cellpose", "MOSAIK_cellpose"),
    ):
        validate_analysis_layer(
            store,
            platform="ATERA",
            segmentation=segmentation,
            table_key=table_key,
            shape_key=shape_key,
        )
    # Per-shape assignment registers the three tables as written.
    before = json.dumps(sdata.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]["cellpose"])
    run_per_shape_assignment_for_dataset(
        dataset_name=DATASET,
        latest_path=store,
        platform="ATERA",
        segmentation_source="cellpose_proseg",
    )
    after = sd.read_zarr(store).attrs[MERXEN_SCHEMA_ATTR]["segmentations"]
    assert json.dumps(after["cellpose"]) == before
    # A second ENRICH keeps the layer.
    enrich_single_latest(
        _enrich_config(route, store, tmp_path / "again"), polygon_n_jobs=2
    )
    assert (
        sd.read_zarr(store).attrs["merxen_atera_cellpose_layer"]["built_at"]
        == check["built_at"]
    )


def test_enrich_refuses_a_store_stitched_from_another_mask(
    route: Route, store: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other.npy"
    np.save(other, np.load(route.mask_path))
    with pytest.raises(RuntimeError, match="another Cellpose mask"):
        enrich_single_latest(
            _enrich_config(route, store, tmp_path / "out").model_copy(
                update={"mask_path": other}
            ),
            polygon_n_jobs=2,
        )


def test_an_unprinted_weight_is_accepted_only_at_one() -> None:
    """ProSeg prints its weight only above 1 (a sparse section's auto weight)."""
    from merxen.segmentation.atera_proseg import PriorWeightMismatchError
    from merxen.segmentation.atera_proseg.runner import check_prior_weight

    check_prior_weight(None, 1.0, 0.02)
    check_prior_weight(None, 1.01, 0.02)
    with pytest.raises(PriorWeightMismatchError, match="printed no prior weight"):
        check_prior_weight(None, 140.0, 0.02)


def test_a_rebuilt_cellpose_layer_drops_its_derived_tables(
    route: Route, store: Path, tmp_path: Path
) -> None:
    from spatialdata.models import TableModel

    from merxen.enrichment.atera_cellpose_layer import build_atera_cellpose_layer
    from merxen.io.spatialdata_io import write_or_replace_element

    enrich_single_latest(
        _enrich_config(route, store, tmp_path / "enrich_out"), polygon_n_jobs=2
    )
    sdata = sd.read_zarr(store, selection=("tables",))
    derived = sdata.tables["table_MOSAIK_cellpose"].copy()
    derived.uns.pop("spatialdata_attrs", None)
    derived = TableModel.parse(
        derived,
        region="MOSAIK_cellpose",
        region_key="region",
        instance_key="instance_id",
    )
    write_or_replace_element(
        sdata, "table_MOSAIK_cellpose_image_quantification", "tables", derived
    )
    del sdata
    record = build_atera_cellpose_layer(
        store,
        route.mask_path,
        pixel_size_um=PS,
        dataset_name=DATASET,
        n_jobs=2,
        model_identity=None,
        force=True,
    )
    assert record["deleted_derived_tables"] == [
        "table_MOSAIK_cellpose_image_quantification"
    ]
    tables = sd.read_zarr(store, selection=("tables",)).tables
    assert "table_MOSAIK_cellpose_image_quantification" not in tables
    assert "table_MOSAIK_cellpose" in tables


def test_points_labels_off_the_mask_are_counted_not_indexed(tmp_path: Path) -> None:
    # Review of AT-7 (F7): a cellpose_assignment above the mask's largest label
    # is reported as unknown instead of raising an IndexError.
    from merxen.enrichment.atera_cellpose_layer import _reduce_file

    path = tmp_path / "points.parquet"
    pq.write_table(
        pa.table(
            {
                CELLPOSE_ASSIGNMENT_COLUMN: pa.array([1, 3, 99, None, 1], pa.uint64()),
                "is_gene": [True, True, True, True, False],
                "feature_name": ["A", "B", "A", "A", "NegControlProbe_1"],
            }
        ),
        path,
    )
    index_of = np.asarray([-1, 0, -1, 1])  # labels 1 and 3 are table rows 0, 1
    reduced = _reduce_file(path, {"A": 0, "B": 1}, index_of, 2)
    assert reduced.n_unknown == 1 and reduced.n_rows == 5
    good = _reduce_file(
        _rewrite(path, tmp_path / "good.parquet"), {"A": 0, "B": 1}, index_of, 2
    )
    assert good.n_unknown == 0
    assert good.keys.tolist() == [0, 3] and good.counts.tolist() == [1, 1]
    assert good.control_rows.tolist() == [0]


def _rewrite(source: Path, target: Path) -> Path:
    table = pq.read_table(source)
    labels = table.column(CELLPOSE_ASSIGNMENT_COLUMN).to_pylist()
    fixed = [None if value == 99 else value for value in labels]
    index = table.column_names.index(CELLPOSE_ASSIGNMENT_COLUMN)
    pq.write_table(
        table.set_column(
            index, CELLPOSE_ASSIGNMENT_COLUMN, pa.array(fixed, pa.uint64())
        ),
        target,
    )
    return target


def test_comparison_counts_independent_stock_masks_without_proseg_on_stock(
    route: Route, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One trained ProSeg result coexists with independent stock/raw cell counts."""
    from merxen.config import CellposeConfig, MaskFilterConfig, TilingConfig
    from merxen.enrichment import enrich as enrich_module
    from merxen.segmentation import model_identity
    from merxen.segmentation.model_identity import (
        CPSAM_V2_SHA256,
        CellposeInferenceContext,
        CellposeModelIdentity,
    )

    vendor = np.asarray(_level0(route.source / "labels" / "cell_labels")[:])
    stock = np.zeros_like(vendor, dtype=np.uint32)
    stock[:, :-1] = vendor[:, 1:]
    stock[stock > 0] = 5000 + stock[stock > 0] * 3
    stock_path = tmp_path / "stock" / "cellpose_masks_tiled.npy"
    stock_path.parent.mkdir()
    np.save(stock_path, stock)
    context = CellposeInferenceContext(
        platform="ATERA",
        species="human",
        channels=["DAPI", "18S"],
        image_key="morphology_focus",
        pixel_size_um=(PS, PS),
        mask_filter=MaskFilterConfig(),
        tiling=TilingConfig(),
    )
    identity = CellposeModelIdentity(
        status="verified",
        model_name="cpsam_v2",
        weights_sha256=CPSAM_V2_SHA256,
        cellpose_version="4.2.1.1",
        torch_version="2.14.0+cu130",
        cellpose_config=CellposeConfig(gpu=False),
        inference_context=context,
    )

    def read_identity(path: Path) -> CellposeModelIdentity:
        if Path(path).resolve() == stock_path.resolve():
            return identity
        return identity.model_copy(update={"weights_sha256": "1" * 64})

    monkeypatch.setattr(model_identity, "read_model_identity", read_identity)
    monkeypatch.setattr(enrich_module, "read_model_identity", read_identity)
    config = stitch_config(
        store,
        route.plan_path,
        route.tile_dirs,
        tmp_path / "stitch",
        prior_kind="cellpose",
    ).model_copy(update={"stock_cellpose_mask_path": stock_path})
    stitch(config)
    for path in points_parquet_dir(store, "transcripts").glob("*.parquet"):
        rows = pq.read_table(path, columns=["x", "y", "stock_cellpose_assignment"])
        expected = lookup_labels(stock, rows["x"].to_numpy(), rows["y"].to_numpy(), PS)
        found = rows["stock_cellpose_assignment"].fill_null(0).to_numpy()
        np.testing.assert_array_equal(found, expected)
    enrich_config = _enrich_config(route, store, tmp_path / "enrich").model_copy(
        update={
            "segmentation_source": "cellpose_comparison",
            "stock_cellpose_mask_path": stock_path,
        }
    )
    enrich_single_latest(enrich_config, polygon_n_jobs=2)
    dataset = sd.read_zarr(store, selection=("tables", "shapes", "labels"))
    for segmentation, table, shape in [
        ("reseg", "table_MOSAIK_proseg", "MOSAIK_proseg"),
        ("proseg_mask", "table_MOSAIK_cellpose", "MOSAIK_cellpose"),
        ("stock_cellpose", "table_MOSAIK_stock_cellpose", "MOSAIK_stock_cellpose"),
        ("original_seg", "table_original", "atera_cell_boundaries"),
    ]:
        assert (
            validate_analysis_layer(
                store,
                platform="ATERA",
                segmentation=segmentation,
                table_key=table,
                shape_key=shape,
            )["n_cells"]
            > 0
        )
    stock_ids = set(dataset.tables["table_MOSAIK_stock_cellpose"].obs["instance_id"])
    trained_ids = set(dataset.tables["table_MOSAIK_cellpose"].obs["instance_id"])
    assert stock_ids == set(np.unique(stock)) - {0}
    assert stock_ids.isdisjoint(trained_ids)
    registry = dataset.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]
    assert (
        registry["stock_cellpose"]["assignment_column"] == "stock_cellpose_assignment"
    )
    assert (
        registry["stock_cellpose"]["model_identity"]["weights_sha256"]
        == CPSAM_V2_SHA256
    )
    assert registry["cellpose"]["assignment_column"] == "cellpose_assignment"
    # The same trained plan/tiles are reused; no stock ProSeg plan was introduced.
    assert read_plan(route.plan_path)["prior"]["mask_path"] == str(
        route.mask_path.resolve()
    )
