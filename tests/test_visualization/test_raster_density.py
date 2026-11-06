"""Streaming density rasters and the ATERA (rasterised) figures.

ATERA figures draw every transcript as density: no plotting function may
sample the points, and every transcript in a window is counted.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import xarray as xr
from click.testing import CliRunner
from test_io.atera_native_fixture import make_native_reference

from merxen.cli.run_visualization import _sanity_assignment_column, visualize_command
from merxen.config import VisualizationSampleConfig
from merxen.io.points_aggregates import points_parquet_dir
from merxen.visualization import raster_density
from merxen.visualization.density_overview import plot_single_transcript_overview
from merxen.visualization.raster_density import (
    ASSIGNED_LAYER,
    UNASSIGNED_LAYER,
    bin_indices,
    composite_rgba,
    grid_shape,
    points_bounds,
    stream_hist2d,
)
from merxen.visualization.sanity_plots import (
    _crop_extent_um,
    _resolve_dataset_mask_affine,
    plot_sanity_crop_panel,
    plot_single_sanity_crop,
)

PIXEL_SIZE = 0.21249222


@pytest.fixture(scope="module")
def reference_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _, latest = make_native_reference(tmp_path_factory.mktemp("native_ref"))
    return latest


@pytest.fixture
def latest(reference_store: Path, tmp_path: Path) -> Path:
    store = tmp_path / "latest_spatialdata.zarr"
    shutil.copytree(reference_store, store)
    return store


def _all_points(store: Path) -> pd.DataFrame:
    directory = points_parquet_dir(store, "transcripts")
    return pd.concat(
        [
            pq.read_table(path).to_pandas()
            for path in sorted(directory.glob("*.parquet"))
        ],
        ignore_index=True,
    )


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #


def test_bin_indices_match_histogram_edges() -> None:
    values = np.array([0.0, 0.5, 0.9999999, 1.0, 2.0, 3.0, -1e-9, 3.0000001, np.nan])
    index = bin_indices(values, 0.0, 1.0, 3)
    assert index.tolist() == [0, 0, 0, 1, 2, 2, -1, -1, -1]


def test_bin_indices_correct_the_floor_estimate_on_float_edges() -> None:
    # Values on float64 edges where floor((v - origin) / bin) is off by one in
    # both directions (2.0 and 4.1 one bin low, 1.8 and 3.5 one bin high).
    origin, bin_um, n_bins = 0.1, 0.1, 50
    values = np.array([2.0, 4.1, 1.8, 3.5])
    estimate = np.floor((values - origin) / bin_um).astype(np.int64)
    edges = origin + bin_um * np.arange(n_bins + 1, dtype=np.float64)
    expected = np.searchsorted(edges, values, side="right") - 1
    assert (estimate < expected).any() and (estimate > expected).any()

    index = bin_indices(values, origin, bin_um, n_bins)
    assert index.tolist() == expected.tolist()
    hist, _ = np.histogram(values, bins=edges)
    assert hist.tolist() == np.bincount(index, minlength=n_bins).tolist()
    rng = np.random.default_rng(0)
    coords = rng.uniform(-3.0, 13.0, 50_000).astype(np.float32)
    edges = 0.1 + 0.37 * np.arange(28)
    expected, _ = np.histogram(coords.astype(np.float64), bins=edges)
    got = np.bincount(
        bin_indices(coords, 0.1, 0.37, 27)[bin_indices(coords, 0.1, 0.37, 27) >= 0],
        minlength=27,
    )
    np.testing.assert_array_equal(got, expected)


def test_grid_shape_covers_the_window_and_refuses_huge_grids() -> None:
    assert grid_shape((0.0, 0.0, 10.0, 5.0), 1.0) == (5, 10)
    assert grid_shape((0.0, 0.0, 10.5, 5.0), 1.0) == (5, 11)
    with pytest.raises(ValueError, match="bins"):
        grid_shape((0.0, 0.0, 1e6, 1e6), 0.01)
    with pytest.raises(ValueError, match="empty"):
        grid_shape((1.0, 0.0, 1.0, 5.0), 1.0)


@pytest.mark.parametrize("use_index", [True, False])
@pytest.mark.parametrize(
    ("bbox", "bin_um"),
    [((-10.0, -1.0, 12.0, 21.0), 0.5), ((2.0, 10.0, 6.0, 16.0), 0.25)],
)
def test_stream_hist2d_equals_histogram2d(
    latest: Path, use_index: bool, bbox: tuple[float, ...], bin_um: float
) -> None:
    raster = stream_hist2d(latest, "transcripts", bbox, bin_um, use_index=use_index)
    points = _all_points(latest)
    expected, _, _ = np.histogram2d(
        points["x"].to_numpy(np.float64),
        points["y"].to_numpy(np.float64),
        bins=[raster.x_edges, raster.y_edges],
    )
    np.testing.assert_array_equal(raster.counts["all"], expected.T.astype(np.uint64))
    assert raster.pruned is use_index
    if use_index and bbox == (2.0, 10.0, 6.0, 16.0):
        # Only the files whose indexed bbox meets the crop are read.
        assert raster.n_files_read < raster.n_files_total


def test_stream_hist2d_assignment_layers_count_every_transcript(latest: Path) -> None:
    raster = stream_hist2d(
        latest,
        "transcripts",
        (-10.0, -1.0, 12.0, 21.0),
        0.5,
        category="original_assignment",
        category_mode="assignment",
    )
    points = _all_points(latest)
    assert raster.total() == len(points)
    assert raster.total(ASSIGNED_LAYER) == int(
        points["original_assignment"].notna().sum()
    )
    assert raster.total(UNASSIGNED_LAYER) == int(
        points["original_assignment"].isna().sum()
    )


def test_stream_hist2d_value_layers(latest: Path) -> None:
    raster = stream_hist2d(
        latest, "transcripts", (-10.0, -1.0, 12.0, 21.0), 1.0, category="is_gene"
    )
    points = _all_points(latest)
    counts = points["is_gene"].astype(str).value_counts()
    assert {name: raster.total(name) for name in raster.counts} == {
        str(name): int(n) for name, n in counts.items()
    }


def test_points_bounds_use_aggregates_or_statistics(latest: Path) -> None:
    from_attrs = points_bounds(latest, "transcripts")
    assert from_attrs == (-8.0625, -0.125, 9.625, 19.28125)
    # A stale index (a file removed from the listing) falls back to the files.
    index_path = latest / "merxen_atera_points_index.json"
    index = json.loads(index_path.read_text())
    index["points"]["transcripts"]["fingerprint"]["n_rows"] += 1
    index_path.write_text(json.dumps(index))
    points = _all_points(latest)
    assert points_bounds(latest, "transcripts") == (
        float(points["x"].min()),
        float(points["y"].min()),
        float(points["x"].max()),
        float(points["y"].max()),
    )
    raster = stream_hist2d(latest, "transcripts", (2.0, 10.0, 6.0, 16.0), 0.5)
    assert raster.pruned is False


def test_coarsen_sums_blocks_and_keeps_every_count(latest: Path) -> None:
    points = _all_points(latest)
    x0, y0 = float(points["x"].min()) - 0.3, float(points["y"].min()) - 0.7
    bbox = (x0, y0, float(points["x"].max()) + 0.2, float(points["y"].max()) + 0.1)
    fine = stream_hist2d(latest, "transcripts", bbox, 0.5)
    assert fine.shape[0] % 3 or fine.shape[1] % 3  # exercises the padding
    coarse = fine.coarsen(3)
    assert coarse.bin_um == pytest.approx(1.5)
    assert coarse.origin_um == fine.origin_um
    assert coarse.shape == (-(-fine.shape[0] // 3), -(-fine.shape[1] // 3))
    assert coarse.total() == fine.total() == len(points)
    expected, _, _ = np.histogram2d(
        points["y"].to_numpy(np.float64),
        points["x"].to_numpy(np.float64),
        bins=[coarse.y_edges, coarse.x_edges],
    )
    np.testing.assert_array_equal(coarse.combined(), expected.astype(np.uint64))
    assert fine.coarsen(1).combined().tolist() == fine.combined().tolist()
    with pytest.raises(ValueError):
        fine.coarsen(0)


def test_composite_rgba_alpha_and_order() -> None:
    low = np.array([[0, 10], [0, 0]])
    high = np.array([[0, 0], [5, 10]])
    rgba = composite_rgba([(low, "#ff0000"), (high, "#ffff00")], vmax={0: 10, 1: 10})
    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 0, 3] == 0.0
    assert rgba[1, 1, 3] == pytest.approx(0.9)
    np.testing.assert_allclose(rgba[0, 1, :3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rgba[1, 1, :3], [1.0, 1.0, 0.0])
    assert 0.0 < rgba[1, 0, 3] < rgba[1, 1, 3]


# --------------------------------------------------------------------------- #
# ATERA image placement
# --------------------------------------------------------------------------- #


def test_one_pixel_image_extent_follows_the_corner_convention() -> None:
    crop = xr.DataArray(
        np.ones((1, 1)), dims=("y", "x"), coords={"y": [0.5], "x": [0.5]}
    )
    rows = ((PIXEL_SIZE, 0.0, 0.0), (0.0, PIXEL_SIZE, 0.0))
    assert _crop_extent_um(crop, *rows, corner=True) == pytest.approx(
        (0.0, PIXEL_SIZE, 0.0, PIXEL_SIZE)
    )
    # The XENIUM/MERSCOPE panels keep the extent between pixel centres.
    assert _crop_extent_um(crop, *rows) == pytest.approx(
        (0.5 * PIXEL_SIZE, 0.5 * PIXEL_SIZE, 0.5 * PIXEL_SIZE, 0.5 * PIXEL_SIZE)
    )
    wide = xr.DataArray(
        np.ones((2, 3)), dims=("y", "x"), coords={"y": [4.5, 5.5], "x": [1.5, 2.5, 3.5]}
    )
    assert _crop_extent_um(wide, *rows, corner=True) == pytest.approx(
        (1 * PIXEL_SIZE, 4 * PIXEL_SIZE, 4 * PIXEL_SIZE, 6 * PIXEL_SIZE)
    )


def test_atera_mask_affine_comes_from_the_store(latest: Path, tmp_path: Path) -> None:
    rows = _resolve_dataset_mask_affine("ATERA", zarr_path=latest)
    assert rows == ((PIXEL_SIZE, 0.0, 0.0), (0.0, PIXEL_SIZE, 0.0))
    with pytest.raises(ValueError, match="store path"):
        _resolve_dataset_mask_affine("ATERA", zarr_path=None)
    empty = tmp_path / "empty.zarr"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _resolve_dataset_mask_affine("ATERA", zarr_path=empty)


# --------------------------------------------------------------------------- #
# ATERA figures: no sampling, every transcript counted
# --------------------------------------------------------------------------- #


def _forbid_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("ATERA figures must not sample transcripts")

    monkeypatch.setattr(pd.DataFrame, "sample", refuse)
    monkeypatch.setattr(dd.DataFrame, "sample", refuse)


class _RasterLog:
    """Record every raster the figures compute."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[tuple[float, ...], Any]] = []
        original = raster_density.stream_hist2d

        def recording(*args: Any, **kwargs: Any) -> Any:
            raster = original(*args, **kwargs)
            bbox = kwargs.get("bbox", args[2] if len(args) > 2 else None)
            self.calls.append((tuple(bbox), raster))
            return raster

        for module in (
            "merxen.visualization.density_overview",
            "merxen.visualization.sanity_plots",
        ):
            monkeypatch.setattr(f"{module}.stream_hist2d", recording)


def _n_in_raster(points: pd.DataFrame, raster: Any) -> int:
    x0, x1, y0, y1 = raster.extent
    x = points["x"].to_numpy(np.float64)
    y = points["y"].to_numpy(np.float64)
    return int(((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)).sum())


def test_atera_transcript_overview_rasterises_every_transcript(
    latest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import spatialdata as sd

    _forbid_sampling(monkeypatch)
    log = _RasterLog(monkeypatch)
    out = plot_single_transcript_overview(
        sd.read_zarr(latest),
        "ATERA",
        tmp_path / "overview.png",
        crop_size_um=8.0,
    )
    assert out.exists()
    points = _all_points(latest)
    # One pass over the section (both section panels) and one over the crop.
    assert len(log.calls) == 2
    for _, raster in log.calls:
        assert raster.total() == _n_in_raster(points, raster)
    # The section panels hold every transcript.
    assert log.calls[0][1].total() == len(points)


def test_atera_sanity_crop_rasterises_assignment(
    latest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import spatialdata as sd

    _forbid_sampling(monkeypatch)
    log = _RasterLog(monkeypatch)
    sdata = sd.read_zarr(latest)
    out = plot_single_sanity_crop(
        sdata,
        "ATERA",
        tmp_path / "sanity.png",
        zarr_path=latest,
        crop_size_um=12.0,
        assignment_shape_key="atera_cell_boundaries",
        assignment_column="original_assignment",
    )
    assert out.exists()
    assert (tmp_path / "sanity_crop_location.png").exists()
    (bbox, raster), *_ = log.calls
    assert set(raster.counts) == {ASSIGNED_LAYER, UNASSIGNED_LAYER}
    points = _all_points(latest)
    inside = points[
        (points["x"] >= raster.extent[0])
        & (points["x"] <= raster.extent[1])
        & (points["y"] >= raster.extent[2])
        & (points["y"] <= raster.extent[3])
    ]
    assert raster.total(ASSIGNED_LAYER) == int(
        inside["original_assignment"].notna().sum()
    )
    assert raster.total(UNASSIGNED_LAYER) == int(
        inside["original_assignment"].isna().sum()
    )


def test_atera_sanity_panel_legend_counts_and_scale_bar(latest: Path) -> None:
    import spatialdata as sd

    sdata = sd.read_zarr(latest)
    fig, ax = plt.subplots()
    try:
        plot_sanity_crop_panel(
            ax,
            sdata,
            "ATERA",
            crop_bbox=(-10.0, -1.0, 12.0, 21.0),
            assignment_shape_key="atera_cell_boundaries",
            zarr_path=latest,
        )
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        points = _all_points(latest)
        n_assigned = int(points["original_assignment"].notna().sum())
        n_unassigned = len(points) - n_assigned
        assert any(f"Assigned tx ({n_assigned:,}," in label for label in labels)
        assert any(f"Unassigned tx ({n_unassigned:,}," in label for label in labels)
        assert any(label.startswith("Original segmentation") for label in labels)
        assert any(text.get_text().endswith(" um") for text in ax.texts)
        assert ax.get_title() == "ATERA"
    finally:
        plt.close(fig)


def test_atera_sanity_background_follows_the_corner_convention(latest: Path) -> None:
    import spatialdata as sd

    sdata = sd.read_zarr(latest)
    image = sd.get_pyramid_levels(sdata.images["morphology_focus"], n=0)
    xs = np.asarray(image.coords["x"].values, dtype=np.float64)
    ys = np.asarray(image.coords["y"].values, dtype=np.float64)
    # SpatialData pixel centres: pixel i sits at i + 0.5.
    np.testing.assert_allclose(xs, np.arange(xs.size) + 0.5)
    bbox = (-10.0, -1.0, 12.0, 21.0)
    fig, ax = plt.subplots()
    try:
        plot_sanity_crop_panel(
            ax,
            sdata,
            "ATERA",
            crop_bbox=bbox,
            assignment_shape_key="atera_cell_boundaries",
            zarr_path=latest,
        )
        background = ax.images[0]
        extent = background.get_extent()
        n_y, n_x = background.get_array().shape[:2]
    finally:
        plt.close(fig)
    # The crop keeps the pixels whose centre lies within the bbox plus 1 px.
    sel_x = xs[(xs >= bbox[0] / PIXEL_SIZE - 1) & (xs <= bbox[2] / PIXEL_SIZE + 1)]
    sel_y = ys[(ys >= bbox[1] / PIXEL_SIZE - 1) & (ys <= bbox[3] / PIXEL_SIZE + 1)]
    assert (n_y, n_x) == (sel_y.size, sel_x.size)
    i0, i1 = int(sel_x.min() - 0.5), int(sel_x.max() - 0.5)
    j0, j1 = int(sel_y.min() - 0.5), int(sel_y.max() - 0.5)
    # Pixel i covers [i * ps, (i + 1) * ps): the extent reaches the outer edges.
    assert extent == pytest.approx(
        (i0 * PIXEL_SIZE, (i1 + 1) * PIXEL_SIZE, j0 * PIXEL_SIZE, (j1 + 1) * PIXEL_SIZE)
    )


def test_atera_sanity_background_refuses_a_missing_pixel_size(
    latest: Path, tmp_path: Path
) -> None:
    import spatialdata as sd

    sdata = sd.read_zarr(latest)
    empty = tmp_path / "empty.zarr"
    empty.mkdir()
    fig, ax = plt.subplots()
    try:
        with pytest.raises(FileNotFoundError):
            plot_sanity_crop_panel(
                ax,
                sdata,
                "ATERA",
                crop_bbox=(-10.0, -1.0, 12.0, 21.0),
                assignment_shape_key="atera_cell_boundaries",
                zarr_path=empty,
            )
    finally:
        plt.close(fig)


def test_sanity_assignment_column_reads_the_registry(latest: Path) -> None:
    import spatialdata as sd

    sdata = sd.read_zarr(latest)
    atera = VisualizationSampleConfig(
        sample_id="ATERA_SYNTH",
        platform="ATERA",
        zarr_path=latest,
        segmentation="original_seg",
        table_key="table_original",
        shape_key="atera_cell_boundaries",
    )
    assert _sanity_assignment_column(atera, sdata) == "original_assignment"
    xenium = VisualizationSampleConfig(
        sample_id="X", platform="XENIUM", zarr_path=latest, segmentation="original_seg"
    )
    assert _sanity_assignment_column(xenium, sdata) is None
    assert _sanity_assignment_column(xenium) is None


@pytest.mark.integration
def test_visualize_cli_on_native_atera_store(
    latest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_sampling(monkeypatch)
    out_dir = tmp_path / "viz"
    config = tmp_path / "visualization_config.json"
    config.write_text(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "pair_id": "ATERA_SYNTH",
                "samples": [
                    {
                        "sample_id": "ATERA_SYNTH_ATERA",
                        "platform": "ATERA",
                        "zarr_path": str(latest),
                        "segmentation": "original_seg",
                        "table_key": "table_original",
                        "shape_key": "atera_cell_boundaries",
                    }
                ],
            }
        )
    )
    result = CliRunner().invoke(
        visualize_command, ["--config", str(config)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    for suffix in (
        "sanity_overlay",
        "sanity_overlay_crop_location",
        "transcript_overview",
        "geometry_hist",
        "cell_violin",
        "assignment_rate_bar",
    ):
        assert (out_dir / f"ATERA_SYNTH_ATERA_{suffix}.png").exists(), suffix
