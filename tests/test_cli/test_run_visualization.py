"""Tests for the sanity-overlay wiring of the visualize command."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from matplotlib.figure import Figure
from shapely.geometry import box

from merxen.cli import run_visualization
from merxen.config import VisualizationConfig, VisualizationSampleConfig
from merxen.visualization import sanity_plots

# Six transcripts inside the 10 x 10 um crop.  ProSeg draws two cells as
# polygons, but its own per-transcript assignment disagrees with polygon
# containment for three transcripts: (2, 2) and (3, 3) sit inside the first
# polygon yet are background, and (5, 5) sits between the polygons yet belongs
# to the second cell.
_TRANSCRIPT_X = [1.0, 2.0, 3.0, 5.0, 8.0, 9.0]
_TRANSCRIPT_Y = [1.0, 2.0, 3.0, 5.0, 8.0, 1.0]
_PROSEG_ASSIGNED = [True, False, False, True, True, False]
_INSIDE_PROSEG_POLYGON = [True, True, True, False, True, False]
_INSIDE_CELLPOSE_POLYGON = [True, True, True, False, False, True]
_INSIDE_ORIGINAL_POLYGON = [False, True, True, True, False, False]
_HYBRID_ASSIGNED = [True, False, True, False, False, True]


def _shapes(*polygons: Any) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": list(polygons)})


def _synthetic_sdata() -> SimpleNamespace:
    """Build a SpatialData-like object carrying every segmentation branch."""
    points = pd.DataFrame(
        {
            "x": _TRANSCRIPT_X,
            "y": _TRANSCRIPT_Y,
            "assignment": pd.Series([0, pd.NA, pd.NA, 1, 1, pd.NA], dtype="UInt32"),
            "background": [not assigned for assigned in _PROSEG_ASSIGNED],
            "hybrid_assignment": pd.Series(
                [7, pd.NA, 8, pd.NA, pd.NA, 9],
                dtype="UInt64",
            ),
            "hybrid_background": [not assigned for assigned in _HYBRID_ASSIGNED],
            "hybrid_assignment_source": [
                "single_mask",
                "outside",
                "proseg_overlap",
                "ambiguous_overlap",
                "outside",
                "single_mask",
            ],
        }
    )
    return SimpleNamespace(
        shapes={
            "MOSAIK_proseg": _shapes(
                box(0.0, 0.0, 4.0, 4.0),
                box(6.0, 6.0, 10.0, 10.0),
            ),
            "MOSAIK_cellpose": _shapes(box(0.0, 0.0, 10.0, 4.5)),
            "MOSAIK_proseg_hybrid": _shapes(box(0.0, 0.0, 10.0, 10.0)),
            "merscope_cell_boundaries": _shapes(box(1.5, 1.5, 6.0, 6.0)),
            "xenium_cell_boundaries": _shapes(box(1.5, 1.5, 6.0, 6.0)),
        },
        points={"transcripts": points},
        images={},
    )


@pytest.fixture
def overlay_recorder(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub everything but the sanity overlay and record what it draws."""
    sdata = _synthetic_sdata()
    recorded: dict[str, Any] = {"crops": [], "legends": {}}

    def _no_plot(*_args: Any, **_kwargs: Any) -> None:
        return None

    qc = {
        "summary": {"pct_assigned": 50.0},
        "geometry_metrics": None,
        "cell_metrics": None,
    }
    monkeypatch.setattr(run_visualization.sd, "read_zarr", lambda _path: sdata)
    monkeypatch.setattr(run_visualization, "compute_dataset_qc", lambda *_a, **_k: qc)
    monkeypatch.setattr(
        run_visualization,
        "compute_gene_summary_from_path",
        lambda *_a, **_k: {"total_counts_df": None, "assigned_counts_df": None},
    )
    monkeypatch.setattr(
        run_visualization,
        "compute_gene_comparison_from_paths",
        lambda *_a, **_k: {"total_normalized_df": None, "assigned_normalized_df": None},
    )
    for name in (
        "plot_gene_abundance",
        "plot_gene_scatter",
        "plot_geometry_histograms",
        "plot_geometry_histograms_comparison",
        "plot_cell_metrics_violin",
        "plot_cell_metrics_violin_comparison",
        "plot_single_transcript_overview",
        "plot_transcript_overview",
        "plot_assignment_bar",
    ):
        monkeypatch.setattr(run_visualization, name, _no_plot)

    crop_points = sanity_plots._crop_points

    def _recording_crop_points(*args: Any, **kwargs: Any) -> Any:
        result = crop_points(*args, **kwargs)
        recorded["crops"].append(result[0])
        return result

    def _recording_save_figure(fig: Figure, output_path: Path | str, **_k: Any) -> Path:
        recorded["legends"][Path(output_path).name] = [
            [text.get_text() for text in ax.get_legend().get_texts()]
            for ax in fig.axes
            if ax.get_legend() is not None
        ]
        return Path(output_path)

    monkeypatch.setattr(sanity_plots, "_crop_points", _recording_crop_points)
    monkeypatch.setattr(sanity_plots, "save_figure", _recording_save_figure)
    return recorded


def _sample(platform: str, segmentation: str, shape_key: str) -> Any:
    return VisualizationSampleConfig(
        sample_id=f"P1_{platform}",
        platform=platform,
        zarr_path=Path(f"P1_{platform}.zarr"),
        segmentation=segmentation,
        table_key=f"table_{shape_key}",
        shape_key=shape_key,
    )


def test_reseg_single_overlay_follows_proseg_assignment_column(
    tmp_path: Path,
    overlay_recorder: dict[str, Any],
) -> None:
    """Reseg overlays should show the transcripts ProSeg itself assigned."""
    sample = _sample("MERSCOPE", "reseg", "MOSAIK_proseg")
    cfg = VisualizationConfig(output_dir=tmp_path, pair_id="P1", samples=[sample])

    run_visualization._write_single_visualizations(cfg, sample)

    (crop,) = overlay_recorder["crops"]
    assert crop["assigned"].tolist() == _PROSEG_ASSIGNED
    assert crop["assigned"].tolist() != _INSIDE_PROSEG_POLYGON
    (labels,) = overlay_recorder["legends"]["P1_MERSCOPE_sanity_overlay.png"]
    assert "Assigned tx (3)" in labels
    assert "Unassigned tx (3)" in labels


def test_reseg_paired_overlay_follows_proseg_assignment_column(
    tmp_path: Path,
    overlay_recorder: dict[str, Any],
) -> None:
    """Both panels of a paired reseg overlay should use ProSeg's assignment."""
    merscope = _sample("MERSCOPE", "reseg", "MOSAIK_proseg")
    xenium = _sample("XENIUM", "reseg", "MOSAIK_proseg")
    cfg = VisualizationConfig(
        output_dir=tmp_path,
        pair_id="P1",
        samples=[merscope, xenium],
    )

    run_visualization._write_paired_visualizations(
        cfg,
        {"MERSCOPE": merscope, "XENIUM": xenium},
    )

    crops = overlay_recorder["crops"]
    assert [crop["assigned"].tolist() for crop in crops] == [_PROSEG_ASSIGNED] * 2
    panel_labels = overlay_recorder["legends"]["P1_sanity_overlay.png"]
    assert len(panel_labels) == 2
    for labels in panel_labels:
        assert "Assigned tx (3)" in labels
        assert "Unassigned tx (3)" in labels


@pytest.mark.parametrize(
    ("segmentation", "shape_key", "expected_assigned"),
    [
        ("proseg_mask", "MOSAIK_cellpose", _INSIDE_CELLPOSE_POLYGON),
        ("original_seg", "merscope_cell_boundaries", _INSIDE_ORIGINAL_POLYGON),
        ("proseg_hybrid", "MOSAIK_proseg_hybrid", _HYBRID_ASSIGNED),
    ],
)
def test_other_segmentation_overlays_keep_their_assignment_source(
    tmp_path: Path,
    overlay_recorder: dict[str, Any],
    segmentation: str,
    shape_key: str,
    expected_assigned: list[bool],
) -> None:
    """Cellpose and original overlays use polygons; hybrid keeps its columns."""
    sample = _sample("MERSCOPE", segmentation, shape_key)
    cfg = VisualizationConfig(output_dir=tmp_path, pair_id="P1", samples=[sample])

    run_visualization._write_single_visualizations(cfg, sample)

    (crop,) = overlay_recorder["crops"]
    assert crop["assigned"].tolist() == expected_assigned
