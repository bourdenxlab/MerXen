"""Tests for sanity overlay plotting helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

from merxen.visualization.sanity_plots import (
    _crop_points,
    _prepare_overlay_image,
    plot_pair_sanity_crops,
    plot_sanity_crop_panel,
    plot_sanity_overlay,
    plot_single_sanity_crop,
)


def test_prepare_overlay_image_promotes_two_channel_input_to_rgb() -> None:
    """Two-channel crops should be padded to a display-safe RGB array."""
    image = np.dstack(
        [
            np.linspace(0, 5000, 16, dtype=np.float32).reshape(4, 4),
            np.linspace(5000, 0, 16, dtype=np.float32).reshape(4, 4),
        ]
    )

    display, cmap = _prepare_overlay_image(image)

    assert display.shape == (4, 4, 3)
    assert cmap is None
    assert np.all((display >= 0.0) & (display <= 1.0))
    assert np.allclose(display[..., 2], 0.0)


def test_plot_sanity_overlay_writes_file_for_two_channel_image(
    tmp_path: Path,
) -> None:
    """Overlay plotting should succeed for two-channel microscopy tiles."""
    image = np.dstack(
        [
            np.arange(64, dtype=np.uint16).reshape(8, 8),
            np.arange(64, 128, dtype=np.uint16).reshape(8, 8),
        ]
    )
    out = tmp_path / "overlay.png"

    result = plot_sanity_overlay(image, out, title="two-channel overlay")

    assert result == out
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_pair_sanity_crops_writes_file(tmp_path: Path) -> None:
    """Paired sanity crop plotting should succeed without image backgrounds."""
    shapes = gpd.GeoDataFrame({"geometry": [box(0.0, 0.0, 10.0, 10.0)]})
    points = pd.DataFrame({"x": [1.0, 5.0], "y": [1.0, 5.0]})
    merscope = SimpleNamespace(
        shapes={"MOSAIK_proseg": shapes},
        points={"transcripts": points},
        images={},
    )
    xenium = SimpleNamespace(
        shapes={"MOSAIK_proseg": shapes},
        points={"transcripts": points},
        images={},
    )
    out = tmp_path / "pair_sanity.png"

    plot_pair_sanity_crops(merscope, xenium, out)

    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert (tmp_path / "pair_sanity_crop_location.png").exists()
    assert (tmp_path / "pair_sanity_crop_location.pdf").exists()


def test_plot_single_sanity_crop_writes_file(tmp_path: Path) -> None:
    """Single-platform sanity crop plotting should write crop and location files."""
    shapes = gpd.GeoDataFrame({"geometry": [box(0.0, 0.0, 10.0, 10.0)]})
    points = pd.DataFrame({"x": [1.0, 5.0], "y": [1.0, 5.0]})
    sdata = SimpleNamespace(
        shapes={"MOSAIK_proseg": shapes},
        points={"transcripts": points},
        images={},
    )
    out = tmp_path / "single_sanity.png"

    plot_single_sanity_crop(sdata, "MERSCOPE", out)

    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert (tmp_path / "single_sanity_crop_location.png").exists()
    assert (tmp_path / "single_sanity_crop_location.pdf").exists()


def test_sanity_crop_panel_uses_clean_labels_and_xenium_scale_bar() -> None:
    """Paired sanity panels should use clean labels and one Xenium scale bar."""
    shapes = gpd.GeoDataFrame({"geometry": [box(0.0, 0.0, 300.0, 300.0)]})
    points = pd.DataFrame({"x": [75.0, 150.0], "y": [75.0, 150.0]})
    sdata = SimpleNamespace(
        shapes={
            "cell_boundaries": shapes,
            "MOSAIK_proseg": shapes,
            "MOSAIK_cellpose": shapes,
            "xenium_cell_boundaries": shapes,
            "xenium_nucleus": shapes,
        },
        points={"transcripts": points},
        images={},
    )
    fig, axes = plt.subplots(1, 2)
    try:
        plot_sanity_crop_panel(axes[0], sdata, "MERSCOPE")
        plot_sanity_crop_panel(axes[1], sdata, "XENIUM")

        assert axes[0].get_title() == "MERSCOPE"
        assert axes[1].get_title() == "XENIUM"
        assert len(axes[0].get_xticks()) == 0
        assert len(axes[0].get_yticks()) == 0
        assert all(text.get_text() != "100 um" for text in axes[0].texts)
        assert any(text.get_text() == "100 um" for text in axes[1].texts)

        legend = axes[1].get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        assert not any("cell_boundaries" in label for label in labels)
        assert not any("nucleus" in label.lower() for label in labels)
        assert any(label.startswith("ProSeg") for label in labels)
        assert any(label.startswith("Cellpose-SAM") for label in labels)
        assert any(label.startswith("Original segmentation") for label in labels)
        handle_colors = {
            mcolors.to_hex(handle.get_color()) for handle in legend.legend_handles
        }
        assert "#2ca02c" in handle_colors
        assert "#9467bd" in handle_colors
        assert "#ff7f0e" in handle_colors
    finally:
        plt.close(fig)


def test_hybrid_sanity_crop_uses_stored_assignment_and_provenance() -> None:
    shapes = gpd.GeoDataFrame({"geometry": [box(0.0, 0.0, 10.0, 10.0)]})
    points = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "hybrid_assignment": pd.Series([1, pd.NA, pd.NA], dtype="UInt64"),
            "hybrid_background": [False, True, True],
            "hybrid_assignment_source": [
                "single_mask",
                "ambiguous_overlap",
                "outside",
            ],
        }
    )
    sdata = SimpleNamespace(
        shapes={"MOSAIK_proseg_hybrid": shapes},
        points={"transcripts": points},
        images={},
    )

    cropped, _, assignment_column = _crop_points(
        sdata,
        (0.0, 0.0, 10.0, 10.0),
        max_points=None,
        random_state=0,
        assignment_shape_key="MOSAIK_proseg_hybrid",
        prefer_aligned_points=False,
        prefer_aligned_assignment=False,
    )

    assert cropped["assigned"].tolist() == [True, False, False]
    assert cropped["assignment_source"].tolist() == [
        "single_mask",
        "ambiguous_overlap",
        "outside",
    ]
    assert assignment_column == "hybrid_assignment"
