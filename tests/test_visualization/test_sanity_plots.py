"""Tests for sanity overlay plotting helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from merxen.visualization.sanity_plots import (
    _prepare_overlay_image,
    plot_pair_sanity_crops,
    plot_sanity_overlay,
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
