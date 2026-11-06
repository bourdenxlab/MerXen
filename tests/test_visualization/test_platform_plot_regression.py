"""XENIUM and MERSCOPE figures keep the exact data they plotted before AT-6.

AT-6 draws ATERA transcripts as density rasters over every transcript
(``PlatformSpec.rasterised_transcript_plots``). That branch must not touch the
XENIUM and MERSCOPE figures. These tests record every array handed to
``Axes.scatter``, ``Axes.imshow`` and ``Axes.plot`` (and the scale-bar text)
while the existing figures are drawn from fixed-seed synthetic data, and compare
a digest of the recording with the digest the pre-AT-6 code produced.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dask.dataframe as dd
import geopandas as gpd
import matplotlib.axes
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box
from spatialdata.models import Image2DModel

from merxen.visualization.density_overview import (
    plot_single_transcript_overview,
    plot_transcript_overview,
)
from merxen.visualization.sanity_plots import (
    plot_pair_sanity_crops,
    plot_single_sanity_crop,
)

# Digests of the recordings made with the code before AT-6 (commit of WP-F).
EXPECTED_DIGESTS: dict[str, str] = {
    "single_overview_XENIUM": (
        "361a815713b40817ee03b5f4ddd5574da07a818f37436d3826be8ea81c229829"
    ),
    "single_overview_MERSCOPE": (
        "9afe6469d6e5eee1de39726408f2378b3e1685d03066cce63331e59a442959a3"
    ),
    "single_sanity_XENIUM": (
        "22536d0039010b1b1ceaa1fdfc03675afb1d0a7a6e2e92b7be84c8ea348ec2d7"
    ),
    "single_sanity_MERSCOPE": (
        "9831df42c3350019e83ad04198a9123b01456171589c5a8407d64f763f910576"
    ),
    "pair_overview": (
        "4702010c5f81c347fb7a27b349e1e2fafe23daa59ac9b04046b4c2eaa47f8a69"
    ),
    "pair_sanity": ("b59de2a646d46c76d10bdfb94d043bb7615dd18ea98892fec1e87e4b1d92e0f9"),
}


def _as_bytes(value: Any) -> bytes:
    if value is None:
        return b"None"
    if isinstance(value, str):
        return value.encode()
    if isinstance(value, (tuple, list)) and all(
        isinstance(item, (int, float, np.floating, np.integer)) for item in value
    ):
        return np.asarray(value, dtype=np.float64).tobytes()
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    array = np.asarray(value)
    if array.dtype == object:
        return repr(array.tolist()).encode()
    return str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()


class _Recorder:
    """Patch the Axes drawing calls and hash what they receive."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.digest = hashlib.sha256()
        self.calls = 0
        for name in ("scatter", "imshow", "plot", "text"):
            original = getattr(matplotlib.axes.Axes, name)
            monkeypatch.setattr(
                matplotlib.axes.Axes, name, self._wrap(name, original), raising=True
            )

    def _wrap(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        recorder = self

        def wrapper(ax: Any, *args: Any, **kwargs: Any) -> Any:
            recorder.calls += 1
            recorder.digest.update(name.encode())
            for arg in args:
                recorder.digest.update(_as_bytes(arg))
            for key in ("c", "color", "s", "alpha", "extent", "origin", "cmap"):
                if key in kwargs:
                    recorder.digest.update(key.encode() + _as_bytes(kwargs[key]))
            norm = kwargs.get("norm")
            if norm is not None:
                recorder.digest.update(
                    type(norm).__name__.encode() + _as_bytes((norm.vmin, norm.vmax))
                )
            return original(ax, *args, **kwargs)

        return wrapper

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def _points(seed: int, n: int = 6000, span: float = 600.0) -> Any:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x": rng.uniform(0.0, span, n),
            "y": rng.uniform(0.0, span, n),
            "assignment": pd.array(
                np.where(rng.random(n) < 0.6, rng.integers(1, 50, n), 0),
                dtype="UInt32",
            ),
        }
    )
    return dd.from_pandas(frame, npartitions=3)


def _shapes(span: float = 600.0) -> gpd.GeoDataFrame:
    boxes = [
        box(x, y, x + 20.0, y + 20.0)
        for x in np.arange(10.0, span - 30.0, 45.0)
        for y in np.arange(10.0, span - 30.0, 45.0)
    ]
    return gpd.GeoDataFrame({"geometry": boxes})


def _image(seed: int, channels: list[str], size: int = 400) -> Any:
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4000, size=(len(channels), size, size), dtype=np.uint16)
    return Image2DModel.parse(
        xr.DataArray(data, dims=("c", "y", "x"), coords={"c": channels})
    )


def _sdata(platform: str, seed: int) -> SimpleNamespace:
    if platform == "XENIUM":
        shape_key, image_key, channels = (
            "xenium_cell_boundaries",
            "morphology_focus",
            ["DAPI", "18S"],
        )
    else:
        shape_key, image_key, channels = (
            "merscope_cell_boundaries",
            "MERSCOPE_z_projection",
            ["DAPI", "PolyT"],
        )
    return SimpleNamespace(
        shapes={shape_key: _shapes(), "MOSAIK_proseg": _shapes()},
        points={"transcripts": _points(seed)},
        images={image_key: _image(seed, channels)},
    )


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Recorder]:
    yield _Recorder(monkeypatch)


def _check(name: str, recorder: _Recorder) -> None:
    assert recorder.calls > 0
    assert recorder.hexdigest() == EXPECTED_DIGESTS[name], name


@pytest.mark.parametrize("platform", ["XENIUM", "MERSCOPE"])
def test_single_transcript_overview_plots_the_same_data(
    tmp_path: Path, recorder: _Recorder, platform: str
) -> None:
    plot_single_transcript_overview(
        _sdata(platform, 3),
        platform,
        tmp_path / "overview.png",
        sample_n=2000,
        crop_sample_n=500,
        crop_size_um=200.0,
        heatmap_bins=40,
    )
    _check(f"single_overview_{platform}", recorder)


@pytest.mark.parametrize("platform", ["XENIUM", "MERSCOPE"])
def test_single_sanity_crop_plots_the_same_data(
    tmp_path: Path, recorder: _Recorder, platform: str
) -> None:
    plot_single_sanity_crop(
        _sdata(platform, 5),
        platform,
        tmp_path / "sanity.png",
        crop_size_um=150.0,
        assignment_shape_key="MOSAIK_proseg",
    )
    _check(f"single_sanity_{platform}", recorder)


def test_pair_transcript_overview_plots_the_same_data(
    tmp_path: Path, recorder: _Recorder
) -> None:
    plot_transcript_overview(
        _sdata("MERSCOPE", 7),
        _sdata("XENIUM", 8),
        tmp_path / "pair_overview.png",
        sample_n=2000,
        crop_sample_n=500,
        crop_bbox_um=(100.0, 100.0, 300.0, 300.0),
        heatmap_bins=40,
    )
    _check("pair_overview", recorder)


def test_pair_sanity_crops_plot_the_same_data(
    tmp_path: Path, recorder: _Recorder
) -> None:
    plot_pair_sanity_crops(
        _sdata("MERSCOPE", 9),
        _sdata("XENIUM", 10),
        tmp_path / "pair_sanity.png",
        crop_size_um=150.0,
        xenium_assignment_column="assignment",
    )
    _check("pair_sanity", recorder)
