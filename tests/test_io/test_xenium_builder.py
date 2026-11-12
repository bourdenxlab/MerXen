"""Tests of the Xenium builder's pre-2.0 channel naming."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, get_channel_names

from merxen.io.builders.xenium import DAPI_CHANNEL, name_pre_v2_dapi_channels


def _image(channels: list[object] | None, n_channels: int = 1) -> object:
    data = np.zeros((n_channels, 8, 8), dtype=np.uint16)
    kwargs = {} if channels is None else {"c_coords": channels}
    return Image2DModel.parse(data, dims=("c", "y", "x"), **kwargs)


def _sdata(**images: object) -> SpatialData:
    return SpatialData(images=images)


def _experiment(path: Path, specs: dict[str, object]) -> Path:
    (path / "experiment.xenium").write_text(json.dumps(specs))
    return path


@pytest.mark.parametrize("version", ["xenium-1.4.0.6", "xenium-1.9.0.0"])
def test_pre_v2_single_channel_images_are_named_dapi(
    tmp_path: Path, version: str
) -> None:
    sdata = _sdata(morphology_focus=_image(None), morphology_mip=_image(None))
    assert list(get_channel_names(sdata.images["morphology_focus"])) == [0]
    _experiment(tmp_path, {"analysis_sw_version": version})

    renamed = name_pre_v2_dapi_channels(sdata, tmp_path)

    assert renamed == ["morphology_focus", "morphology_mip"]
    for key in renamed:
        assert list(get_channel_names(sdata.images[key])) == [DAPI_CHANNEL]


def test_multiscale_image_is_named_at_every_scale(tmp_path: Path) -> None:
    data = np.zeros((1, 64, 64), dtype=np.uint16)
    image = Image2DModel.parse(data, dims=("c", "y", "x"), scale_factors=[2, 2])
    sdata = _sdata(morphology_focus=image)
    _experiment(tmp_path, {"analysis_sw_version": "xenium-1.4.0.6"})

    assert name_pre_v2_dapi_channels(sdata, tmp_path) == ["morphology_focus"]

    tree = sdata.images["morphology_focus"]
    for scale in tree.children.values():
        assert list(scale["image"].coords["c"].values) == [DAPI_CHANNEL]


@pytest.mark.parametrize(
    "specs",
    [
        {"analysis_sw_version": "xenium-2.0.0.10"},
        {"analysis_sw_version": "xenium-3.0.0.15"},
        # A Xenium Ranger re-segmentation version takes precedence, as in the
        # reader.
        {
            "analysis_sw_version": "xenium-1.4.0.6",
            "xenium_ranger": {"version": "xenium-3.0.1.1"},
        },
    ],
)
def test_v2_and_later_versions_are_left_alone(
    tmp_path: Path, specs: dict[str, object]
) -> None:
    sdata = _sdata(morphology_focus=_image(None))
    _experiment(tmp_path, specs)

    assert name_pre_v2_dapi_channels(sdata, tmp_path) == []
    assert list(get_channel_names(sdata.images["morphology_focus"])) == [0]


@pytest.mark.parametrize(
    "specs", [{"analysis_sw_version": "experimental-build"}, {}, None]
)
def test_unparsable_versions_are_treated_as_pre_v2(
    tmp_path: Path, specs: dict[str, object] | None
) -> None:
    """spatialdata-io 0.6 reads the pre-2.0 single-file images for these."""
    sdata = _sdata(morphology_focus=_image(None))
    if specs is not None:
        _experiment(tmp_path, specs)

    assert name_pre_v2_dapi_channels(sdata, tmp_path) == ["morphology_focus"]
    assert list(get_channel_names(sdata.images["morphology_focus"])) == [DAPI_CHANNEL]


@pytest.mark.parametrize(
    ("channels", "n_channels"),
    [(["DAPI"], 1), (["nuclei"], 1), ([0, 1], 2)],
)
def test_named_or_multichannel_images_are_left_alone(
    tmp_path: Path, channels: list[object], n_channels: int
) -> None:
    sdata = _sdata(morphology_focus=_image(channels, n_channels))
    _experiment(tmp_path, {"analysis_sw_version": "xenium-1.4.0.6"})

    assert name_pre_v2_dapi_channels(sdata, tmp_path) == []
    assert list(get_channel_names(sdata.images["morphology_focus"])) == channels


def test_other_images_are_not_renamed(tmp_path: Path) -> None:
    sdata = _sdata(morphology_focus=_image(None), aligned_if=_image(None))
    _experiment(tmp_path, {"analysis_sw_version": "xenium-1.4.0.6"})

    assert name_pre_v2_dapi_channels(sdata, tmp_path) == ["morphology_focus"]
    assert list(get_channel_names(sdata.images["aligned_if"])) == [0]
