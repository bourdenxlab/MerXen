"""The convert-czi command: argument handling, scenes, names, pixel size, summary."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

from merxen.cli.run_convert_czi import convert_czi_command
from merxen.io import czi
from merxen.io.ome_tiff import read_ome_tiff_info

FIXTURES = Path(__file__).resolve().parents[1] / "test_data" / "czi"

SUMMARY_KEYS = {
    "input_path",
    "output_path",
    "scene_index",
    "scene_name",
    "bbox_xywh",
    "shape_cyx",
    "level_shapes_yx",
    "channel_names",
    "name_sources",
    "channel_name_warnings",
    "pixel_size_um",
    "pixel_size_source",
    "metadata_pixel_size_um",
    "z_planes_projected",
    "pixel_type",
    "compression",
    "czifile_version",
    "seconds",
}


@pytest.fixture(params=["in_process", "runtime"])
def backend_args(
    request: pytest.FixtureRequest, real_path: Callable[[str], Path]
) -> list[str]:
    """Extra CLI arguments selecting czifile in this process or the CZI runtime."""
    if request.param == "in_process":
        pytest.importorskip("czifile")
        return []
    value = os.environ.get("MERXEN_CZI_PYTHON")
    return ["--czi-python", value if value else str(real_path("MERXEN_CZI_PYTHON"))]


def run(args: list[str]) -> tuple[int, str]:
    result = CliRunner().invoke(convert_czi_command, args)
    return result.exit_code, result.output


def test_output_suffix_is_checked_first(tmp_path: Path) -> None:
    code, output = run(
        [
            "--input",
            str(FIXTURES / "one_scene_hole.czi"),
            "--output",
            str(tmp_path / "x.tif"),
        ]
    )
    assert code == 2
    assert ".ome.tif" in output


def test_convert_single_scene(tmp_path: Path, backend_args: list[str]) -> None:
    out = tmp_path / "out.ome.tif"
    summary = tmp_path / "s.json"
    code, output = run(
        [
            "--input",
            str(FIXTURES / "one_scene_hole.czi"),
            "--output",
            str(out),
            "--tile-size",
            "16",
            "--workers",
            "2",
            "--summary-json",
            str(summary),
            *backend_args,
        ]
    )
    assert code == 0, output
    assert read_ome_tiff_info(out).channel_names == ("DAPI", "AF568", "AF647")
    data = json.loads(summary.read_text())
    assert set(data) == SUMMARY_KEYS
    assert data["pixel_size_source"] == "metadata"
    assert data["name_sources"] == ["Name", "Name", "Name"]
    assert data["scene_index"] == 0
    assert data["shape_cyx"] == [3, 80, 96]
    assert "Converted scene 0 (96×80 px, 3 channels: DAPI, AF568, AF647)" in output


def test_several_scenes_need_a_choice(tmp_path: Path, backend_args: list[str]) -> None:
    source = str(FIXTURES / "two_scenes_zstd.czi")
    out = tmp_path / "out.ome.tif"
    code, output = run(["--input", source, "--output", str(out), *backend_args])
    assert code == 1
    assert "2 scenes" in output
    assert "x=2000, y=3000" in output
    assert not out.exists()

    code, output = run(
        [
            "--input",
            source,
            "--output",
            str(out),
            "--scene",
            "1",
            "--channel-names",
            "DAPI;AF568;AF647",
            "--tile-size",
            "16",
            *backend_args,
        ]
    )
    assert code == 0, output
    assert read_ome_tiff_info(out).channel_names == ("DAPI", "AF568", "AF647")

    code, output = run(
        [
            "--input",
            source,
            "--output",
            str(tmp_path / "bad.ome.tif"),
            "--scene",
            "0",
            "--channel-names",
            "a;b",
            *backend_args,
        ]
    )
    assert code == 1
    assert "detected candidates" in output
    assert not (tmp_path / "bad.ome.tif").exists()


def test_missing_pixel_size(tmp_path: Path, backend_args: list[str]) -> None:
    source = str(FIXTURES / "no_scaling.czi")
    out = tmp_path / "out.ome.tif"
    code, output = run(["--input", source, "--output", str(out), *backend_args])
    assert code == 1
    assert "--pixel-size-um" in output
    code, output = run(
        [
            "--input",
            source,
            "--output",
            str(out),
            "--pixel-size-um",
            "0.5",
            "--tile-size",
            "16",
            *backend_args,
        ]
    )
    assert code == 0, output
    assert read_ome_tiff_info(out).pixel_size_um == pytest.approx((0.5, 0.5))


def test_no_runtime_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(czi, "czifile_available", lambda: False)
    monkeypatch.delenv("MERXEN_CZI_PYTHON", raising=False)
    code, output = run(
        [
            "--input",
            str(FIXTURES / "one_scene_hole.czi"),
            "--output",
            str(tmp_path / "out.ome.tif"),
        ]
    )
    assert code == 1
    assert "MERXEN_CZI_PYTHON" in output
