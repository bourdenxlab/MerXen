"""Tests for the Zeiss CZI reader/converter and its runtime bridge."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merxen import local_config
from merxen.io import czi
from merxen.io.channel_names import resolve_channel_names
from merxen.io.czi import (
    CziChannelMeta,
    CziConversionResult,
    CziFileInfo,
    CziRuntimeUnavailableError,
    CziSceneMeta,
    czi_channel_candidates,
    czi_runtime_python,
    describe_scenes,
    parse_channel_metadata,
    parse_scene_names,
    resolve_czi_pixel_size,
    resolve_scene,
    run_convert_czi_scene,
    run_read_czi_info,
)
from merxen.io.ome_tiff import open_ome_tiff_level, read_ome_tiff_info

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "test_data" / "czi"


@pytest.fixture(params=["in_process", "runtime"])
def czi_python(
    request: pytest.FixtureRequest, real_path: Callable[[str], Path]
) -> str | None:
    """Backend of the pixel tests: czifile in this process, or the CZI runtime."""
    if request.param == "in_process":
        pytest.importorskip("czifile")
        return None
    # The environment value (CI: GITHUB_ENV) is taken first; otherwise the value
    # captured before the test isolation (environment or .env) is used, or the
    # test skips.
    value = os.environ.get("MERXEN_CZI_PYTHON")
    return value if value else str(real_path("MERXEN_CZI_PYTHON"))


def ref_levels(array: np.ndarray, n_sub: int) -> list[np.ndarray]:
    """Recursive exact 2x2 reference pyramid (edge pad, (sum + 2) // 4)."""
    levels = [array]
    for _ in range(n_sub):
        out = []
        for plane in levels[-1]:
            h, w = plane.shape
            padded = np.pad(plane, ((0, h % 2), (0, w % 2)), mode="edge")
            total = np.zeros((padded.shape[0] // 2, padded.shape[1] // 2), np.uint64)
            for dy in (0, 1):
                for dx in (0, 1):
                    total += padded[dy::2, dx::2].astype(np.uint64)
            out.append(((total + 2) // 4).astype(array.dtype))
        levels.append(np.stack(out))
    return levels


# --------------------------------------------------------------------------
# Pure tests (no czifile)
# --------------------------------------------------------------------------

USER_XML = """
<ImageDocument><Metadata><Information><Image><Dimensions><Channels>
  <Channel Id="Channel:0" Name="DAPI"><Fluor>DAPI</Fluor>
    <ExcitationWavelength>353</ExcitationWavelength>
    <EmissionWavelength>465</EmissionWavelength><PixelType>Gray16</PixelType></Channel>
  <Channel Id="Channel:1" Name="AF568"><Fluor>Alexa Fluor 568</Fluor>
    <ExcitationWavelength>578</ExcitationWavelength>
    <EmissionWavelength>603</EmissionWavelength><PixelType>Gray16</PixelType></Channel>
  <Channel Id="Channel:2" Name="AF647"><Fluor>Alexa Fluor 647</Fluor>
    <ExcitationWavelength>653</ExcitationWavelength>
    <EmissionWavelength>668</EmissionWavelength><PixelType>Gray16</PixelType></Channel>
</Channels></Dimensions></Image></Information>
<DisplaySetting><Channels>
  <Channel Id="Channel:0" Name="DAPI"><ShortName>DAPI</ShortName></Channel>
  <Channel Id="Channel:1" Name="AF568"><ShortName>AF568</ShortName></Channel>
  <Channel Id="Channel:2" Name="AF647"><ShortName>AF647</ShortName></Channel>
</Channels></DisplaySetting></Metadata></ImageDocument>
"""


def make_info(
    *,
    pixel_size: tuple[float, float] | None = (0.5, 0.5),
    channels: list[CziChannelMeta] | None = None,
    name: str = "scan.czi",
) -> CziFileInfo:
    return CziFileInfo(
        path=Path(name),
        scenes=(CziSceneMeta(0, "ScanRegion0", (-1000, 500, 72, 56)),),
        channels=tuple(
            channels if channels is not None else parse_channel_metadata(USER_XML)
        ),
        sizes={"T": 1, "C": 3, "Z": 1, "S": 1},
        dtype="uint16",
        pixel_size_um=pixel_size,
        pixel_type="Gray16",
        compression="Uncompressed",
        czifile_version="2026.10.14",
    )


def test_parse_channel_metadata_user_layout() -> None:
    channels = parse_channel_metadata(USER_XML)
    assert channels == [
        CziChannelMeta(
            0, "Channel:0", "DAPI", "DAPI", None, "DAPI", 353.0, 465.0, "Gray16", None
        ),
        CziChannelMeta(
            1,
            "Channel:1",
            "AF568",
            "Alexa Fluor 568",
            None,
            "AF568",
            578.0,
            603.0,
            "Gray16",
            None,
        ),
        CziChannelMeta(
            2,
            "Channel:2",
            "AF647",
            "Alexa Fluor 647",
            None,
            "AF647",
            653.0,
            668.0,
            "Gray16",
            None,
        ),
    ]
    resolved = resolve_channel_names(czi_channel_candidates(make_info()))
    assert resolved.names == ("DAPI", "AF568", "AF647")
    assert resolved.sources == ("Name", "Name", "Name")


def test_parse_channel_metadata_fallbacks() -> None:
    xml = """
    <Root><Information><Image><Dimensions><Channels>
      <Channel Id="Channel:0"><Name>NeuN</Name>
        <DetectionWavelength><Ranges>500-550</Ranges></DetectionWavelength></Channel>
      <Channel Id="Channel:1"><DetectionWavelength>
        <Ranges>abc</Ranges></DetectionWavelength></Channel>
    </Channels></Dimensions></Image></Information>
    <DisplaySetting><Channels>
      <Channel Id="Channel:0"><DyeName>Alexa Fluor 488</DyeName>
        <DyeMaxExcitation>499</DyeMaxExcitation></Channel>
    </Channels></DisplaySetting></Root>
    """
    first, second = parse_channel_metadata(xml)
    assert first.name == "NeuN"
    assert first.emission_nm == 525.0
    assert first.dye_name == "Alexa Fluor 488"
    assert first.excitation_nm == 499.0
    assert second.short_name is None
    assert second.emission_nm is None
    assert parse_channel_metadata("<Root><Information/></Root>") == []


def test_parse_scene_names() -> None:
    xml = (
        "<Root><Information><Image><Dimensions><S><Scenes>"
        '<Scene Index="0" Name="ScanRegion0"/>'
        '<Scene Index="2"><Name>Region B</Name></Scene>'
        "</Scenes></S></Dimensions></Image></Information></Root>"
    )
    assert parse_scene_names(xml) == {0: "ScanRegion0", 2: "Region B"}
    assert parse_scene_names("<Root/>") == {}


SCENES = (
    CziSceneMeta(0, "ScanRegion0", (-1000, 500, 72, 56)),
    CziSceneMeta(1, None, (2000, 3000, 72, 56)),
)


def test_resolve_scene() -> None:
    with pytest.raises(ValueError) as caught:
        resolve_scene(SCENES, None)
    message = str(caught.value)
    assert "2 scenes" in message
    assert "0 'ScanRegion0' (72×56 px at x=-1000, y=500)" in message
    assert "1 (72×56 px at x=2000, y=3000)" in message
    assert resolve_scene(SCENES, "1") == 1
    assert resolve_scene(SCENES, " scanregion0 ") == 0
    with pytest.raises(ValueError, match="not found"):
        resolve_scene(SCENES, "7")
    with pytest.raises(ValueError, match="not found"):
        resolve_scene(SCENES, "nope")
    assert resolve_scene(SCENES[1:], None) == 1
    twins = (CziSceneMeta(0, "A", (0, 0, 1, 1)), CziSceneMeta(1, "A", (0, 0, 1, 1)))
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_scene(twins, "a")
    with pytest.raises(ValueError, match="no image scenes"):
        resolve_scene((), None)
    assert describe_scenes(SCENES).startswith("0 'ScanRegion0' (72×56 px")


def test_resolve_czi_pixel_size(caplog: pytest.LogCaptureFixture) -> None:
    assert resolve_czi_pixel_size(make_info(pixel_size=(0.34511, 0.34511)), None) == (
        0.34511,
        "metadata",
    )
    assert (
        resolve_czi_pixel_size(make_info(pixel_size=(0.5, 0.5002)), None)[1]
        == "metadata"
    )
    with pytest.raises(ValueError, match="differ"):
        resolve_czi_pixel_size(make_info(pixel_size=(0.5, 0.501)), None)
    with pytest.raises(ValueError, match="reference_pixel_size_um"):
        resolve_czi_pixel_size(
            make_info(pixel_size=None), None, override_hint="reference_pixel_size_um"
        )
    with caplog.at_level(logging.WARNING):
        assert resolve_czi_pixel_size(make_info(), 0.6) == (0.6, "override")
    assert "differs from the metadata" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert resolve_czi_pixel_size(make_info(), 0.5) == (0.5, "override")
    assert caplog.text == ""
    for bad in (0.0, float("nan")):
        with pytest.raises(ValueError, match="finite"):
            resolve_czi_pixel_size(make_info(), bad)


def test_dict_round_trip() -> None:
    info = make_info()
    assert CziFileInfo.from_dict(info.to_dict()) == info
    assert CziFileInfo.from_dict(json.loads(json.dumps(info.to_dict()))) == info
    none_scale = make_info(pixel_size=None)
    assert CziFileInfo.from_dict(none_scale.to_dict()) == none_scale
    result = CziConversionResult(
        output_path=Path("out.ome.tif"),
        shape_cyx=(3, 56, 72),
        level_shapes_yx=((56, 72), (28, 36)),
        scene_index=1,
        scene_name=None,
        bbox_xywh=(2000, 3000, 72, 56),
        z_planes_projected=3,
        pixel_type="Gray16",
        compression="Zstd1",
        czifile_version="2026.10.14",
        seconds=1.5,
    )
    assert (
        CziConversionResult.from_dict(json.loads(json.dumps(result.to_dict())))
        == result
    )


def test_czi_runtime_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(czi, "czifile_available", lambda: False)
    monkeypatch.delenv("MERXEN_CZI_PYTHON", raising=False)
    monkeypatch.setenv("MERXEN_ENV_FILE", "none")
    local_config.clear_cache()
    with pytest.raises(CziRuntimeUnavailableError, match="MERXEN_CZI_PYTHON"):
        czi_runtime_python()

    monkeypatch.setenv("MERXEN_CZI_PYTHON", "/runtime/python")
    assert czi_runtime_python() == Path("/runtime/python")

    monkeypatch.setenv("MERXEN_CZI_PYTHON", "in_process")
    with pytest.raises(CziRuntimeUnavailableError):
        czi_runtime_python()

    monkeypatch.delenv("MERXEN_CZI_PYTHON")
    env_file = tmp_path / "env"
    env_file.write_text("MERXEN_CZI_PYTHON=/from/dotenv/python\n", encoding="utf-8")
    monkeypatch.setenv("MERXEN_ENV_FILE", str(env_file))
    local_config.clear_cache()
    assert czi_runtime_python() == Path("/from/dotenv/python")
    assert czi_runtime_python("/explicit/python") == Path("/explicit/python")

    monkeypatch.setattr(czi, "czifile_available", lambda: True)
    assert czi_runtime_python() is None


def test_main_reports_errors_as_json(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    request.write_text(json.dumps({"path": str(tmp_path / "missing.czi")}))
    code = czi.main(["info", "--request", str(request), "--result", str(result)])
    payload = json.loads(result.read_text())
    assert code == 2
    assert payload["ok"] is False
    assert payload["message"]
    assert not list(tmp_path.glob("*.tmp"))


def test_bridge_does_not_expose_the_base_site_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_path: Callable[[str], Path],
) -> None:
    """The runtime must see only the merxen package, never its parent directory."""
    interpreter = os.environ.get("MERXEN_CZI_PYTHON")
    if not interpreter:
        try:
            interpreter = str(real_path("MERXEN_CZI_PYTHON"))
        except pytest.skip.Exception:
            interpreter = sys.executable
    own = subprocess.run(
        [interpreter, "-c", "import imagecodecs; print(imagecodecs.__file__)"],
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    )
    if own.returncode != 0:
        pytest.skip("the interpreter has no imagecodecs")
    site = tmp_path / "site-packages"
    (site / "imagecodecs").mkdir(parents=True)
    (site / "imagecodecs" / "__init__.py").write_text('__version__ = "0.0.dummy"\n')
    (site / "merxen").symlink_to(czi._merxen_package_dir(), target_is_directory=True)
    monkeypatch.setattr(czi, "_merxen_package_dir", lambda: site / "merxen")
    # A development PYTHONPATH would put another copy of merxen's dependencies first.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    modules = czi._run_runtime(Path(interpreter), "probe", {})["modules"]
    assert modules["imagecodecs"] == os.path.realpath(own.stdout.strip())
    assert not str(modules["imagecodecs"]).startswith(str(site))
    assert (
        Path(modules["merxen"])
        .resolve()
        .is_relative_to(Path(czi.__file__).resolve().parent.parent)
    )


# --------------------------------------------------------------------------
# Backend tests (czifile in this process or through the CZI runtime)
# --------------------------------------------------------------------------


def test_info_two_scenes(czi_python: str | None) -> None:
    info = run_read_czi_info(FIXTURES / "two_scenes_zstd.czi", python=czi_python)
    assert info.scenes == (
        CziSceneMeta(0, None, (-1000, 500, 72, 56)),
        CziSceneMeta(1, None, (2000, 3000, 72, 56)),
    )
    assert len(info.channels) == 3
    assert all(ch.name is None for ch in info.channels)
    assert info.sizes["C"] == 3
    assert info.sizes["Z"] == 3
    assert info.sizes.get("T", 1) == 1
    assert info.dtype == "uint16"
    assert info.pixel_size_um == pytest.approx((0.5, 0.5))
    assert info.pixel_type == "Gray16"
    assert info.compression == "Zstd1"
    assert info.czifile_version == "2026.10.14"
    resolved = resolve_channel_names(czi_channel_candidates(info))
    assert resolved.names == ("c0", "c1", "c2")
    assert resolved.sources == ("fallback",) * 3


def test_info_hole(czi_python: str | None) -> None:
    info = run_read_czi_info(FIXTURES / "one_scene_hole.czi", python=czi_python)
    assert info.scenes == (CziSceneMeta(0, None, (-131, 20, 96, 80)),)
    assert [ch.name for ch in info.channels] == ["DAPI", "AF568", "AF647"]
    assert info.sizes["Z"] == 1
    assert info.pixel_size_um == pytest.approx((0.34511, 0.34511))
    assert info.compression == "Uncompressed"


CONVERSIONS = [
    ("two_scenes_zstd", 0, 1),
    ("two_scenes_zstd", 0, 3),
    ("two_scenes_zstd", 1, 3),
    ("one_scene_hole", 0, 1),
    ("one_scene_hole", 0, 3),
]


def convert(
    name: str, scene: int, workers: int, out: Path, czi_python: str | None
) -> CziConversionResult:
    return run_convert_czi_scene(
        FIXTURES / f"{name}.czi",
        out,
        python=czi_python,
        scene=scene,
        channel_names=["A", "B", "C"],
        pixel_size_um=0.5,
        image_name="fixture_reference",
        tile_size=16,
        band_height=16,
        min_level_size=16,
        workers=workers,
    )


@pytest.mark.parametrize(("name", "scene", "workers"), CONVERSIONS)
def test_conversion_matches_oracle(
    name: str, scene: int, workers: int, tmp_path: Path, czi_python: str | None
) -> None:
    oracle = np.load(FIXTURES / "oracle.npz")
    out = tmp_path / "out.ome.tif"
    result = convert(name, scene, workers, out, czi_python)
    expected = oracle[f"{name}__s{scene}"]

    info = read_ome_tiff_info(out)
    shapes = (
        ((56, 72), (28, 36), (14, 18), (7, 9))
        if name == "two_scenes_zstd"
        else ((80, 96), (40, 48), (20, 24), (10, 12))
    )
    assert info.level_shapes_cyx == tuple((3, *s) for s in shapes)
    assert info.channel_names == ("A", "B", "C")
    assert info.pixel_size_um == pytest.approx((0.5, 0.5))
    assert np.array_equal(open_ome_tiff_level(info, 0).values, expected)
    for k, reference in enumerate(ref_levels(expected, 3)):
        assert np.array_equal(open_ome_tiff_level(info, k).values, reference), k

    assert result.scene_index == scene
    assert result.bbox_xywh == tuple(oracle[f"{name}__bbox"][scene].tolist())
    assert result.z_planes_projected == (3 if name == "two_scenes_zstd" else 1)
    assert result.pixel_type == "Gray16"
    assert result.compression == (
        "Zstd1" if name == "two_scenes_zstd" else "Uncompressed"
    )
    assert result.czifile_version == "2026.10.14"
    assert result.seconds >= 0
    assert result.shape_cyx == (3, *shapes[0])
    assert result.level_shapes_yx == shapes
    if name == "one_scene_hole" and workers == 3:
        level0 = open_ome_tiff_level(info, 0).values
        assert not level0[:, 0:40, 48:96].any()
        assert level0[1, 0, 0] == 2000


def test_conversion_spot_values(tmp_path: Path, czi_python: str | None) -> None:
    out = tmp_path / "a.ome.tif"
    convert("two_scenes_zstd", 0, 2, out, czi_python)
    two = open_ome_tiff_level(read_ome_tiff_info(out), 0).values
    assert two[0, 0, 0] == 150
    assert two[0, 30, 36] == 103
    out = tmp_path / "b.ome.tif"
    convert("two_scenes_zstd", 1, 2, out, czi_python)
    assert open_ome_tiff_level(read_ome_tiff_info(out), 0).values[2, 30, 36] == 2113
    out = tmp_path / "c.ome.tif"
    convert("one_scene_hole", 0, 2, out, czi_python)
    hole = open_ome_tiff_level(read_ome_tiff_info(out), 0).values
    assert hole[0, 10, 60] == 0
    assert hole[1, 0, 0] == 2000
    assert hole[0, 45, 10] == 103
    assert hole[2, 79, 95] == 4259


def test_conversion_is_deterministic(tmp_path: Path, czi_python: str | None) -> None:
    a, b = tmp_path / "a.ome.tif", tmp_path / "b.ome.tif"
    convert("one_scene_hole", 0, 1, a, czi_python)
    convert("one_scene_hole", 0, 3, b, czi_python)
    assert a.read_bytes() == b.read_bytes()


def test_unsupported_files(tmp_path: Path, czi_python: str | None) -> None:
    with pytest.raises(ValueError, match="not supported in v1"):
        run_read_czi_info(FIXTURES / "rgb_bgr24.czi", python=czi_python)
    with pytest.raises(ValueError, match=r"unsupported dimensions \{'T': 2\}"):
        run_read_czi_info(FIXTURES / "time_t2.czi", python=czi_python)
    info = run_read_czi_info(FIXTURES / "no_scaling.czi", python=czi_python)
    assert info.pixel_size_um is None
    with pytest.raises(ValueError, match="no pixel size"):
        resolve_czi_pixel_size(info, None)

    source = FIXTURES / "two_scenes_zstd.czi"
    out = tmp_path / "x.ome.tif"
    common: dict[str, Any] = {
        "python": czi_python,
        "pixel_size_um": 0.5,
        "image_name": "x",
        "tile_size": 16,
        "band_height": 16,
        "min_level_size": 16,
    }
    with pytest.raises(ValueError, match="does not exist"):
        run_convert_czi_scene(
            source, out, scene=5, channel_names=["A", "B", "C"], **common
        )
    with pytest.raises(ValueError, match=r"2 channel name\(s\)"):
        run_convert_czi_scene(source, out, scene=0, channel_names=["A", "B"], **common)
    with pytest.raises(ValueError, match="channel name"):
        run_convert_czi_scene(
            source, out, scene=0, channel_names=["A", "a", "B"], **common
        )
    assert not out.exists()


PROXY_PRELUDE = textwrap.dedent(
    """
    import json, logging, sys
    from pathlib import Path
    from types import SimpleNamespace
    from merxen.io import czi

    real_open = czi._open_czi
    """
)


def run_proxy_script(body: str, czi_python: str | None, *args: str) -> dict[str, Any]:
    """Run ``PROXY_PRELUDE + body`` where czifile is (the runtime interpreter).

    The script prints one JSON object on its last stdout line.
    """
    code = PROXY_PRELUDE + textwrap.dedent(body)
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run(
        [czi_python or sys.executable, "-c", code, *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return payload


def test_multi_file_czi_is_rejected(czi_python: str | None) -> None:
    result = run_proxy_script(
        """
        class Proxy:
            def __init__(self, path):
                self._real = real_open(path)
                self.header = SimpleNamespace(
                    file_part=1, file_guid="a", primary_file_guid="b"
                )

            def __getattr__(self, name):
                return getattr(self._real, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._real.close()

        czi._open_czi = Proxy
        try:
            czi.read_czi_info(Path(sys.argv[1]))
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}))
        else:
            print(json.dumps({"error": None}))
        """,
        czi_python,
        str(FIXTURES / "one_scene_hole.czi"),
    )
    assert result["error"] is not None
    assert "multi-file CZI" in result["error"]


def test_missing_channel_plane_is_zero_filled_and_warned_once(
    tmp_path: Path, czi_python: str | None
) -> None:
    out = tmp_path / "out.ome.tif"
    result = run_proxy_script(
        """
        class Image:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def __call__(self, **kwargs):
                if kwargs.get("C") == 1:
                    raise ValueError("selection matches no subblocks")
                return self._real(**kwargs)

        class Handle:
            def __init__(self, path):
                self._real = real_open(path)
                self.header = self._real.header

            def __getattr__(self, name):
                return getattr(self._real, name)

            @property
            def scenes(self):
                real = self._real.scenes

                class Scenes:
                    def __call__(self, *args, **kwargs):
                        return real(*args, **kwargs)

                    def __getitem__(self, key):
                        return Image(real[key])

                    def items(self):
                        return [(k, Image(v)) for k, v in real.items()]

                    def __getattr__(self, name):
                        return getattr(real, name)

                return Scenes()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._real.close()

        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        logging.getLogger().addHandler(Capture())
        czi._open_czi = Handle
        czi.convert_czi_scene(
            Path(sys.argv[1]), Path(sys.argv[2]), scene=0,
            channel_names=["A", "B", "C"], pixel_size_um=0.5, image_name="x",
            tile_size=16, band_height=16, min_level_size=16, workers=3,
        )
        print(json.dumps({"warnings": [m for m in records if "no data for" in m]}))
        """,
        czi_python,
        str(FIXTURES / "one_scene_hole.czi"),
        str(out),
    )
    assert len(result["warnings"]) == 1
    assert "channel 1, plane 0" in result["warnings"][0]
    expected = np.load(FIXTURES / "oracle.npz")["one_scene_hole__s0"]
    level0 = open_ome_tiff_level(read_ome_tiff_info(out), 0).values
    assert not level0[1].any()
    assert expected[1].any()
    assert np.array_equal(level0[0], expected[0])
    assert np.array_equal(level0[2], expected[2])


def test_relative_paths_are_resolved_in_the_callers_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, czi_python: str | None
) -> None:
    """The runtime starts elsewhere; relative paths must still mean the caller's."""
    (tmp_path / "in.czi").symlink_to(FIXTURES / "one_scene_hole.czi")
    monkeypatch.chdir(tmp_path)
    info = run_read_czi_info(Path("in.czi"), python=czi_python)
    assert info.scenes[0].bbox_xywh == (-131, 20, 96, 80)
    run_convert_czi_scene(
        Path("in.czi"),
        Path("rel.ome.tif"),
        python=czi_python,
        scene=0,
        channel_names=["A", "B", "C"],
        pixel_size_um=0.5,
        image_name="fixture_reference",
        tile_size=16,
        band_height=16,
        min_level_size=16,
        workers=2,
        tmp_dir=Path("pyramid-tmp"),
    )
    expected = np.load(FIXTURES / "oracle.npz")["one_scene_hole__s0"]
    assert (tmp_path / "rel.ome.tif").is_file()
    level0 = open_ome_tiff_level(read_ome_tiff_info(tmp_path / "rel.ome.tif"), 0)
    assert np.array_equal(level0.values, expected)


def test_xml_is_never_logged(
    caplog: pytest.LogCaptureFixture, czi_python: str | None
) -> None:
    path = FIXTURES / "one_scene_hole.czi"
    if czi_python is None:
        with caplog.at_level(logging.DEBUG):
            czi.read_czi_info(path)
        assert "<ImageDocument" not in caplog.text
        return
    code = textwrap.dedent(
        """
        import logging, sys
        from pathlib import Path
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
        from merxen.io import czi
        czi.read_czi_info(Path(sys.argv[1]))
        """
    )
    shim_env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run(
        [czi_python, "-c", code, str(path)],
        env=shim_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "<ImageDocument" not in proc.stderr + proc.stdout
