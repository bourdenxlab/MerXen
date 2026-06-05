"""Tests for SpatialData build-step orchestration."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from merxen.config import MerscopeBuildConfig, SpatialDataBuildConfig
from merxen.io.builders.merscope import write_merscope_spatialdata
from merxen.io.builders.pipeline import build_spatialdata_artifact
from merxen.io.image_source import MERSCOPE_ZPROJ_IMAGE_NAME


def test_build_spatialdata_artifact_reuses_existing_persistent_zarr(
    tmp_path: Path,
) -> None:
    """A persistent cached zarr should be staged without rebuilding."""
    persistent_zarr = tmp_path / "cache" / "source.zarr"
    persistent_zarr.mkdir(parents=True)
    output_path = tmp_path / "staged" / "source.zarr"
    cfg = SpatialDataBuildConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        input_path=tmp_path / "missing_raw_dir",
        output_path=output_path,
        persistent_output_path=persistent_zarr,
    )

    out = build_spatialdata_artifact(cfg)

    assert out == output_path
    assert output_path.exists()
    assert output_path.resolve() == persistent_zarr.resolve()


def test_build_spatialdata_artifact_force_requires_raw_input(
    tmp_path: Path,
) -> None:
    """Force rebuild should fail if only an existing zarr was supplied."""
    persistent_zarr = tmp_path / "cache" / "source.zarr"
    persistent_zarr.mkdir(parents=True)
    cfg = SpatialDataBuildConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        input_path=persistent_zarr,
        output_path=tmp_path / "staged" / "source.zarr",
        persistent_output_path=persistent_zarr,
    )

    with pytest.raises(ValueError, match="force-rerun"):
        build_spatialdata_artifact(cfg, force_rerun=True)


def test_build_spatialdata_artifact_dispatches_to_platform_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The orchestrator should call the correct platform writer for raw input."""
    raw_dir = tmp_path / "merscope_raw"
    raw_dir.mkdir()
    output_path = tmp_path / "stage" / "source.zarr"
    calls: list[tuple[Path, Path]] = []

    fake_module = types.ModuleType("merxen.io.builders.merscope")

    def _fake_write_merscope_spatialdata(**kwargs: object) -> Path:
        input_path = Path(kwargs["input_path"])
        final_output_path = Path(kwargs["output_path"])
        final_output_path.mkdir(parents=True)
        calls.append((input_path, final_output_path))
        return final_output_path

    fake_module.write_merscope_spatialdata = _fake_write_merscope_spatialdata  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "merxen.io.builders.merscope", fake_module)

    cfg = SpatialDataBuildConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        input_path=raw_dir,
        output_path=output_path,
    )

    out = build_spatialdata_artifact(cfg, force_rerun=True)

    assert out == output_path
    assert calls == [(raw_dir, output_path)]


def test_merscope_writer_builds_projection_without_reader_mosaics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MERSCOPE raw builds should not ask spatialdata_io to materialize z planes."""
    raw_dir = tmp_path / "merscope_raw"
    raw_dir.mkdir()
    output_path = tmp_path / "source.zarr"
    calls: dict[str, object] = {}
    projection = object()
    written: list[tuple[object, Path, bool | None]] = []

    fake_spatialdata_io = types.ModuleType("spatialdata_io")

    def _fake_merscope_reader(*args: object, **kwargs: object) -> object:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return SimpleNamespace(images={"unexpected_z_plane": object()})

    fake_spatialdata_io.merscope = _fake_merscope_reader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spatialdata_io", fake_spatialdata_io)
    monkeypatch.setattr(
        "merxen.io.builders.merscope._load_merscope_projection_from_raw",
        lambda **kwargs: projection,
    )
    monkeypatch.setattr(
        "merxen.io.builders.merscope.write_spatialdata_zarr",
        lambda sdata, path, overwrite=None: written.append((sdata, path, overwrite)),
    )

    write_merscope_spatialdata(
        input_path=raw_dir,
        output_path=output_path,
        build_config=MerscopeBuildConfig(z_layers=[0, 1]),
    )

    assert calls["kwargs"] == {
        "z_layers": [0, 1],
        "region_name": None,
        "slide_name": None,
        "mosaic_images": False,
    }
    assert len(written) == 1
    sdata, path, overwrite = written[0]
    assert path == output_path
    assert overwrite is True
    assert sdata.images == {MERSCOPE_ZPROJ_IMAGE_NAME: projection}
