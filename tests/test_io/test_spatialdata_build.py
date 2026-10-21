"""Tests for SpatialData build-step orchestration."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest

from merxen.config import (
    MerscopeBeadRemovalConfig,
    MerscopeBuildConfig,
    SpatialDataBuildConfig,
)
from merxen.io.bead_removal.provenance import (
    IMAGE_PREPROCESSING_ATTR,
    build_image_preprocessing_record,
)
from merxen.io.builders.merscope import (
    ProjectionBuildResult,
    _build_iterative_projection_store,
    write_merscope_spatialdata,
)
from merxen.io.builders.pipeline import build_spatialdata_artifact
from merxen.io.image_source import MERSCOPE_ZPROJ_IMAGE_NAME
from merxen.path_utils import remove_path


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


def test_build_spatialdata_artifact_accepts_direct_vzg2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A direct .vzg2 file should be treated as a MERSCOPE source input."""
    archive_path = tmp_path / "source.vzg2"
    archive_path.touch()
    output_path = tmp_path / "stage" / "source.zarr"
    calls: list[Path] = []

    fake_module = types.ModuleType("merxen.io.builders.merscope")

    def _fake_write_merscope_spatialdata(**kwargs: object) -> Path:
        source = Path(kwargs["input_path"])
        destination = Path(kwargs["output_path"])
        destination.mkdir(parents=True)
        calls.append(source)
        return destination

    fake_module.write_merscope_spatialdata = _fake_write_merscope_spatialdata  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "merxen.io.builders.merscope", fake_module)
    cfg = SpatialDataBuildConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        input_path=archive_path,
        output_path=output_path,
    )

    result = build_spatialdata_artifact(cfg, force_rerun=True)

    assert result == output_path
    assert calls == [archive_path]


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
    projection_tmp = tmp_path / "projection_tmp"
    projection_tmp.mkdir()
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
        lambda **kwargs: ProjectionBuildResult(projection, projection_tmp),
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
    assert not projection_tmp.exists()


def test_iterative_merscope_projection_store_bounds_active_planes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Iterative projection should produce the same max without stacking z planes."""
    planes: dict[tuple[str, int], np.ndarray] = {
        ("DAPI", 0): np.array([[1, 5], [3, 2]], dtype=np.uint16),
        ("DAPI", 1): np.array([[4, 2], [9, 1]], dtype=np.uint16),
        ("DAPI", 2): np.array([[0, 8], [7, 6]], dtype=np.uint16),
        ("PolyT", 0): np.array([[10, 1], [2, 3]], dtype=np.uint16),
        ("PolyT", 1): np.array([[4, 12], [1, 5]], dtype=np.uint16),
        ("PolyT", 2): np.array([[6, 7], [8, 0]], dtype=np.uint16),
    }
    opened: list[tuple[str, int]] = []

    def _fake_open_plane(**kwargs: object) -> da.Array:
        stain = str(kwargs["stain"])
        z_layer = int(kwargs["z_layer"])
        opened.append((stain, z_layer))
        return da.from_array(planes[(stain, z_layer)], chunks=(1, 1))

    monkeypatch.setattr(
        "merxen.io.builders.merscope._open_merscope_channel_plane_array",
        _fake_open_plane,
    )

    result = _build_iterative_projection_store(
        images_dir=tmp_path,
        stainings=["DAPI", "PolyT"],
        z_layers=[0, 1, 2],
        image_models_kwargs={"chunks": (1, 1, 1), "scale_factors": [2]},
        temp_root=tmp_path,
    )

    try:
        projection = result.image.data.compute()
        expected = np.stack(
            [
                np.maximum.reduce(
                    [planes[("DAPI", 0)], planes[("DAPI", 1)], planes[("DAPI", 2)]]
                ),
                np.maximum.reduce(
                    [
                        planes[("PolyT", 0)],
                        planes[("PolyT", 1)],
                        planes[("PolyT", 2)],
                    ]
                ),
            ],
            axis=0,
        )

        assert result.temp_path is not None
        assert result.temp_path.exists()
        assert tuple(result.image.dims) == ("c", "y", "x")
        assert list(result.image.coords["c"].values) == ["DAPI", "PolyT"]
        assert opened == [
            ("DAPI", 0),
            ("DAPI", 0),
            ("DAPI", 1),
            ("DAPI", 2),
            ("PolyT", 0),
            ("PolyT", 1),
            ("PolyT", 2),
        ]
        np.testing.assert_array_equal(projection, expected)
    finally:
        if result.temp_path is not None:
            remove_path(result.temp_path)


def _fake_spatialdata_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the MERSCOPE reader return a SpatialData stand-in without images."""
    fake = types.ModuleType("spatialdata_io")
    fake.merscope = lambda *args, **kwargs: SimpleNamespace(images={})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spatialdata_io", fake)


def test_bead_removal_refuses_z_layers_with_the_bead_plane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Plane 0 is read for detection only, so it cannot be projected."""
    _fake_spatialdata_io(monkeypatch)

    with pytest.raises(ValueError, match="start at plane 1"):
        write_merscope_spatialdata(
            input_path=tmp_path,
            output_path=tmp_path / "source.zarr",
            build_config=MerscopeBuildConfig(
                z_layers=[0, 1, 2, 3],
                bead_removal=MerscopeBeadRemovalConfig(enabled=True),
            ),
        )


def test_bead_removal_refuses_z_layers_above_seven(tmp_path: Path) -> None:
    """Planes above 7 have no template."""
    with pytest.raises(ValueError, match="up to 7"):
        write_merscope_spatialdata(
            input_path=tmp_path,
            output_path=tmp_path / "source.zarr",
            build_config=MerscopeBuildConfig(
                z_layers=[1, 8],
                bead_removal=MerscopeBeadRemovalConfig(enabled=True),
            ),
        )


def test_bead_removal_z_layer_refusal_is_not_applied_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A disabled request keeps the existing z_layers behaviour."""
    _fake_spatialdata_io(monkeypatch)
    monkeypatch.setattr(
        "merxen.io.builders.merscope._load_merscope_projection_from_raw",
        lambda **kwargs: ProjectionBuildResult(object(), None),
    )
    written: list[object] = []
    monkeypatch.setattr(
        "merxen.io.builders.merscope.write_spatialdata_zarr",
        lambda sdata, path, overwrite=None: written.append(sdata),
    )

    write_merscope_spatialdata(
        input_path=tmp_path,
        output_path=tmp_path / "source.zarr",
        build_config=MerscopeBuildConfig(
            z_layers=[0, 1, 9], bead_removal=MerscopeBeadRemovalConfig()
        ),
    )

    assert len(written) == 1


def test_bead_removal_projection_is_not_wired_yet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An accepted request must not silently fall back to the plain projection."""
    _fake_spatialdata_io(monkeypatch)
    monkeypatch.setattr(
        "merxen.io.builders.merscope._load_merscope_projection_from_raw",
        lambda **kwargs: pytest.fail("the plain projection must not be built"),
    )

    with pytest.raises(NotImplementedError, match="later work package"):
        write_merscope_spatialdata(
            input_path=tmp_path,
            output_path=tmp_path / "source.zarr",
            build_config=MerscopeBuildConfig(
                z_layers=[1, 2, 3],
                bead_removal=MerscopeBeadRemovalConfig(enabled=True),
            ),
        )


def test_merscope_build_writes_the_preprocessing_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every raw-TIFF build records its preprocessing, also without beads."""
    _fake_spatialdata_io(monkeypatch)
    monkeypatch.setattr(
        "merxen.io.builders.merscope._load_merscope_projection_from_raw",
        lambda **kwargs: ProjectionBuildResult(
            object(), None, stainings=("PolyT", "DAPI")
        ),
    )
    written: list[object] = []
    monkeypatch.setattr(
        "merxen.io.builders.merscope.write_spatialdata_zarr",
        lambda sdata, path, overwrite=None: written.append(sdata),
    )

    write_merscope_spatialdata(
        input_path=tmp_path,
        output_path=tmp_path / "source.zarr",
        build_config=MerscopeBuildConfig(z_layers=[3, 1, 2]),
    )

    record = written[0].attrs[IMAGE_PREPROCESSING_ATTR]  # type: ignore[attr-defined]
    assert record["z_layers"] == [3, 1, 2]
    assert record["stains"] == ["DAPI", "PolyT"]
    assert record["bead_removal"] == {"enabled": False, "fill": {"enabled": False}}


def _cached_store(path: Path, record: dict | None) -> Path:
    """Write a stand-in cached zarr whose root attributes hold ``record``."""
    path.mkdir(parents=True)
    attributes = {} if record is None else {IMAGE_PREPROCESSING_ATTR: record}
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attributes})
    )
    return path


def _guard_config(
    tmp_path: Path,
    cached: Path,
    bead_removal: MerscopeBeadRemovalConfig | None,
    *,
    platform: str = "MERSCOPE",
    raw: bool = False,
) -> SpatialDataBuildConfig:
    raw_dir = tmp_path / "raw"
    if raw:
        raw_dir.mkdir()
    return SpatialDataBuildConfig(
        dataset_name="P1_MERSCOPE",
        platform=platform,  # type: ignore[arg-type]
        input_path=raw_dir,
        output_path=tmp_path / "staged" / "source.zarr",
        persistent_output_path=cached,
        merscope=MerscopeBuildConfig(bead_removal=bead_removal),
    )


def _on_record(*, fill: bool = False) -> dict:
    return build_image_preprocessing_record(
        z_layers=[1, 2, 3],
        stainings=["PolyT", "DAPI"],
        bead_removal_cfg=MerscopeBeadRemovalConfig(enabled=True, fill=fill),
    )


def _off_record() -> dict:
    return build_image_preprocessing_record(
        z_layers=[1, 2, 3], stainings=["PolyT", "DAPI"], bead_removal_cfg=None
    )


def test_reuse_guard_refuses_a_cache_built_with_bead_removal(tmp_path: Path) -> None:
    """A corrected cache must not silently serve a request without correction."""
    cached = _cached_store(tmp_path / "cache" / "source.zarr", _on_record())
    cfg = _guard_config(tmp_path, cached, None)

    with pytest.raises(ValueError, match="--force_spatialdata_build"):
        build_spatialdata_artifact(cfg)

    assert not cfg.output_path.exists()
    assert cached.exists()


def test_reuse_guard_refuses_a_legacy_cache_for_a_bead_request(
    tmp_path: Path,
) -> None:
    """A cache without a record predates bead removal, so it is uncorrected."""
    cached = _cached_store(tmp_path / "cache" / "source.zarr", None)
    cfg = _guard_config(tmp_path, cached, MerscopeBeadRemovalConfig(enabled=True))

    with pytest.raises(ValueError, match="built with bead removal off"):
        build_spatialdata_artifact(cfg)


@pytest.mark.parametrize("record_kind", ["legacy", "off"])
def test_reuse_guard_reuses_a_cache_without_bead_removal(
    tmp_path: Path, record_kind: str
) -> None:
    """Default runs against legacy or uncorrected caches are unchanged."""
    record = None if record_kind == "legacy" else _off_record()
    cached = _cached_store(tmp_path / "cache" / "source.zarr", record)
    cfg = _guard_config(tmp_path, cached, None)

    out = build_spatialdata_artifact(cfg)

    assert out.resolve() == cached.resolve()


def test_reuse_guard_reuses_a_matching_bead_cache(tmp_path: Path) -> None:
    cached = _cached_store(tmp_path / "cache" / "source.zarr", _on_record())
    cfg = _guard_config(tmp_path, cached, MerscopeBeadRemovalConfig(enabled=True))

    out = build_spatialdata_artifact(cfg)

    assert out.resolve() == cached.resolve()


@pytest.mark.parametrize(
    ("recorded_fill", "requested_fill"), [(False, True), (True, False)]
)
def test_reuse_guard_refuses_a_different_fill_setting(
    tmp_path: Path, recorded_fill: bool, requested_fill: bool
) -> None:
    cached = _cached_store(
        tmp_path / "cache" / "source.zarr", _on_record(fill=recorded_fill)
    )
    cfg = _guard_config(
        tmp_path,
        cached,
        MerscopeBeadRemovalConfig(enabled=True, fill=requested_fill),
    )

    with pytest.raises(ValueError, match="merscope_bead_removal_fill"):
        build_spatialdata_artifact(cfg)


def test_reuse_guard_checks_an_already_staged_output(tmp_path: Path) -> None:
    """The staged copy is reused when no persistent cache exists, so it is checked."""
    staged = _cached_store(tmp_path / "staged" / "source.zarr", _on_record())
    cfg = SpatialDataBuildConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        input_path=tmp_path / "raw",
        output_path=staged,
    )

    with pytest.raises(ValueError, match="--force_spatialdata_build"):
        build_spatialdata_artifact(cfg)


def test_reuse_guard_lets_a_forced_rebuild_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A forced rebuild replaces the mismatching cache; nothing is deleted early."""
    cached = _cached_store(tmp_path / "cache" / "source.zarr", _on_record())
    cfg = _guard_config(tmp_path, cached, None, raw=True)
    calls: list[Path] = []

    fake_module = types.ModuleType("merxen.io.builders.merscope")

    def _fake_write(**kwargs: object) -> Path:
        destination = Path(kwargs["output_path"])  # type: ignore[arg-type]
        destination.mkdir(parents=True)
        calls.append(destination)
        return destination

    fake_module.write_merscope_spatialdata = _fake_write  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "merxen.io.builders.merscope", fake_module)

    build_spatialdata_artifact(cfg, force_rerun=True)

    assert calls == [cached]


def test_reuse_guard_ignores_other_platforms(tmp_path: Path) -> None:
    cached = _cached_store(tmp_path / "cache" / "source.zarr", _on_record())
    cfg = _guard_config(tmp_path, cached, None, platform="XENIUM")

    out = build_spatialdata_artifact(cfg)

    assert out.resolve() == cached.resolve()
