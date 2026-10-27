"""Prepare step: pixel-size and normalisation rules, store plan, end-to-end runs."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest
import tifffile
import zarr
from spatialdata import read_zarr
from spatialdata.transformations import get_transformation

from merxen.io import czi
from merxen.io.ome_tiff import read_ome_tiff_info
from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.io.spatialdata_schema import validate_merxen_schema
from merxen.manual_registration import prepare as prepare_module
from merxen.manual_registration.config import (
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
)
from merxen.manual_registration.prepare import (
    _normalisation_reasons,
    _ome_band_reader,
    _plan_store_update,
    _resolve_pixel_size,
    _round_pixel_size,
    build_prepared_store,
    prepare_image,
    run_manual_registration_prepare,
)
from merxen.manual_registration.schema import (
    CONVENTIONS,
    LABELLING_ATTR,
    STORE_ATTR,
    StoreAttrs,
    canonical_json,
    labelling_content_sha256,
    read_root_attrs,
    write_root_attr,
)
from merxen.viewer_cache.build import ViewerCacheParams, build_image_pyramid_caches

FIXTURES = Path(__file__).resolve().parents[1] / "test_data" / "czi"

REF = np.random.default_rng(1).integers(0, 4000, (3, 1280, 1152), dtype=np.uint16)
MOV = np.random.default_rng(2).integers(0, 4000, (2, 1100, 1300), dtype=np.uint16)
REF_NAMES = ["DAPI", "AF568", "AF647"]
MOV_NAMES = ["Hoechst 33342", "AF568"]


def _metadata(names: list[str], pixel_size_um: float | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"axes": "CYX", "Channel": {"Name": names}}
    if pixel_size_um is not None:
        metadata["PhysicalSizeX"] = pixel_size_um
        metadata["PhysicalSizeY"] = pixel_size_um
    return metadata


def _write_tiled(
    path: Path,
    image_cyx: np.ndarray,
    names: list[str],
    pixel_size_um: float | None,
    tile: int = 256,
) -> Path:
    tifffile.imwrite(
        path,
        image_cyx,
        tile=(tile, tile),
        metadata=_metadata(names, pixel_size_um),
    )
    return path


def _write_strip(
    path: Path,
    image_cyx: np.ndarray,
    names: list[str],
    pixel_size_um: float | None,
) -> Path:
    tifffile.imwrite(
        path,
        image_cyx,
        rowsperstrip=image_cyx.shape[1],
        metadata=_metadata(names, pixel_size_um),
    )
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    ref = _write_tiled(tmp_path / "ref.ome.tif", REF, REF_NAMES, 0.5)
    mov = _write_strip(tmp_path / "mov.ome.tif", MOV, MOV_NAMES, 0.4)
    return ref, mov


def _config(tmp_path: Path, **overrides: Any) -> ManualRegistrationPrepareConfig:
    ref, mov = tmp_path / "ref.ome.tif", tmp_path / "mov.ome.tif"
    values: dict[str, Any] = {
        "dataset_name": "T1",
        "reference": {"path": ref},
        "moving": {"path": mov},
        "prepared_dir": tmp_path / "prepared",
        "output_dir": tmp_path / "out",
        "chunk_size": 512,
        "tile_size": 256,
        "band_height": 512,
        "min_level_size": 256,
        "workers": 1,
    }
    values.update(overrides)
    return ManualRegistrationPrepareConfig.model_validate(values)


def _labelling(store: Path, revision: int = 1, extra_pairs: int = 0) -> dict[str, Any]:
    attrs = read_root_attrs(store)[STORE_ATTR]
    pairs = [
        {
            "id": i + 1,
            "colour": "#ff0000",
            "enabled": True,
            "reference_xy_px": [100.0 + i, 200.0],
            "moving_xy_px": [110.0 + i, 210.0],
        }
        for i in range(extra_pairs)
    ]
    payload: dict[str, Any] = {
        "schema": "merxen.manual_registration.labelling",
        "schema_version": 1,
        "status": "finished",
        "revision": revision,
        "updated_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:00Z",
        "written_by": {"software": "test", "version": "0"},
        "store_grid_digest": attrs["grid_digest"],
        "images": {
            role: {
                "shape_yx": attrs["images"][role]["shape_yx"],
                "pixel_size_um": attrs["images"][role]["pixel_size_um"],
            }
            for role in ("reference", "moving")
        },
        "coarse": {
            "rotation_deg": 0.0,
            "mirror_x": False,
            "offset_um": [0.0, 0.0],
            "matrix_xy": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "crops": {"reference": None, "moving": None},
        "landmarks": {"next_id": extra_pairs + 1, "colour_seed": 0, "pairs": pairs},
    }
    payload["content_sha256"] = labelling_content_sha256(payload)
    return payload


def _inject(store: Path, payload: dict[str, Any]) -> None:
    with spatialdata_write_lock(store):
        write_root_attr(store, LABELLING_ATTR, payload)


def _root_json(store: Path) -> tuple[bytes, int]:
    path = store / "zarr.json"
    return path.read_bytes(), path.stat().st_mtime_ns


def _tree_mtimes(store: Path) -> dict[str, int]:
    return {
        str(p.relative_to(store)): p.stat().st_mtime_ns
        for p in sorted(store.rglob("*"))
        if p.is_file()
    }


def _store_attrs(store: Path) -> dict[str, Any]:
    return read_root_attrs(store)[STORE_ATTR]


def _leftovers(prepared: Path) -> list[str]:
    return sorted(
        p.name
        for p in prepared.iterdir()
        if p.name.startswith(".") and (".tmp-" in p.name or ".old-" in p.name)
    )


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_round_pixel_size_strips_float_noise() -> None:
    assert _round_pixel_size(3.4511e-07 * 1e6) == 0.34511


def test_resolve_pixel_size_cases() -> None:
    assert _resolve_pixel_size((0.5, 0.5), None, role="moving") == (
        0.5,
        "metadata",
        [],
    )
    assert _resolve_pixel_size((0.5, 0.5), 0.504, role="moving") == (
        0.504,
        "override",
        [],
    )
    value, source, warnings = _resolve_pixel_size((0.5, 0.5), 0.51, role="moving")
    assert (value, source) == (0.51, "override")
    assert len(warnings) == 1
    assert "1 %" in warnings[0]
    with pytest.raises(ValueError, match="moving_pixel_size_um"):
        _resolve_pixel_size(None, None, role="moving")
    with pytest.raises(ValueError, match="X and Y"):
        _resolve_pixel_size((0.5, 0.52), None, role="moving")
    assert _resolve_pixel_size((0.5, 0.52), 0.5, role="moving")[:2] == (
        0.5,
        "override",
    )


def test_normalisation_reasons() -> None:
    def reasons(
        shapes: list[tuple[int, int]],
        *,
        tiled: bool = True,
        metadata: float | None = 0.5,
        override: float | None = None,
    ) -> list[str]:
        return _normalisation_reasons(
            is_tiled=tiled,
            level_shapes_yx=shapes,
            metadata_pixel_size_um=metadata,
            override_pixel_size_um=override,
        )

    assert reasons([(1280, 1152)]) == []
    assert reasons([(1280, 1152)], tiled=False) == ["untiled"]
    assert reasons([(3000, 4100)]) == ["insufficient_pyramid"]
    assert reasons([(3000, 4100), (1500, 2050)]) == ["insufficient_pyramid"]
    assert reasons([(3000, 4100), (1500, 2050), (750, 1025)]) == []
    assert reasons([(6561, 6561), (2187, 2187), (729, 729)]) == ["insufficient_pyramid"]
    assert reasons([(1280, 1152)], metadata=None, override=0.5) == [
        "pixel_size_missing"
    ]
    assert reasons([(1280, 1152)], override=0.51) == ["pixel_size_override"]
    assert reasons([(1280, 1152)], override=0.504) == []
    assert reasons([(1280, 1152)], tiled=False, metadata=None, override=0.5) == [
        "untiled",
        "pixel_size_missing",
    ]


def _old_attrs(
    *,
    grid_digest: str = "sha256:" + "a" * 64,
    content_digest: str = "sha256:" + "b" * 64,
    labelling: dict[str, Any] | None = None,
    store_extra: dict[str, Any] | None = None,
    moving_shape: tuple[int, int] = (1100, 1300),
    moving_pixel: float = 0.4,
) -> dict[str, Any]:
    def record(
        shape: tuple[int, int], pixel: float, fingerprint: str
    ) -> dict[str, Any]:
        return {
            "shape_yx": list(shape),
            "pixel_size_um": pixel,
            "source": {"fingerprint": "sha256:" + fingerprint * 64},
            "czi": None,
        }

    store = {
        "schema": "merxen.manual_registration.store",
        "schema_version": 1,
        "kind": "prepared",
        "grid_digest": grid_digest,
        "content_digest": content_digest,
        "images": {
            "reference": record((1280, 1152), 0.5, "1"),
            "moving": record(moving_shape, moving_pixel, "2"),
        },
        **(store_extra or {}),
    }
    attrs: dict[str, Any] = {STORE_ATTR: store}
    if labelling is not None:
        attrs[LABELLING_ATTR] = labelling
    return attrs


def _new_entries(
    moving_shape: tuple[int, int] = (1100, 1300), moving_pixel: float = 0.4
) -> dict[str, dict[str, Any]]:
    return {
        "reference": {
            "source_fingerprint": "sha256:" + "1" * 64,
            "scene_index": None,
            "shape_yx": [1280, 1152],
            "pixel_size_um": 0.5,
        },
        "moving": {
            "source_fingerprint": "sha256:" + "2" * 64,
            "scene_index": None,
            "shape_yx": list(moving_shape),
            "pixel_size_um": moving_pixel,
        },
    }


def _plan(
    old_attrs: dict[str, Any],
    *,
    exists: bool = True,
    intact: bool = True,
    grid_digest: str = "sha256:" + "a" * 64,
    content_digest: str | None = "sha256:" + "b" * 64,
    force: bool = False,
    entries: dict[str, dict[str, Any]] | None = None,
) -> Any:
    return _plan_store_update(
        dataset_name="T1",
        store_path=Path("prepared/T1_prepared.zarr"),
        exists=exists,
        intact=intact,
        old_attrs=old_attrs,
        new_grid_entries=entries or _new_entries(),
        grid_digest=grid_digest,
        content_digest=content_digest,
        force=force,
    )


_LAB = {"status": "finished", "revision": 4, "schema_version": 1}


def test_plan_store_update_table() -> None:
    # 1. no store
    plan = _plan({}, exists=False)
    assert (plan.action, plan.status, plan.carry_labelling, plan.old_store) == (
        "build",
        "built",
        None,
        "none",
    )
    # 2. not a prepared store
    for attrs in ({}, {STORE_ATTR: "x"}, {STORE_ATTR: {"schema": "other"}}):
        plan = _plan(attrs)
        assert plan.action == "refuse"
        assert "is not a MerXen prepared store" in plan.message
        forced = _plan(attrs, force=True)
        assert (forced.action, forced.status, forced.old_store) == (
            "build",
            "forced",
            "supersede",
        )
        assert forced.carry_labelling is None
    result_kind = _old_attrs()
    result_kind[STORE_ATTR]["kind"] = "result"
    assert _plan(result_kind).action == "refuse"
    # 3. newer schema
    newer = _old_attrs(store_extra={"schema_version": 2})
    assert "newer MerXen" in _plan(newer).message
    assert _plan(newer, force=True).status == "forced"
    newer_lab = _old_attrs(labelling={**_LAB, "schema_version": 2})
    assert _plan(newer_lab).action == "refuse"
    assert _plan(newer_lab, force=True).old_store == "supersede"
    # 4. force: carry iff shapes and pixel sizes match
    labelled = _old_attrs(labelling=_LAB)
    plan = _plan(labelled, force=True)
    assert (plan.action, plan.status, plan.old_store) == (
        "build",
        "forced",
        "supersede",
    )
    assert plan.carry_labelling == _LAB
    assert plan.warnings == ()
    changed = _plan(
        labelled, force=True, entries=_new_entries(moving_pixel=0.45), grid_digest="x"
    )
    assert changed.carry_labelling is None
    assert "labelling not carried over" in changed.warnings[0]
    assert "moving" in changed.warnings[0]
    unlabelled = _plan(_old_attrs(), force=True, entries=_new_entries((1, 2)))
    assert unlabelled.carry_labelling is None
    assert unlabelled.warnings == ()
    # 5. unchanged
    plan = _plan(labelled)
    assert (plan.action, plan.status, plan.old_store) == (
        "unchanged",
        "unchanged",
        "none",
    )
    # a broken store is rebuilt, not skipped
    assert _plan(labelled, intact=False).action == "build"
    # the preflight (no content digest) never skips
    assert _plan(labelled, content_digest=None).action == "build"
    # 6. same grid, other content
    plan = _plan(labelled, content_digest="sha256:" + "c" * 64)
    assert (plan.status, plan.carry_labelling, plan.old_store) == (
        "rebuilt_kept_labelling",
        _LAB,
        "delete",
    )
    plan = _plan(_old_attrs(), content_digest="sha256:" + "c" * 64)
    assert (plan.status, plan.carry_labelling, plan.old_store) == (
        "built",
        None,
        "delete",
    )
    # 7. grid differs under a labelling
    other = "sha256:" + "d" * 64
    changed_content = "sha256:" + "c" * 64
    plan = _plan(
        labelled,
        grid_digest=other,
        content_digest=changed_content,
        entries=_new_entries(moving_shape=(1000, 1300)),
    )
    assert plan.action == "refuse"
    assert "moving.shape_yx" in plan.message
    assert "--manual_registration_force_prepare true" in plan.message
    assert "--force" in plan.message
    assert "status finished, revision 4" in plan.message
    forced = _plan(
        labelled,
        grid_digest=other,
        entries=_new_entries(moving_shape=(1000, 1300)),
        force=True,
    )
    assert forced.status == "forced"
    assert forced.carry_labelling is None
    # 8. grid differs, no labelling
    plan = _plan(_old_attrs(), grid_digest=other, content_digest=changed_content)
    assert (plan.action, plan.status, plan.carry_labelling, plan.old_store) == (
        "build",
        "built",
        None,
        "delete",
    )


def test_single_strip_band_reader_decodes_each_plane_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_strip(tmp_path / "strip.ome.tif", MOV, MOV_NAMES, 0.4)
    info = read_ome_tiff_info(path)
    real = prepare_module.open_ome_tiff_level
    computed: list[tuple[int, ...]] = []

    def counting(block: np.ndarray, block_info: Any = None) -> np.ndarray:
        computed.append(tuple(block_info[None]["chunk-location"]))
        return block

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        level = real(*args, **kwargs)
        return level.copy(data=da.map_blocks(counting, level.data, dtype=level.dtype))

    monkeypatch.setattr(prepare_module, "open_ome_tiff_level", wrapper)
    read_band = _ome_band_reader(info, single_strip=True)

    for c, y0, h in ((0, 0, 512), (0, 512, 512), (0, 1024, 76), (1, 0, 512)):
        np.testing.assert_array_equal(read_band(c, y0, h), MOV[c, y0 : y0 + h])

    assert len(computed) == len(set(computed)), "a plane chunk was decoded twice"
    assert {location[0] for location in computed} == {0, 1}

    computed.clear()
    plain = _ome_band_reader(info, single_strip=False)
    for c, y0, h in ((0, 0, 512), (1, 1024, 76)):
        np.testing.assert_array_equal(plain(c, y0, h), MOV[c, y0 : y0 + h])


@pytest.mark.parametrize(
    "case",
    [
        "scene_on_tiff",
        "png",
        "wrong_override_length",
        "registration_channel",
        "z_stack",
        "same_file",
        "no_pixel_size",
    ],
)
def test_input_errors(tmp_path: Path, case: str) -> None:
    _inputs(tmp_path)
    overrides: dict[str, Any] = {}
    match = ""
    if case == "scene_on_tiff":
        overrides["moving"] = {"path": tmp_path / "mov.ome.tif", "scene": "1"}
        match = r"\[T1\] moving image mov.ome.tif: scene is only valid for .czi inputs"
    elif case == "png":
        (tmp_path / "x.png").write_bytes(b"not an image")
        overrides["moving"] = {"path": tmp_path / "x.png"}
        match = "unsupported image file type 'x.png'"
    elif case == "wrong_override_length":
        overrides["reference"] = {
            "path": tmp_path / "ref.ome.tif",
            "channel_names": ["A", "B"],
        }
        match = r"\[T1\] reference image ref.ome.tif: channel_names has 2 entries"
    elif case == "registration_channel":
        overrides["reference"] = {
            "path": tmp_path / "ref.ome.tif",
            "registration_channel": "GFP",
        }
        match = r"'GFP'.*\['DAPI', 'AF568', 'AF647'\]"
    elif case == "z_stack":
        stack = np.zeros((3, 64, 64), dtype=np.uint16)
        tifffile.imwrite(tmp_path / "z.ome.tif", stack, metadata={"axes": "ZYX"})
        overrides["moving"] = {"path": tmp_path / "z.ome.tif"}
        match = r"\[T1\] moving image z.ome.tif: .*non-singleton axes"
    elif case == "same_file":
        overrides["moving"] = {"path": tmp_path / "ref.ome.tif"}
        match = "reference and moving are the same file"
    elif case == "no_pixel_size":
        _write_strip(tmp_path / "nopx.ome.tif", MOV, MOV_NAMES, None)
        overrides["moving"] = {"path": tmp_path / "nopx.ome.tif"}
        match = "moving_pixel_size_um"
    with pytest.raises(ValueError, match=match):
        run_manual_registration_prepare(_config(tmp_path, **overrides))
    assert not (tmp_path / "prepared" / "T1_prepared.zarr").exists()


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_prepare_builds_store_layout_and_attrs(tmp_path: Path) -> None:
    ref_path, _ = _inputs(tmp_path)
    outputs = run_manual_registration_prepare(_config(tmp_path))
    prepared = tmp_path / "prepared"
    store = prepared / "T1_prepared.zarr"
    assert outputs["store"] == store
    summary = json.loads(outputs["summary"].read_text())
    assert summary["status"] == "built"

    sdata = read_zarr(store)
    assert set(sdata.images) == {"reference", "moving"}
    assert not (sdata.tables or sdata.labels or sdata.points or sdata.shapes)
    reference, moving = sdata.images["reference"], sdata.images["moving"]
    assert list(reference.coords["c"].values) == ["DAPI", "AF568"]
    assert reference.dtype == np.uint16
    assert reference.shape == (2, 1280, 1152)
    np.testing.assert_array_equal(reference.values, REF[[0, 1]])
    assert zarr.open_array(store / "images/reference/s0").chunks == (1, 512, 512)
    matrix = get_transformation(reference, "global").to_affine_matrix(
        ("x", "y"), ("x", "y")
    )
    np.testing.assert_allclose(matrix, np.diag([0.5, 0.5, 1.0]))
    assert list(moving.coords["c"].values) == ["DAPI", "AF568"]
    assert moving.shape == (2, 1100, 1300)
    np.testing.assert_array_equal(moving.values, MOV)
    matrix = get_transformation(moving, "global").to_affine_matrix(
        ("x", "y"), ("x", "y")
    )
    np.testing.assert_allclose(matrix, np.diag([0.4, 0.4, 1.0]))

    attrs = _store_attrs(store)
    StoreAttrs.model_validate(attrs)
    assert attrs["kind"] == "prepared"
    assert attrs["dataset_name"] == "T1"
    assert attrs["conventions"] == CONVENTIONS
    for key in ("grid_digest", "content_digest"):
        assert attrs[key].startswith("sha256:") and len(attrs[key]) == 71
    ref_rec, mov_rec = attrs["images"]["reference"], attrs["images"]["moving"]
    assert ref_rec["ome_tiff"]["path"] == str(ref_path)
    assert ref_rec["ome_tiff"]["written_by_prepare"] is False
    assert ref_rec["ome_tiff"]["exact_level_factors"] is False
    assert ref_rec["source"]["format"] == "ome_tiff"
    assert ref_rec["czi"] is None
    assert ref_rec["channel_names"] == ["DAPI", "AF568"]
    assert ref_rec["canonical_channel_names"] == ["DAPI", "AF568", "AF647"]
    assert ref_rec["pixel_size_source"] == "metadata"
    assert mov_rec["ome_tiff"]["path"].endswith("prepared/T1_moving.ome.tif")
    assert mov_rec["ome_tiff"]["written_by_prepare"] is True
    assert mov_rec["ome_tiff"]["exact_level_factors"] is True
    assert mov_rec["ome_tiff"]["level_shapes_yx"] == [
        [1100, 1300],
        [550, 650],
        [275, 325],
        [138, 163],
    ]
    assert mov_rec["source_channel_names"] == ["Hoechst 33342", "AF568"]
    assert mov_rec["canonical_channel_names"] == ["DAPI", "AF568"]
    assert mov_rec["registration_channel"] == "DAPI"
    assert attrs["channel_matching"] == {
        "reference_kept": ["DAPI", "AF568"],
        "reference_dropped": ["AF647"],
        "moving_unmatched": [],
    }
    validate_merxen_schema(read_zarr(store))

    assert (prepared / "T1_moving.prepare.json").exists()
    assert not (prepared / "T1_reference.ome.tif").exists()
    assert not (prepared / "T1_reference.prepare.json").exists()
    assert str(tmp_path) not in (prepared / "T1_moving.prepare.json").read_text()
    durable = prepared / "T1_prepare_summary.json"
    assert (
        durable.read_text() == (tmp_path / "out" / "prepare_summary.json").read_text()
    )
    assert _leftovers(prepared) == []
    assert not [p for p in prepared.iterdir() if p.name.startswith(".tmp-")]


@pytest.mark.integration
def test_grid_digest_from_probes_equals_store_digest(tmp_path: Path) -> None:
    _inputs(tmp_path)
    config = _config(tmp_path)
    run_manual_registration_prepare(config)
    entries = {
        role: prepare_module._grid_entry(
            source_fingerprint=probe.source_fingerprint,
            scene_index=probe.scene_index,
            shape_yx=probe.shape_cyx[1:],
            pixel_size_um=probe.pixel_size_um,
        )
        for role, probe in (
            (
                role,
                prepare_module._inspect_image(
                    getattr(config, role), role=role, config=config
                ),
            )
            for role in ("reference", "moving")
        )
    }
    assert (
        prepare_module._grid_digest(entries)
        == _store_attrs(config.store_path)["grid_digest"]
    )


@pytest.mark.integration
def test_rerun_is_a_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _inputs(tmp_path)
    config = _config(tmp_path)
    run_manual_registration_prepare(config)
    store = config.store_path
    prepared = config.prepared_dir
    root_before = _root_json(store)
    tree_before = _tree_mtimes(store)
    ome = prepared / "T1_moving.ome.tif"
    prov = prepared / "T1_moving.prepare.json"
    mtimes = (ome.stat().st_mtime_ns, prov.stat().st_mtime_ns)

    def must_not_convert(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the conversion was not skipped")

    monkeypatch.setattr(prepare_module, "write_pyramidal_ome_tiff", must_not_convert)
    outputs = run_manual_registration_prepare(config)

    assert json.loads(outputs["summary"].read_text())["status"] == "unchanged"
    assert _root_json(store) == root_before
    assert _tree_mtimes(store) == tree_before
    assert (ome.stat().st_mtime_ns, prov.stat().st_mtime_ns) == mtimes


@pytest.mark.integration
def test_content_change_rebuilds_and_keeps_labelling(tmp_path: Path) -> None:
    _inputs(tmp_path)
    run_manual_registration_prepare(_config(tmp_path))
    store = tmp_path / "prepared" / "T1_prepared.zarr"
    first = _store_attrs(store)
    labelling = _labelling(store, extra_pairs=2)
    _inject(store, labelling)

    outputs = run_manual_registration_prepare(_config(tmp_path, chunk_size=256))

    assert json.loads(outputs["summary"].read_text())["status"] == (
        "rebuilt_kept_labelling"
    )
    attrs = read_root_attrs(store)
    assert canonical_json(attrs[LABELLING_ATTR]) == canonical_json(labelling)
    assert zarr.open_array(store / "images/reference/s0").chunks == (1, 256, 256)
    assert attrs[STORE_ATTR]["grid_digest"] == first["grid_digest"]
    assert attrs[STORE_ATTR]["content_digest"] != first["content_digest"]
    assert _leftovers(tmp_path / "prepared") == []

    overridden = _config(
        tmp_path,
        chunk_size=256,
        moving={"path": tmp_path / "mov.ome.tif", "channel_names": ["DAPI", "AF568b"]},
    )
    outputs = run_manual_registration_prepare(overridden)
    assert json.loads(outputs["summary"].read_text())["status"] == (
        "rebuilt_kept_labelling"
    )
    matching = _store_attrs(store)["channel_matching"]
    assert matching["moving_unmatched"] == ["AF568b"]
    assert matching["reference_kept"] == ["DAPI"]


@pytest.mark.integration
def test_grid_change_with_labelling_is_refused_before_conversion(
    tmp_path: Path,
) -> None:
    _, mov = _inputs(tmp_path)
    run_manual_registration_prepare(_config(tmp_path))
    store = tmp_path / "prepared" / "T1_prepared.zarr"
    _inject(store, _labelling(store))
    ome = tmp_path / "prepared" / "T1_moving.ome.tif"
    root_before = _root_json(store)
    ome_before = (ome.read_bytes(), ome.stat().st_mtime_ns)
    other = np.random.default_rng(3).integers(0, 4000, MOV.shape, dtype=np.uint16)
    _write_strip(mov, other, MOV_NAMES, 0.4)

    with pytest.raises(ValueError, match="moving.source_fingerprint") as excinfo:
        run_manual_registration_prepare(_config(tmp_path))

    assert "--manual_registration_force_prepare true" in str(excinfo.value)
    assert _root_json(store) == root_before
    assert (ome.read_bytes(), ome.stat().st_mtime_ns) == ome_before
    assert _leftovers(tmp_path / "prepared") == []


@pytest.mark.integration
def test_force_supersedes_and_carries_labelling_when_shapes_match(
    tmp_path: Path,
) -> None:
    _, mov = _inputs(tmp_path)
    run_manual_registration_prepare(_config(tmp_path))
    prepared = tmp_path / "prepared"
    store = prepared / "T1_prepared.zarr"
    labelling = _labelling(store)
    _inject(store, labelling)
    other = np.random.default_rng(3).integers(0, 4000, MOV.shape, dtype=np.uint16)
    _write_strip(mov, other, MOV_NAMES, 0.4)

    outputs = run_manual_registration_prepare(_config(tmp_path, force=True))

    summary = json.loads(outputs["summary"].read_text())
    assert summary["status"] == "forced"
    superseded = list(prepared.glob("T1_prepared.zarr.superseded-*"))
    assert len(superseded) == 1
    assert summary["superseded_store"] == str(superseded[0])
    assert canonical_json(
        read_root_attrs(superseded[0])[LABELLING_ATTR]
    ) == canonical_json(labelling)
    assert canonical_json(read_root_attrs(store)[LABELLING_ATTR]) == canonical_json(
        labelling
    )
    assert _leftovers(prepared) == []


@pytest.mark.integration
def test_force_does_not_carry_labelling_when_pixel_size_changes(
    tmp_path: Path,
) -> None:
    _inputs(tmp_path)
    run_manual_registration_prepare(_config(tmp_path))
    prepared = tmp_path / "prepared"
    store = prepared / "T1_prepared.zarr"
    _inject(store, _labelling(store))
    moving = {"path": tmp_path / "mov.ome.tif", "pixel_size_um": 0.45}

    with pytest.raises(ValueError, match="moving.pixel_size_um"):
        run_manual_registration_prepare(_config(tmp_path, moving=moving))

    outputs = run_manual_registration_prepare(
        _config(tmp_path, moving=moving, force=True)
    )

    summary = json.loads(outputs["summary"].read_text())
    assert summary["status"] == "forced"
    assert LABELLING_ATTR not in read_root_attrs(store)
    assert any("labelling not carried over" in w for w in summary["warnings"])
    (superseded,) = prepared.glob("T1_prepared.zarr.superseded-*")
    assert LABELLING_ATTR in read_root_attrs(superseded)
    assert str(superseded) in " ".join(summary["warnings"])


@pytest.mark.integration
def test_existing_non_prepared_directory_is_refused(tmp_path: Path) -> None:
    _inputs(tmp_path)
    prepared = tmp_path / "prepared"
    store = prepared / "T1_prepared.zarr"
    store.mkdir(parents=True)
    (store / "dummy.txt").write_text("keep me")

    with pytest.raises(ValueError, match="is not a MerXen prepared store"):
        run_manual_registration_prepare(_config(tmp_path))
    assert (store / "dummy.txt").exists()

    outputs = run_manual_registration_prepare(_config(tmp_path, force=True))

    assert json.loads(outputs["summary"].read_text())["status"] == "forced"
    (kept,) = prepared.glob("T1_prepared.zarr.superseded-*")
    assert (kept / "dummy.txt").read_text() == "keep me"
    assert (store / "images").is_dir()


@pytest.mark.integration
def test_newer_store_schema_is_refused(tmp_path: Path) -> None:
    _inputs(tmp_path)
    config = _config(tmp_path)
    run_manual_registration_prepare(config)
    store = config.store_path
    newer = {**_store_attrs(store), "schema_version": 2}
    with spatialdata_write_lock(store):
        write_root_attr(store, STORE_ATTR, newer)

    with pytest.raises(ValueError, match="newer MerXen"):
        run_manual_registration_prepare(config)


@pytest.mark.integration
def test_viewer_caches_on_prepared_store_keep_labelling(tmp_path: Path) -> None:
    _inputs(tmp_path)
    config = _config(tmp_path)
    run_manual_registration_prepare(config)
    store = config.store_path

    result = build_image_pyramid_caches(store, params=ViewerCacheParams(min_size=256))

    assert result == {"image_pyramids": {"moving": "built", "reference": "built"}}
    labelling = _labelling(store)
    _inject(store, labelling)
    outputs = run_manual_registration_prepare(config)
    assert json.loads(outputs["summary"].read_text())["status"] == "unchanged"
    assert canonical_json(read_root_attrs(store)[LABELLING_ATTR]) == canonical_json(
        labelling
    )
    keys = {str(k) for k in read_zarr(store).images}
    assert "_napari_compare_imgpyr__reference__ds4" in keys
    assert "_napari_compare_imgpyr__moving__ds4" in keys


@pytest.mark.integration
def test_labelling_saved_during_rebuild_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inputs(tmp_path)
    config = _config(tmp_path)
    run_manual_registration_prepare(config)
    store = config.store_path
    first = _labelling(store, revision=1, extra_pairs=1)
    second = _labelling(store, revision=2, extra_pairs=2)
    _inject(store, first)
    real_write = prepare_module.SpatialData.write

    def write_then_save(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = real_write(self, *args, **kwargs)
        _inject(store, second)  # a viewer save between the plan and the swap
        return result

    monkeypatch.setattr(prepare_module.SpatialData, "write", write_then_save)
    outputs = run_manual_registration_prepare(_config(tmp_path, chunk_size=256))

    assert json.loads(outputs["summary"].read_text())["status"] == (
        "rebuilt_kept_labelling"
    )
    assert canonical_json(read_root_attrs(store)[LABELLING_ATTR]) == canonical_json(
        second
    )
    assert zarr.open_array(store / "images/reference/s0").chunks == (1, 256, 256)
    assert _leftovers(config.prepared_dir) == []


# --------------------------------------------------------------------------
# CZI input
# --------------------------------------------------------------------------


@pytest.fixture(params=["in_process", "runtime"])
def czi_python(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    real_path: Callable[[str], Path],
) -> str | None:
    """Backend of the CZI test: czifile in this process, or the CZI runtime."""
    if request.param == "in_process":
        if not czi.czifile_available():
            pytest.skip("czifile is not importable in this environment")
        return None
    value = os.environ.get("MERXEN_CZI_PYTHON")
    interpreter = value if value else str(real_path("MERXEN_CZI_PYTHON"))
    monkeypatch.setenv("MERXEN_CZI_PYTHON", interpreter)
    if czi.czifile_available():
        # Force the bridge even where czifile is importable.
        monkeypatch.setattr(czi, "czifile_available", lambda: False)
    return interpreter


@pytest.mark.integration
def test_czi_input(
    tmp_path: Path, czi_python: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_tiled(tmp_path / "ref.ome.tif", REF, REF_NAMES, 0.5)
    oracle = np.load(FIXTURES / "oracle.npz")
    expected = oracle["one_scene_hole__s0"]
    bbox = [int(v) for v in oracle["one_scene_hole__bbox"][0]]
    config = _config(
        tmp_path,
        moving={"path": FIXTURES / "one_scene_hole.czi"},
        tile_size=16,
        band_height=16,
        min_level_size=16,
        chunk_size=64,
    )

    outputs = run_manual_registration_prepare(config)

    store = outputs["store"]
    record = _store_attrs(store)["images"]["moving"]
    assert record["source"]["format"] == "czi"
    assert record["ome_tiff"]["written_by_prepare"] is True
    assert set(record["czi"]) == {
        "scene_index",
        "scene_name",
        "bbox_xywh",
        "z_planes_projected",
        "pixel_type",
        "compression",
    }
    assert record["czi"]["scene_index"] == 0
    assert record["czi"]["bbox_xywh"] == bbox
    assert record["pixel_size_um"] == 0.34511
    moving = read_zarr(store).images["moving"]
    np.testing.assert_array_equal(moving.values, expected)
    provenance = json.loads(
        (config.prepared_dir / "T1_moving.prepare.json").read_text()
    )
    assert provenance["conversion"]["converter"] == "czifile==2026.10.14"
    assert str(tmp_path) not in json.dumps(provenance)

    def must_not_convert(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the conversion was not skipped")

    monkeypatch.setattr(prepare_module, "run_convert_czi_scene", must_not_convert)
    rerun = run_manual_registration_prepare(config)
    assert json.loads(rerun["summary"].read_text())["status"] == "unchanged"


def test_czi_input_without_a_runtime_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_tiled(tmp_path / "ref.ome.tif", REF, REF_NAMES, 0.5)
    monkeypatch.setattr(czi, "czifile_available", lambda: False)
    monkeypatch.delenv("MERXEN_CZI_PYTHON", raising=False)
    config = _config(tmp_path, moving={"path": FIXTURES / "one_scene_hole.czi"})

    with pytest.raises(ValueError, match=r"\[T1\] moving image .*MERXEN_CZI_PYTHON"):
        run_manual_registration_prepare(config)


# --------------------------------------------------------------------------
# Direct API use
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_prepare_image_and_store_api(tmp_path: Path) -> None:
    _inputs(tmp_path)
    config = _config(tmp_path)
    spec_ref = ManualRegistrationImageInput(path=tmp_path / "ref.ome.tif")
    spec_mov = ManualRegistrationImageInput(path=tmp_path / "mov.ome.tif")

    reference = prepare_image(spec_ref, role="reference", config=config)
    moving = prepare_image(spec_mov, role="moving", config=config)
    details: dict[str, Any] = {}
    store, status = build_prepared_store(
        config=config, reference=reference, moving=moving, details_out=details
    )

    assert status == "built"
    assert any(
        "rewritten as a tiled pyramidal OME-TIFF (untiled" in w for w in moving.warnings
    )
    assert details["superseded_store"] is None
    assert details["grid_digest"] == _store_attrs(store)["grid_digest"]
    with pytest.raises(ValueError, match="role must be"):
        prepare_image(spec_ref, role="fixed", config=config)
