"""Tests for the manual-registration store-attrs schema and its helpers."""

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import re
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from pydantic import ValidationError

from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.manual_registration import schema
from merxen.manual_registration.schema import (
    CoarseTransform,
    CropRect,
    LandmarkPair,
    UnsupportedSchemaVersionError,
    canonical_json,
    canonical_sha256,
    file_fingerprint,
    labelling_content_sha256,
    pair_colour,
    parse_labelling_attrs,
    parse_store_attrs,
    read_root_attrs,
    read_root_attrs_locked,
    utc_now,
    write_root_attr,
)

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "test_data"
    / "manual_registration"
    / "golden_v1.json"
)
LABELLING_DIGEST = (
    "sha256:f6b16674a10854ab06ff385a6a349ce325d874abf0e6c9d78e0a3b7683e328ad"
)
_STARTUP_TIMEOUT_SECONDS = 120.0
_BLOCKED_WINDOW_SECONDS = 1.0


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data


@pytest.fixture
def labelling(golden: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(golden["examples"]["labelling"])  # type: ignore[no-any-return]


@pytest.fixture
def prepared(golden: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(golden["examples"]["store_attr_prepared"])  # type: ignore[no-any-return]


@pytest.fixture
def result(golden: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(golden["examples"]["store_attr_result"])  # type: ignore[no-any-return]


def test_constants_equal_the_contract_literals() -> None:
    assert schema.SCHEMA_VERSION == 1
    assert schema.STORE_ATTR == "merxen_manual_registration_store"
    assert schema.LABELLING_ATTR == "merxen_manual_registration"
    assert schema.RESULT_ATTR == "merxen_manual_registration_result"
    assert schema.STORE_SCHEMA == "merxen.manual_registration.store"
    assert schema.LABELLING_SCHEMA == "merxen.manual_registration.labelling"
    assert schema.RESULT_SCHEMA == "merxen.manual_registration.result"
    assert schema.LAYER_METADATA_KEY == "merxen_manual_registration"
    assert schema.MIN_LANDMARK_PAIRS == 8
    assert schema.DEFAULT_OUTLIER_FLOOR_UM == 10.0
    assert schema.OUTLIER_K == 3.0
    assert schema.MAX_OUTLIER_FRACTION == 0.25
    assert schema.MIN_USED_PAIRS == 6
    assert schema.CROP_SNAP_PX == 64
    assert schema.CROP_MIN_SIZE_PX == 1024
    assert schema.DATASET_ID_PATTERN == r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
    assert schema.DIGEST_PATTERN == r"^sha256:[0-9a-f]{64}$"
    assert schema.COLOUR_PATTERN == r"^#[0-9a-f]{6}$"
    assert schema.CONVENTIONS == {
        "pixel": "level0_xy_integer_is_pixel_centre",
        "matrix": "moving_level0_px_xy_to_reference_level0_px_xy",
        "rotation": "positive_ccw_on_screen_y_down",
        "crop": "half_open_level0_px",
    }


def test_crop_rect_validation_and_accessors() -> None:
    crop = CropRect(x0=10, y0=20, x1=110, y1=70)
    assert (crop.width(), crop.height()) == (100, 50)
    assert crop.as_tuple() == (10, 20, 110, 70)
    for bad in (
        {"x0": 5, "y0": 0, "x1": 5, "y1": 10},
        {"x0": 0, "y0": 9, "x1": 10, "y1": 3},
        {"x0": -1, "y0": 0, "x1": 10, "y1": 10},
        {"x0": 0, "y0": 0, "x1": 10, "y1": -4},
    ):
        with pytest.raises(ValidationError):
            CropRect(**bad)
    CropRect(x0=0, y0=0, x1=200, y1=100).validate_within((100, 200))
    with pytest.raises(ValueError, match="exceeds the image"):
        CropRect(x0=0, y0=0, x1=201, y1=10).validate_within((100, 200))
    with pytest.raises(ValidationError):
        CropRect.model_validate({"x0": 0, "y0": 0, "x1": 5, "y1": 5, "extra": 1})


def test_coarse_transform_wraps_and_validates() -> None:
    assert CoarseTransform(rotation_deg=190).rotation_deg == pytest.approx(-170)
    assert CoarseTransform(rotation_deg=-180).rotation_deg == 180
    with pytest.raises(ValidationError):
        CoarseTransform(rotation_deg=float("nan"))
    with pytest.raises(ValidationError):
        CoarseTransform(offset_um=(float("inf"), 0.0))
    ignored = CoarseTransform.model_validate({"rotation_deg": 5, "unknown": 1})
    assert ignored.rotation_deg == 5


def test_landmark_pair_validation() -> None:
    valid = {
        "id": 1,
        "colour": "#fff400",
        "reference_xy_px": [1.0, 2.0],
        "moving_xy_px": [3.0, 4.0],
    }
    assert LandmarkPair.model_validate(valid).enabled is True
    for bad in (
        {**valid, "colour": "#FFF400"},
        {**valid, "id": 0},
        {**valid, "moving_xy_px": [float("nan"), 1.0]},
    ):
        with pytest.raises(ValidationError):
            LandmarkPair.model_validate(bad)


def test_parse_labelling_attrs_accepts_the_example(labelling: dict[str, Any]) -> None:
    parsed = parse_labelling_attrs(labelling)
    dumped = parsed.model_dump(mode="json", by_alias=True)
    assert dumped["schema"] == "merxen.manual_registration.labelling"
    assert [pair.id for pair in parsed.enabled_pairs()] == [1]
    assert parsed.images.reference.shape_yx == (55012, 76544)


def test_parse_labelling_attrs_ignores_unknown_top_level_keys(
    labelling: dict[str, Any],
) -> None:
    labelling["future_field"] = {"a": 1}
    assert parse_labelling_attrs(labelling).revision == 9


def test_parse_labelling_attrs_rejects_other_versions(
    labelling: dict[str, Any],
) -> None:
    labelling["schema_version"] = 2
    with pytest.raises(UnsupportedSchemaVersionError, match="newer"):
        parse_labelling_attrs(labelling)
    labelling["schema_version"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        parse_labelling_attrs(labelling)
    labelling["schema_version"] = True
    with pytest.raises(ValueError, match="positive integer"):
        parse_labelling_attrs(labelling)


def test_parse_labelling_attrs_rejects_inconsistent_payloads(
    labelling: dict[str, Any],
) -> None:
    duplicate = copy.deepcopy(labelling)
    duplicate["landmarks"]["pairs"][1]["id"] = 1
    with pytest.raises(ValidationError, match="duplicate landmark ids"):
        parse_labelling_attrs(duplicate)

    low_next = copy.deepcopy(labelling)
    low_next["landmarks"]["next_id"] = 2
    with pytest.raises(ValidationError, match="next_id must exceed"):
        parse_labelling_attrs(low_next)

    status = copy.deepcopy(labelling)
    status["status"] = "done"
    with pytest.raises(ValidationError):
        parse_labelling_attrs(status)

    missing = copy.deepcopy(labelling)
    del missing["images"]["moving"]
    with pytest.raises(ValidationError):
        parse_labelling_attrs(missing)


def test_a_draft_may_have_no_pairs(labelling: dict[str, Any]) -> None:
    labelling["status"] = "draft"
    labelling["landmarks"]["pairs"] = []
    assert parse_labelling_attrs(labelling).enabled_pairs() == []


def test_parse_store_attrs_accepts_both_examples(
    prepared: dict[str, Any], result: dict[str, Any]
) -> None:
    parsed = parse_store_attrs(prepared)
    assert parsed.kind == "prepared"
    assert parsed.images["moving"].czi is not None
    assert parsed.model_dump(mode="json", by_alias=True)["schema"] == (
        "merxen.manual_registration.store"
    )
    assert parse_store_attrs(result).prepared_store is not None


def test_parse_store_attrs_rejects_inconsistent_payloads(
    prepared: dict[str, Any], result: dict[str, Any]
) -> None:
    other_kind = copy.deepcopy(prepared)
    other_kind["kind"] = "other"
    with pytest.raises(ValidationError):
        parse_store_attrs(other_kind)

    no_grid = copy.deepcopy(prepared)
    del no_grid["grid_digest"]
    with pytest.raises(ValidationError, match="prepared store needs"):
        parse_store_attrs(no_grid)

    no_run = copy.deepcopy(result)
    del no_run["run_id"]
    with pytest.raises(ValidationError, match="result store needs"):
        parse_store_attrs(no_run)

    no_moving = copy.deepcopy(prepared)
    del no_moving["images"]["moving"]
    with pytest.raises(ValidationError, match="exactly 'reference' and 'moving'"):
        parse_store_attrs(no_moving)

    wrong_role = copy.deepcopy(prepared)
    wrong_role["images"]["reference"]["role"] = "moving"
    with pytest.raises(ValidationError, match="role"):
        parse_store_attrs(wrong_role)

    wrong_channel = copy.deepcopy(prepared)
    wrong_channel["images"]["reference"]["registration_channel"] = "GFP"
    with pytest.raises(ValidationError, match="registration_channel"):
        parse_store_attrs(wrong_channel)

    newer = copy.deepcopy(prepared)
    newer["schema_version"] = 7
    with pytest.raises(UnsupportedSchemaVersionError):
        parse_store_attrs(newer)


def test_czi_metadata_may_lack_pixel_type_and_compression(
    prepared: dict[str, Any],
) -> None:
    nulled = copy.deepcopy(prepared)
    nulled["images"]["reference"]["czi"]["pixel_type"] = None
    nulled["images"]["reference"]["czi"]["compression"] = None
    removed = copy.deepcopy(prepared)
    del removed["images"]["reference"]["czi"]["pixel_type"]
    del removed["images"]["reference"]["czi"]["compression"]
    for payload in (nulled, removed):
        czi = parse_store_attrs(payload).images["reference"].czi
        assert czi is not None
        dumped = czi.model_dump(mode="json")
        assert dumped["pixel_type"] is None
        assert dumped["compression"] is None


def test_canonical_json_reference_values() -> None:
    obj = {"b": 1, "a": [1.5, None, True], "c": "é"}
    assert canonical_json(obj) == '{"a":[1.5,null,true],"b":1,"c":"\\u00e9"}'
    assert canonical_sha256(obj) == (
        "sha256:5474a01554484b7b914824bd129578c9ad3b12149779e565293fc0a01b137a53"
    )
    reordered = {"c": "é", "a": [1.5, None, True], "b": 1}
    assert canonical_sha256(reordered) == canonical_sha256(obj)
    assert canonical_sha256(json.loads(canonical_json(obj))) == canonical_sha256(obj)
    with pytest.raises(ValueError):
        canonical_json(float("nan"))


def test_labelling_content_sha256_reference_and_sensitivity(
    labelling: dict[str, Any],
) -> None:
    assert labelling_content_sha256(labelling) == LABELLING_DIGEST

    def digest_after(mutate: Any) -> str:
        payload = copy.deepcopy(labelling)
        mutate(payload)
        return labelling_content_sha256(payload)

    unchanged = [
        lambda p: p.__setitem__("revision", 99),
        lambda p: p.__setitem__("updated_at", "2030-01-01T00:00:00Z"),
        lambda p: p["coarse"].__setitem__("matrix_xy", None),
        lambda p: p.__setitem__("status", "draft"),
    ]
    for mutate in unchanged:
        assert digest_after(mutate) == LABELLING_DIGEST

    changed = [
        lambda p: p["landmarks"]["pairs"][0].__setitem__("reference_xy_px", [1.0, 2.0]),
        lambda p: p["crops"]["reference"].__setitem__("x0", 1001),
        lambda p: p["landmarks"]["pairs"][1].__setitem__("enabled", True),
        lambda p: p["landmarks"]["pairs"][0].__setitem__("colour", "#000000"),
        lambda p: p["coarse"].__setitem__("offset_um", [152.5, -88.0]),
    ]
    for mutate in changed:
        assert digest_after(mutate) != LABELLING_DIGEST

    broken = copy.deepcopy(labelling)
    del broken["landmarks"]
    with pytest.raises(ValueError, match="lacks"):
        labelling_content_sha256(broken)


def test_file_fingerprint_reference_values(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(bytes(range(256)) * 4096)
    assert file_fingerprint(path) == (
        "sha256:1a8a4c0bfd349a939bacc8fbc5712fbfc86047d0eec5c9b069a9bcc77821e630"
    )
    assert file_fingerprint(path, chunk_bytes=1000) == (
        "sha256:937904f97e066f976311b935adca8e5538ad041854fc9f6cfa8e5bc7016128a0"
    )
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert file_fingerprint(empty) == (
        "sha256:c375221ef40f4c8f08492605a3d0b0757b08cb252cd0a324bdd5554bc1d90a04"
    )
    with pytest.raises(ValueError):
        file_fingerprint(path, chunk_bytes=0)


def test_file_fingerprint_ignores_path_and_mtime_but_sees_content(
    tmp_path: Path,
) -> None:
    original = tmp_path / "a.bin"
    original.write_bytes(b"abc" * 1000)
    copy_path = tmp_path / "sub" / "b.bin"
    copy_path.parent.mkdir()
    copy_path.write_bytes(original.read_bytes())
    os.utime(copy_path, (1_000_000_000, 1_000_000_000))
    assert file_fingerprint(copy_path) == file_fingerprint(original)
    copy_path.write_bytes(original.read_bytes() + b"x")
    assert file_fingerprint(copy_path) != file_fingerprint(original)


def test_pair_colour_reference_values() -> None:
    assert [pair_colour(i, 4211) for i in range(1, 13)] == [
        "#fff400",
        "#b219ff",
        "#40ff87",
        "#ff1400",
        "#194aff",
        "#a0ff40",
        "#ff00cb",
        "#19ffeb",
        "#ffb640",
        "#5400ff",
        "#19ff22",
        "#ff4070",
    ]
    assert [pair_colour(i, 0) for i in range(1, 13)] == [
        "#004aff",
        "#9fff19",
        "#ff40e7",
        "#00ffd4",
        "#ff9619",
        "#6f40ff",
        "#0bff00",
        "#ff1966",
        "#40b8ff",
        "#eaff00",
        "#cf19ff",
        "#40ff9f",
    ]


def test_utc_now_format() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_now())


# ---------------------------------------------------------------------------
# root attrs


def test_read_root_attrs_zarr_v3(tmp_path: Path) -> None:
    store = tmp_path / "v3.zarr"
    zarr.open_group(str(store), mode="w").attrs.update({"a": 1})
    assert read_root_attrs(store) == {"a": 1}


def test_read_root_attrs_zarr_v2(tmp_path: Path) -> None:
    store = tmp_path / "v2.zarr"
    store.mkdir()
    (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    assert read_root_attrs(store) == {}
    (store / ".zattrs").write_text(json.dumps({"b": 2}))
    assert read_root_attrs(store) == {"b": 2}


def test_read_root_attrs_errors(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(FileNotFoundError, match="not a Zarr group"):
        read_root_attrs(plain)
    array = tmp_path / "array.zarr"
    array.mkdir()
    (array / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "array"})
    )
    with pytest.raises(ValueError, match="not a Zarr group"):
        read_root_attrs(array)


def _consolidated_group(path: Path) -> None:
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update({"other": {"keep": True}})
    zarr.consolidate_metadata(str(path))


def test_write_root_attr_keeps_other_keys_and_consolidated_metadata(
    tmp_path: Path,
) -> None:
    store = tmp_path / "s.zarr"
    _consolidated_group(store)
    payload = {"schema_version": 1, "pairs": [(1.0, 2.0)], "flag": True}
    with spatialdata_write_lock(store):
        write_root_attr(store, schema.LABELLING_ATTR, payload)
    raw = json.loads((store / "zarr.json").read_text())
    assert raw["attributes"]["other"] == {"keep": True}
    assert raw["consolidated_metadata"] is not None
    stored = read_root_attrs(store)[schema.LABELLING_ATTR]
    assert stored == {"schema_version": 1, "pairs": [[1.0, 2.0]], "flag": True}


def test_write_root_attr_needs_the_lock(tmp_path: Path) -> None:
    store = tmp_path / "s.zarr"
    _consolidated_group(store)
    with pytest.raises(RuntimeError, match="hold spatialdata_write_lock"):
        write_root_attr(store, "key", {"a": 1})
    assert "key" not in read_root_attrs(store)


def test_write_root_attr_rejects_non_json_payloads_without_touching_the_file(
    tmp_path: Path,
) -> None:
    store = tmp_path / "s.zarr"
    _consolidated_group(store)
    before = (store / "zarr.json").read_bytes()
    with spatialdata_write_lock(store):
        with pytest.raises(ValueError):
            write_root_attr(store, "key", {"a": float("nan")})
        with pytest.raises(ValueError, match="not plain JSON"):
            write_root_attr(store, "key", {"a": np.int64(3)})
    assert (store / "zarr.json").read_bytes() == before


def test_write_root_attr_never_creates_a_store(tmp_path: Path) -> None:
    store = tmp_path / "missing.zarr"
    with spatialdata_write_lock(store), pytest.raises(FileNotFoundError):
        write_root_attr(store, "key", {"a": 1})
    assert not store.exists()


@pytest.mark.integration
def test_write_root_attr_on_a_spatialdata_store(tmp_path: Path) -> None:
    import spatialdata as sd
    from spatialdata.models import Image2DModel

    store = tmp_path / "sdata.zarr"
    image = Image2DModel.parse(np.zeros((1, 64, 64), np.uint16), dims=("c", "y", "x"))
    sdata = sd.SpatialData(images={"img": image})
    sdata.attrs["other"] = {"a": 1}
    sdata.write(store)

    payload = {"schema_version": 1, "status": "draft"}
    with spatialdata_write_lock(store):
        write_root_attr(store, schema.LABELLING_ATTR, payload)

    reread = sd.read_zarr(store)
    assert reread.attrs["other"] == {"a": 1}
    assert reread.attrs[schema.LABELLING_ATTR] == payload
    assert "spatialdata_attrs" in read_root_attrs(store)


def _hold_lock(store: str, held: Event, release: Event) -> None:
    with spatialdata_write_lock(store):
        held.set()
        release.wait(timeout=_STARTUP_TIMEOUT_SECONDS)


def _read_locked(store: str, ready: Event, finished: Event) -> None:
    ready.set()
    read_root_attrs_locked(Path(store))
    finished.set()


@pytest.mark.store_lock
def test_read_root_attrs_locked_waits_for_a_writer(tmp_path: Path) -> None:
    store = tmp_path / "s.zarr"
    _consolidated_group(store)
    context = multiprocessing.get_context("spawn")
    held, release = context.Event(), context.Event()
    ready, finished = context.Event(), context.Event()
    holder = context.Process(
        target=_hold_lock, args=(str(store), held, release), daemon=True
    )
    reader = context.Process(
        target=_read_locked, args=(str(store), ready, finished), daemon=True
    )
    holder.start()
    try:
        assert held.wait(timeout=_STARTUP_TIMEOUT_SECONDS)
        reader.start()
        assert ready.wait(timeout=_STARTUP_TIMEOUT_SECONDS)
        assert not finished.wait(timeout=_BLOCKED_WINDOW_SECONDS)
        release.set()
        assert finished.wait(timeout=_STARTUP_TIMEOUT_SECONDS)
    finally:
        release.set()
        for process in (holder, reader):
            if process.ident is not None:
                process.join(timeout=_STARTUP_TIMEOUT_SECONDS)
                if process.is_alive():
                    process.kill()
                    process.join()
    assert holder.exitcode == 0
    assert reader.exitcode == 0
