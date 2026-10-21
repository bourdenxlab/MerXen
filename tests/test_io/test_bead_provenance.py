"""Tests for the bead-removal provenance record and the cache-reuse guard."""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merxen.config import MerscopeBeadRemovalConfig
from merxen.io.bead_removal.provenance import (
    IMAGE_PREPROCESSING_ATTR,
    bead_removal_identity,
    build_image_preprocessing_record,
    check_bead_removal_reuse,
    identity_from_record,
    read_image_preprocessing_attr,
)

_LOGGER = "merxen.io.bead_removal.provenance"
_OFF = {"enabled": False, "fill": {"enabled": False}}


def _cfg(*, fill: bool = False, **kwargs: Any) -> MerscopeBeadRemovalConfig:
    """Return an enabled bead-removal config."""
    return MerscopeBeadRemovalConfig(enabled=True, fill=fill, **kwargs)


def _record(
    cfg: MerscopeBeadRemovalConfig | None,
    *,
    z_layers: tuple[int, ...] = (1, 2, 3),
    **kwargs: Any,
) -> dict[str, Any]:
    return build_image_preprocessing_record(
        z_layers=z_layers,
        stainings=["PolyT", "DAPI", "Anti-X"],
        bead_removal_cfg=cfg,
        **kwargs,
    )


@pytest.fixture
def fake_fill_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a stand-in fill module so the fill identity can be exercised."""
    module = types.ModuleType("merxen.io.bead_removal.fill")
    module.FillParams = lambda enabled=True: types.SimpleNamespace(  # type: ignore[attr-defined]
        enabled=enabled
    )
    module.fill_params_sha256 = lambda params: "fillhash-1"  # type: ignore[attr-defined]
    module.support_px = lambda params: {"total": 37}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "merxen.io.bead_removal.fill", module)
    return module


# --- record ---------------------------------------------------------------


def test_record_without_bead_removal_is_metadata_only() -> None:
    record = _record(None)

    assert record["schema"] == 1
    assert record["source"] == "raw_tiff"
    assert record["z_layers"] == [1, 2, 3]
    # The storage order of the stains is not deterministic, so they are sorted.
    assert record["stains"] == ["Anti-X", "DAPI", "PolyT"]
    assert record["bead_removal"] == _OFF
    assert "merxen" in record["writer_versions"]
    json.dumps(record)


def test_record_for_a_disabled_config_equals_no_config() -> None:
    disabled = MerscopeBeadRemovalConfig()

    assert _record(disabled)["bead_removal"] == _record(None)["bead_removal"]


def test_record_with_bead_removal_is_json_safe_and_sorted() -> None:
    calibration = {
        "calibration_sha256": "abc123",
        "rho": {"DAPI": np.float32(0.11)},
        "n_beads": np.int64(7),
    }
    run = {"chunks": np.int64(4), "beads": np.array([1, 2])}

    record = _record(_cfg(stains=["PolyT", "DAPI"]), calibration=calibration, run=run)
    bead = record["bead_removal"]

    assert bead["enabled"] is True
    assert bead["method"] == "psf_subtract"
    assert bead["calibration_sha256"] == "abc123"
    assert bead["corrected_stains"] == ["DAPI", "PolyT"]
    assert bead["calibration"]["n_beads"] == 7
    assert bead["run"]["beads"] == [1, 2]
    assert bead["fill"] == {"enabled": False}
    assert bead["backend"] == {"device": "cpu"}
    json.dumps(record)


def test_record_fill_identity(fake_fill_module: types.ModuleType) -> None:
    bead = _record(_cfg(fill=True))["bead_removal"]

    assert bead["fill"] == {
        "enabled": True,
        "params_sha256": "fillhash-1",
        "support_px": 37,
    }


def test_record_fill_without_fill_module_still_records_the_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "merxen.io.bead_removal.fill", None)

    fill = _record(_cfg(fill=True))["bead_removal"]["fill"]

    assert fill == {"enabled": True, "params_sha256": None, "support_px": None}


def test_params_hash_tracks_output_relevant_config_only() -> None:
    base = bead_removal_identity(_cfg())["params_sha256"]

    assert bead_removal_identity(_cfg(max_workers=2))["params_sha256"] == base
    assert bead_removal_identity(_cfg(fill=True))["params_sha256"] == base
    assert (
        bead_removal_identity(_cfg(stains=["DAPI", "PolyT"]))["params_sha256"] == base
    )
    assert bead_removal_identity(_cfg(halo_px=320))["params_sha256"] != base
    assert bead_removal_identity(_cfg(stains=["PolyT"]))["params_sha256"] != base


def test_identity_from_record_reads_legacy_as_off() -> None:
    assert identity_from_record(None) == _OFF
    assert identity_from_record({}) == _OFF
    assert identity_from_record(_record(None)) == _OFF


def test_identity_round_trips_through_the_record() -> None:
    cfg = _cfg()

    identity = identity_from_record(_record(cfg))

    assert identity == bead_removal_identity(cfg)


# --- reading the attribute ------------------------------------------------


def test_read_attr_from_zarr_v3_root(tmp_path: Path) -> None:
    store = tmp_path / "source.zarr"
    store.mkdir()
    record = _record(_cfg())
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "attributes": {IMAGE_PREPROCESSING_ATTR: record}})
    )

    assert read_image_preprocessing_attr(store) == record


def test_read_attr_from_zarr_v2_root(tmp_path: Path) -> None:
    store = tmp_path / "source.zarr"
    store.mkdir()
    record = _record(None)
    (store / ".zattrs").write_text(json.dumps({IMAGE_PREPROCESSING_ATTR: record}))

    assert read_image_preprocessing_attr(store) == record


def test_read_attr_follows_a_symlinked_store(tmp_path: Path) -> None:
    store = tmp_path / "cache" / "source.zarr"
    store.mkdir(parents=True)
    record = _record(None)
    (store / "zarr.json").write_text(
        json.dumps({"attributes": {IMAGE_PREPROCESSING_ATTR: record}})
    )
    link = tmp_path / "staged.zarr"
    link.symlink_to(store, target_is_directory=True)

    assert read_image_preprocessing_attr(link) == record


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not json",
        json.dumps({"attributes": {}}),
        json.dumps({"attributes": {"other": 1}}),
        json.dumps({"attributes": {IMAGE_PREPROCESSING_ATTR: "text"}}),
        json.dumps({"attributes": None}),
    ],
)
def test_read_attr_reads_legacy_or_broken_metadata_as_none(
    tmp_path: Path, content: str | None
) -> None:
    store = tmp_path / "source.zarr"
    store.mkdir()
    if content is not None:
        (store / "zarr.json").write_text(content)

    assert read_image_preprocessing_attr(store) is None


def test_read_attr_of_a_missing_store_is_none(tmp_path: Path) -> None:
    assert read_image_preprocessing_attr(tmp_path / "absent.zarr") is None


# --- reuse guard ----------------------------------------------------------

# State name -> (recorded attribute, requested config).
_RECORDED = {
    "legacy": lambda: None,
    "off": lambda: _record(None),
    "on": lambda: _record(_cfg()),
    "on_fill": lambda: _record(_cfg(fill=True)),
}
_REQUESTED = {
    "none": lambda: None,
    "off": lambda: MerscopeBeadRemovalConfig(),
    "on": lambda: _cfg(),
    "on_fill": lambda: _cfg(fill=True),
}
_BEADS_ON = {"on", "on_fill"}


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("requested_name", sorted(_REQUESTED))
@pytest.mark.parametrize("recorded_name", sorted(_RECORDED))
def test_guard_matrix(
    recorded_name: str,
    requested_name: str,
    force: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorded = _RECORDED[recorded_name]()
    requested = _REQUESTED[requested_name]()
    recorded_on = recorded_name in _BEADS_ON
    requested_on = requested_name in _BEADS_ON
    mismatch = recorded_on != requested_on or (
        recorded_on and (recorded_name == "on_fill") != (requested_name == "on_fill")
    )

    def run() -> None:
        check_bead_removal_reuse(
            recorded, requested, force=force, source=Path("cache/source.zarr")
        )

    if mismatch and not force:
        with pytest.raises(ValueError, match="--force_spatialdata_build"):
            run()
        return
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        run()
    if mismatch:
        assert "Rebuilding the cached SpatialData" in caplog.text
    # Matching settings, including legacy against off, are reused silently.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_guard_error_names_the_cache_and_both_states() -> None:
    with pytest.raises(ValueError) as excinfo:
        check_bead_removal_reuse(
            _record(_cfg()), None, source=Path("cache/source.zarr")
        )

    message = str(excinfo.value)
    assert "cache/source.zarr" in message
    assert "bead removal on but off was requested" in message
    assert "merscope_bead_removal" in message


def test_guard_fill_error_points_at_the_fill_param() -> None:
    with pytest.raises(ValueError, match="merscope_bead_removal_fill"):
        check_bead_removal_reuse(_record(_cfg()), _cfg(fill=True))


def test_guard_legacy_store_with_bead_request_is_an_error() -> None:
    with pytest.raises(ValueError, match="built with bead removal off"):
        check_bead_removal_reuse(None, _cfg())


def test_guard_warns_on_a_different_method_version(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorded = _record(_cfg())
    recorded["bead_removal"]["version"] = 0

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        check_bead_removal_reuse(recorded, _cfg())

    assert "method version 0" in caplog.text


def test_guard_warns_on_a_different_parameter_hash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        check_bead_removal_reuse(_record(_cfg()), _cfg(halo_px=320))

    assert "parameter hash" in caplog.text


def test_guard_warns_on_a_different_fill_hash(
    caplog: pytest.LogCaptureFixture, fake_fill_module: types.ModuleType
) -> None:
    recorded = _record(_cfg(fill=True))
    recorded["bead_removal"]["fill"]["params_sha256"] = "older"

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        check_bead_removal_reuse(recorded, _cfg(fill=True))

    assert "fill parameter hash older" in caplog.text


def test_guard_warns_on_a_different_backend_but_never_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorded = _record(_cfg(), device="cpu")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        check_bead_removal_reuse(recorded, _cfg(), requested_device="cuda")

    assert "backend device cpu" in caplog.text


def test_guard_warns_on_a_different_z_range_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        check_bead_removal_reuse(
            _record(None, z_layers=(1, 2)), None, requested_z_layers=[1, 2, 3]
        )

    assert "projected from z planes [1, 2]" in caplog.text


def test_guard_is_silent_for_matching_z_range_and_legacy_store(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        check_bead_removal_reuse(
            _record(None, z_layers=(3, 1, 2)), None, requested_z_layers=[1, 2, 3]
        )
        check_bead_removal_reuse(None, None, requested_z_layers=[1, 2, 3])

    assert caplog.text == ""
