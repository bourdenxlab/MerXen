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
    STORE_SIDECAR_NAME,
    bead_removal_identity,
    bead_removal_table_metadata,
    build_image_preprocessing_record,
    check_bead_removal_reuse,
    check_segmentation_sidecar,
    check_store_segmentation_sidecar,
    identity_from_record,
    read_image_preprocessing_attr,
    remove_segmentation_sidecar,
    segmentation_identity,
    segmentation_sidecar_path,
    write_segmentation_sidecar,
    write_store_segmentation_sidecar,
)

_LOGGER = "merxen.io.bead_removal.provenance"
_OFF = {"enabled": False, "fill": {"enabled": False}}


def _cfg(*, fill: bool = False, **kwargs: Any) -> MerscopeBeadRemovalConfig:
    """Return an enabled bead-removal config.

    The fill is off unless asked for, and then flat unless a mode is given: the
    default of the config is the hybrid fill, which these tests name explicitly.
    """
    if fill:
        kwargs.setdefault("fill_mode", "flat")
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


_GPU_BACKEND = {
    "device": "cuda",
    "update_order": "colour",
    "torch": "2.14.0+cu130",
    "triton": "3.8.0",
    "cuda": "13.0",
    "gpu_name": "Test GPU",
}


def test_record_and_identity_carry_the_gpu_backend_versions() -> None:
    record = _record(_cfg(), backend=_GPU_BACKEND)

    assert record["bead_removal"]["backend"] == _GPU_BACKEND
    assert identity_from_record(record)["backend"] == _GPU_BACKEND
    assert bead_removal_identity(_cfg(), backend=_GPU_BACKEND)["backend"] == (
        _GPU_BACKEND
    )
    json.dumps(record)


def test_the_backend_record_replaces_the_device_argument() -> None:
    record = _record(_cfg(), device="cpu", backend=_GPU_BACKEND)

    assert record["bead_removal"]["backend"]["device"] == "cuda"


def test_the_device_does_not_change_the_parameter_hash() -> None:
    cpu = bead_removal_identity(_cfg(device="cpu"))
    cuda = bead_removal_identity(_cfg(device="cuda"))
    auto = bead_removal_identity(_cfg(device="auto"))

    assert cpu["params_sha256"] == cuda["params_sha256"] == auto["params_sha256"]


def test_guard_warns_once_per_changed_library_version_and_never_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorded = _record(_cfg(), backend=_GPU_BACKEND)
    requested = {**_GPU_BACKEND, "torch": "2.6.0+cu124", "cuda": "12.4"}

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        check_bead_removal_reuse(recorded, _cfg(), requested_backend=requested)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert any("backend torch 2.14.0+cu130" in text for text in warnings)
    assert any("backend cuda 13.0" in text for text in warnings)
    assert all("is reused as built" in text for text in warnings)


def test_guard_is_silent_for_the_same_backend_versions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorded = _record(_cfg(), backend=_GPU_BACKEND)
    requested = {
        key: _GPU_BACKEND[key] for key in ("device", "torch", "triton", "cuda")
    }

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        check_bead_removal_reuse(recorded, _cfg(), requested_backend=requested)

    assert caplog.text == ""


def test_guard_warns_when_the_device_changes_from_cuda_to_cpu(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorded = _record(_cfg(), backend=_GPU_BACKEND)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        check_bead_removal_reuse(recorded, _cfg(), requested_backend={"device": "cpu"})

    assert "backend device cuda" in caplog.text


def test_guard_does_not_compare_an_unresolved_auto_device(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorded = _record(_cfg(), backend=_GPU_BACKEND)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        check_bead_removal_reuse(recorded, _cfg(device="auto"), requested_backend={})

    assert caplog.text == ""


def test_a_library_version_change_leaves_the_segmentation_identity_alone() -> None:
    recorded = _record(
        _cfg(), calibration={"calibration_sha256": "cal-1"}, backend=_GPU_BACKEND
    )
    newer = json.loads(json.dumps(recorded))
    newer["bead_removal"]["backend"].update(torch="2.6.0", triton="3.2.0", cuda="12.4")

    identity = segmentation_identity(recorded)
    assert identity is not None
    assert identity["device"] == "cuda"
    assert segmentation_identity(newer) == identity


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


# ----------------------------------------------------------------------------
# Segmentation sidecar and downstream table metadata (D6, WP8)
# ----------------------------------------------------------------------------


def _enabled_record(
    *, calibration_sha256: str = "cal-1", device: str = "cpu", **cfg_kwargs: Any
) -> dict[str, Any]:
    return _record(
        _cfg(**cfg_kwargs),
        calibration={"calibration_sha256": calibration_sha256},
        device=device,
    )


class TestSegmentationIdentity:
    def test_off_and_legacy_are_the_same_image(self) -> None:
        assert segmentation_identity(None) is None
        assert segmentation_identity(_record(None)) is None

    def test_enabled_keeps_what_the_image_depends_on(self) -> None:
        identity = segmentation_identity(_enabled_record())
        assert identity is not None
        assert identity["version"] == 1
        assert identity["calibration_sha256"] == "cal-1"
        assert identity["device"] == "cpu"
        assert identity["fill"] == {"enabled": False}
        # Stain lists, z range and run statistics do not define the image.
        assert set(identity) == {
            "method",
            "version",
            "params_sha256",
            "calibration_sha256",
            "fill",
            "device",
        }

    def test_device_and_calibration_distinguish_images(self) -> None:
        base = segmentation_identity(_enabled_record())
        assert segmentation_identity(_enabled_record(device="cuda")) != base
        assert segmentation_identity(_enabled_record(calibration_sha256="x")) != base

    def test_library_versions_are_not_compared(self) -> None:
        record = _enabled_record()
        other = json.loads(json.dumps(record))
        other["bead_removal"]["backend"]["torch"] = "9.9"
        other["writer_versions"] = {"merxen": "other"}
        assert segmentation_identity(other) == segmentation_identity(record)

    def test_fill_is_part_of_the_identity(self, fake_fill_module: object) -> None:
        with_fill = segmentation_identity(_enabled_record(fill=True))
        assert with_fill is not None
        assert with_fill["fill"]["enabled"] is True
        assert with_fill != segmentation_identity(_enabled_record())


class TestSegmentationSidecar:
    @pytest.mark.parametrize("nuclei", [False, True])
    def test_path_sits_next_to_the_mask(self, tmp_path: Path, nuclei: bool) -> None:
        mask = tmp_path / "masks.npy"
        path = segmentation_sidecar_path(mask, nuclei=nuclei)
        assert path.parent == tmp_path
        assert path.name.endswith("image_preprocessing.json")
        assert (path.name.startswith("nuclei_")) is nuclei

    def test_cell_and_nuclei_sidecars_are_independent(self, tmp_path: Path) -> None:
        mask = tmp_path / "masks.npy"
        write_segmentation_sidecar(mask, _enabled_record())
        # Nuclei masks in the same directory were never recorded.
        with pytest.raises(ValueError, match="predates the correction"):
            check_segmentation_sidecar(mask, _enabled_record(), nuclei=True)
        check_segmentation_sidecar(mask, _enabled_record())

    def test_roundtrip_matches(self, tmp_path: Path) -> None:
        mask = tmp_path / "masks.npy"
        record = _enabled_record()
        path = write_segmentation_sidecar(mask, record)
        assert json.loads(path.read_text())["bead_removal"] == (
            segmentation_identity(record)
        )
        check_segmentation_sidecar(mask, record, dataset_name="D")

    def test_write_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        write_segmentation_sidecar(tmp_path / "masks.npy", _enabled_record())
        assert [p.name for p in tmp_path.iterdir()] == ["image_preprocessing.json"]

    def test_off_sidecar_records_null(self, tmp_path: Path) -> None:
        path = write_segmentation_sidecar(tmp_path / "masks.npy", None)
        assert json.loads(path.read_text())["bead_removal"] is None
        check_segmentation_sidecar(tmp_path / "masks.npy", _record(None))

    @pytest.mark.parametrize(
        ("stored", "source"),
        [
            (None, "enabled"),
            ("enabled", None),
            ("enabled", "device"),
            ("enabled", "calibration"),
        ],
    )
    def test_mismatch_raises_with_remedy(
        self, tmp_path: Path, stored: str | None, source: str | None
    ) -> None:
        records: dict[str | None, dict[str, Any] | None] = {
            None: None,
            "enabled": _enabled_record(),
            "device": _enabled_record(device="cuda"),
            "calibration": _enabled_record(calibration_sha256="other"),
        }
        mask = tmp_path / "masks.npy"
        write_segmentation_sidecar(mask, records[stored])
        with pytest.raises(ValueError, match="new --outdir") as err:
            check_segmentation_sidecar(mask, records[source], dataset_name="D")
        message = str(err.value)
        assert "[D]" in message
        # The remedy names every place that holds the stale image and both ways
        # to recompute (there is no --force-rerun in Nextflow).
        assert "segmentation/" in message and "latest/" in message
        assert "--force_proseg_rerun" in message
        assert "merxen cellpose-segment --force-rerun" in message

    def test_fill_mismatch_raises(
        self, tmp_path: Path, fake_fill_module: object
    ) -> None:
        mask = tmp_path / "masks.npy"
        write_segmentation_sidecar(mask, _enabled_record(fill=True))
        with pytest.raises(ValueError, match="with fill"):
            check_segmentation_sidecar(mask, _enabled_record())

    def test_missing_sidecar_with_source_enabled_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="predates the correction"):
            check_segmentation_sidecar(tmp_path / "masks.npy", _enabled_record())

    @pytest.mark.parametrize("record", [None, "off"])
    def test_missing_sidecar_with_source_off_or_legacy_reuses(
        self, tmp_path: Path, record: str | None
    ) -> None:
        source = None if record is None else _record(None)
        check_segmentation_sidecar(tmp_path / "masks.npy", source)
        assert not segmentation_sidecar_path(tmp_path / "masks.npy").exists()

    @pytest.mark.parametrize("content", ["{not json", "[1]", '{"bead_removal": 3}'])
    def test_unreadable_sidecar_raises(self, tmp_path: Path, content: str) -> None:
        mask = tmp_path / "masks.npy"
        segmentation_sidecar_path(mask).write_text(content)
        with pytest.raises(ValueError, match="new --outdir"):
            check_segmentation_sidecar(mask, None)

    def test_remove_is_idempotent(self, tmp_path: Path) -> None:
        mask = tmp_path / "masks.npy"
        remove_segmentation_sidecar(mask)
        write_segmentation_sidecar(mask, None)
        remove_segmentation_sidecar(mask)
        assert not segmentation_sidecar_path(mask).exists()

    VETO = {"version": 1, "a0_min": 400.0, "ring_px": [10.0, 25.0]}

    def test_the_bead_mask_veto_is_recorded_only_when_applied(
        self, tmp_path: Path
    ) -> None:
        record = _enabled_record()
        without = write_segmentation_sidecar(tmp_path / "a" / "masks.npy", record)
        with_veto = write_segmentation_sidecar(
            tmp_path / "b" / "masks.npy", record, bead_mask_veto=self.VETO
        )
        # A run without the veto writes the sidecar it always wrote.
        assert set(json.loads(without.read_text())) == {"schema", "bead_removal"}
        assert json.loads(with_veto.read_text())["bead_mask_veto"] == self.VETO

    def test_a_matching_veto_is_reused(self, tmp_path: Path) -> None:
        mask = tmp_path / "masks.npy"
        record = _enabled_record()
        write_segmentation_sidecar(mask, record, bead_mask_veto=self.VETO)
        check_segmentation_sidecar(mask, record, bead_mask_veto=self.VETO)

    @pytest.mark.parametrize(
        ("stored", "requested", "message"),
        [
            (None, VETO, "computed without the bead-mask veto but this run applies"),
            (VETO, None, "computed with the bead-mask veto .* applies without"),
            (VETO, {**VETO, "a0_min": 500.0}, "A0 >= 400.0.*A0 >= 500.0"),
        ],
    )
    def test_a_different_veto_is_refused_with_the_usual_remedy(
        self,
        tmp_path: Path,
        stored: dict[str, Any] | None,
        requested: dict[str, Any] | None,
        message: str,
    ) -> None:
        mask = tmp_path / "masks.npy"
        record = _enabled_record()
        write_segmentation_sidecar(mask, record, bead_mask_veto=stored)
        with pytest.raises(ValueError, match=message) as err:
            check_segmentation_sidecar(
                mask, record, dataset_name="D", bead_mask_veto=requested
            )
        assert "merxen cellpose-segment --force-rerun" in str(err.value)
        assert "new --outdir" in str(err.value)

    def test_the_nuclei_sidecar_and_the_store_sidecar_know_no_veto(
        self, tmp_path: Path
    ) -> None:
        record = _enabled_record()
        write_segmentation_sidecar(tmp_path / "masks.npy", record, nuclei=True)
        check_segmentation_sidecar(tmp_path / "masks.npy", record, nuclei=True)
        write_store_segmentation_sidecar(tmp_path, record)
        check_store_segmentation_sidecar(tmp_path, record)


class TestStoreSegmentationSidecar:
    """The record inside a ProSeg store of the image its prior masks came from."""

    def test_round_trip_and_no_temporary_file(self, tmp_path: Path) -> None:
        store = tmp_path / "latest.zarr"
        store.mkdir()
        record = _enabled_record()

        path = write_store_segmentation_sidecar(store, record)

        assert path == store / STORE_SIDECAR_NAME
        assert json.loads(path.read_text())["bead_removal"] == segmentation_identity(
            record
        )
        assert [p.name for p in store.iterdir()] == [STORE_SIDECAR_NAME]
        check_store_segmentation_sidecar(store, record, dataset_name="D")

    def test_the_sidecar_survives_a_store_rewrite(self, tmp_path: Path) -> None:
        """It is not an element, so the contract upgrade copies it along."""
        from merxen.io.spatialdata_io import _copy_non_spatialdata_sidecars

        store = tmp_path / "raw.zarr"
        (store / "points").mkdir(parents=True)
        write_store_segmentation_sidecar(store, _enabled_record())
        rewritten = tmp_path / "latest.zarr"
        rewritten.mkdir()

        _copy_non_spatialdata_sidecars(store, rewritten)

        check_store_segmentation_sidecar(rewritten, _enabled_record())
        assert (rewritten / STORE_SIDECAR_NAME).is_file()
        assert not (rewritten / "points").exists()

    @pytest.mark.parametrize(
        ("stored", "source"),
        [
            (None, "enabled"),
            ("enabled", None),
            ("enabled", "device"),
            ("enabled", "calibration"),
        ],
    )
    def test_mismatch_raises_with_the_remedy(
        self, tmp_path: Path, stored: str | None, source: str | None
    ) -> None:
        records: dict[str | None, dict[str, Any] | None] = {
            None: None,
            "enabled": _enabled_record(),
            "device": _enabled_record(device="cuda"),
            "calibration": _enabled_record(calibration_sha256="other"),
        }
        write_store_segmentation_sidecar(tmp_path, records[stored])
        with pytest.raises(ValueError, match="--force_proseg_rerun") as err:
            check_store_segmentation_sidecar(
                tmp_path, records[source], dataset_name="D"
            )
        assert "[D]" in str(err.value) and "latest/" in str(err.value)

    def test_missing_sidecar_with_source_enabled_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="predates the correction"):
            check_store_segmentation_sidecar(tmp_path, _enabled_record())

    @pytest.mark.parametrize("record", [None, "off"])
    def test_missing_sidecar_with_source_off_or_legacy_reuses(
        self, tmp_path: Path, record: str | None
    ) -> None:
        check_store_segmentation_sidecar(
            tmp_path, None if record is None else _record(None)
        )
        assert not (tmp_path / STORE_SIDECAR_NAME).exists()

    @pytest.mark.parametrize("content", ["{not json", "[1]", '{"bead_removal": 3}'])
    def test_unreadable_sidecar_raises(self, tmp_path: Path, content: str) -> None:
        (tmp_path / STORE_SIDECAR_NAME).write_text(content)
        with pytest.raises(ValueError, match="--force_proseg_rerun"):
            check_store_segmentation_sidecar(tmp_path, None)


class TestBeadRemovalTableMetadata:
    def test_none_when_off_or_legacy(self) -> None:
        assert bead_removal_table_metadata(None) is None
        assert bead_removal_table_metadata(_record(None)) is None

    def test_enabled_lists_corrected_stains(self) -> None:
        metadata = bead_removal_table_metadata(_enabled_record())
        assert metadata == {
            "bead_corrected_stains": ["DAPI", "PolyT"],
            "bead_removal_version": 1,
            "bead_removal_fill": False,
        }

    def test_fill_flag(self, fake_fill_module: object) -> None:
        metadata = bead_removal_table_metadata(_enabled_record(fill=True))
        assert metadata is not None
        assert metadata["bead_removal_fill"] is True


# --- hybrid fill mode ------------------------------------------------------
def test_flat_fill_identity_is_unchanged_and_hybrid_records_its_mode() -> None:
    flat = bead_removal_identity(_cfg(fill=True))
    hybrid = bead_removal_identity(_cfg(fill=True, fill_mode="hybrid"))

    # A flat fill block has no mode key (as before the hybrid mode existed).
    assert "mode" not in flat["fill"]
    assert hybrid["fill"]["mode"] == "hybrid"
    assert hybrid["fill"]["enabled"] is True
    assert hybrid["fill"]["params_sha256"] != flat["fill"]["params_sha256"]
    assert hybrid["fill"]["support_px"] <= 254
    # The mode belongs to the fill record, not to the algorithm parameter hash.
    assert hybrid["params_sha256"] == flat["params_sha256"]
    # It survives the record and is read back.
    record = _record(_cfg(fill=True, fill_mode="hybrid"))
    assert record["bead_removal"]["fill"]["mode"] == "hybrid"
    assert identity_from_record(record)["fill"]["mode"] == "hybrid"
    json.dumps(record)


def test_hybrid_mode_is_recorded_without_the_fill_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "merxen.io.bead_removal.fill", None)

    fill = _record(_cfg(fill=True, fill_mode="hybrid"))["bead_removal"]["fill"]

    assert fill == {
        "enabled": True,
        "params_sha256": None,
        "support_px": None,
        "mode": "hybrid",
    }


@pytest.mark.parametrize(
    ("recorded_mode", "requested_mode"), [("flat", "hybrid"), ("hybrid", "flat")]
)
def test_guard_refuses_another_fill_mode(
    recorded_mode: str, requested_mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    recorded = _record(_cfg(fill=True, fill_mode=recorded_mode))
    requested = _cfg(fill=True, fill_mode=requested_mode)

    with pytest.raises(ValueError, match="merscope_bead_removal_fill_mode") as caught:
        check_bead_removal_reuse(recorded, requested, source=Path("cache/s.zarr"))
    assert f"fill mode {recorded_mode} but {requested_mode}" in str(caught.value)
    # A rebuild is allowed to replace it.
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        check_bead_removal_reuse(recorded, requested, force=True)
    assert "fill mode was" in caplog.text


def test_guard_accepts_the_same_hybrid_mode_and_a_legacy_flat_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hybrid = _cfg(fill=True, fill_mode="hybrid")
    legacy_flat = _record(_cfg(fill=True))
    # A record written before the mode existed has no mode key at all.
    assert "mode" not in legacy_flat["bead_removal"]["fill"]

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        check_bead_removal_reuse(_record(hybrid), hybrid)
        check_bead_removal_reuse(legacy_flat, _cfg(fill=True))

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_segmentation_identity_tells_the_fill_modes_apart() -> None:
    flat = segmentation_identity(_record(_cfg(fill=True)))
    hybrid = segmentation_identity(_record(_cfg(fill=True, fill_mode="hybrid")))

    assert flat is not None and hybrid is not None
    assert flat != hybrid
    assert hybrid["fill"]["mode"] == "hybrid"
