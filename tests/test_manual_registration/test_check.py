"""Synthetic end to end of the landmark check: prepare, label, check, upload.

Pairs are rendered by ``make_synthetic_pair`` with channels DAPI, AF568 and AF647 and
independent nuclei (a 30 um / 1.5 mm warp, 2048 px at 2 um); the labelling has two
planted misclicks (300 um and 1 mm).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from click.testing import CliRunner

from merxen.cli.run_manual_registration_register import (
    manual_registration_check_command,
)
from merxen.io.ome_tiff import open_ome_tiff_level, read_ome_tiff_info
from merxen.manual_registration import check as check_module
from merxen.manual_registration import field_fit
from merxen.manual_registration.check import (
    StaleLabellingError,
    read_labelling_file,
    run_check,
    write_check_attr,
)
from merxen.manual_registration.config import (
    ManualRegistrationCheckConfig,
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
)
from merxen.manual_registration.geometry import coarse_from_matrix
from merxen.manual_registration.prepare import run_manual_registration_prepare
from merxen.manual_registration.schema import (
    CHECK_ATTR,
    LABELLING_ATTR,
    STORE_ATTR,
    CoarseTransform,
    LandmarkPair,
    parse_check_attrs,
    read_root_attrs,
)
from merxen.manual_registration.simulate import (
    SyntheticPair,
    inject_labelling,
    make_synthetic_pair,
    sample_tissue_points,
    simulate_landmark_pairs,
)

pytestmark = pytest.mark.integration

NAME = "SYN1"
P_REF = 2.0
P_MOV = 1.6
ROLES = "at8=AF568;map2=AF647;dapi=DAPI"
WORKING_TO_LEVEL0 = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
MISCLICK_SMALL_ID = 5
MISCLICK_GROSS_ID = 9


@dataclass
class Prepared:
    """A prepared synthetic pair with its labelling."""

    root: Path
    pair: SyntheticPair
    store: Path
    pairs: list[LandmarkPair]
    coarse: CoarseTransform


def _moving_level(pair: SyntheticPair, level: int) -> np.ndarray:
    info = read_ome_tiff_info(pair.moving_path)
    return np.asarray(open_ome_tiff_level(info, level).values[0], dtype=np.float64)


def _prepare(root: Path) -> Prepared:
    pair = make_synthetic_pair(
        root / "images",
        channel_names=("DAPI", "AF568", "AF647"),
        independent_nuclei=True,
        warp_amplitude_um=30.0,
        warp_wavelength_um=1500.0,
        rotation_deg=178.0,
        offset_um=(40.0, -25.0),
        seed=3,
    )
    outputs = run_manual_registration_prepare(
        ManualRegistrationPrepareConfig(
            dataset_name=NAME,
            reference=ManualRegistrationImageInput(path=pair.reference_path),
            moving=ManualRegistrationImageInput(path=pair.moving_path),
            prepared_dir=root / NAME / "prepared",
            output_dir=root / "prepare_out",
            workers=1,
        )
    )
    points = sample_tissue_points(
        _moving_level(pair, 1),
        WORKING_TO_LEVEL0,
        n=26,
        seed=1,
        min_separation_px=150.0 / P_MOV,
    )
    pairs = simulate_landmark_pairs(
        points,
        pair.moving_to_reference,
        noise_um=3.0,
        reference_pixel_size_um=P_REF,
        seed=1,
        outliers={
            MISCLICK_SMALL_ID: (300.0, 0.0),
            MISCLICK_GROSS_ID: (1000.0, 0.0),
        },
        disabled=(25, 26),
    )
    fitted = coarse_from_matrix(
        pair.true_matrix,
        moving_shape_yx=pair.moving_shape_yx,
        reference_shape_yx=pair.reference_shape_yx,
        moving_pixel_size_um=P_MOV,
        reference_pixel_size_um=P_REF,
    )[0]
    coarse = CoarseTransform(
        rotation_deg=fitted.rotation_deg + 2.0,
        mirror_x=fitted.mirror_x,
        offset_um=(fitted.offset_um[0] + 50.0, fitted.offset_um[1] - 30.0),
    )
    inject_labelling(outputs["store"], pairs, coarse=coarse)
    return Prepared(root, pair, outputs["store"], pairs, coarse)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    return _prepare(tmp_path_factory.mktemp("manual_registration_check"))


@pytest.fixture
def store(prepared: Prepared, tmp_path: Path) -> Path:
    """A private copy of the prepared store (the checks write its attributes)."""
    target = tmp_path / f"{NAME}_prepared.zarr"
    shutil.copytree(prepared.store, target)
    return target


def _config(store: Path, out: Path, **overrides: Any) -> ManualRegistrationCheckConfig:
    values: dict[str, Any] = {
        "dataset_name": NAME,
        "prepared_store_path": store,
        "output_dir": out,
        "channel_roles": ROLES,
        "windows": False,
        "n_suggestions": 6,
    }
    values.update(overrides)
    return ManualRegistrationCheckConfig(**values)


def _root_bytes(store: Path) -> bytes:
    return (store / "zarr.json").read_bytes()


def _load(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


@pytest.fixture
def first(store: Path, tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """The default check of the store's own labelling (attribute written)."""
    out = tmp_path / "check_store"
    return store, run_check(_config(store, out))


def test_run_check_writes_a_valid_current_attribute(
    first: tuple[Path, dict[str, Path]],
) -> None:
    store, paths = first
    attrs = read_root_attrs(store)
    check = parse_check_attrs(attrs[CHECK_ATTR])
    assert check.labelling_source == "store"
    assert check.dataset_name == NAME
    assert check.store_content_digest == attrs[STORE_ATTR]["content_digest"]
    assert check.channel_roles is not None
    assert check.channel_roles["reference"]["map2"] == "AF647"
    assert paths["json"].name == f"{NAME}_cortex_serial_check.json"
    assert paths["png"].exists() and paths["png"].with_suffix(".pdf").exists()
    document = _load(paths["json"])
    assert set(document) == set(attrs[CHECK_ATTR]) | {"details"}
    assert {k: v for k, v in document.items() if k != "details"} == attrs[CHECK_ATTR]
    assert set(document["details"]) == {
        "click_estimates",
        "windows",
        "field_model",
        "clicks_only_hyperparameters",
        "timings_s",
        "warnings",
    }
    assert document["details"]["windows"] is None
    assert len(check.pairs) == 26
    assert check.pairs["25"].pair_class == "disabled"
    assert check.pairs["1"].pair_class in {"ok", "kept_confirmed", "check"}
    assert len(check.suggestions) == 6
    assert all(s.sd_um > 0.0 for s in check.suggestions)
    assert check.field.n_windows == 0
    assert check.algorithm["image_evidence"] == "image_evidence_bp20_v1"
    assert sum(check.field.verdict_counts.values()) == 26
    # The two disabled pairs are not checked; a click near an image edge may not be.
    assert 2 <= check.field.verdict_counts["not_checked"] <= 8


def test_the_one_millimetre_misclick_is_flagged(
    first: tuple[Path, dict[str, Path]],
) -> None:
    store, _paths = first
    check = parse_check_attrs(read_root_attrs(store)[CHECK_ATTR])
    gross = check.pairs[str(MISCLICK_GROSS_ID)]
    assert gross.pair_class in {"likely_misclick", "check"}
    assert gross.loo_um > 500.0
    assert gross.p_value < 0.01
    small = check.pairs[str(MISCLICK_SMALL_ID)]
    assert small.loo_um > 100.0
    clean = [
        pair.loo_um
        for key, pair in check.pairs.items()
        if int(key) not in (MISCLICK_SMALL_ID, MISCLICK_GROSS_ID, 25, 26)
    ]
    assert np.median(clean) < 100.0


def test_no_write_attr_leaves_the_attributes_byte_identical(
    store: Path, tmp_path: Path
) -> None:
    before = _root_bytes(store)
    paths = run_check(_config(store, tmp_path / "out", write_attr=False))
    assert _root_bytes(store) == before
    assert CHECK_ATTR not in read_root_attrs(store)
    assert paths["json"].exists()


def test_a_task_output_dir_gets_a_copy_of_the_json(store: Path, tmp_path: Path) -> None:
    paths = run_check(
        _config(
            store,
            tmp_path / "out",
            write_attr=False,
            image_check=False,
            task_output_dir=tmp_path / "check_out",
        )
    )
    assert paths["task_json"].read_bytes() == paths["json"].read_bytes()
    assert paths["task_json"].parent == tmp_path / "check_out"


def test_a_labelling_changed_between_read_and_write_is_stale(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared: Prepared
) -> None:
    original = check_module.check_payload

    def edit_then_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original(*args, **kwargs)
        moved = list(prepared.pairs)
        moved[0] = moved[0].model_copy(
            update={"moving_xy_px": (moved[0].moving_xy_px[0] + 50.0, 100.0)}
        )
        inject_labelling(store, moved, coarse=prepared.coarse)
        return payload

    monkeypatch.setattr(check_module, "check_payload", edit_then_build)
    out = tmp_path / "out"
    with pytest.raises(
        StaleLabellingError, match="changed while the check ran"
    ) as info:
        run_check(_config(store, out, image_check=False))
    assert CHECK_ATTR not in read_root_attrs(store)
    assert (out / f"{NAME}_cortex_serial_check.json").exists()
    assert info.value.run is not None and info.value.run.attr_written is False


def test_the_cli_exits_three_on_a_stale_labelling(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared: Prepared
) -> None:
    original = check_module.check_payload

    def edit_then_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original(*args, **kwargs)
        inject_labelling(store, prepared.pairs[:-1], coarse=prepared.coarse)
        return payload

    monkeypatch.setattr(check_module, "check_payload", edit_then_build)
    result = CliRunner().invoke(
        manual_registration_check_command,
        [
            "--store",
            str(store),
            "--out-dir",
            str(tmp_path / "out"),
            "--no-image-check",
            "--no-windows",
        ],
    )
    assert result.exit_code == 3, result.output
    assert "changed while the check ran" in result.output
    assert "check attribute written: no" in result.output
    assert CHECK_ATTR not in read_root_attrs(store)
    assert (tmp_path / "out" / f"{NAME}_cortex_serial_check.json").exists()


def test_write_check_attr_refuses_a_payload_of_another_store_digest(
    first: tuple[Path, dict[str, Path]],
) -> None:
    store, paths = first
    payload = {k: v for k, v in _load(paths["json"]).items() if k != "details"}
    payload["store_content_digest"] = "sha256:" + "0" * 64
    before = _root_bytes(store)
    with pytest.raises(StaleLabellingError, match="prepared again"):
        write_check_attr(store, payload)
    assert _root_bytes(store) == before


def test_a_copy_of_the_stores_zarr_json_gives_the_same_check(
    first: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    store, paths = first
    upload = tmp_path / "upload" / "zarr.json"
    upload.parent.mkdir()
    shutil.copy(store / "zarr.json", upload)
    before = _root_bytes(store)
    attrs_before = read_root_attrs(store)
    uploaded = run_check(_config(store, tmp_path / "check_file", labelling_path=upload))
    assert _root_bytes(store) == before
    assert read_root_attrs(store) == attrs_before
    from_store = _load(paths["json"])
    from_file = _load(uploaded["json"])
    assert from_file["labelling_source"] == "file"
    assert from_store["labelling_source"] == "store"
    for key in ("pairs", "field", "suggestions"):
        assert from_file[key] == from_store[key], key
    ignored = {"written_at", "labelling_source", "outputs", "details"}
    assert {k: v for k, v in from_file.items() if k not in ignored} == {
        k: v for k, v in from_store.items() if k not in ignored
    }


def test_a_bare_labelling_payload_and_a_top_level_object_are_accepted(
    store: Path, tmp_path: Path
) -> None:
    attrs = read_root_attrs(store)
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(attrs[LABELLING_ATTR]))
    top = tmp_path / "top.json"
    top.write_text(
        json.dumps(
            {LABELLING_ATTR: attrs[LABELLING_ATTR], STORE_ATTR: attrs[STORE_ATTR]}
        )
    )
    labelling, store_attr = read_labelling_file(bare)
    assert labelling == attrs[LABELLING_ATTR] and store_attr is None
    labelling, store_attr = read_labelling_file(top)
    assert labelling == attrs[LABELLING_ATTR] and store_attr == attrs[STORE_ATTR]
    kwargs: dict[str, Any] = {"image_check": False}
    one = run_check(_config(store, tmp_path / "a", labelling_path=bare, **kwargs))
    two = run_check(_config(store, tmp_path / "b", labelling_path=top, **kwargs))
    assert _load(one["json"])["pairs"] == _load(two["json"])["pairs"]
    assert CHECK_ATTR not in read_root_attrs(store)


def test_a_labelling_file_of_another_preparation_exits_one(
    store: Path, tmp_path: Path
) -> None:
    upload = tmp_path / "zarr.json"
    attrs = read_root_attrs(store)
    other = json.loads(json.dumps(attrs))
    other[STORE_ATTR]["content_digest"] = "sha256:" + "1" * 64
    upload.write_text(json.dumps({"node_type": "group", "attributes": other}))
    result = CliRunner().invoke(
        manual_registration_check_command,
        [
            "--store",
            str(store),
            "--labelling",
            str(upload),
            "--out-dir",
            str(tmp_path / "out"),
            "--no-image-check",
            "--no-windows",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "different preparation of this store" in result.output
    assert "download the store again" in result.output
    assert not (tmp_path / "out").exists()


def test_a_labelling_made_on_other_images_is_refused_by_the_reader(
    store: Path, tmp_path: Path
) -> None:
    attrs = read_root_attrs(store)
    other = json.loads(json.dumps(attrs[LABELLING_ATTR]))
    other["images"]["reference"]["shape_yx"] = [1000, 1000]
    upload = tmp_path / "bare.json"
    upload.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="landmarks were placed on a different"):
        run_check(
            _config(store, tmp_path / "out", labelling_path=upload, image_check=False)
        )


def test_a_labelling_file_works_on_a_store_without_its_own_labelling(
    store: Path, tmp_path: Path
) -> None:
    attrs = read_root_attrs(store)
    upload = tmp_path / "upload.json"
    upload.write_text(json.dumps({"attributes": attrs}))
    group = zarr.open_group(str(store), mode="a")
    del group.attrs[LABELLING_ATTR]
    assert LABELLING_ATTR not in read_root_attrs(store)
    paths = run_check(
        _config(store, tmp_path / "out", labelling_path=upload, image_check=False)
    )
    assert parse_check_attrs(_load(paths["json"])).labelling_source == "file"
    assert CHECK_ATTR not in read_root_attrs(store)
    # Without a file the store's own labelling is missing.
    with pytest.raises(ValueError, match="no landmarks"):
        run_check(_config(store, tmp_path / "again", image_check=False))


def test_a_payload_can_be_checked_directly(store: Path, tmp_path: Path) -> None:
    payload = read_root_attrs(store)[LABELLING_ATTR]
    paths = run_check(
        _config(store, tmp_path / "out", image_check=False),
        labelling_payload=payload,
    )
    check = parse_check_attrs(_load(paths["json"]))
    assert check.labelling_source == "file"
    assert CHECK_ATTR not in read_root_attrs(store)


def test_a_landmarks_only_check_needs_no_roles(store: Path, tmp_path: Path) -> None:
    paths = run_check(
        _config(
            store,
            tmp_path / "out",
            channel_roles="",
            image_check=False,
            windows=False,
        )
    )
    check = parse_check_attrs(_load(paths["json"]))
    assert check.channel_roles is None
    assert check.algorithm["image_evidence"] is None
    assert set(p.verdict for p in check.pairs.values()) == {"not_checked"}
    assert check.field.loo_um_clicks_only is None


def test_the_image_check_without_roles_is_refused(store: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="need channel roles"):
        run_check(_config(store, tmp_path / "out", channel_roles=""))


def test_windows_enter_the_field(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(field_fit, "WINDOW_SIZES_UM", (200.0, 300.0))
    paths = run_check(
        _config(
            store,
            tmp_path / "out",
            windows=True,
            image_check=False,
            row_settings="field_window_step_um=150;field_window_search_um=150",
            write_attr=False,
        )
    )
    document = _load(paths["json"])
    windows = document["details"]["windows"]
    assert windows is not None and windows["n_candidates"] > 0
    assert document["field"]["n_windows"] == windows["n_accepted_windows"]
    assert document["algorithm"]["image_evidence"] == "image_evidence_bp20_v1"
    if document["field"]["n_windows"]:
        classes = document["details"]["field_model"]["observations"]["class"]
        assert classes.count("window") == document["field"]["n_windows"]
        assert document["field"]["loo_um_clicks_only"] is not None


def test_a_forced_prepare_rebuild_drops_the_check_attribute(
    prepared: Prepared, first: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    store, _paths = first
    assert CHECK_ATTR in read_root_attrs(store)
    # Prepare writes only the store and labelling attributes: a rebuilt store has no
    # check, which is the documented behaviour (the check is rerun, not carried).
    outputs = run_manual_registration_prepare(
        ManualRegistrationPrepareConfig(
            dataset_name=NAME,
            reference=ManualRegistrationImageInput(path=prepared.pair.reference_path),
            moving=ManualRegistrationImageInput(path=prepared.pair.moving_path),
            prepared_dir=store.parent,
            output_dir=tmp_path / "prepare_again",
            force=True,
            workers=1,
        )
    )
    rebuilt = read_root_attrs(outputs["store"])
    assert STORE_ATTR in rebuilt and CHECK_ATTR not in rebuilt
