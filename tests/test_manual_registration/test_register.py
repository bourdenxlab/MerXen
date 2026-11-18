"""Landmark fit, store validation and the pure helpers of the register step."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from merxen.image_registration import (
    pull_points_through_registration,
    push_points_through_registration,
    registration_output_paths,
)
from merxen.manual_registration import register as register_module
from merxen.manual_registration.config import (
    ManualRegistrationRunConfig,
    ManualRegistrationSettings,
)
from merxen.manual_registration.presets import resolve_settings
from merxen.manual_registration.register import (
    LandmarkFitError,
    LandmarkFitReport,
    _core_qc_listing,
    _landmark_records,
    _landmarks_dataframe,
    _outputs_from_stored,
    _plain_json,
    _provenance_digest,
    _publish,
    _remove_core_qc_files,
    _remove_superseded_siblings,
    _residual_summary,
    _result_store_attr,
    _reusable,
    _validate_prepared_store,
    fit_landmarks,
    fit_summary,
    run_manual_registration,
)
from merxen.manual_registration.register import (
    _stored_result as read_stored_result,
)
from merxen.manual_registration.schema import (
    LABELLING_ATTR,
    STORE_ATTR,
    CoarseTransform,
    CropRect,
    LabellingAttrs,
    Materialization,
    StoreAttrs,
    UnsupportedSchemaVersionError,
    labelling_content_sha256,
    parse_labelling_attrs,
    parse_store_attrs,
    read_root_attrs,
)
from merxen.manual_registration.simulate import inject_labelling

from ._mr_helpers import (
    A2_COARSE,
    A2_MOVING,
    A2_MOVING_SHAPE,
    A2_PIXEL_SIZE,
    A2_REFERENCE_SHAPE,
    a2_pairs,
    a2_store,
    a2_truth_matrix,
    fake_store,
    labelling_dict,
    make_pairs,
    store_attr_dict,
)

RUN_ID = "SYN1_same_section"


def _settings(
    preset: str = "same_section", row_settings: str = ""
) -> ManualRegistrationSettings:
    return resolve_settings(
        preset,
        reference_pixel_size_um=A2_PIXEL_SIZE,
        reference_channel="DAPI",
        row_settings=row_settings,
    )


def _store() -> Any:
    return parse_store_attrs(
        store_attr_dict(
            A2_REFERENCE_SHAPE, A2_MOVING_SHAPE, A2_PIXEL_SIZE, A2_PIXEL_SIZE
        )
    )


def _labelling(
    pairs: Any = None, coarse: CoarseTransform = A2_COARSE, **kwargs: Any
) -> LabellingAttrs:
    payload = labelling_dict(a2_pairs() if pairs is None else pairs, coarse, **kwargs)
    return parse_labelling_attrs(payload)


def _fit(
    labelling: Any = None, settings: Any = None, **kwargs: Any
) -> LandmarkFitReport:
    return fit_landmarks(
        labelling or _labelling(),
        _store(),
        settings or _settings(),
        **kwargs,
    )


def _truth_pairs(moving: list[tuple[float, float]], scale: float = 1.0) -> Any:
    matrix = a2_truth_matrix()
    points = np.asarray(moving, dtype=float)
    reference = points @ matrix[:2, :2].T + matrix[:2, 2]
    centroid = reference.mean(axis=0)
    reference = centroid + scale * (reference - centroid)
    return make_pairs(moving, reference)


# --------------------------------------------------------------------------
# fit_landmarks
# --------------------------------------------------------------------------


def test_default_fit_trims_the_planted_outlier() -> None:
    report = _fit(run_id=RUN_ID)
    assert int(report.used.sum()) == 9
    assert list(np.flatnonzero(report.outlier)) == [4]
    np.testing.assert_allclose(
        report.matrix_full[:2, :2],
        [[-0.999861, 0.017468], [-0.017468, -0.999861]],
        atol=1e-3,
    )
    np.testing.assert_allclose(
        report.matrix_full[:2, 2], [76009.005205, 54892.785195], atol=1e-2
    )
    assert report.loo_um[4] == pytest.approx(41.0328, abs=1e-3)
    assert "1 landmark pair(s) flagged as outlier and excluded: 5" in report.warnings
    summary = fit_summary(report, store=_store(), settings=_settings())
    assert summary["n_used"] == 9
    assert summary["rotation_deg"] == pytest.approx(178.99913, abs=1e-4)
    assert summary["n_outliers"] == 1
    assert summary["n_enabled"] == 10
    assert summary["viewer_outlier_floor_um"] is None
    assert all(type(v) in (float, int, bool, str, type(None)) for v in summary.values())


def test_fewer_than_eight_enabled_pairs_fail() -> None:
    labelling = _labelling(a2_pairs(disabled=(1, 2, 3)))
    with pytest.raises(LandmarkFitError, match="7 enabled") as excinfo:
        _fit(labelling, run_id=RUN_ID)
    assert str(excinfo.value).startswith(f"[{RUN_ID}]")
    assert excinfo.value.report is None


def test_collinear_pairs_fail_with_a_report() -> None:
    moving = [(5000.0 + 6000.0 * i, 20000.0 + 0.5 * i) for i in range(10)]
    labelling = _labelling(_truth_pairs(moving))
    with pytest.raises(LandmarkFitError, match="collinear") as excinfo:
        _fit(labelling)
    assert excinfo.value.report is not None


def test_wrong_flip_is_detected() -> None:
    labelling = _labelling(
        coarse=CoarseTransform(rotation_deg=179.0, mirror_x=True, offset_um=(150, -80))
    )
    with pytest.raises(LandmarkFitError, match="mirrored"):
        _fit(labelling)


def test_scale_off_by_three_percent_depends_on_the_tolerance() -> None:
    pairs = _truth_pairs(list(A2_MOVING), scale=1.03)
    labelling = _labelling(pairs)
    with pytest.raises(LandmarkFitError, match="pixel-size ratio") as excinfo:
        _fit(labelling)
    assert "same_section:matrix_scale_tolerance=0.05" in str(excinfo.value)
    _fit(labelling, _settings("serial_section"))
    _fit(
        labelling,
        _settings(row_settings="same_section:matrix_scale_tolerance=0.05"),
    )


def test_outlier_policies() -> None:
    with pytest.raises(LandmarkFitError, match="flagged as outliers: 5") as excinfo:
        _fit(settings=_settings(row_settings="landmark_outlier_policy=fail"))
    assert excinfo.value.report is not None
    report = _fit(settings=_settings(row_settings="landmark_outlier_policy=keep"))
    assert int(report.used.sum()) == 10
    assert list(np.flatnonzero(report.outlier)) == [4]
    assert any("kept" in w for w in report.warnings)


def test_outside_crop_policies() -> None:
    crop = {"reference": CropRect(x0=0, y0=0, x1=70000, y1=61000)}
    labelling = _labelling(crops=crop)
    used = _fit(labelling)
    assert used.outside_crop[0] and used.used[0]
    assert any("outside a crop are used" in w and "1" in w for w in used.warnings)
    excluded = _fit(labelling, _settings(row_settings="landmarks_outside_crop=exclude"))
    assert not excluded.used[0]
    assert 0 not in excluded.fit_index
    assert excluded.outside_crop[0]
    with pytest.raises(LandmarkFitError, match="outside a crop"):
        _fit(labelling, _settings(row_settings="landmarks_outside_crop=fail"))


def test_disabled_pairs_stay_out_of_the_fit() -> None:
    report = _fit(_labelling(a2_pairs(disabled=(9, 10))))
    assert 8 not in report.fit_index and 9 not in report.fit_index
    np.testing.assert_array_equal(report.loo_um[8:], report.residuals_um[8:])
    assert not report.used[8:].any()
    assert report.enabled.sum() == 8


def test_large_rotation_difference_warns() -> None:
    labelling = _labelling(
        coarse=CoarseTransform(rotation_deg=150.0, mirror_x=False, offset_um=(150, -80))
    )
    report = _fit(labelling)
    assert any("differs from the coarse rotation" in w for w in report.warnings)


def test_messages_carry_the_run_id() -> None:
    labelling = _labelling(a2_pairs(disabled=(1, 2, 3)))
    with pytest.raises(LandmarkFitError) as excinfo:
        _fit(labelling, run_id="SYN1_same_section")
    assert "[SYN1_same_section]" in str(excinfo.value)
    with pytest.raises(LandmarkFitError) as excinfo:
        _fit(labelling)
    assert str(excinfo.value).startswith("[SYN1]")


def test_viewer_floor_warning() -> None:
    labelling = _labelling(outlier_floor_um=10.0)
    report = _fit(labelling, _settings("serial_section"))
    assert any(
        "viewer's outlier floor was 10 um, this fit uses 30 um" in w
        for w in report.warnings
    )
    summary = fit_summary(
        report,
        store=_store(),
        settings=_settings("serial_section"),
        labelling=labelling,
    )
    assert summary["viewer_outlier_floor_um"] == 10
    same = _fit(labelling)
    assert not any("viewer's outlier floor" in w for w in same.warnings)
    assert not any("viewer's outlier floor" in w for w in _fit().warnings)


# --------------------------------------------------------------------------
# _validate_prepared_store
# --------------------------------------------------------------------------


def _validate(
    store: Path, **kwargs: Any
) -> tuple[StoreAttrs, LabellingAttrs, list[str]]:
    kwargs.setdefault("check_fingerprints", False)
    return _validate_prepared_store(store, run_id=RUN_ID, dataset_name="SYN1", **kwargs)


def test_valid_store_passes(tmp_path: Path) -> None:
    store, labelling, warnings = _validate(a2_store(tmp_path))
    assert store.kind == "prepared" and labelling.status == "finished"
    assert warnings == []


def test_missing_store_and_missing_attrs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manual_registration_prepare"):
        _validate(tmp_path / "absent.zarr")
    empty = fake_store(tmp_path / "empty.zarr", {})
    with pytest.raises(ValueError, match="not a manual-registration store"):
        _validate(empty)


def test_newer_schema_versions_are_refused(tmp_path: Path) -> None:
    attrs = store_attr_dict(
        A2_REFERENCE_SHAPE,
        A2_MOVING_SHAPE,
        A2_PIXEL_SIZE,
        A2_PIXEL_SIZE,
        schema_version=2,
    )
    labelling = labelling_dict(a2_pairs(), A2_COARSE)
    store = fake_store(
        tmp_path / "a.zarr", {STORE_ATTR: attrs, LABELLING_ATTR: labelling}
    )
    with pytest.raises(UnsupportedSchemaVersionError, match="newer") as excinfo:
        _validate(store)
    assert str(excinfo.value).startswith(f"[{RUN_ID}]")

    good_store = store_attr_dict(
        A2_REFERENCE_SHAPE, A2_MOVING_SHAPE, A2_PIXEL_SIZE, A2_PIXEL_SIZE
    )
    newer = labelling_dict(a2_pairs(), A2_COARSE, schema_version=2)
    store2 = fake_store(
        tmp_path / "b.zarr", {STORE_ATTR: good_store, LABELLING_ATTR: newer}
    )
    with pytest.raises(UnsupportedSchemaVersionError, match="newer") as excinfo:
        _validate(store2)
    assert str(excinfo.value).startswith(f"[{RUN_ID}]")

    before = read_root_attrs(store)
    with pytest.raises(UnsupportedSchemaVersionError):
        inject_labelling(store, a2_pairs())
    assert read_root_attrs(store) == before


def test_store_kind_dataset_and_labelling_state(tmp_path: Path) -> None:
    result = fake_store(
        tmp_path / "r.zarr",
        {
            STORE_ATTR: store_attr_dict(
                A2_REFERENCE_SHAPE,
                A2_MOVING_SHAPE,
                A2_PIXEL_SIZE,
                A2_PIXEL_SIZE,
                kind="result",
            )
        },
    )
    with pytest.raises(ValueError, match="result store"):
        _validate(result)
    with pytest.raises(ValueError, match="dataset"):
        _validate_prepared_store(
            a2_store(tmp_path),
            run_id=RUN_ID,
            dataset_name="SYN2",
            check_fingerprints=False,
        )
    bare = fake_store(
        tmp_path / "bare.zarr",
        {
            STORE_ATTR: store_attr_dict(
                A2_REFERENCE_SHAPE, A2_MOVING_SHAPE, A2_PIXEL_SIZE, A2_PIXEL_SIZE
            )
        },
    )
    with pytest.raises(ValueError, match="Finish labelling"):
        _validate(bare)
    draft = a2_store(tmp_path, status="draft", name="draft.zarr")
    with pytest.raises(ValueError, match="Finish labelling"):
        _validate(draft)
    _, labelling, _ = _validate(draft, require_finished=False)
    assert labelling.status == "draft"


def test_shape_mismatch_is_refused(tmp_path: Path) -> None:
    labelling = labelling_dict(a2_pairs(), A2_COARSE, ref_shape=(100, 100))
    store = fake_store(
        tmp_path / "s.zarr",
        {
            STORE_ATTR: store_attr_dict(
                A2_REFERENCE_SHAPE, A2_MOVING_SHAPE, A2_PIXEL_SIZE, A2_PIXEL_SIZE
            ),
            LABELLING_ATTR: labelling,
        },
    )
    with pytest.raises(ValueError, match="different reference image"):
        _validate(store)


def test_fingerprint_mismatch_and_content_sha_warning(tmp_path: Path) -> None:
    files = {}
    for role in ("reference", "moving"):
        path = tmp_path / f"{role}.ome.tif"
        path.write_bytes(b"x" * 100)
        files[role] = path
    attrs = store_attr_dict(
        A2_REFERENCE_SHAPE,
        A2_MOVING_SHAPE,
        A2_PIXEL_SIZE,
        A2_PIXEL_SIZE,
        ome_paths=files,
    )
    labelling = labelling_dict(a2_pairs(), A2_COARSE)
    store = fake_store(
        tmp_path / "f.zarr", {STORE_ATTR: attrs, LABELLING_ATTR: labelling}
    )
    _validate(store, check_fingerprints=True)
    files["moving"].write_bytes(b"y" * 100)
    with pytest.raises(ValueError, match="re-run prepare"):
        _validate(store, check_fingerprints=True)
    files["moving"].unlink()
    with pytest.raises(FileNotFoundError, match="re-run prepare"):
        _validate(store, check_fingerprints=True)

    wrong = dict(labelling, content_sha256="sha256:" + "0" * 64)
    store2 = fake_store(tmp_path / "w.zarr", {STORE_ATTR: attrs, LABELLING_ATTR: wrong})
    _, _, warnings = _validate(store2)
    assert any("content_sha256 differs" in w for w in warnings)


# --------------------------------------------------------------------------
# _landmark_records and friends
# --------------------------------------------------------------------------


def _report_for(pairs: Any) -> LandmarkFitReport:
    n = len(pairs)
    return LandmarkFitReport(
        robust=cast(Any, None),
        pairs=pairs,
        enabled=np.array([p.enabled for p in pairs]),
        outside_crop=np.zeros(n, dtype=bool),
        matrix_full=np.eye(3),
        warnings=[],
        fit_index=np.arange(n),
        used=np.array([p.enabled for p in pairs]),
        outlier=np.zeros(n, dtype=bool),
        residuals_um=np.zeros(n),
        loo_um=np.arange(n, dtype=float),
    )


MATRIX = np.array([[1.2, 0.3, 50.0], [-0.3, 1.2, 80.0], [0.0, 0.0, 1.0]])
WORKING = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])


def _flow() -> np.ndarray:
    ys, xs = np.mgrid[0:200, 0:260]
    return np.stack([3.0 * np.sin(xs / 40.0), 2.5 * np.cos(ys / 35.0)]).astype(
        np.float32
    )


def test_records_without_a_flow_equal_the_affine() -> None:
    moving = [(100.0, 100.0), (300.0, 200.0), (500.0, 400.0)]
    pairs = make_pairs(moving, [(10.0, 10.0)] * 3)
    records = _landmark_records(
        _report_for(pairs),
        reference_pixel_size_um=0.5,
        reference_origin_xy=(64, 128),
        moving_origin_xy=(0, 0),
        landmark_full=np.eye(3),
        final_affine_crop=MATRIX,
        flow_yx=None,
        flow_working_to_reference=None,
    )
    for record, (mx, my) in zip(records, moving, strict=True):
        affine = MATRIX @ [mx, my, 1.0]
        expected = [affine[0] + 64, affine[1] + 128]
        np.testing.assert_allclose(
            record["moving_in_reference_xy_px"]["affine"], expected
        )
        assert (
            record["moving_in_reference_xy_px"]["final"]
            == record["moving_in_reference_xy_px"]["affine"]
        )
        assert record["final_includes_flow"] is False
        assert record["push_converged"] is True
        assert record["residual_um"]["final"] == record["residual_um"]["affine"]
    assert records[2]["residual_um"]["loo"] == 2.0


def test_records_with_a_flow_push_in_the_crop_frame() -> None:
    inside = [(100.0, 100.0), (300.0, 200.0)]
    outside = [(5000.0, 5000.0)]
    moving = inside + outside
    pairs = make_pairs(moving, [(10.0, 10.0)] * 3)
    flow = _flow()
    records = _landmark_records(
        _report_for(pairs),
        reference_pixel_size_um=0.5,
        reference_origin_xy=(64, 128),
        moving_origin_xy=(0, 0),
        landmark_full=np.eye(3),
        final_affine_crop=MATRIX,
        flow_yx=flow,
        flow_working_to_reference=WORKING,
    )
    pushed, _ = push_points_through_registration(
        np.asarray(inside),
        final_matrix=MATRIX,
        flow_yx=flow,
        working_to_reference=WORKING,
    )
    for record, expected in zip(records[:2], pushed + [64, 128], strict=True):
        np.testing.assert_allclose(
            record["moving_in_reference_xy_px"]["final"], expected, atol=1e-9
        )
        assert record["final_includes_flow"] is True
        assert (
            record["moving_in_reference_xy_px"]["final"]
            != (record["moving_in_reference_xy_px"]["affine"])
        )
    far = records[2]
    assert far["final_includes_flow"] is False
    np.testing.assert_allclose(
        far["moving_in_reference_xy_px"]["final"],
        far["moving_in_reference_xy_px"]["affine"],
    )


def test_records_with_a_moving_origin_use_the_crop_frame() -> None:
    # Moving crop origin (448, 64) and reference crop origin (192, 320): the push
    # runs on the moving points shifted into the moving crop frame, and the result
    # is shifted back by the reference origin.
    origin_r, origin_m = (192, 320), (448, 64)
    # The last point lies inside the flow grid in the crop frame only: without
    # the moving origin subtracted it would fall beyond the grid's right edge.
    edge = np.linalg.inv(MATRIX) @ np.array([600.0, 300.0, 1.0])
    crop_points = np.array([[100.0, 100.0], [300.0, 200.0], [5000.0, 5000.0], edge[:2]])
    assert (MATRIX @ [*(edge[:2] + origin_m), 1.0])[0] > 4 * 259 + 1.5
    pairs = make_pairs(crop_points + origin_m, [(10.0, 10.0)] * 4)
    flow = _flow()
    kwargs: dict[str, Any] = {
        "reference_pixel_size_um": 0.5,
        "reference_origin_xy": origin_r,
        "moving_origin_xy": origin_m,
        "landmark_full": np.eye(3),
        "final_affine_crop": MATRIX,
    }
    records = _landmark_records(
        _report_for(pairs), flow_yx=flow, flow_working_to_reference=WORKING, **kwargs
    )
    inside = [0, 1, 3]
    pushed, _ = push_points_through_registration(
        crop_points[inside],
        final_matrix=MATRIX,
        flow_yx=flow,
        working_to_reference=WORKING,
    )
    affine = crop_points @ MATRIX[:2, :2].T + MATRIX[:2, 2] + origin_r
    for i, record in enumerate(records):
        np.testing.assert_allclose(
            record["moving_in_reference_xy_px"]["affine"], affine[i], atol=1e-9
        )
    for k, i in enumerate(inside):
        np.testing.assert_allclose(
            records[i]["moving_in_reference_xy_px"]["final"],
            pushed[k] + origin_r,
            atol=1e-9,
        )
        assert records[i]["final_includes_flow"] is True
        # Pulling the recorded position back (crop frame) returns the point.
        back = pull_points_through_registration(
            np.asarray([records[i]["moving_in_reference_xy_px"]["final"]]) - origin_r,
            final_matrix=MATRIX,
            flow_yx=flow,
            working_to_reference=WORKING,
        )
        np.testing.assert_allclose(back[0], crop_points[i], atol=1e-3)
    assert records[2]["final_includes_flow"] is False
    np.testing.assert_allclose(
        records[2]["moving_in_reference_xy_px"]["final"],
        records[2]["moving_in_reference_xy_px"]["affine"],
    )
    # Without a flow the final position is the affine one, in full reference px.
    plain = _landmark_records(
        _report_for(pairs), flow_yx=None, flow_working_to_reference=None, **kwargs
    )
    for i, record in enumerate(plain):
        np.testing.assert_allclose(
            record["moving_in_reference_xy_px"]["final"], affine[i], atol=1e-9
        )


def test_residual_summary_groups_and_empty_groups() -> None:
    def record(enabled: bool, used: bool, value: float) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "used_in_fit": used,
            "residual_um": {"landmark": value, "affine": value, "final": value / 2},
        }

    summary = _residual_summary(
        [record(True, True, 2.0), record(True, True, 4.0), record(False, False, 10.0)]
    )
    assert summary["used"]["landmark"] == {
        "n": 2,
        "median": 3.0,
        "p90": pytest.approx(3.8),
        "max": 4.0,
    }
    assert summary["held_out"]["final"]["median"] == 5.0
    empty = _residual_summary([record(True, True, 1.0)])
    assert empty["held_out"]["landmark"] == {
        "n": 0,
        "median": None,
        "p90": None,
        "max": None,
    }


def test_provenance_digest_depends_on_its_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    store = _store()
    payload = labelling_dict(a2_pairs(), A2_COARSE)
    sha = labelling_content_sha256(payload)

    def digest(**overrides: Any) -> str:
        kwargs: dict[str, Any] = {
            "preset": "same_section",
            "run_label": None,
            "settings": settings,
            "labelling_sha": sha,
            "store": store,
        }
        kwargs.update(overrides)
        return _provenance_digest(**kwargs)

    base = digest()
    assert base == digest()
    moved = a2_pairs()
    moved[1] = moved[1].model_copy(update={"reference_xy_px": (46117.596, 48370.654)})
    moved_sha = labelling_content_sha256(labelling_dict(moved, A2_COARSE))
    assert digest(labelling_sha=moved_sha) != base
    assert digest(run_label="x") != base
    monkeypatch.setitem(register_module.PRESET_VERSIONS, "same_section", "v2")
    assert digest() != base


# Digests computed on the commit before the field settings existed (a fixed
# synthetic input: the A.2 labelling and store). They must not move: the older
# presets' results stay valid after the upgrade (plan D10).
PINNED_PROVENANCE_DIGESTS = {
    "same_section": (
        "sha256:7a96957247462bc5e0e685a7f95c919446646cc363c802eb687f7e822266c465"
    ),
    "serial_section": (
        "sha256:717fec80e1f7ee6a6ee575e0ca19d2f7c6615142b96ba6b5ef77905537934f17"
    ),
}
PINNED_PROVENANCE_DIGESTS_OVERRIDDEN = {
    "same_section": (
        "sha256:fbda9a608ee8132171031891f276ecd2fd5fc8dae6c7af55f1599b576d7b2fa7"
    ),
    "serial_section": (
        "sha256:0436f758e7c0f895d35a8bb555b30325015e0aea331c8f04dfe0e7bea154be61"
    ),
}


@pytest.mark.parametrize("preset", ["same_section", "serial_section"])
def test_provenance_digest_of_the_older_presets_is_unchanged(preset: str) -> None:
    sha = labelling_content_sha256(labelling_dict(a2_pairs(), A2_COARSE))
    digest = _provenance_digest(
        preset=preset,
        run_label=None,
        settings=_settings(preset),
        labelling_sha=sha,
        store=_store(),
    )
    assert digest == PINNED_PROVENANCE_DIGESTS[preset]
    overridden = _provenance_digest(
        preset=preset,
        run_label="x",
        settings=_settings(preset, "landmark_outlier_k=4"),
        labelling_sha=sha,
        store=_store(),
    )
    assert overridden == PINNED_PROVENANCE_DIGESTS_OVERRIDDEN[preset]


def test_provenance_settings_drop_only_field_keys_at_their_defaults() -> None:
    from merxen.manual_registration.config import FIELD_SETTINGS_KEYS
    from merxen.manual_registration.register import _provenance_settings

    plain = _provenance_settings(_settings("same_section"))
    assert not set(plain) & set(FIELD_SETTINGS_KEYS)
    changed = _provenance_settings(
        _settings("same_section", "field_outlier_p=0.02;materialize=preview")
    )
    assert changed["field_outlier_p"] == 0.02
    assert changed["materialize"] == "preview"
    assert "field_kernel" not in changed


def test_provenance_digest_of_a_field_run_sees_the_roles_and_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from merxen.manual_registration.image_evidence import ChannelRoles

    sha = labelling_content_sha256(labelling_dict(a2_pairs(), A2_COARSE))
    settings = _settings("cortex_serial")
    store = _store()

    def digest(roles: ChannelRoles | None, settings: Any = settings) -> str:
        return _provenance_digest(
            preset="cortex_serial",
            run_label=None,
            settings=settings,
            labelling_sha=sha,
            store=store,
            roles=roles,
        )

    one = ChannelRoles(
        reference={"at8": "AF568", "map2": "AF647", "dapi": "DAPI"},
        moving={"at8": "AF568", "map2": "AF647", "dapi": "DAPI"},
    )
    two = ChannelRoles(reference=one.reference, moving={**one.moving, "at8": "X"})
    assert digest(one) == digest(one)
    assert digest(one) != digest(two)
    assert digest(one) != digest(None)
    other = _settings("cortex_serial", "field_outlier_p=0.02")
    assert digest(one, other) != digest(one)
    # A field store written by an older layout (v1: empty moving pyramid) is not
    # reused, whatever the preset.
    current = digest(one)
    monkeypatch.setattr(register_module, "FIELD_STORE_ALGORITHM", "field_store_v1")
    assert digest(one) != current


def _run_config(tmp_path: Path, **overrides: Any) -> ManualRegistrationRunConfig:
    values: dict[str, Any] = {
        "dataset_name": "SYN1",
        "preset": "same_section",
        "prepared_store_path": tmp_path / "SYN1_prepared.zarr",
        "result_dir": tmp_path / "result",
        "output_dir": tmp_path / "out",
    }
    values.update(overrides)
    return ManualRegistrationRunConfig(**values)


@pytest.fixture
def read_store_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make step 1 of register return the A.2 store and labelling (no images)."""
    payload = labelling_dict(a2_pairs(), A2_COARSE)

    def stub(*_args: Any, **_kwargs: Any) -> Any:
        return _store(), parse_labelling_attrs(payload), payload, []

    monkeypatch.setattr(register_module, "_read_prepared_store", stub)


@pytest.mark.parametrize(
    ("row_settings", "message"),
    [
        ("materialize=preview", "only for robust-field runs"),
        ("field_model=robust_gp;field_image_check=false", "skip the DAPI refinement"),
    ],
)
def test_register_checks_the_cross_field_rules(
    tmp_path: Path, read_store_stub: None, row_settings: str, message: str
) -> None:
    config = _run_config(tmp_path, row_settings=row_settings)
    with pytest.raises(ValueError, match=message):
        run_manual_registration(config)


def test_register_cortex_serial_needs_channel_roles(
    tmp_path: Path, read_store_stub: None
) -> None:
    config = _run_config(tmp_path, preset="cortex_serial")
    with pytest.raises(ValueError, match="preset cortex_serial needs channel roles"):
        run_manual_registration(config)


def test_register_reports_a_bad_roles_text_before_anything_else(
    tmp_path: Path, read_store_stub: None
) -> None:
    config = _run_config(
        tmp_path, preset="cortex_serial", channel_roles="at8=AF568;map2=NOPE;dapi=DAPI"
    )
    with pytest.raises(ValueError, match="channel roles: channel 'NOPE'"):
        run_manual_registration(config)


def test_plain_json_and_publish(tmp_path: Path) -> None:
    value = {
        "a": np.float32(1.5),
        "b": np.int64(3),
        "c": float("nan"),
        "d": (1, 2),
        "e": np.array([1.0, np.inf]),
        "f": tmp_path,
        "g": np.bool_(True),
    }
    plain = _plain_json(value)
    assert plain == {
        "a": 1.5,
        "b": 3,
        "c": None,
        "d": [1, 2],
        "e": [1.0, None],
        "f": str(tmp_path),
        "g": True,
    }
    assert type(plain["g"]) is bool
    source = tmp_path / "a.txt"
    source.write_text("hello")
    target = _publish(source, tmp_path / "sub" / "b.txt")
    assert target.read_text() == "hello"
    assert [p.name for p in target.parent.iterdir()] == ["b.txt"]


# --------------------------------------------------------------------------
# reuse, core QC files and leftover stores
# --------------------------------------------------------------------------

DIGEST_NEW = "sha256:" + "1" * 64
DIGEST_OLD = "sha256:" + "2" * 64


def _stored_result(
    directory: Path,
    *,
    durable_digest: str | None = DIGEST_NEW,
    durable_status: str = "complete",
) -> dict[str, Any]:
    outputs: dict[str, str] = {}
    for key, name in (
        ("landmark_matrix_csv", "lm.csv"),
        ("final_matrix_csv", "final.csv"),
        ("landmarks_csv", "landmarks.csv"),
        ("summary_json", "summary.json"),
    ):
        path = directory / name
        path.write_text("x\n")
        outputs[key] = str(path)
    if durable_digest is not None:
        Path(outputs["summary_json"]).write_text(
            json.dumps({"status": durable_status, "provenance_digest": durable_digest})
        )
    return {"status": "complete", "provenance_digest": DIGEST_NEW, "outputs": outputs}


def test_reusable_needs_the_durable_summary_of_the_same_run(tmp_path: Path) -> None:
    assert _reusable(None, DIGEST_NEW) is False
    stored = _stored_result(tmp_path)
    assert _reusable(stored, DIGEST_NEW) is True
    assert _reusable(stored, DIGEST_OLD) is False
    assert _reusable({**stored, "status": "failed"}, DIGEST_NEW) is False


@pytest.mark.parametrize(
    ("durable_digest", "durable_status"),
    [
        (DIGEST_OLD, "complete"),  # the publish after the swap never ran
        (DIGEST_NEW, "failed"),
        (None, "complete"),  # not a JSON object
    ],
)
def test_reusable_refuses_a_stale_or_broken_durable_summary(
    tmp_path: Path, durable_digest: str | None, durable_status: str
) -> None:
    stored = _stored_result(
        tmp_path, durable_digest=durable_digest, durable_status=durable_status
    )
    assert _reusable(stored, DIGEST_NEW) is False


def test_reusable_refuses_a_missing_output_file(tmp_path: Path) -> None:
    stored = _stored_result(tmp_path)
    Path(stored["outputs"]["final_matrix_csv"]).unlink()
    assert _reusable(stored, DIGEST_NEW) is False
    stored = _stored_result(tmp_path)
    Path(stored["outputs"]["summary_json"]).unlink()
    assert _reusable(stored, DIGEST_NEW) is False


def _list_core_qc(
    stored: dict[str, Any], directory: Path, *keys: str
) -> dict[str, Path]:
    """List core QC files on ``stored``, written before its durable summary."""
    summary = Path(stored["outputs"]["summary_json"])
    before = summary.stat().st_mtime_ns - 5_000_000_000
    listed: dict[str, Path] = {}
    for key in keys:
        path = directory / f"{key}.bin"
        path.write_bytes(b"x")
        os.utime(path, ns=(before, before))
        listed[key] = path
    stored["outputs"]["core_qc"] = {key: str(path) for key, path in listed.items()}
    return listed


def test_reusable_refuses_a_listed_core_qc_file_that_is_missing(
    tmp_path: Path,
) -> None:
    # A run that failed in the core deleted the files of the result it left current.
    stored = _stored_result(tmp_path)
    assert _reusable(stored, DIGEST_NEW) is True  # no core_qc key: nothing to check
    listed = _list_core_qc(stored, tmp_path, "registration_qc_png", "nonrigid_flow_npz")
    assert _reusable(stored, DIGEST_NEW) is True
    listed["nonrigid_flow_npz"].unlink()
    assert _reusable(stored, DIGEST_NEW) is False
    stored["outputs"]["core_qc"] = {}
    assert _reusable(stored, DIGEST_NEW) is True


def test_reusable_refuses_a_listed_core_qc_file_written_after_the_summary(
    tmp_path: Path,
) -> None:
    # A run with other inputs that failed after the core rewrote the listed files
    # in place (same names, other content) and left the store and the summary.
    stored = _stored_result(tmp_path)
    listed = _list_core_qc(stored, tmp_path, "registration_qc_png", "nonrigid_flow_npz")
    summary = Path(stored["outputs"]["summary_json"])
    assert _reusable(stored, DIGEST_NEW) is True
    flow = listed["nonrigid_flow_npz"]
    flow.write_bytes(b"other run")  # now: later than the summary
    assert flow.stat().st_mtime_ns > summary.stat().st_mtime_ns
    assert _reusable(stored, DIGEST_NEW) is False
    # The same instant is not later: a complete run keeps its files and the summary
    # (copied with its modification time) side by side.
    stat = summary.stat()
    os.utime(flow, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert _reusable(stored, DIGEST_NEW) is True
    # A newer summary (a later, complete run) makes older files current again.
    os.utime(flow, ns=(stat.st_atime_ns, stat.st_mtime_ns - 1))
    os.utime(summary, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert _reusable(stored, DIGEST_NEW) is True


def _touch_core_qc(qc_paths: dict[str, Path], *keys: str) -> None:
    for key in keys:
        qc_paths[key].write_bytes(b"x")
        if qc_paths[key].suffix == ".png":
            qc_paths[key].with_suffix(".pdf").write_bytes(b"x")


def test_core_qc_files_of_an_earlier_run_are_removed(tmp_path: Path) -> None:
    qc_paths = registration_output_paths(
        tmp_path, "SYN1_same_section", "moving_registered"
    )
    other = registration_output_paths(
        tmp_path, "SYN1_serial_section", "moving_registered"
    )
    keys = ("qc_plot", "local_shifts", "nonrigid_flow", "nonrigid_qc_plot")
    _touch_core_qc(qc_paths, *keys)
    _touch_core_qc(other, *keys)
    _remove_core_qc_files(qc_paths)
    left = {path.name for path in tmp_path.iterdir()}
    assert left == {path.name for key in keys for path in (other[key],)} | {
        other["qc_plot"].with_suffix(".pdf").name,
        other["nonrigid_qc_plot"].with_suffix(".pdf").name,
    }
    _remove_core_qc_files(qc_paths)  # nothing left: still fine


def _outcome(*, measured: bool, flow: bool) -> Any:
    return SimpleNamespace(
        refinement=SimpleNamespace(
            before=object() if measured else None,
            working_to_reference=np.eye(3) if measured else None,
        ),
        nonrigid=SimpleNamespace(
            result=object() if flow else None,
            working_to_reference=np.eye(3) if flow else None,
        ),
    )


def test_core_qc_is_listed_from_the_outcome_not_from_the_disk(tmp_path: Path) -> None:
    qc_paths = registration_output_paths(
        tmp_path, "SYN1_same_section", "moving_registered"
    )
    _touch_core_qc(
        qc_paths, "qc_plot", "local_shifts", "nonrigid_flow", "nonrigid_qc_plot"
    )
    # A flow that was not computed this run: the leftover files are not listed.
    listing = _core_qc_listing(qc_paths, _outcome(measured=True, flow=False))
    assert set(listing) == {"registration_qc_png", "registration_local_shifts_csv"}
    assert listing["registration_qc_png"] == str(qc_paths["qc_plot"])
    # Nothing measured and no flow: nothing is listed although the files exist.
    assert _core_qc_listing(qc_paths, _outcome(measured=False, flow=False)) == {}
    both = _core_qc_listing(qc_paths, _outcome(measured=True, flow=True))
    assert set(both) == {
        "registration_qc_png",
        "registration_local_shifts_csv",
        "nonrigid_flow_npz",
        "nonrigid_qc_png",
    }
    # Claimed by the outcome but missing on disk: not listed either.
    qc_paths["nonrigid_flow"].unlink()
    assert "nonrigid_flow_npz" not in _core_qc_listing(
        qc_paths, _outcome(measured=True, flow=True)
    )


def test_leftover_superseded_stores_are_removed(tmp_path: Path) -> None:
    final = tmp_path / "SYN1_same_section_registered.zarr"
    final.mkdir()
    old = final.with_name(final.name + ".superseded-20260101T000000Z")
    older = final.with_name(final.name + ".superseded-20250101T000000Z-2")
    for path in (old, older):
        path.mkdir()
        (path / "zarr.json").write_text("{}")
    kept = final.with_name(final.name + ".superseded-20260202T000000Z")
    kept.mkdir()
    other_run = (
        tmp_path / "SYN1_serial_section_registered.zarr.superseded-20260101T000000Z"
    )
    other_run.mkdir()
    _remove_superseded_siblings(final, keep=kept)
    assert not old.exists() and not older.exists()
    assert kept.exists() and other_run.exists() and final.exists()
    _remove_superseded_siblings(final)
    assert not kept.exists() and other_run.exists()


# --------------------------------------------------------------------------
# field-run additions: store attr, records, reuse without a store
# --------------------------------------------------------------------------


def _result_attr(**kwargs: Any) -> dict[str, Any]:
    return _result_store_attr(
        store=_store(),
        config=_run_config(Path("/unused")),
        store_path=Path("/unused/store.zarr"),
        crop_reference=CropRect(x0=64, y0=128, x1=1088, y1=1152),
        crop_moving=CropRect(x0=0, y0=0, x1=512, y1=512),
        kept_reference_names=["DAPI"],
        **kwargs,
    )


def test_result_store_attr_without_materialization_is_unchanged() -> None:
    attr = _result_attr()
    assert "materialization" not in attr["images"]["reference"]
    assert attr["images"]["moving"]["shape_yx"] == [1024, 1024]


def test_result_store_attr_records_the_materialization_and_element_shape() -> None:
    record = Materialization(mode="preview", level_factor=16, pixel_size_um=5.52)
    attr = _result_attr(materialization=record, element_shape_yx=(64, 64))
    for role in ("reference", "moving"):
        image = attr["images"][role]
        assert image["materialization"] == {
            "mode": "preview",
            "level_factor": 16,
            "pixel_size_um": 5.52,
        }
        assert image["shape_yx"] == [64, 64]
    # The record keeps the reference level-0 pixel size and the crop (contract C4).
    assert attr["images"]["reference"]["pixel_size_um"] == A2_PIXEL_SIZE
    assert attr["images"]["reference"]["crop"]["x0"] == 64
    parsed = parse_store_attrs(attr)
    assert parsed.images["moving"].materialization == record


def test_similarity_records_have_no_field_and_the_csv_columns_stay_empty() -> None:
    pairs = make_pairs([(100.0, 100.0)], [(10.0, 10.0)])
    records = _landmark_records(
        _report_for(pairs),
        reference_pixel_size_um=0.5,
        reference_origin_xy=(0, 0),
        moving_origin_xy=(0, 0),
        landmark_full=np.eye(3),
        final_affine_crop=np.eye(3),
        flow_yx=None,
        flow_working_to_reference=None,
    )
    assert records[0]["field"] is None
    frame = _landmarks_dataframe(records)
    assert list(frame.columns[-4:]) == [
        "field_weight",
        "field_p_value",
        "field_verdict",
        "field_class",
    ]
    assert frame.iloc[0, -4:].isna().all()
    # A record written before M1 has no field key at all.
    del records[0]["field"]
    assert _landmarks_dataframe(records).iloc[0, -4:].isna().all()


def test_landmarks_csv_columns_carry_the_field_values() -> None:
    pairs = make_pairs([(100.0, 100.0)], [(10.0, 10.0)])
    records = _landmark_records(
        _report_for(pairs),
        reference_pixel_size_um=0.5,
        reference_origin_xy=(0, 0),
        moving_origin_xy=(0, 0),
        landmark_full=np.eye(3),
        final_affine_crop=np.eye(3),
        flow_yx=None,
        flow_working_to_reference=None,
    )
    records[0]["field"] = {
        "weight": None,
        "p_value": 0.25,
        "clicks_only": None,
        "verdict": "confirmed",
        "class": "ok",
        "loo_vector_moving_um": [1.0, 2.0],
    }
    row = _landmarks_dataframe(records).iloc[0]
    assert row["field_p_value"] == 0.25
    assert (row["field_verdict"], row["field_class"]) == ("confirmed", "ok")
    assert row["field_weight"] is None or np.isnan(row["field_weight"])


def test_a_run_without_a_store_is_read_from_its_durable_summary(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    assert read_stored_result(tmp_path / "missing.zarr", summary_json=summary) is None
    summary.write_text("not json")
    assert read_stored_result(tmp_path / "missing.zarr", summary_json=summary) is None
    summary.write_text(json.dumps({"schema": "other"}))
    assert read_stored_result(tmp_path / "missing.zarr", summary_json=summary) is None
    summary.write_text(
        json.dumps(
            {"schema": "merxen.manual_registration.result", "status": "complete"}
        )
    )
    stored = read_stored_result(tmp_path / "missing.zarr", summary_json=summary)
    assert stored is not None and stored["status"] == "complete"


def test_reuse_of_a_field_result_needs_its_model_and_flow_files(
    tmp_path: Path,
) -> None:
    stored = _stored_result(tmp_path)
    for key, name in (
        ("landmark_field_json", "model.json"),
        ("field_flow_npz", "flow.npz"),
    ):
        path = tmp_path / name
        path.write_text("x")
        # Durable files of a summary-only result must not be newer than its summary.
        os.utime(path, (1_000_000_000, 1_000_000_000))
        stored["outputs"][key] = str(path)
    assert _reusable(stored, DIGEST_NEW) is True
    (tmp_path / "flow.npz").unlink()
    assert _reusable(stored, DIGEST_NEW) is False
    (tmp_path / "flow.npz").write_text("x")
    os.utime(tmp_path / "flow.npz", (1_000_000_000, 1_000_000_000))
    (tmp_path / "model.json").unlink()
    assert _reusable(stored, DIGEST_NEW) is False


def test_outputs_from_stored_lists_the_field_files_and_no_store_for_none(
    tmp_path: Path,
) -> None:
    stored = _stored_result(tmp_path)
    stored["outputs"]["landmark_field_json"] = str(tmp_path / "model.json")
    stored["outputs"]["field_flow_npz"] = str(tmp_path / "flow.npz")
    stored["outputs"]["field_qc_png"] = str(tmp_path / "qc.png")
    paths = _outputs_from_stored(stored, RUN_ID, tmp_path / "out")
    assert "result_store" not in paths
    assert paths["landmark_field_json"] == tmp_path / "model.json"
    assert paths["field_flow_npz"] == tmp_path / "flow.npz"
    assert paths["field_qc_png"] == tmp_path / "qc.png"
    assert (
        paths["task_summary_json"].name == f"{RUN_ID}_manual_registration_summary.json"
    )
